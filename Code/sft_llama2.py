# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Fine-Tune Llama2-7b on SE paired dataset
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    is_torch_npu_available,
    is_torch_xpu_available,
    set_seed,
)

from trl import SFTConfig, SFTTrainer
from trl.trainer import ConstantLengthDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

@dataclass
class ScriptArguments:
    """
    存储脚本的参数：
    模型名称，路径，可改为本地路径
    数据集名称，路径，下载后可改为本地路径
    特定的子目录，可修改
    size_valid_set
    seq_length
    以上两个参数可根据GPU情况适当调整
    """
    model_name: Optional[str] = field(default="/mnt/sdd/OpenSourceLLM/Model/Llama-2-7b-hf", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(default="/mnt/sdd/OpenSourceLLM/Dataset/stack-exchange-paired", metadata={"help": "the dataset name"})
    subset: Optional[str] = field(default="data/finetune", metadata={"help": "the subset to use"})
    split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    size_valid_set: Optional[int] = field(default=4000, metadata={"help": "the size of the validation set"})
    streaming: Optional[bool] = field(default=True, metadata={"help": "whether to stream the dataset"})
    shuffle_buffer: Optional[int] = field(default=5000, metadata={"help": "the shuffle buffer size"})
    seq_length: Optional[int] = field(default=1024, metadata={"help": "the sequence length"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})
    use_bnb: Optional[bool] = field(default=True, metadata={"help": "whether to use BitsAndBytes"})

    # LoraConfig
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

"""
参数解析和LoRA配置:
"""
parser = HfArgumentParser((ScriptArguments, SFTConfig))
script_args, training_args = parser.parse_args_into_dataclasses()
peft_config = LoraConfig(
    r=script_args.lora_r,
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

"""
配置检查: 确保不能同时使用group_by_length和packing两个参数
group_by_length 按输入序列的长度对数据进行分组，以减少序列的填充（padding），从而提高训练效率。
packing 则是将较短的序列合并成一个固定长度的批次，以充分利用序列长度。
"""
if training_args.group_by_length and training_args.packing:
    raise ValueError("Cannot use both packing and group by length")

"""
配置检查: gradient_checkpointing 是否为 True
gradient_checkpointing 是一种在训练中减少内存消耗的技术，通过延迟部分计算，
将一些中间结果重新计算以节省显存。然而，在此代码中并不支持此功能（会引发回溯错误）。
"""
# `gradient_checkpointing` was True by default until `1f3314`, but it's actually not used.
# `gradient_checkpointing=True` will cause `Variable._execution_engine.run_backward`.
if training_args.gradient_checkpointing:
    raise ValueError("gradient_checkpointing not supported")

"""
设置随机种子，保证实验可重复性
"""
set_seed(training_args.seed)


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    计算数据集中字符与 token（单词或子词）之间的平均比率，即每个 token 包含多少字符。
    通过知道字符与 token 的比例，可以更好地控制输入序列的长度。
    返回值：字符与 token 的平均比率，用于后续数据格式的控制。
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    计算并打印模型中可训练参数和总参数的数量，以及可训练参数占当模型仅针总参数的百分比。
    这种统计在微调或训练深度学习模型时很常用，特别是对一部分参数进行微调时
    （如低秩自适应 LoRA 训练）。
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_sample_text(example):
    """
    Prepare the text from a sample of the dataset.
    用于将数据集中的一个样本转换为模型输入的标准格式文本。
    它的具体作用是将数据集中的问题和回答字段提取并格式化，以便统一输入格式。
    """
    text = f"Question: {example['question']}\n\nAnswer: {example['response_j']}"
    return text


def create_datasets(tokenizer, args, seed=None):
    """
    用于加载和准备训练与验证数据集，以便用于模型微调。
    """
    dataset = load_dataset(
        args.dataset_name,
        data_dir=args.subset,
        split=args.split,
        # use_auth_token=True,
        num_proc=args.num_workers if not args.streaming else None,
        streaming=args.streaming,
    )
    if args.streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(args.size_valid_set)
        train_data = dataset.skip(args.size_valid_set)
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=seed)
    else:
        dataset = dataset.train_test_split(test_size=0., seed=seed)
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset

"""
加载并量化基础模型 (base_model) 以便在显存有限的设备上更高效地运行。
"""
bnb_config = None
if script_args.use_bnb:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

base_model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=bnb_config,
    device_map={"": Accelerator().local_process_index},
    trust_remote_code=True,
    # use_auth_token=True,
)
base_model.config.use_cache = False

"""
加载分词器并设置填充参数
"""
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

"""
创建训练和验证数据集
"""
train_dataset, eval_dataset = create_datasets(tokenizer, script_args, seed=training_args.seed)

"""
初始化 SFTTrainer 进行模型微调
"""
trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    max_seq_length=None,
    formatting_func=prepare_sample_text,
    tokenizer=tokenizer,
    # processing_class=tokenizer,
    args=training_args,
)

"""
训练和保存模型
"""
trainer.train()
trainer.save_model(training_args.output_dir)


"""
保存最终 checkpoint 并释放内存
"""
output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)

# Free memory for merging weights
del base_model
if is_torch_xpu_available():
    torch.xpu.empty_cache()
elif is_torch_npu_available():
    torch.npu.empty_cache()
else:
    torch.cuda.empty_cache()

"""
合并微调权重并保存最终模型
"""
model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
model = model.merge_and_unload()

output_merged_dir = os.path.join(training_args.output_dir, "final_merged_checkpoint")
model.save_pretrained(output_merged_dir, safe_serialization=True)
