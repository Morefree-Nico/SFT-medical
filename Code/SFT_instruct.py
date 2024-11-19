from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from trl.trainer import ConstantLengthDataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    set_seed,
)
from dataclasses import dataclass, field
from typing import Optional
from peft import AutoPeftModelForCausalLM, LoraConfig
import os
import torch
from accelerate import Accelerator

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

@dataclass
class Arguments:
    # 加载数据集和模型 参数
    model_name:Optional[str] = field(default="/mnt/sdb_mount/daixinwei/LLM/Model/Llama-3.2-1B-Instruct/",metadata={"help":"模型路径"})
    dataset_name:Optional[str] = field(default="/mnt/sdb_mount/daixinwei/LLM/Dataset/medical_llama2_instruct_dataset_short",metadata={"help":"数据集路径"})
    split: Optional[str] = field(default="train", metadata={"help": "加载数据集"})
    size_valid_set: Optional[int] = field(default=200, metadata={"help": "验证集大小"})
    size_test_set: Optional[int] = field(default=200, metadata={"help": "测试集大小, 会保存"})
    streaming: Optional[bool] = field(default=True, metadata={"help": "是否流式加载数据集"})
    shuffle_buffer: Optional[int] = field(default=2000, metadata={"help": "数据集的混洗缓冲区大小"})
    seq_length: Optional[int] = field(default=1024, metadata={"help": "限制输入文本的最大长度"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "并行处理的进程数"})
    use_bnb: Optional[bool] = field(default=True, metadata={"help": "是否启用量化"})
    test_set_path:Optional[str] = field(default="/mnt/sdb_mount/daixinwei/SFT/SFT-medical/Dataset_test",metadata={"help":"测试集保存的类路径"})
    torch_dtype: Optional[str] = field(default="auto", metadata={"help": "加载模型时的dtype"}) # torch.bfloat16
    
    # LORA 参数
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "LORA缩放因子"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "LoRA 层的 dropout 概率"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "LoRA 中低秩矩阵的秩"})
    
    # 训练过程 SFTConfig 参数
    output_dir:Optional[str] = field(default="/mnt/sdb_mount/daixinwei/SFT/SFT-medical/Output",metadata={"help":"训练结果输出路径"})
    max_steps: Optional[int] = field(default=800, metadata={"help": "最大训练步数"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "日志记录的步数间隔"})
    save_steps: Optional[int] = field(default=50, metadata={"help": "模型保存的步数间隔"})
    per_device_train_batch_size: Optional[int] = field(default=2, metadata={"help": "每个设备上的训练批次大小"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "每个设备上的验证批次大小"})
    gradient_accumulation_steps: Optional[int] = field(default=1, metadata={"help": "梯度累积步数"})
    gradient_checkpointing: Optional[bool] = field(default=False, metadata={"help": "是否启用梯度检查点"})
    group_by_length: Optional[bool] = field(default=False, metadata={"help": "是否按长度分组批次"})
    learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "学习率"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "学习率调度类型即学习率随训练过程的变化方式"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "学习率预热步数"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "权重衰减率"})
    optim: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "优化器类型"})
    bf16: Optional[bool] = field(default=False, metadata={"help": "是否启用 bfloat16 精度"})
    fp16: Optional[bool] = field(default=True, metadata={"help": "是否启用 fp16 16-bit 混合精度"})
    remove_unused_columns: Optional[bool] = field(default=False, metadata={"help": "是否移除未使用的列"})
    run_name: Optional[str] = field(default="sft_instruct", metadata={"help": "实验运行名称"})
    report_to: Optional[str] = field(default="wandb", metadata={"help": "日志报告平台"})
    seed: Optional[int] = field(default=42, metadata={"help": "随机种子"})

    
parser = HfArgumentParser(Arguments)
args = parser.parse_args_into_dataclasses()[0]

# 保证实验可重复性
set_seed(args.seed)

def prepare_sample_text(example):
    """
    用于将数据集中的一个样本转换为模型输入的标准格式文本，并加上系统信息（instruction）。
    """
    # 系统信息，告诉模型它的身份和任务
    instruction = "Instruction: You are a medical professional. Answer the question truthfully."

    # 将系统信息和实际问题与答案拼接
    text = f"{instruction}\n\nQuestion: {example['input']}\n\nAnswer: {example['output']}"
    
    return text


def chars_token_radio(dataset, tokenizer, number_examples=200):
    """
    计算一个每个token包含几个字符, 这个参数ConstantLengthDataset()需要, 用于提前分配空间
    """
    # 初始化 字符和token数量
    total_characters = 0
    total_tokens = 0

    # 循环200条，样本计算字符数量和token数量
    for _, example in tqdm(zip(range(number_examples), iter(dataset)), total=number_examples):
        text = prepare_sample_text(example=example)
        total_characters = total_characters + len(text)
        if tokenizer.is_fast:
            total_tokens = total_tokens + len(tokenizer(text).tokens())
        else:
            total_tokens = total_tokens + len(tokenizer.tokenize(text))

    # 计算比率
    radio = total_characters / total_tokens

    return radio

def save_streaming_subset(subset, args):
    """
    保存流式加载的数据集到本地。
    """
    # 将流式数据集的样本逐条加载到内存
    samples = [sample for sample in subset]
    
    # 转换为标准 `Dataset` 格式
    saved_dataset = Dataset.from_dict({k: [sample[k] for sample in samples] for k in samples[0].keys()})
    
    # 保存到指定路径
    saved_dataset.save_to_disk(args.test_set_path)
    print(f"测试集成功保存到 {args.test_set_path}")

def Create_datasets(tokenizer, args, seed=None):
    """
    加载和处理数据集
    """
    dataset = load_dataset(
        path=args.dataset_name,
        split=args.split,
        streaming=args.streaming,
        num_proc=args.num_workers if not args.streaming else None,
    )
    # 如果支持流式加载数据
    if args.streaming:
        print("---------正在以流式加载和保存数据集---------")
        test_data = dataset.take(args.size_test_set) # 需要迭代到内存中才能再次保存
        save_streaming_subset(test_data,args)
        print("测试集保存成功")
        remain_data = dataset.skip(args.size_test_set)
        valid_data = remain_data.take(args.size_valid_set)
        train_data = remain_data.skip(args.size_valid_set)
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=seed)
    else:
        print("---------正在以非流式常规方式加载和保存数据集---------")
        dataset_split = dataset.train_test_split(test_size=0.1, seed=seed)
        test_data = dataset_split['test']
        test_data.save_to_disk(args.test_set_path)
        print("测试集保存成功")
        remain_data = dataset_split['train'].train_test_split(test_size=0.1, seed=seed)
        train_data = remain_data["train"]
        valid_data = remain_data["test"]

    train_dataset = ConstantLengthDataset(
        tokenizer=tokenizer,
        dataset=train_data,
        infinite=True,
        formatting_func=prepare_sample_text,
        seq_length=args.seq_length,
        chars_per_token=chars_token_radio(dataset=train_data, tokenizer=tokenizer)
    )

    valid_dataset = ConstantLengthDataset(
        tokenizer=tokenizer,
        dataset=valid_data,
        infinite=False,
        formatting_func=prepare_sample_text,
        seq_length=args.seq_length,
        chars_per_token=chars_token_radio(dataset=valid_data, tokenizer=tokenizer)
    )

    return train_dataset, valid_dataset

# LORA配置
peft_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

# 量化的参数，如果选择量化的话
bnb_config = None
if args.use_bnb:
    # 量化模型-参数配置
    print("已启用模型量化!")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        # bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_compute_dtype=torch.float16
    )

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=args.model_name,
    quantization_config=bnb_config,
    device_map={"": Accelerator().local_process_index},
    trust_remote_code=True,
)
# 禁用缓存 适合训练  反之 适合推理
model.config.use_cache = False


# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=args.model_name,
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

# 创建训练集和验证集
train_dataset, eval_dataset = Create_datasets(tokenizer=tokenizer, args=args, seed=args.seed)

# 训练参数配置
training_args = SFTConfig(
    output_dir=args.output_dir,
    max_steps=args.max_steps,
    logging_steps=args.logging_steps,
    save_steps=args.save_steps,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    gradient_checkpointing=args.gradient_checkpointing,
    group_by_length=args.group_by_length,
    learning_rate=args.learning_rate,
    lr_scheduler_type=args.lr_scheduler_type,
    warmup_steps=args.warmup_steps,
    weight_decay=args.weight_decay,
    optim=args.optim,
    bf16=args.bf16,
    fp16=args.fp16,
    remove_unused_columns=args.remove_unused_columns,
    run_name=args.run_name,
    report_to=args.report_to,
    packing=False, # default to False
)

response_template = " Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# 初始化 Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    max_seq_length=512,
    peft_config=peft_config,
    tokenizer=tokenizer,
    formatting_func=prepare_sample_text,
    args=training_args,
    data_collator=collator,
)

trainer.train()  # 启动训练

# 保存训练后的模型
trainer.save_model(args.output_dir)

# 保存训练过程中的检查点
output_dir = os.path.join(args.output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)

# Free memory for merging weights
del model  # 删除原始的模型，释放内存

# 清理显存
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # 清空 GPU 显存

# 加载并合并微调权重
model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=args.torch_dtype)
model = model.merge_and_unload()

# 保存合并后的模型
output_merged_dir = os.path.join(args.output_dir, "final_merged_checkpoint")
model.save_pretrained(output_merged_dir, safe_serialization=True)

# 释放内存
del model  # 删除合并后的模型
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # 清空 GPU 显存
