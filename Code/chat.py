import transformers
from transformers import AutoTokenizer
import os

# 模型路径
model_path = "/mnt/sdb_mount/daixinwei/LLM/Model/Llama-3.2-3B-Instruct"
model_sft_path = "/mnt/sdb_mount/daixinwei/SFT/huanhuan-chat/llama3_1_instruct_lora/checkpoint-699"

# 初始化分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_path)

prompt = "嬛嬛，有人欺负你吗？"

messages = [
        {"role": "system", "content": "假设你是皇帝身边的女人--甄嬛。"},
        {"role": "user", "content": prompt}
]

# 初始化模型 pipeline
pipe = transformers.pipeline(
    task="text-generation",
    tokenizer=tokenizer,
    model=model_sft_path,
    torch_dtype="auto",
    device_map="auto",
)

output = pipe(
    messages,
    max_new_tokens=512,
)

print(output[-1]["generated_text"][-1]["content"])