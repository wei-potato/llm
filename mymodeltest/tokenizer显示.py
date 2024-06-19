from transformers import AutoConfig, AutoTokenizer

# 指定模型名称和本地路径
model_name = "FlagAlpha/Atom-7B-Chat"
local_path = ".FlagAlpha/Atom-7B-Chat"

# 下载并保存配置文件到本地路径
config = AutoConfig.from_pretrained(model_name, cache_dir=local_path)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_path, use_fast=True)

