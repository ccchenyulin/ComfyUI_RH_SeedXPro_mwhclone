from transformers import AutoTokenizer
import os
import json

# 让用户输入模型路径
model_path = input("请输入 Seed-X-PPO-7B 模型的路径: ").strip()
if not os.path.exists(model_path):
    print(f"错误：路径 '{model_path}' 不存在")
    exit(1)

# 直接保存到原目录
save_path = model_path

print(f"正在从 {model_path} 加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)

print("✅ 加载成功！正在生成配置文件...")

# 1. 生成 tokenizer_config.json
tokenizer_config = {
    "tokenizer_class": tokenizer.__class__.__name__,
    "model_max_length": tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') else 8192,
    "bos_token": tokenizer.bos_token,
    "eos_token": tokenizer.eos_token,
    "pad_token": tokenizer.pad_token if tokenizer.pad_token is not None else tokenizer.eos_token,
    "unk_token": tokenizer.unk_token,
    "clean_up_tokenization_spaces": False,
    "legacy": False
}
tokenizer_config = {k: v for k, v in tokenizer_config.items() if v is not None}

config_file_path = os.path.join(save_path, "tokenizer_config.json")
with open(config_file_path, "w", encoding="utf-8") as f:
    json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
print(f"✅ 已生成: {config_file_path}")

# 2. 生成 special_tokens_map.json（非必须但推荐）
special_tokens_map = {
    "bos_token": tokenizer.bos_token,
    "eos_token": tokenizer.eos_token,
    "pad_token": tokenizer.pad_token if tokenizer.pad_token is not None else tokenizer.eos_token,
    "unk_token": tokenizer.unk_token
}
special_tokens_map = {k: v for k, v in special_tokens_map.items() if v is not None}

special_map_path = os.path.join(save_path, "special_tokens_map.json")
with open(special_map_path, "w", encoding="utf-8") as f:
    json.dump(special_tokens_map, f, indent=2, ensure_ascii=False)
print(f"✅ 已生成: {special_map_path}")

print("\n🎉 配置文件已生成在模型目录中！现在可以去另一个环境测试了。")
