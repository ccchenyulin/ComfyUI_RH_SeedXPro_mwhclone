# ComfyUI SeedXPro 翻译节点

## 功能特性

- **简单易用**: 开箱即用，无需复杂配置
- **无特殊依赖**: 除标准依赖外，无需安装额外的特殊依赖包
- **即插即用**: 只需放入ComfyUI的custom_nodes目录即可使用
- 支持多语言翻译，基于 ByteDance-Seed/Seed-X-PPO-7B 模型
- **自动模型下载**: 首次使用时自动从 Hugging Face 下载模型到 `models/Seed-X-PPO-7B` 目录
- 无需手动下载模型文件

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用说明

1. 将此插件放入 ComfyUI 的 `custom_nodes` 目录
2. 首次运行时，系统会自动检查并下载 `ByteDance-Seed/Seed-X-PPO-7B` 模型
3. 模型会下载到 ComfyUI 的 `models/Seed-X-PPO-7B` 目录
4. 下载完成后即可正常使用翻译功能

## Tokenizer 配置

> **重要提示：适用于 transformers 4.57.6 及以上版本**

新版本 `transformers`（4.57.6+）需要 `tokenizer_config.json` 配置文件。  
生成脚本 `generate_tokenizer_config.py` 位于本插件同级目录下。

你有两种方式提供所需文件：

### 方式一：自动生成
`generate_tokenizer_config.py` 会生成 `tokenizer_config.json` 和 `special_tokens_map.json`（后者不是必须的，但最好有，以兼容更旧版本）。

### 方式二：手动创建
在模型目录（例如 `/mnt/d/qwen2511/Seed-X-PPO-7B`）下新建一个名为 `tokenizer_config.json` 的文件，并填入以下内容：

```json
{
  "add_bos_token": true,
  "add_eos_token": false,
  "added_tokens_decoder": {
    "0": {
      "content": "<unk>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "1": {
      "content": "<s>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "2": {
      "content": "</s>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    }
  },
  "bos_token": "<s>",
  "clean_up_tokenization_spaces": false,
  "eos_token": "</s>",
  "model_max_length": 32768,
  "pad_token": null,
  "sp_model_kwargs": {},
  "tokenizer_class": "MistralTokenizer",
  "unk_token": "<unk>",
  "use_default_system_prompt": false
}
```

> **注意**：**不需要**创建 `special_tokens_map.json`，上述文件已包含所有必要信息。

### 与旧版 transformers 的兼容性
对于旧版本（例如 `transformers 4.54.0`）：  
如果找不到 `tokenizer_config.json`，系统会自动回退使用 `tokenizer.json`。

## 注意事项

- 模型文件较大（约13GB），首次下载需要一定时间和网络带宽
- 确保有足够的磁盘空间存储模型文件
- 需要 CUDA 环境运行模型
- 如果使用 `transformers >= 4.57.6`，请注意 tokenizer 配置要求
