import subprocess
import folder_paths
import os
import re

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import torch

def ensure_model_downloaded(model_path, repo_id="ByteDance-Seed/Seed-X-PPO-7B"):
    """
    Ensure model is downloaded, auto-download if not exists
    """
    if not os.path.exists(model_path):
        print(f"Model directory does not exist: {model_path}")
        print(f"Downloading model from Hugging Face: {repo_id}")
        try:
            # Create parent directory
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Download model
            snapshot_download(
                repo_id=repo_id,
                local_dir=model_path,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print(f"Model download completed: {model_path}")
        except Exception as e:
            print(f"Model download failed: {e}")
            raise e
    else:
        print(f"Model already exists: {model_path}")

def split_text_into_chunks(text, split_mode="By Sentence", max_chunk_size=400):
    """
    Split long text into smaller chunks for translation
    split_mode: "By Sentence" (default) or "By Danbooru Tag"
    """
    if len(text) <= max_chunk_size:
        return [text]
    
    if split_mode == "By Danbooru Tag":
        # 按英文逗号切分，并去除每个tag前后的空格
        tags = [tag.strip() for tag in text.split(',')]
        chunks = []
        current_chunk = ""
        
        for tag in tags:
            if not tag:  # 跳过空tag
                continue
            
            # 拼接测试：如果是第一个tag直接赋值，否则加逗号
            separator = ", " if current_chunk else ""
            test_chunk = current_chunk + separator + tag
            
            if len(test_chunk) <= max_chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                # 如果单个tag就超长，也先放进去，后面会有硬切逻辑兜底
                current_chunk = tag
        
        if current_chunk:
            chunks.append(current_chunk)
            
    else:
        # 原有逻辑：按句子切分
        sentences = re.split(r'[.!?。！？]\s*', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) <= max_chunk_size:
                current_chunk += sentence + ". " if sentence else ""
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". " if sentence else ""
        
        if current_chunk:
            chunks.append(current_chunk.strip())

    # 通用兜底逻辑：如果通过上述逻辑切分后的块依然超长，强制按字符切分
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chunk_size:
            final_chunks.append(chunk)
        else:
            # Split long chunk by characters
            for i in range(0, len(chunk), max_chunk_size):
                final_chunks.append(chunk[i:i + max_chunk_size])
    
    return final_chunks

def extract_translation_from_output(output, dst_code):
    """
    Extract translation from model output using multiple strategies
    """
    # Strategy 1: Standard pattern with language code
    patterns = [
        f'<{dst_code}>(.*?)(?:<(?!/)|$)',  # Stop at next tag or end
        f'<{dst_code}>(.*)',               # Everything after the tag
        f'{dst_code}>(.*?)(?:<|$)',        # Without opening bracket
        f'<{dst_code}>\s*(.*?)(?:\n\n|$)', # Stop at double newline or end
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
        if match:
            result = match.group(1).strip()
            if result and len(result) > 0:
                return result
    
    # Strategy 2: Find text after the language code marker
    lines = output.split('\n')
    found_marker = False
    result_lines = []
    
    for line in lines:
        if f'<{dst_code}>' in line or f'{dst_code}>' in line:
            found_marker = True
            # Extract text after the marker in the same line
            parts = re.split(f'<{dst_code}>|{dst_code}>', line, 1)
            if len(parts) > 1 and parts[1].strip():
                result_lines.append(parts[1].strip())
            continue
        
        if found_marker:
            # Stop if we hit another language tag
            if re.search(r'<[a-z]{2}>', line, re.IGNORECASE):
                break
            result_lines.append(line)
    
    if result_lines:
        return '\n'.join(result_lines).strip()
    
    return None

def translate_single_chunk(chunk, src, dst, dst_code, model, tokenizer, max_new_tokens_limit, chunk_index, split_mode):
    """
    Translate a single chunk of text
    Returns: (translation_result, input_token_count, output_token_count)
    """
    if split_mode == "By Danbooru Tag":
        # Danbooru 标签专用模板
        message = f"Translate the following {src} comma-separated tags into {dst}:\n{chunk} <{dst_code}>"
    else:
        # 普通文本模板
        message = f"Translate the following {src} text into {dst}:\n{chunk} <{dst_code}>"
    
    inputs = tokenizer(message, return_tensors="pt").to("cuda")
    input_token_length = inputs['input_ids'].shape[1]
    
    # 统计输入信息
    raw_char_count = len(chunk)
    print("-" * 30)
    print(f"[Chunk {chunk_index}] 处理中...")
    print(f"  [输入] 字符数: {raw_char_count} | Token数 (含Prompt): {input_token_length}")

    # 优化：使用真实的输入Token数进行估算
    calculated_max = min(max_new_tokens_limit, max(150, int(input_token_length * 2.0)))
    print(f"  [设置] max_new_tokens: {calculated_max}")
    
    # Multiple attempts with different parameters
    for attempt in range(2):
        try:
            if attempt == 0:
                # First attempt: greedy decoding
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=calculated_max,
                    do_sample=False,
                    temperature=1.0,
                    repetition_penalty=1.05,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            else:
                # Second attempt: with sampling
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=calculated_max,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            res = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract translation
            translation = extract_translation_from_output(res, dst_code)
            
            if translation and len(translation.strip()) > 0:
                # 统计翻译结果的Token数
                output_tokens = tokenizer.encode(translation, return_tensors="pt").shape[1]
                
                print(f"  [输出] 字符数: {len(translation)} | Token数: {output_tokens}")
                print(f"  [结果] 翻译成功 (Attempt {attempt + 1})")
                
                # 返回译文和统计数据
                return translation, input_token_length, output_tokens
            else:
                print(f"  [警告] Attempt {attempt + 1} 未能提取到有效翻译")
                
        except Exception as e:
            print(f"  [错误] Attempt {attempt + 1} 发生异常: {e}")
            continue
    
    # If all attempts failed
    print(f"  [失败] 所有尝试均失败")
    return f"[Translation failed for: {chunk}]", input_token_length, 0

def translate(**kwargs):
    try:
        prompt = kwargs.get('prompt')
        original_length = len(prompt)
        
        # 接收新参数
        split_mode = kwargs.get('split_mode')
        max_new_tokens_limit = kwargs.get('max_new_tokens')
        
        # Only remove truly problematic control characters
        prompt = re.sub(r'[\x00\x01-\x08\x0b\x0c\x0e-\x1f\x7f]', '', prompt)
        
        if len(prompt) != original_length:
            removed_count = original_length - len(prompt)
            print(f"Warning: Removed {removed_count} problematic control character(s) from input")
        
        if not prompt.strip():
            return "Error: Empty input after cleaning"
        
        src = kwargs.get('from')
        dst = kwargs.get('to')
        dst_code = kwargs.get('dst_code')
        model_path = os.path.join(folder_paths.models_dir, 'Seed-X-PPO-7B')

        # Ensure model is downloaded
        ensure_model_downloaded(model_path)

        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to('cuda')
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as model_error:
            error_msg = f"Failed to load model from {model_path}. Error: {model_error}"
            print(error_msg)
            return f'Model loading failed: {error_msg}. Please delete model directory and re-download.'

        print(f"=" * 50)
        print(f"[任务开始] 源语言: {src} -> 目标语言: {dst} | 切分模式: {split_mode}")
        print(f"=" * 50)
        
        # 切分文本
        chunks = split_text_into_chunks(prompt, split_mode=split_mode, max_chunk_size=400)
        print(f"[预处理] 文本已切分为 {len(chunks)} 个块")
        
        # 【新增】输出每个切分块的原文内容
        print("分别为：")
        for idx, chunk_content in enumerate(chunks, 1):
            print(f"  第 {idx} 块：{chunk_content}")
        
        # 初始化全局统计变量
        total_input_tokens = 0
        total_output_tokens = 0
        translated_chunks = []
        
        # 循环处理
        for i, chunk in enumerate(chunks):
            translation, in_tok, out_tok = translate_single_chunk(
                chunk, src, dst, dst_code, 
                model, tokenizer, max_new_tokens_limit, 
                i + 1, split_mode
            )
            translated_chunks.append(translation)
            total_input_tokens += in_tok
            total_output_tokens += out_tok
            
        # 拼接结果
        if split_mode == "By Danbooru Tag":
            result = ', '.join(translated_chunks)
        else:
            result = ' '.join(translated_chunks)
        
        # 打印最终统计报告
        print(f"=" * 50)
        print(f"[任务完成] 最终统计报告")
        print(f"  总输入 Token 数: {total_input_tokens}")
        print(f"  总输出 Token 数: {total_output_tokens}")
        print(f"  总计消耗 Token 数: {total_input_tokens + total_output_tokens}")
        print(f"=" * 50)
        
        return result
        
    except Exception as e:
        print(f"Translation error: {e}")
        import traceback
        traceback.print_exc()
        return f'Translation failed: {str(e)}'

class RH_SeedXPro_Translator:

    language_code_map = {
        "Arabic": "ar",
        "French": "fr",
        "Malay": "ms",
        "Russian": "ru",
        "Czech": "cs",
        "Croatian": "hr",
        "Norwegian Bokmal": "nb",
        "Swedish": "sv",
        "Danish": "da",
        "Hungarian": "hu",
        "Dutch": "nl",
        "Thai": "th",
        "German": "de",
        "Indonesian": "id",
        "Norwegian": "no",
        "Turkish": "tr",
        "English": "en",
        "Italian": "it",
        "Polish": "pl",
        "Ukrainian": "uk",
        "Spanish": "es",
        "Japanese": "ja",
        "Portuguese": "pt",
        "Vietnamese": "vi",
        "Finnish": "fi",
        "Korean": "ko",
        "Romanian": "ro",
        "Chinese": "zh"
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "may the force be with you",
                    "tooltip": "需要翻译的原文文本。如果是Danbooru标签模式，请输入用英文逗号分隔的标签。"
                }),
                "from": (list(cls.language_code_map.keys()), {
                    'default': 'English',
                    "tooltip": "源语言，即输入文本使用的语言。"
                }),
                "to": (list(cls.language_code_map.keys()), {
                    'default': 'Chinese',
                    "tooltip": "目标语言，即你希望翻译成的语言。"
                }),
                "split_mode": (["By Sentence", "By Danbooru Tag"], {
                    'default': 'By Sentence',
                    "tooltip": "文本切分模式。\n- By Sentence: 按标点符号切分（适合普通文章）。\n- By Danbooru Tag: 按英文逗号切分（适合标签翻译）。"
                }),
                "max_new_tokens": ("INT", {
                    "default": 1024, 
                    "min": 128, 
                    "max": 4096,
                    "tooltip": "每轮翻译生成的最大Token数上限。如果译文被截断，请适当增大此数值。"
                }),
                "seed": ("INT", {
                    "default": 28, 
                    "min": 0, 
                    "max": 0xffffffffffffffff,
                    "tooltip": "随机种子。用于复现相同的翻译结果，通常保持默认即可。"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("content",)
    FUNCTION = "translate"

    CATEGORY = "Runninghub/SeedXPro"
    TITLE = "RunningHub SeedXPro Translator"

    def translate(self, **kwargs):
        kwargs['dst_code'] = self.language_code_map[kwargs.get('to')]
        res = translate(**kwargs)
        return (res,)

NODE_CLASS_MAPPINGS = {
    "RunningHub SeedXPro Translator": RH_SeedXPro_Translator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RunningHub SeedXPro Translator": "RunningHub SeedXPro Translator",
} 
