import subprocess
import folder_paths
import os
import re
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import torch

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  本地词典：加载 & 查询
# ─────────────────────────────────────────────

def load_local_dictionary(dict_path):
    """
    从 txt 文件加载本地翻译词典。
    格式：每行 "英文原文=中文翻译"，忽略空行和以 # 开头的注释行。
    返回 dict，key 为小写英文，value 为翻译。
    """
    dictionary = {}
    if not dict_path or not dict_path.strip():
        return dictionary
    dict_path = dict_path.strip()
    if not os.path.exists(dict_path):
        logger.warning(f"[Dictionary] 词典文件不存在，跳过加载: {dict_path}")
        return dictionary
    try:
        with open(dict_path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    logger.warning(f"[Dictionary] 第 {lineno} 行格式不正确，已跳过: {line!r}")
                    continue
                src_word, _, tgt_word = line.partition("=")
                src_word = src_word.strip()
                tgt_word = tgt_word.strip()
                if src_word and tgt_word:
                    dictionary[src_word.lower()] = tgt_word
        logger.info(f"[Dictionary] 词典加载完成，共 {len(dictionary)} 条词条: {dict_path}")
    except Exception as e:
        logger.error(f"[Dictionary] 词典加载失败: {e}", exc_info=True)
    return dictionary


def translate_tags_with_dict(tags, dictionary):
    """
    对一组 Danbooru tag 使用本地词典进行翻译。
    返回:
        dict_hits   : {原始tag: 译文}  — 词典命中
        model_needed: [原始tag]        — 需要模型翻译
    """
    dict_hits = {}
    model_needed = []
    for tag in tags:
        key = tag.strip().lower()
        if key in dictionary:
            dict_hits[tag] = dictionary[key]
        else:
            model_needed.append(tag)
    return dict_hits, model_needed


# ─────────────────────────────────────────────
#  模型下载
# ─────────────────────────────────────────────

def ensure_model_downloaded(model_path, repo_id="ByteDance-Seed/Seed-X-PPO-7B", mirror_url=None):
    """Ensure model is downloaded, auto-download if not exists"""
    logger.info(f"[Model Check] Starting model existence check: {model_path}")
    if not os.path.exists(model_path):
        logger.warning(f"[Model Check] Model directory does not exist: {model_path}")
        logger.info(f"[Download] Starting download from Hugging Face: {repo_id}")
        try:
            parent_dir = os.path.dirname(model_path)
            logger.info(f"[Download] Creating parent directory: {parent_dir}")
            os.makedirs(parent_dir, exist_ok=True)
            download_kwargs = {
                "repo_id": repo_id,
                "local_dir": model_path,
                "local_dir_use_symlinks": False,
                "resume_download": True
            }
            if mirror_url:
                logger.info(f"[Download] Using mirror: {mirror_url}")
                download_kwargs["endpoint"] = mirror_url
            snapshot_download(**download_kwargs)
            logger.info(f"[Download] Model download completed successfully: {model_path}")
        except Exception as e:
            logger.error(f"[Download] Model download failed: {e}", exc_info=True)
            raise e
    else:
        logger.info(f"[Model Check] Model already exists: {model_path}")


# ─────────────────────────────────────────────
#  文本切分
# ─────────────────────────────────────────────

def split_text_into_chunks(text, split_mode="By Sentence", max_chunk_size=400):
    """Split long text into smaller chunks for translation"""
    logger.debug(f"[Split] Starting text split, mode: {split_mode}, max size: {max_chunk_size}")
    if len(text) <= max_chunk_size:
        logger.debug(f"[Split] Text is short enough, no split needed")
        return [text]

    if split_mode == "By Danbooru Tag":
        tags = [tag.strip() for tag in text.split(',')]
        chunks = []
        current_chunk = ""
        for tag in tags:
            if not tag:
                continue
            separator = ", " if current_chunk else ""
            test_chunk = current_chunk + separator + tag
            if len(test_chunk) <= max_chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = tag
        if current_chunk:
            chunks.append(current_chunk)
    else:
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

    # 兜底：硬切超长块
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chunk_size:
            final_chunks.append(chunk)
        else:
            for i in range(0, len(chunk), max_chunk_size):
                final_chunks.append(chunk[i:i + max_chunk_size])

    logger.info(f"[Split] Text split into {len(final_chunks)} chunks")
    return final_chunks


# ─────────────────────────────────────────────
#  模型输出解析
# ─────────────────────────────────────────────

def extract_translation_from_output(output, dst_code):
    """Extract translation from model output using multiple strategies"""
    patterns = [
        f'<{dst_code}>(.*?)(?:<(?!/)|$)',
        f'<{dst_code}>(.*)',
        f'{dst_code}>(.*?)(?:<|$)',
        f'<{dst_code}>\s*(.*?)(?:\n\n|$)',
    ]
    for pattern in patterns:
        match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
        if match:
            result = match.group(1).strip()
            if result and len(result) > 0:
                return result

    lines = output.split('\n')
    found_marker = False
    result_lines = []
    for line in lines:
        if f'<{dst_code}>' in line or f'{dst_code}>' in line:
            found_marker = True
            parts = re.split(f'<{dst_code}>|{dst_code}>', line, 1)
            if len(parts) > 1 and parts[1].strip():
                result_lines.append(parts[1].strip())
            continue
        if found_marker:
            if re.search(r'<[a-z]{2}>', line, re.IGNORECASE):
                break
            result_lines.append(line)

    if result_lines:
        return '\n'.join(result_lines).strip()
    return None


# ─────────────────────────────────────────────
#  单块翻译（模型）
# ─────────────────────────────────────────────

def translate_single_chunk(chunk, src, dst, dst_code, model, tokenizer,
                           max_new_tokens_limit, chunk_index, split_mode):
    """
    Translate a single chunk of text via model.
    Returns: (translation_result, input_token_count, output_token_count)
    """
    logger.debug(f"[Chunk {chunk_index}] Starting translation")
    if split_mode == "By Danbooru Tag":
        message = f"Translate the following {src} comma-separated tags into {dst}:\n{chunk} <{dst_code}>"
    else:
        message = f"Translate the following {src} text into {dst}:\n{chunk} <{dst_code}>"

    inputs = tokenizer(message, return_tensors="pt").to("cuda")
    input_token_length = inputs['input_ids'].shape[1]

    raw_char_count = len(chunk)
    logger.info("-" * 30)
    logger.info(f"[Chunk {chunk_index}] Processing...")
    logger.info(f"  [Input] Chars: {raw_char_count} | Tokens (w/ Prompt): {input_token_length}")

    calculated_max = min(max_new_tokens_limit, max(150, int(input_token_length * 2.0)))
    logger.info(f"  [Config] max_new_tokens: {calculated_max}")

    for attempt in range(2):
        try:
            logger.debug(f"[Chunk {chunk_index}] Attempt {attempt + 1} started")
            if attempt == 0:
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
            translation = extract_translation_from_output(res, dst_code)

            if translation and len(translation.strip()) > 0:
                output_tokens = tokenizer.encode(translation, return_tensors="pt").shape[1]
                logger.info(f"  [Output] Chars: {len(translation)} | Tokens: {output_tokens}")
                logger.info(f"  [Result] Translation succeeded (Attempt {attempt + 1})")
                return translation, input_token_length, output_tokens
            else:
                logger.warning(f"  [Warning] Attempt {attempt + 1} failed to extract valid translation")

        except Exception as e:
            logger.error(f"  [Error] Attempt {attempt + 1} exception: {e}", exc_info=True)
            continue

    logger.error(f"  [Failed] All attempts failed for chunk {chunk_index}")
    return f"[Translation failed for: {chunk}]", input_token_length, 0


# ─────────────────────────────────────────────
#  Danbooru Tag 专用翻译流程（词典 + 模型混合）
# ─────────────────────────────────────────────

def translate_danbooru_tags(prompt, src, dst, dst_code,
                            model, tokenizer, max_new_tokens_limit,
                            dictionary):
    """
    Danbooru Tag 模式的完整翻译流程：
    1. 把所有 tag 先过词典
    2. 未命中的 tag 组成新文本，送模型翻译
    3. 将词典译文和模型译文按原始顺序拼合
    返回: (最终译文字符串, 模型映射字符串, total_input_tokens, total_output_tokens)
    """
    # ── 1. 解析所有 tag（保留原始顺序）──
    all_tags_raw = [t.strip() for t in prompt.split(',')]
    all_tags = [t for t in all_tags_raw if t]  # 去空

    total_tags = len(all_tags)

    # ── 2. 词典查询 ──
    dict_hits, model_needed = translate_tags_with_dict(all_tags, dictionary)

    dict_count  = len(dict_hits)
    model_count = len(model_needed)

    # ── 3. 控制台统计输出 ──
    logger.info("=" * 60)
    logger.info(f"[Danbooru Stats] 共 {total_tags} 个 tag")
    logger.info(f"  词典命中: {dict_count} 个  |  需模型翻译: {model_count} 个")
    logger.info("-" * 60)

    if dict_hits:
        logger.info("[词典翻译] 命中的 tag：")
        for orig, trans in dict_hits.items():
            logger.info(f"  '{orig}'  →  '{trans}'")
    else:
        logger.info("[词典翻译] 无命中")

    if model_needed:
        logger.info("[模型翻译] 需要模型处理的 tag：")
        for tag in model_needed:
            logger.info(f"  '{tag}'")
    else:
        logger.info("[模型翻译] 无需模型处理")
    logger.info("=" * 60)

    # ── 4. 模型翻译（仅处理未命中的 tag）──
    total_input_tokens  = 0
    total_output_tokens = 0
    model_results = {}   # {原始tag: 译文}

    if model_needed and model is not None:
        # 把需要模型翻译的 tag 重新组成文本，切块后翻译
        model_text  = ", ".join(model_needed)
        chunks      = split_text_into_chunks(model_text,
                                             split_mode="By Danbooru Tag",
                                             max_chunk_size=400)
        chunk_translations = []
        for i, chunk in enumerate(chunks):
            trans, in_tok, out_tok = translate_single_chunk(
                chunk, src, dst, dst_code,
                model, tokenizer, max_new_tokens_limit,
                i + 1, "By Danbooru Tag"
            )
            chunk_translations.append((chunk, trans))
            total_input_tokens  += in_tok
            total_output_tokens += out_tok

        # 把模型翻译结果与原始 tag 一一对应
        # chunk 内的 tag 以逗号分隔，译文也以逗号（或顿号）分隔
        model_trans_list = []
        for chunk_src, chunk_trans in chunk_translations:
            src_tags_in_chunk  = [t.strip() for t in chunk_src.split(',') if t.strip()]
            # 译文可能用中文顿号或英文逗号分隔，统一处理
            trans_parts = [t.strip() for t in re.split(r'[,、，]', chunk_trans) if t.strip()]
            # 对齐：若数量不匹配，按顺序尽量对齐，多余的合并到最后一个
            for idx, src_tag in enumerate(src_tags_in_chunk):
                if idx < len(trans_parts):
                    model_trans_list.append((src_tag, trans_parts[idx]))
                else:
                    # 翻译数量不足，标记失败
                    model_trans_list.append((src_tag, f"[翻译失败: {src_tag}]"))

        # 记录模型结果
        for src_tag, trans in model_trans_list:
            model_results[src_tag] = trans

        # 控制台输出模型翻译结果
        logger.info("[模型翻译] 翻译结果：")
        for src_tag, trans in model_results.items():
            logger.info(f"  '{src_tag}'  →  '{trans}'")
        logger.info("=" * 60)

    # ── 5. 按原始顺序拼合结果 ──
    final_parts = []
    for tag in all_tags:
        if tag in dict_hits:
            final_parts.append(dict_hits[tag])
        elif tag in model_results:
            final_parts.append(model_results[tag])
        else:
            # 既没词典命中也没模型结果（model=None 且未命中时）
            final_parts.append(tag)

    result = "、".join(final_parts)

    # ── 6. 构建模型翻译映射字符串（按原始顺序）──
    mapping_lines = []
    for tag in model_needed:   # model_needed 保持输入顺序
        trans = model_results.get(tag, f"[翻译失败: {tag}]")
        mapping_lines.append(f"{tag}={trans}")
    mapping_str = "\n".join(mapping_lines)

    return result, mapping_str, total_input_tokens, total_output_tokens


# ─────────────────────────────────────────────
#  主翻译入口
# ─────────────────────────────────────────────

def translate(**kwargs):
    try:
        logger.info("=" * 60)
        logger.info("[Task] New translation task received")
        logger.info("=" * 60)

        prompt = kwargs.get('prompt')
        original_length = len(prompt)

        split_mode           = kwargs.get('split_mode')
        max_new_tokens_limit = kwargs.get('max_new_tokens')
        huggingface_mirror   = kwargs.get('huggingface_mirror', 'Official')
        model_path           = kwargs.get('model_path')
        dict_path            = kwargs.get('dict_path', '')   # 新增词典路径

        mirror_url_map = {
            "Official": None,
            "HF Mirror (hf-mirror.com)": "https://hf-mirror.com"
        }
        mirror_url = mirror_url_map.get(huggingface_mirror)

        if not model_path or not model_path.strip():
            error_msg = "Model path cannot be empty! Please input a valid model path in the node."
            logger.error(f"[Preprocess] {error_msg}")
            return f"Error: {error_msg}", ""
        model_path = model_path.strip()
        logger.info(f"[Preprocess] Using model path from node: {model_path}")

        logger.info(f"[Preprocess] Original text length: {original_length} chars")
        prompt = re.sub(r'[\x00\x01-\x08\x0b\x0c\x0e-\x1f\x7f]', '', prompt)

        if len(prompt) != original_length:
            removed_count = original_length - len(prompt)
            logger.warning(f"[Preprocess] Removed {removed_count} problematic control character(s)")

        if not prompt.strip():
            logger.error("[Preprocess] Error: Empty input after cleaning")
            return "Error: Empty input after cleaning", ""

        src      = kwargs.get('from')
        dst      = kwargs.get('to')
        dst_code = kwargs.get('dst_code')

        # ── 加载本地词典 ──
        dictionary = load_local_dictionary(dict_path)

        # ── Danbooru Tag 模式：词典优先 ──
        if split_mode == "By Danbooru Tag":
            # 判断是否所有 tag 都能被词典覆盖，若是则无需加载模型
            all_tags = [t.strip() for t in prompt.split(',') if t.strip()]
            _, model_needed = translate_tags_with_dict(all_tags, dictionary)

            model     = None
            tokenizer = None

            if model_needed:
                # 有 tag 需要模型翻译，才加载模型
                ensure_model_downloaded(model_path, mirror_url=mirror_url)
                try:
                    logger.info("[Model Load] Starting to load model and tokenizer...")
                    model     = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to('cuda')
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    logger.info("[Model Load] Model and tokenizer loaded successfully")
                except Exception as model_error:
                    error_msg = f"Failed to load model from {model_path}. Error: {model_error}"
                    logger.error(f"[Model Load] {error_msg}", exc_info=True)
                    return f'Model loading failed: {error_msg}.', ""

            result, mapping_str, total_input_tokens, total_output_tokens = translate_danbooru_tags(
                prompt, src, dst, dst_code,
                model, tokenizer, max_new_tokens_limit,
                dictionary
            )

        else:
            # ── 普通文本模式（原有逻辑不变）──
            ensure_model_downloaded(model_path, mirror_url=mirror_url)
            try:
                logger.info("[Model Load] Starting to load model and tokenizer...")
                model     = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to('cuda')
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                logger.info("[Model Load] Model and tokenizer loaded successfully")
            except Exception as model_error:
                error_msg = f"Failed to load model from {model_path}. Error: {model_error}"
                logger.error(f"[Model Load] {error_msg}", exc_info=True)
                return f'Model loading failed: {error_msg}.', ""

            logger.info(f"=" * 60)
            logger.info(f"[Task Start] Source: {src} -> Target: {dst} | Split mode: {split_mode}")
            logger.info(f"=" * 60)

            chunks = split_text_into_chunks(prompt, split_mode=split_mode, max_chunk_size=400)
            logger.info(f"[Preprocess] Text split into {len(chunks)} chunks")
            for idx, chunk_content in enumerate(chunks, 1):
                logger.info(f"  Chunk {idx}: {chunk_content}")

            total_input_tokens  = 0
            total_output_tokens = 0
            translated_chunks   = []

            for i, chunk in enumerate(chunks):
                translation, in_tok, out_tok = translate_single_chunk(
                    chunk, src, dst, dst_code,
                    model, tokenizer, max_new_tokens_limit,
                    i + 1, split_mode
                )
                translated_chunks.append(translation)
                total_input_tokens  += in_tok
                total_output_tokens += out_tok

            result = ' '.join(translated_chunks)
            mapping_str = ""   # 普通文本模式无模型映射输出

        # ── 最终统计 ──
        logger.info(f"=" * 60)
        logger.info(f"[Task Complete] Final statistics")
        logger.info(f"  Total input tokens:    {total_input_tokens}")
        logger.info(f"  Total output tokens:   {total_output_tokens}")
        logger.info(f"  Total tokens consumed: {total_input_tokens + total_output_tokens}")
        logger.info(f"=" * 60)

        return result, mapping_str

    except Exception as e:
        logger.error(f"[Fatal] Translation error: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return f'Translation failed: {str(e)}', ""


# ─────────────────────────────────────────────
#  ComfyUI 节点定义
# ─────────────────────────────────────────────

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
                "model_path": ("STRING", {
                    "multiline": False,
                    "default": "/mnt/d/qwen2511/Seed-X-PPO-7B",
                    "tooltip": "模型本地存放的绝对路径。若路径下无模型文件，会自动从Hugging Face下载到该路径。"
                }),
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
                    "tooltip": "文本切分模式。\n- By Sentence: 按标点符号切分（适合普通文章）。\n- By Danbooru Tag: 按英文逗号切分，优先使用本地词典翻译（适合标签翻译）。"
                }),
                "max_new_tokens": ("INT", {
                    "default": 1024,
                    "min": 128,
                    "max": 4096,
                    "tooltip": "每轮翻译生成的最大Token数上限。如果译文被截断，请适当增大此数值。"
                }),
                "huggingface_mirror": (["Official", "HF Mirror (hf-mirror.com)"], {
                    'default': 'Official',
                    "tooltip": "Hugging Face下载镜像源。如果官方下载慢，可以选择镜像源加速。"
                }),
                "seed": ("INT", {
                    "default": 28,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "随机种子。用于复现相同的翻译结果，通常保持默认即可。"
                }),
            },
            "optional": {
                # 新增：本地词典路径，仅 By Danbooru Tag 模式下生效
                "dict_path": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": (
                        "【仅 By Danbooru Tag 模式生效】本地翻译词典 txt 文件的绝对路径。\n"
                        "格式：每行一条，英文原文=中文翻译，例如：\n"
                        "  tileable=可平铺拼贴\n"
                        "  widescreen=宽屏幕\n"
                        "词典中存在的 tag 将直接使用词典翻译，其余 tag 交由模型翻译。\n"
                        "留空则全部使用模型翻译。"
                    )
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("content", "model_mapping")
    FUNCTION = "translate"

    CATEGORY = "Runninghub/SeedXPro"
    TITLE = "RunningHub SeedXPro Translator"

    def translate(self, **kwargs):
        kwargs['dst_code'] = self.language_code_map[kwargs.get('to')]
        res, mapping = translate(**kwargs)
        return (res, mapping)


NODE_CLASS_MAPPINGS = {
    "RunningHub SeedXPro Translator": RH_SeedXPro_Translator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RunningHub SeedXPro Translator": "RunningHub SeedXPro Translator",
}
