# token_utils.py
# Token估算工具函数

from typing import Tuple
from collections import OrderedDict
import hashlib
import threading

_tiktoken_available = False
_encoder = None

DEFAULT_MAX_TOKENS = {
    "gpt-4": 8192,
    "gpt-4-turbo": 128000,
    "gpt-3.5-turbo": 4096,
    "claude-3": 200000,
    "gemini-1.5": 1000000,
    "default": 4096
}

_TOKEN_CACHE_MAX_SIZE = 500
_token_cache: OrderedDict = OrderedDict()
_token_cache_lock = threading.Lock()

try:
    import tiktoken
    _tiktoken_available = True
    _encoder = tiktoken.get_encoding("cl100k_base")
except ImportError:
    pass


def _get_cache_key(text: str) -> str:
    """生成缓存键"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def _cache_get(text: str) -> Tuple[bool, int]:
    """
    从缓存获取 token 数
    
    Returns:
        (是否命中, token数)
    """
    key = _get_cache_key(text)
    with _token_cache_lock:
        if key in _token_cache:
            _token_cache.move_to_end(key)
            return True, _token_cache[key]
    return False, 0


def _cache_set(text: str, tokens: int):
    """设置缓存"""
    key = _get_cache_key(text)
    with _token_cache_lock:
        if key in _token_cache:
            _token_cache.move_to_end(key)
        else:
            if len(_token_cache) >= _TOKEN_CACHE_MAX_SIZE:
                _token_cache.popitem(last=False)
            _token_cache[key] = tokens


def clear_token_cache():
    """清空 token 缓存"""
    with _token_cache_lock:
        _token_cache.clear()


def get_token_cache_stats() -> dict:
    """获取缓存统计"""
    with _token_cache_lock:
        return {
            "size": len(_token_cache),
            "max_size": _TOKEN_CACHE_MAX_SIZE
        }


def estimate_tokens(text: str, use_tiktoken: bool = True, use_cache: bool = True) -> int:
    """
    估算文本的token数量
    
    优先使用tiktoken精确计算，不可用时使用估算公式
    
    现代分词器（如 GPT-4o, Gemini 1.5, GLM-4）的Token消耗：
    - 中文：约 1.0-1.2 tokens/字符
    - 英文：约 0.25-0.3 tokens/字符
    
    Args:
        text: 输入文本
        use_tiktoken: 是否尝试使用tiktoken（默认True）
        use_cache: 是否使用缓存（默认True）
    
    Returns:
        估算的token数量
    """
    if not text:
        return 0
    
    if use_cache:
        hit, cached_tokens = _cache_get(text)
        if hit:
            return cached_tokens
    
    if use_tiktoken and _tiktoken_available and _encoder:
        try:
            tokens = len(_encoder.encode(text))
            if use_cache:
                _cache_set(text, tokens)
            return tokens
        except Exception:
            pass
    
    tokens = _estimate_tokens_fallback(text)
    if use_cache:
        _cache_set(text, tokens)
    return tokens


def estimate_tokens_with_limit(
    text: str, 
    max_tokens: int = None,
    model: str = None,
    use_tiktoken: bool = True
) -> Tuple[int, bool, int]:
    """
    估算文本的token数量并检查是否超限
    
    Args:
        text: 输入文本
        max_tokens: 最大token限制（优先使用）
        model: 模型名称（用于获取默认限制）
        use_tiktoken: 是否尝试使用tiktoken
    
    Returns:
        (token数量, 是否超限, 最大限制)
    """
    if not text:
        return 0, False, max_tokens or DEFAULT_MAX_TOKENS.get(model, DEFAULT_MAX_TOKENS["default"])
    
    tokens = estimate_tokens(text, use_tiktoken)
    
    if max_tokens is None:
        max_tokens = DEFAULT_MAX_TOKENS.get(model, DEFAULT_MAX_TOKENS["default"])
    
    exceeds = tokens > max_tokens
    
    return tokens, exceeds, max_tokens


def truncate_to_token_limit(
    text: str, 
    max_tokens: int,
    use_tiktoken: bool = True,
    suffix: str = "..."
) -> str:
    """
    将文本截断到指定token限制内
    
    Args:
        text: 输入文本
        max_tokens: 最大token限制
        use_tiktoken: 是否尝试使用tiktoken
        suffix: 截断后缀
    
    Returns:
        截断后的文本
    """
    if not text:
        return text
    
    tokens, exceeds, _ = estimate_tokens_with_limit(text, max_tokens, use_tiktoken=use_tiktoken)
    
    if not exceeds:
        return text
    
    if use_tiktoken and _tiktoken_available and _encoder:
        try:
            encoded = _encoder.encode(text)
            truncated = encoded[:max_tokens - len(suffix)]
            return _encoder.decode(truncated) + suffix
        except Exception:
            pass
    
    estimated_chars_per_token = len(text) / max(tokens, 1)
    target_chars = int((max_tokens - len(suffix)) * estimated_chars_per_token)
    
    return text[:target_chars] + suffix


def check_text_fits_model(
    text: str, 
    model: str,
    buffer_tokens: int = 100
) -> Tuple[bool, int, int]:
    """
    检查文本是否适合指定模型的上下文限制
    
    Args:
        text: 输入文本
        model: 模型名称
        buffer_tokens: 保留的缓冲token数（用于响应）
    
    Returns:
        (是否适合, 当前token数, 最大允许token数)
    """
    max_tokens = DEFAULT_MAX_TOKENS.get(model, DEFAULT_MAX_TOKENS["default"])
    effective_max = max_tokens - buffer_tokens
    
    tokens = estimate_tokens(text)
    
    fits = tokens <= effective_max
    
    return fits, tokens, effective_max


def _estimate_tokens_fallback(text: str) -> int:
    """
    Token估算的Fallback方案（当tiktoken不可用时）
    
    系数说明：
    - 中文：1.2 tokens/字符（适配现代BPE分词器）
    - 英文：0.3 tokens/字符（适配现代BPE分词器）
    - 其他字符：0.5 tokens/字符（保守估计）
    
    Args:
        text: 输入文本
    
    Returns:
        估算的token数量
    """
    if not text:
        return 0
    
    chinese_chars = 0
    ascii_chars = 0
    other_chars = 0
    
    for c in text:
        if '\u4e00' <= c <= '\u9fff':
            chinese_chars += 1
        elif ord(c) < 128:
            ascii_chars += 1
        else:
            other_chars += 1
    
    return int(chinese_chars * 1.2 + ascii_chars * 0.3 + other_chars * 0.5)


def is_tiktoken_available() -> bool:
    """检查tiktoken是否可用"""
    return _tiktoken_available


def get_tokenizer_info() -> dict:
    """获取分词器信息"""
    return {
        "tiktoken_available": _tiktoken_available,
        "encoder": "cl100k_base" if _tiktoken_available else "fallback",
        "chinese_coefficient": 1.2 if not _tiktoken_available else "exact",
        "english_coefficient": 0.3 if not _tiktoken_available else "exact",
    }
