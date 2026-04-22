# token_utils.py
# Token估算工具函数

_tiktoken_available = False
_encoder = None

try:
    import tiktoken
    _tiktoken_available = True
    _encoder = tiktoken.get_encoding("cl100k_base")
except ImportError:
    pass


def estimate_tokens(text: str, use_tiktoken: bool = True) -> int:
    """
    估算文本的token数量
    
    优先使用tiktoken精确计算，不可用时使用估算公式
    
    现代分词器（如 GPT-4o, Gemini 1.5, GLM-4）的Token消耗：
    - 中文：约 1.0-1.2 tokens/字符
    - 英文：约 0.25-0.3 tokens/字符
    
    Args:
        text: 输入文本
        use_tiktoken: 是否尝试使用tiktoken（默认True）
    
    Returns:
        估算的token数量
    """
    if not text:
        return 0
    
    if use_tiktoken and _tiktoken_available and _encoder:
        try:
            return len(_encoder.encode(text))
        except Exception:
            pass
    
    return _estimate_tokens_fallback(text)


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
