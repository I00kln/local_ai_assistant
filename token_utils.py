# token_utils.py
# Token估算工具函数


def estimate_tokens(text: str) -> int:
    """
    估算文本的token数量
    
    中文约 1.5-2 tokens/字符，英文约 0.25 tokens/字符
    
    Args:
        text: 输入文本
    
    Returns:
        估算的token数量
    """
    if not text:
        return 0
    
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars
    
    return int(chinese_chars * 2 + other_chars * 0.5)
