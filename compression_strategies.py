# compression_strategies.py
# 压缩策略模块 - 从 async_processor.py 拆分
import re
import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List

try:
    from logger import get_logger
except ImportError:
    import logging
    def get_logger():
        return logging.getLogger('MemorySystem')


class CompressionStrategy(ABC):
    """压缩策略抽象接口"""
    
    @abstractmethod
    def compress(self, text: str) -> Optional[str]:
        """压缩文本，返回 None 表示无法压缩"""
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查策略是否可用"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称"""


class LLMCompressionStrategy(CompressionStrategy):
    """
    LLM 压缩策略 - 带熔断器
    
    熔断器状态机：
    - CLOSED: 正常状态，允许请求
    - OPEN: 熔断状态，拒绝请求
    - HALF_OPEN: 半开状态，允许探测请求
    
    常量说明：
    - COMPRESSOR_CHECK_INTERVAL: 压缩器可用性检查间隔（秒）
    - LONG_TEXT_THRESHOLD: 长文本阈值，超过此值使用云端压缩
    - CHUNK_TEXT_THRESHOLD: 分块压缩阈值，超过此值分块处理
    - CHUNK_SIZE: 分块大小
    """
    
    COMPRESSOR_CHECK_INTERVAL = 300
    LONG_TEXT_THRESHOLD = 10000
    CHUNK_TEXT_THRESHOLD = 6000
    CHUNK_SIZE = 3000
    MAX_COMPRESSION_RATIO = 0.7
    LONG_TEXT_COMPRESSION_RATIO = 0.8
    
    _DEFAULT_CB_THRESHOLD = 5
    _DEFAULT_CB_RESET_TIMEOUT = 300
    _DEFAULT_CB_HALF_OPEN_WINDOW = 30
    
    def __init__(self, mem_config):
        self._mem_config = mem_config
        self._llm_client = None
        self._last_check = 0
        self._failure_count = 0
        self._available = False
        self._log = get_logger()
        
        self._cb_half_open_window = self._DEFAULT_CB_HALF_OPEN_WINDOW
        
        self._circuit_breaker = {
            "failures": 0,
            "last_failure": 0,
            "open_until": 0,
            "threshold": self._DEFAULT_CB_THRESHOLD,
            "reset_timeout": self._DEFAULT_CB_RESET_TIMEOUT
        }
        try:
            async_config = mem_config.async_processor
            self._circuit_breaker["threshold"] = getattr(async_config, 'circuit_breaker_threshold', self._DEFAULT_CB_THRESHOLD)
            self._circuit_breaker["reset_timeout"] = getattr(async_config, 'circuit_breaker_reset_timeout', self._DEFAULT_CB_RESET_TIMEOUT)
        except Exception:
            pass
    
    def compress(self, text: str) -> Optional[str]:
        if not self.is_available():
            return None
        try:
            result = self._compress_text(text)
            self._reset_circuit_breaker()
            return result
        except Exception as e:
            self._record_failure(str(e))
            return None
    
    def is_available(self) -> bool:
        current_time = time.time()
        
        if current_time < self._circuit_breaker["open_until"]:
            if current_time > self._circuit_breaker["open_until"] - self._cb_half_open_window:
                if self._probe_connection():
                    self._log.info("CIRCUIT_BREAKER_HALF_OPEN_SUCCESS")
                    return True
            return False
        
        check_interval = self._get_check_interval()
        if current_time - self._last_check < check_interval:
            return self._available
        
        self._last_check = current_time
        self._available = self._probe_connection()
        return self._available
    
    def _get_check_interval(self) -> int:
        """获取检查间隔"""
        try:
            return self._mem_config.async_processor.compressor_check_interval
        except Exception:
            return self.COMPRESSOR_CHECK_INTERVAL
    
    def _probe_connection(self) -> bool:
        """探测连接是否可用"""
        try:
            if self._llm_client is None:
                self._init_llm_client()
            
            if self._llm_client is None:
                return False
            
            return self._llm_client.is_available()
        except Exception:
            return False
    
    def _init_llm_client(self):
        """初始化 LLM 客户端"""
        try:
            from llm_client import get_llm_client
            self._llm_client = get_llm_client()
        except Exception:
            pass
    
    def _record_failure(self, error: str):
        """记录失败"""
        self._circuit_breaker["failures"] += 1
        self._circuit_breaker["last_failure"] = time.time()
        
        if self._circuit_breaker["failures"] >= self._circuit_breaker["threshold"]:
            self._circuit_breaker["open_until"] = time.time() + self._circuit_breaker["reset_timeout"]
            self._log.warning(
                "CIRCUIT_BREAKER_OPENED",
                failures=self._circuit_breaker["failures"],
                reset_timeout=self._circuit_breaker["reset_timeout"]
            )
    
    def _reset_circuit_breaker(self):
        """重置熔断器"""
        self._circuit_breaker["failures"] = 0
        self._circuit_breaker["open_until"] = 0
    
    @property
    def name(self) -> str:
        return "llm"
    
    def _compress_text(self, text: str) -> Optional[str]:
        """压缩文本"""
        if not text:
            return None
        
        text_len = len(text)
        
        if text_len > self.LONG_TEXT_THRESHOLD:
            return self._compress_long_text(text)
        
        if text_len > self.CHUNK_TEXT_THRESHOLD:
            return self._compress_chunked_text(text)
        
        return self._compress_single_text(text)
    
    def _compress_single_text(self, text: str) -> Optional[str]:
        """压缩单个文本块"""
        try:
            from prompts import get_prompt_manager
            prompt_manager = get_prompt_manager()
            
            memory_list = [line.strip() for line in text.split('\n') if line.strip()]
            
            if len(memory_list) > 1:
                messages = prompt_manager.get_compression_prompt(memory_list)
            else:
                messages = [
                    {"role": "system", "content": "你是一个记忆压缩专家。"},
                    {"role": "user", "content": f"""请将以下对话记录压缩为简洁的摘要，保留所有关键信息。

原始内容：
{text}

压缩要求：
1. 保留所有专有名词、人名、地名、数值
2. 保留因果关系和关键决策
3. 去除冗余和修饰性内容
4. 压缩后长度约为原文的30%-50%

压缩结果："""}
                ]
            
            result = self._llm_client.chat(messages, max_tokens=500)
            return result.strip() if result else None
            
        except Exception:
            return None
    
    def _compress_chunked_text(self, text: str) -> Optional[str]:
        """分块压缩长文本"""
        try:
            chunks = []
            
            lines = text.split('\n')
            current_chunk = []
            current_len = 0
            
            for line in lines:
                if current_len + len(line) > self.CHUNK_SIZE and current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_len = 0
                current_chunk.append(line)
                current_len += len(line)
            
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
            
            compressed_chunks = []
            for chunk in chunks:
                compressed = self._compress_single_text(chunk)
                if compressed:
                    compressed_chunks.append(compressed)
            
            if not compressed_chunks:
                return None
            
            merged = '\n'.join(compressed_chunks)
            
            if len(merged) < len(text) * self.MAX_COMPRESSION_RATIO:
                return self._compress_single_text(merged)
            
            return merged
            
        except Exception:
            return None
    
    def _compress_long_text(self, text: str) -> Optional[str]:
        """压缩超长文本"""
        try:
            try:
                from config import config
                if config.cloud.enabled and config.cloud.api_key:
                    return self._cloud_compress_text(text)
            except Exception:
                pass
            
            chunked_result = self._compress_chunked_text(text)
            if chunked_result and len(chunked_result) < len(text) * self.LONG_TEXT_COMPRESSION_RATIO:
                return chunked_result
            
            return None
            
        except Exception:
            return None
    
    def _cloud_compress_text(self, text: str) -> Optional[str]:
        """使用云端压缩"""
        try:
            from cloud_client import CloudClientFactory
            from config import config
            
            cloud_client = CloudClientFactory.create(
                provider=config.cloud.provider,
                api_key=config.cloud.api_key,
                model=config.cloud.model,
                base_url=config.cloud.base_url
            )
            
            if not cloud_client or not cloud_client.is_available():
                return None
            
            prompt = f"""请将以下对话记录压缩为简洁的摘要，保留所有关键信息。

原始内容：
{text}

压缩要求：
1. 保留所有专有名词、人名、地名、数值
2. 保留因果关系和关键决策
3. 去除冗余和修饰性内容
4. 压缩后长度约为原文的30%-50%

压缩结果："""
            
            result = cloud_client.chat([{"role": "user", "content": prompt}])
            return result.strip() if result else None
            
        except Exception:
            return None


class RuleBasedCompressionStrategy(CompressionStrategy):
    """
    规则压缩策略（降级方案）
    
    当 LLM 不可用时使用规则进行压缩
    
    常量说明：
    - MAX_KEY_SENTENCES: 最大保留关键句数
    - TARGET_RATIO_BUFFER: 目标压缩比率缓冲值
    """
    
    MAX_KEY_SENTENCES = 5
    TARGET_RATIO_BUFFER = 0.1
    
    def __init__(self, mem_config):
        self._mem_config = mem_config
    
    def compress(self, text: str) -> Optional[str]:
        parts = re.split(r'([。！？\n])', text)
        
        sentences = []
        for i in range(0, len(parts) - 1, 2):
            sentence = parts[i].strip()
            punctuation = parts[i + 1] if i + 1 < len(parts) else ''
            if sentence:
                sentences.append(sentence + punctuation)
        
        if len(parts) % 2 == 1 and parts[-1].strip():
            sentences.append(parts[-1].strip())
        
        key_sentences = []
        key_patterns = self._mem_config.compression.key_patterns.split('|')
        
        for sentence in sentences:
            for pattern in key_patterns:
                if re.search(pattern, sentence):
                    if sentence not in key_sentences:
                        key_sentences.append(sentence)
                    break
        
        target_ratio = self._mem_config.compression.target_ratio + self.TARGET_RATIO_BUFFER
        max_segment = self._mem_config.compression.max_segment_length
        
        if key_sentences:
            compressed = ''.join(key_sentences[:self.MAX_KEY_SENTENCES])
            if len(compressed) < len(text) * target_ratio:
                return compressed
        
        if len(text) > max_segment:
            return text[:max_segment] + '...'
        
        return None
    
    def is_available(self) -> bool:
        return True
    
    @property
    def name(self) -> str:
        return "rule_based"


class CompressionStrategyChain:
    """
    压缩策略链 - 按优先级尝试多种策略
    
    策略优先级：
    1. LLM 压缩（高质量，需要可用性）
    2. 规则压缩（降级方案，始终可用）
    """
    
    def __init__(self, strategies: List[CompressionStrategy]):
        self._strategies = strategies
    
    def compress(self, text: str) -> Tuple[Optional[str], str]:
        """
        尝试压缩文本
        
        Returns:
            (压缩结果, 使用的策略名称)
        """
        for strategy in self._strategies:
            if strategy.is_available():
                result = strategy.compress(text)
                if result:
                    return result, strategy.name
        return None, "none"


def create_compression_chain(mem_config) -> CompressionStrategyChain:
    """
    创建压缩策略链
    
    Args:
        mem_config: 记忆配置
    
    Returns:
        压缩策略链实例
    """
    return CompressionStrategyChain([
        LLMCompressionStrategy(mem_config),
        RuleBasedCompressionStrategy(mem_config)
    ])
