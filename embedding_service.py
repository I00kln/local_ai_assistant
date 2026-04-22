# embedding_service.py
# 共享的 ONNX 嵌入服务 - 单例模式
import os
import threading
import hashlib
from collections import OrderedDict
from typing import List, Optional
import numpy as np


class EmbeddingService:
    """
    共享的 ONNX 嵌入服务
    
    单例模式，确保 ONNX 模型只加载一次
    供 VectorStore 和 NonsenseFilter 共享使用
    
    缓存策略：
    - 使用 OrderedDict 实现 LRU 淘汰
    - 缓存满时淘汰最久未使用的条目
    - 使用稳定的哈希键（MD5）
    """
    
    _instance: Optional['EmbeddingService'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._tokenizer = None
        self._session = None
        self._dimension = None
        self._model_lock = threading.Lock()
        self._fallback_mode = False
        self._init_error = None
        self._embedding_cache = OrderedDict()
        self._cache_max_size = 1000
        self._initialized = True
    
    def _get_cache_key(self, text: str) -> str:
        """
        生成稳定的缓存键
        
        使用 MD5 替代 Python hash()，确保跨进程稳定性
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _ensure_initialized(self):
        """
        确保模型已初始化（延迟加载）
        
        降级策略：
        - 模型文件不存在时，使用随机向量降级模式
        - 记录错误信息，便于后续修复
        - 系统仍可启动，但向量检索功能受限
        """
        if self._session is not None or self._fallback_mode:
            return
        
        with self._model_lock:
            if self._session is not None or self._fallback_mode:
                return
            
            try:
                from config import config
                import onnxruntime as ort
                from tokenizers import Tokenizer
                
                model_path = config.onnx_model_path
                
                tokenizer_path = os.path.join(model_path, "tokenizer.json")
                model_file_path = os.path.join(model_path, "model.onnx")
                
                if not os.path.exists(tokenizer_path):
                    raise FileNotFoundError(f"Tokenizer 文件不存在: {tokenizer_path}")
                
                if not os.path.exists(model_file_path):
                    raise FileNotFoundError(f"模型文件不存在: {model_file_path}")
                
                self._tokenizer = Tokenizer.from_file(tokenizer_path)
                self._tokenizer.enable_padding()
                self._tokenizer.enable_truncation(max_length=512)
                
                self._session = ort.InferenceSession(
                    model_file_path,
                    providers=['CPUExecutionProvider']
                )
                
                self._dimension = config.embedding_dimension
                print("[EmbeddingService] ONNX嵌入模型加载完成")
                
            except FileNotFoundError as e:
                self._init_error = str(e)
                print(f"[警告] EmbeddingService 嵌入模型文件缺失，启用降级模式: {e}")
                self._enable_fallback()
                
            except ImportError as e:
                self._init_error = str(e)
                print(f"[警告] EmbeddingService 依赖库缺失，启用降级模式: {e}")
                self._enable_fallback()
                
            except Exception as e:
                self._init_error = str(e)
                print(f"[错误] EmbeddingService 嵌入模型加载失败，启用降级模式: {e}")
                self._enable_fallback()
    
    def _enable_fallback(self):
        """
        启用降级模式
        
        使用随机向量替代真实嵌入：
        - 系统可以启动
        - 向量检索功能不可用（相似度计算无意义）
        - 应尽快修复模型文件
        """
        from config import config
        self._fallback_mode = True
        self._dimension = config.embedding_dimension
        print(f"[EmbeddingService] 降级模式已启用，向量维度: {self._dimension}")
    
    @property
    def is_available(self) -> bool:
        """检查服务是否可用"""
        self._ensure_initialized()
        return self._session is not None
    
    @property
    def is_fallback(self) -> bool:
        """检查是否处于降级模式"""
        return self._fallback_mode
    
    @property
    def dimension(self) -> int:
        """获取向量维度"""
        if self._dimension is None:
            from config import config
            self._dimension = config.embedding_dimension
        return self._dimension
    
    def embed(self, texts: List[str], use_cache: bool = True) -> List[List[float]]:
        """
        生成文本嵌入向量
        
        Args:
            texts: 文本列表
            use_cache: 是否使用缓存
        
        Returns:
            嵌入向量列表
        """
        self._ensure_initialized()
        
        if not texts:
            return []
        
        result = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []
        
        if use_cache:
            for idx, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                if cache_key in self._embedding_cache:
                    self._embedding_cache.move_to_end(cache_key)
                    result[idx] = self._embedding_cache[cache_key]
                else:
                    uncached_indices.append(idx)
                    uncached_texts.append(text)
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = texts
        
        if uncached_texts:
            if self._fallback_mode:
                new_embeddings = self._generate_fallback_embeddings(uncached_texts)
            else:
                new_embeddings = self._generate_real_embeddings(uncached_texts)
            
            for idx, text, embedding in zip(uncached_indices, uncached_texts, new_embeddings):
                result[idx] = embedding
                
                if use_cache:
                    cache_key = self._get_cache_key(text)
                    if len(self._embedding_cache) >= self._cache_max_size:
                        self._embedding_cache.popitem(last=False)
                    self._embedding_cache[cache_key] = embedding
        
        return result
    
    def embed_single(self, text: str, use_prefix: bool = False) -> np.ndarray:
        """
        生成单个文本的嵌入向量
        
        Args:
            text: 文本
            use_prefix: 是否添加前缀（用于废话过滤器）
        
        Returns:
            嵌入向量 (numpy array)
        """
        self._ensure_initialized()
        
        if self._fallback_mode:
            return self._generate_fallback_embedding(text)
        
        text_to_embed = f"为这个句子生成表示：{text}" if use_prefix else text
        embeddings = self._generate_real_embeddings([text_to_embed])
        return np.array(embeddings[0], dtype=np.float32)
    
    def _generate_real_embeddings(self, texts: List[str]) -> List[List[float]]:
        """生成真实的嵌入向量"""
        import time
        from metrics import get_metrics_collector
        
        start_time = time.perf_counter()
        
        encoded = self._tokenizer.encode_batch(texts)
        
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        
        inputs_onnx = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
        outputs = self._session.run(None, inputs_onnx)
        
        last_hidden_state = outputs[0]
        embeddings = last_hidden_state[:, 0, :]
        
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-12)
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        get_metrics_collector().record_embedding_latency(duration_ms)
        
        return embeddings.tolist()
    
    def _generate_fallback_embeddings(self, texts: List[str]) -> List[List[float]]:
        """生成降级模式的嵌入向量"""
        embeddings = []
        for text in texts:
            embedding = self._generate_fallback_embedding(text)
            embeddings.append(embedding.tolist())
        return embeddings
    
    def _generate_fallback_embedding(self, text: str) -> np.ndarray:
        """生成单个文本的降级模式嵌入向量"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        np.random.seed(int(text_hash[:8], 16))
        embedding = np.random.randn(self._dimension).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def clear_cache(self):
        """清空嵌入缓存"""
        self._embedding_cache.clear()
    
    def get_cache_stats(self) -> dict:
        """获取缓存统计信息"""
        return {
            "size": len(self._embedding_cache),
            "max_size": self._cache_max_size,
            "hit_rate": 0.0
        }
    
    def cleanup(self):
        """
        清理资源
        
        释放 ONNX 模型会话和缓存
        用于应用关闭时的资源回收
        """
        with self._model_lock:
            if self._session is not None:
                del self._session
                self._session = None
                print("[EmbeddingService] ONNX 会话已释放")
            
            if self._tokenizer is not None:
                del self._tokenizer
                self._tokenizer = None
                print("[EmbeddingService] Tokenizer 已释放")
            
            self._embedding_cache.clear()
            print("[EmbeddingService] 缓存已清空")


def get_embedding_service() -> EmbeddingService:
    """获取 EmbeddingService 单例"""
    return EmbeddingService()
