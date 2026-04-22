# vector_store.py
# L2 向量存储层 - ChromaDB 实现
import os
import threading
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

from config import config


class ONNXEmbeddingFunction:
    """
    自定义嵌入函数 - 使用现有的ONNX BGE模型
    
    ChromaDB要求嵌入函数实现__call__方法
    
    特性：
    - 延迟加载：首次使用时才加载模型
    - 线程安全：使用锁保护初始化
    - 降级模式：模型加载失败时使用随机向量
    """
    
    def __init__(self):
        self._tokenizer = None
        self._session = None
        self._dimension = None
        self._lock = threading.Lock()
        self.name = "onnx_bge"
        self._fallback_mode = False
        self._init_error = None
    
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
        
        with self._lock:
            if self._session is not None or self._fallback_mode:
                return
            
            try:
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
                print("ONNX嵌入模型延迟加载完成")
                
            except FileNotFoundError as e:
                self._init_error = str(e)
                print(f"[警告] 嵌入模型文件缺失，启用降级模式: {e}")
                self._enable_fallback()
                
            except ImportError as e:
                self._init_error = str(e)
                print(f"[警告] 依赖库缺失，启用降级模式: {e}")
                self._enable_fallback()
                
            except Exception as e:
                self._init_error = str(e)
                print(f"[错误] 嵌入模型加载失败，启用降级模式: {e}")
                self._enable_fallback()
    
    def _enable_fallback(self):
        """
        启用降级模式
        
        使用随机向量替代真实嵌入：
        - 系统可以启动
        - 向量检索功能不可用（相似度计算无意义）
        - 应尽快修复模型文件
        """
        self._fallback_mode = True
        self._dimension = config.embedding_dimension
        print("[降级] 嵌入模型使用随机向量模式，请尽快修复模型文件")
    
    def is_fallback_mode(self) -> bool:
        """检查是否处于降级模式"""
        return self._fallback_mode
    
    def get_init_error(self) -> Optional[str]:
        """获取初始化错误信息"""
        return self._init_error
    
    @property
    def tokenizer(self):
        self._ensure_initialized()
        return self._tokenizer
    
    @property
    def session(self):
        self._ensure_initialized()
        return self._session
    
    @property
    def dimension(self):
        self._ensure_initialized()
        return self._dimension
    
    def __call__(self, texts: List[str]) -> List[List[float]]:
        """
        将文本列表转换为嵌入向量列表
        
        降级模式：
        - 使用基于文本哈希的确定性随机向量
        - 相同文本始终生成相同向量
        - 向量检索功能受限
        """
        import time
        import numpy as np
        from metrics import get_metrics_collector
        
        self._ensure_initialized()
        
        start_time = time.perf_counter()
        
        if self._fallback_mode:
            result = self._generate_fallback_embeddings(texts)
        else:
            texts_with_prefix = [f"为这个句子生成表示：{t}" for t in texts]
            
            encoded = self._tokenizer.encode_batch(texts_with_prefix)
            
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
            
            result = embeddings.tolist()
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        get_metrics_collector().record_embedding_latency(duration_ms)
        
        return result
    
    def _generate_fallback_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        生成降级模式的嵌入向量
        
        使用文本哈希生成确定性随机向量：
        - 相同文本始终生成相同向量
        - 不同文本生成不同向量
        - 向量维度与正常模式一致
        """
        import numpy as np
        import hashlib
        
        embeddings = []
        for text in texts:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            np.random.seed(int(text_hash[:8], 16))
            embedding = np.random.randn(self._dimension).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding.tolist())
        
        return embeddings


class VectorStore:
    """
    L2 向量存储层 - ChromaDB
    
    功能：
    - 向量存储与检索
    - 元数据过滤
    - 自动持久化
    - 线程安全
    - 内置去重
    """
    
    def __init__(self, collection_name: str = "memories", persist_directory: str = None):
        """
        初始化向量存储
        
        Args:
            collection_name: 集合名称
            persist_directory: 持久化目录（默认使用config配置）
        """
        if persist_directory is None:
            persist_directory = config.chroma_persist_dir
        
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # 确保目录存在
        os.makedirs(persist_directory, exist_ok=True)
        
        # 初始化ChromaDB
        self._init_chroma()
        
        # 线程锁（虽然ChromaDB内部线程安全，但用于保护批量操作）
        self.lock = threading.RLock()
        
        print(f"向量存储初始化完成: {persist_directory}")
    
    def _init_chroma(self):
        """初始化ChromaDB客户端和集合"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # 创建持久化客户端
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # 创建自定义嵌入函数
            self.embedding_function = ONNXEmbeddingFunction()
            
            # 获取或创建集合（不使用embedding_function参数，手动处理嵌入）
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "ip"}
            )
            
            print(f"集合 '{self.collection_name}' 加载完成，当前文档数: {self.collection.count()}")
            
        except ImportError:
            raise ImportError("请安装 chromadb: pip install chromadb")
        except Exception as e:
            raise RuntimeError(f"ChromaDB初始化失败: {e}")
    
    def add(
        self, 
        texts: List[str], 
        metadatas: List[Dict[str, Any]] = None,
        ids: List[str] = None,
        sqlite_store = None
    ) -> List[str]:
        """
        添加文档到向量库
        
        Args:
            texts: 文本列表
            metadatas: 元数据列表
            ids: 文档ID列表（不提供则自动生成）
            sqlite_store: SQLite存储实例（用于降级）
        
        Returns:
            文档ID列表
        
        降级策略：
        - 磁盘满时，降级到 SQLite 存储
        - 连接失败时，记录错误并返回空列表
        """
        if not texts:
            return []
        
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in texts]
        
        if metadatas is None:
            metadatas = [{
                "timestamp": datetime.now().isoformat(),
                "type": "memory"
            } for _ in texts]
        else:
            for meta in metadatas:
                if "timestamp" not in meta:
                    meta["timestamp"] = datetime.now().isoformat()
        
        try:
            embeddings = self.embedding_function(texts)
            
            with self.lock:
                self.collection.add(
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
                
                print(f"添加 {len(texts)} 条记忆到向量库，当前总数: {self.collection.count()}")
                
                return ids
                
        except OSError as e:
            error_msg = str(e).lower()
            if "no space left" in error_msg or "disk full" in error_msg or "enospc" in error_msg:
                print(f"[严重] ChromaDB 磁盘满，降级到 SQLite 存储")
                
                if sqlite_store:
                    try:
                        from sqlite_store import MemoryRecord
                        for text, meta in zip(texts, metadatas or []):
                            record = MemoryRecord(
                                text=text,
                                metadata=meta,
                                is_vectorized=-1
                            )
                            sqlite_store.add(record)
                        print(f"[降级] {len(texts)} 条记忆已存入 SQLite（待后续升级）")
                        return []
                    except Exception as sqlite_error:
                        print(f"[错误] SQLite 降级存储失败: {sqlite_error}")
                        return []
                else:
                    print(f"[错误] 无 SQLite 存储，记忆丢失")
                    return []
            raise
            
        except Exception as e:
            print(f"[错误] ChromaDB 写入失败: {e}")
            
            if sqlite_store:
                try:
                    from sqlite_store import MemoryRecord
                    for text, meta in zip(texts, metadatas or []):
                        record = MemoryRecord(
                            text=text,
                            metadata=meta,
                            is_vectorized=0
                        )
                        sqlite_store.add(record)
                    print(f"[降级] {len(texts)} 条记忆已存入 SQLite（待重试）")
                    return []
                except Exception:
                    pass
            
            return []
    
    def search(
        self, 
        query: str, 
        n_results: int = None,
        where: Dict = None,
        where_document: Dict = None
    ) -> List[Dict[str, Any]]:
        """
        搜索相似文档
        
        Args:
            query: 查询文本
            n_results: 返回结果数量
            where: 元数据过滤条件
            where_document: 文档内容过滤条件
        
        Returns:
            搜索结果列表
        """
        if n_results is None:
            n_results = config.max_retrieve_results
        
        if self.collection.count() == 0:
            return []
        
        # 生成查询嵌入
        query_embedding = self.embedding_function([query])[0]
        
        # 执行搜索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self.collection.count()),
            where=where,
            where_document=where_document,
            include=["documents", "metadatas", "distances"]
        )
        
        # 格式化结果
        formatted_results = []
        if results and results["ids"]:
            for i, doc_id in enumerate(results["ids"][0]):
                formatted_results.append({
                    "id": doc_id,
                    "text": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "similarity": results["distances"][0][i] if results["distances"] else 0.0
                })
        
        return formatted_results
    
    def search_by_embedding(
        self,
        embedding: List[float],
        n_results: int = None,
        where: Dict = None
    ) -> List[Dict[str, Any]]:
        """
        通过嵌入向量搜索
        
        Args:
            embedding: 嵌入向量
            n_results: 返回结果数量
            where: 元数据过滤条件
        
        Returns:
            搜索结果列表
        """
        if n_results is None:
            n_results = config.max_retrieve_results
        
        if self.collection.count() == 0:
            return []
        
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(n_results, self.collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        formatted_results = []
        if results and results["ids"]:
            for i, doc_id in enumerate(results["ids"][0]):
                formatted_results.append({
                    "id": doc_id,
                    "text": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "similarity": results["distances"][0][i] if results["distances"] else 0.0
                })
        
        return formatted_results
    
    def get(self, ids: List[str] = None, where: Dict = None) -> List[Dict[str, Any]]:
        """
        获取文档
        
        Args:
            ids: 文档ID列表
            where: 元数据过滤条件
        
        Returns:
            文档列表
        """
        results = self.collection.get(
            ids=ids,
            where=where,
            include=["documents", "metadatas"]
        )
        
        formatted_results = []
        if results and results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                formatted_results.append({
                    "id": doc_id,
                    "text": results["documents"][i] if results["documents"] else "",
                    "metadata": results["metadatas"][i] if results["metadatas"] else {}
                })
        
        return formatted_results
    
    def delete(
        self, 
        ids: List[str] = None, 
        where: Dict = None,
        sqlite_store = None
    ) -> int:
        """
        删除文档
        
        Args:
            ids: 文档ID列表
            where: 元数据过滤条件
            sqlite_store: SQLite存储实例（可选，同步删除SQLite记录）
        
        Returns:
            删除的文档数量
        """
        if ids is None and where is None:
            return 0
        
        if ids:
            count = len(ids)
            
            if sqlite_store:
                for doc_id in ids:
                    sqlite_store.delete_by_vector_id(doc_id)
        else:
            results = self.collection.get(where=where)
            count = len(results["ids"]) if results["ids"] else 0
            
            if sqlite_store and results["ids"]:
                for doc_id in results["ids"]:
                    sqlite_store.delete_by_vector_id(doc_id)
        
        self.collection.delete(ids=ids, where=where)
        
        return count
    
    def update(
        self,
        ids: List[str],
        texts: List[str] = None,
        metadatas: List[Dict] = None
    ):
        """
        更新文档
        
        Args:
            ids: 文档ID列表
            texts: 新文本列表
            metadatas: 新元数据列表
        """
        self.collection.update(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
    
    def count(self) -> int:
        """获取文档数量"""
        return self.collection.count()
    
    def clear(self):
        """清空集合"""
        # 删除并重建集合
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "ip"}
        )
        print(f"集合 '{self.collection_name}' 已清空")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        count = self.collection.count()
        
        # 获取所有文档的时间范围
        if count > 0:
            results = self.collection.get(
                limit=count,
                include=["metadatas"]
            )
            
            timestamps = []
            for meta in (results.get("metadatas") or []):
                if meta and "timestamp" in meta:
                    timestamps.append(meta["timestamp"])
            
            oldest = min(timestamps) if timestamps else None
            newest = max(timestamps) if timestamps else None
        else:
            oldest = newest = None
        
        return {
            "count": count,
            "oldest_memory": oldest,
            "newest_memory": newest,
            "persist_directory": self.persist_directory
        }
    
    def deduplicate(self, sqlite_store=None) -> int:
        """
        去重 - 基于文本内容
        
        Args:
            sqlite_store: SQLite存储实例（可选，用于同步删除）
        
        Returns:
            移除的重复文档数量
        """
        if self.collection.count() == 0:
            return 0
        
        # 获取所有文档
        results = self.collection.get(
            include=["documents", "metadatas"]
        )
        
        if not results["ids"]:
            return 0
        
        # 找出重复的文档
        seen_texts = {}
        duplicate_ids = []
        
        for i, text in enumerate(results["documents"]):
            doc_id = results["ids"][i]
            if text in seen_texts:
                duplicate_ids.append(doc_id)
            else:
                seen_texts[text] = doc_id
        
        # 删除重复文档
        if duplicate_ids:
            self.collection.delete(ids=duplicate_ids)
            
            # 同步删除SQLite中的记录
            if sqlite_store:
                for dup_id in duplicate_ids:
                    sqlite_store.delete_by_vector_id(dup_id)
            
            print(f"去重完成：移除 {len(duplicate_ids)} 条重复记忆")
        
        return len(duplicate_ids)
    
    def __len__(self) -> int:
        return self.collection.count()
    
    def get_all_ids(self, limit: int = 1000) -> List[str]:
        """
        获取所有文档ID
        
        Args:
            limit: 最大返回数量
        
        Returns:
            文档ID列表
        """
        try:
            results = self.collection.get(limit=limit)
            return results.get("ids", [])
        except Exception as e:
            print(f"获取文档ID失败: {e}")
            return []
    
    def get_metadata(self, doc_id: str) -> Optional[Dict]:
        """
        获取指定文档的元数据
        
        Args:
            doc_id: 文档ID
        
        Returns:
            元数据字典，如果不存在返回None
        """
        try:
            results = self.collection.get(ids=[doc_id])
            metadatas = results.get("metadatas", [])
            if metadatas:
                return metadatas[0]
            return None
        except Exception as e:
            print(f"获取元数据失败: {e}")
            return None
    
    def update_metadata(self, doc_id: str, metadata: Dict) -> bool:
        """
        更新指定文档的元数据
        
        Args:
            doc_id: 文档ID
            metadata: 新的元数据（会与现有元数据合并）
        
        Returns:
            是否更新成功
        """
        try:
            existing = self.get_metadata(doc_id)
            if existing:
                existing.update(metadata)
                self.collection.update(ids=[doc_id], metadatas=[existing])
                return True
            return False
        except Exception as e:
            print(f"更新元数据失败: {e}")
            return False
    
    def pre_warm(self):
        """
        预热嵌入模型
        
        在后台线程中加载ONNX模型，避免首次使用时的阻塞
        """
        try:
            dummy_text = "预热模型"
            self.embedding_function([dummy_text])
            print("向量存储：嵌入模型预热完成")
        except Exception as e:
            print(f"向量存储：预热失败: {e}")
    
    def close(self):
        """
        关闭向量存储，释放资源
        
        用于会话结束或清空时的资源回收
        """
        with self.lock:
            try:
                # ChromaDB没有显式close方法
                # 但可以清理引用
                if hasattr(self.client, '_server'):
                    del self.client._server
            except Exception:
                pass
            
            # 清空嵌入模型缓存
            if hasattr(self.embedding_function, '_session'):
                self.embedding_function._session = None
                self.embedding_function._tokenizer = None


# 全局实例
_vector_store: Optional[VectorStore] = None
_instance_lock = threading.Lock()


def get_vector_store() -> VectorStore:
    """
    获取全局向量存储实例（线程安全单例）
    """
    global _vector_store
    if _vector_store is None:
        with _instance_lock:
            if _vector_store is None:
                _vector_store = VectorStore()
    return _vector_store


def reset_vector_store():
    """
    重置全局向量存储实例
    
    用于清空会话或重新初始化
    """
    global _vector_store
    with _instance_lock:
        if _vector_store is not None:
            _vector_store.close()
            _vector_store = None
