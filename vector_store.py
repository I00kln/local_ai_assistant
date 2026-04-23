# vector_store.py
# L2 向量存储层 - ChromaDB 实现
import os
import time
import threading
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

from config import config
from memory_tags import MemoryTags


class ONNXEmbeddingFunction:
    """
    自定义嵌入函数 - 使用共享的 EmbeddingService
    
    ChromaDB要求嵌入函数实现__call__方法
    
    特性：
    - 使用共享的 EmbeddingService 单例
    - 延迟加载：首次使用时才加载模型
    - 线程安全：使用锁保护初始化
    - 降级模式：模型加载失败时使用随机向量
    """
    
    def __init__(self):
        self._embedding_service = None
        self.name = "onnx_bge"
    
    def _ensure_initialized(self):
        """确保 EmbeddingService 已初始化"""
        if self._embedding_service is not None:
            return
        
        from embedding_service import get_embedding_service
        self._embedding_service = get_embedding_service()
    
    def is_fallback_mode(self) -> bool:
        """检查是否处于降级模式"""
        self._ensure_initialized()
        return self._embedding_service.is_fallback
    
    def get_init_error(self) -> Optional[str]:
        """获取初始化错误信息"""
        self._ensure_initialized()
        return self._embedding_service._init_error
    
    @property
    def tokenizer(self):
        self._ensure_initialized()
        return self._embedding_service._tokenizer
    
    @property
    def session(self):
        self._ensure_initialized()
        return self._embedding_service._session
    
    @property
    def dimension(self):
        self._ensure_initialized()
        return self._embedding_service.dimension
    
    def __call__(self, texts: List[str]) -> List[List[float]]:
        """
        将文本列表转换为嵌入向量列表
        
        委托给共享的 EmbeddingService
        """
        self._ensure_initialized()
        return self._embedding_service.embed(texts, use_cache=True)


class VectorStore:
    """
    L2 向量存储层 - ChromaDB
    
    功能：
    - 向量存储与检索
    - 元数据过滤
    - 自动持久化
    - 线程安全
    - 内置去重
    - 异步预热（不阻塞主线程）
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
        
        os.makedirs(persist_directory, exist_ok=True)
        
        self._prewarm_complete = False
        self._prewarm_error: Optional[str] = None
        self._prewarm_lock = threading.Lock()
        
        self._init_chroma()
        
        self.lock = threading.RLock()
        
        print(f"向量存储初始化完成: {persist_directory}")
    
    def _init_chroma(self):
        """初始化ChromaDB客户端和集合"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            self.embedding_function = ONNXEmbeddingFunction()
            
            print("[向量存储] 正在后台预加载ONNX嵌入模型...")
            prewarm_thread = threading.Thread(
                target=self._prewarm_embedding,
                daemon=True,
                name="EmbeddingPrewarm"
            )
            prewarm_thread.start()
            
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "ip",
                    "hnsw:M": 24,
                    "hnsw:construction_ef": 200,
                    "hnsw:search_ef": 100,
                }
            )
            
            print(f"集合 '{self.collection_name}' 加载完成，当前文档数: {self.collection.count()}")
            
        except ImportError:
            raise ImportError("请安装 chromadb: pip install chromadb")
        except Exception as e:
            raise RuntimeError(f"ChromaDB初始化失败: {e}")
    
    def _prewarm_embedding(self):
        """
        后台预热嵌入模型
        
        在独立线程中执行，不阻塞主线程
        记录预热状态，便于后续检查
        """
        try:
            self.embedding_function._ensure_initialized()
            with self._prewarm_lock:
                self._prewarm_complete = True
            print("[向量存储] ONNX 嵌入模型预热完成")
        except Exception as e:
            with self._prewarm_lock:
                self._prewarm_error = str(e)
            print(f"[向量存储] ONNX 嵌入模型预热失败: {e}")
    
    def is_prewarm_complete(self) -> bool:
        """检查预热是否完成"""
        with self._prewarm_lock:
            return self._prewarm_complete
    
    def get_prewarm_error(self) -> Optional[str]:
        """获取预热错误信息"""
        with self._prewarm_lock:
            return self._prewarm_error
    
    def wait_for_prewarm(self, timeout: float = 30.0) -> bool:
        """
        等待预热完成
        
        Args:
            timeout: 超时时间（秒）
        
        Returns:
            是否预热成功
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self._prewarm_lock:
                if self._prewarm_complete:
                    return True
                if self._prewarm_error:
                    return False
            time.sleep(0.1)
        return False
    
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
                MemoryTags.TIMESTAMP: datetime.now().isoformat(),
                "type": "memory"
            } for _ in texts]
        else:
            for meta in metadatas:
                if MemoryTags.TIMESTAMP not in meta:
                    meta[MemoryTags.TIMESTAMP] = datetime.now().isoformat()
        
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
                if meta and MemoryTags.TIMESTAMP in meta:
                    timestamps.append(meta[MemoryTags.TIMESTAMP])
            
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
