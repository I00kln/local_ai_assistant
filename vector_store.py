import os
import errno
import json
import time
import uuid
import hashlib
import threading
import shutil
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from config import config
from logger import get_logger
from memory_tags import MemoryTags
from models import MemoryRecord
from thread_pool_manager import submit_io_task


MIN_DISK_SPACE_MB = 100
MIN_DISK_SPACE_RATIO = 0.05


def check_disk_space(path: str, min_mb: int = MIN_DISK_SPACE_MB) -> Tuple[bool, int]:
    """
    检查磁盘空间是否足够
    
    Args:
        path: 检查路径
        min_mb: 最小所需空间（MB）
    
    Returns:
        (是否足够, 可用空间MB)
    """
    try:
        usage = shutil.disk_usage(path)
        free_mb = usage.free // (1024 * 1024)
        return free_mb >= min_mb, free_mb
    except Exception:
        return True, -1


def _serialize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    序列化 metadata 中的嵌套字典
    
    ChromaDB 不支持嵌套 dict 作为 metadata 值，
    需要将所有 dict 类型序列化为 JSON 字符串
    
    Args:
        metadata: 原始 metadata
    
    Returns:
        序列化后的 metadata（所有嵌套 dict 转为 JSON 字符串）
    """
    if not metadata:
        return metadata
    
    result = {}
    for key, value in metadata.items():
        if isinstance(value, dict):
            result[key] = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, list):
            try:
                json.dumps(value)
                result[key] = value
            except (TypeError, ValueError):
                result[key] = json.dumps(value, ensure_ascii=False)
        else:
            result[key] = value
    
    return result


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
        return self._embedding_service.embed(texts, use_cache=True, allow_fallback=True)


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
        
        self._log = get_logger()
        self._prewarm_complete = False
        self._prewarm_error: Optional[str] = None
        self._prewarm_lock = threading.Lock()
        self._warmup_started = False
        
        self._init_chroma()
        
        self.lock = threading.RLock()
        
        self._register_lifecycle()
        
        print(f"向量存储初始化完成: {persist_directory}")
    
    def _register_lifecycle(self):
        """注册到生命周期管理器"""
        try:
            from lifecycle_manager import get_lifecycle_manager, ServicePriority
            lifecycle = get_lifecycle_manager()
            lifecycle.register(
                name="vector_store",
                cleanup_fn=self.close,
                priority=ServicePriority.NORMAL,
                timeout=3.0
            )
        except Exception:
            pass
    
    def warmup(self):
        """
        预热嵌入模型
        
        应在系统启动时显式调用，而非在构造函数中自动启动后台线程
        """
        if self._warmup_started:
            return
        self._warmup_started = True
        
        print("[向量存储] 正在后台预加载ONNX嵌入模型...")
        submit_io_task(self._prewarm_embedding)
    
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
    
    @staticmethod
    def _generate_content_hash(text: str) -> str:
        """
        生成内容哈希（用于去重）
        
        使用文本哈希生成唯一标识，
        用于检测相同或相似内容的重复记录。
        
        Args:
            text: 文本内容
        
        Returns:
            格式为 16 位 16 进制字符串的内容哈希
        """
        normalized_text = text.strip().lower()
        return hashlib.sha256(normalized_text.encode('utf-8')).hexdigest()[:16]
    
    @staticmethod
    def _generate_memory_id(text: str = None) -> Tuple[str, str]:
        """
        生成记忆 ID（业务 ID + 内容哈希）
        
        设计原则：
        - 业务 ID (UUID): 主键，确保全局唯一
        - 内容哈希: 用于去重检测，而非主键
        
        优势：
        - 微小文本变化不会导致完全不同的 ID
        - 可以通过内容哈希检测相似内容
        - 支持幂等写入和去重
        
        Args:
            text: 文本内容（可选，用于生成内容哈希）
        
        Returns:
            (business_id, content_hash) 元组
        """
        business_id = str(uuid.uuid4())
        content_hash = VectorStore._generate_content_hash(text) if text else ""
        return business_id, content_hash
    
    @staticmethod
    def _generate_deterministic_id(text: str) -> str:
        """
        生成确定性 ID（向后兼容方法）
        
        保留此方法用于向后兼容，但建议使用 _generate_memory_id
        
        Args:
            text: 文本内容
        
        Returns:
            格式为 "mem_{hash[:32]}" 的确定性 ID
        """
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        return f"mem_{text_hash[:32]}"
    
    BATCH_SIZE = 500
    
    def add(
        self, 
        texts: List[str], 
        metadatas: List[Dict[str, Any]] = None,
        ids: List[str] = None,
        sqlite_store = None,
        upsert: bool = True
    ) -> List[str]:
        """
        添加文档到向量库（带分块处理）
        
        Args:
            texts: 文本列表
            metadatas: 元数据列表
            ids: 文档ID列表（不提供则自动生成确定性ID）
            sqlite_store: SQLite存储实例（用于降级）
            upsert: 是否使用 upsert 模式（幂等写入），默认 True
        
        Returns:
            文档ID列表
        
        幂等性保证：
        - 使用确定性 ID：mem_{text_hash[:16]}
        - upsert=True 时，相同 ID 的文档会被更新而非重复添加
        - 崩溃恢复后重新写入不会产生重复向量
        
        分块策略：
        - 每批最多 500 条记录，防止 Payload 过大
        - 分块写入，单块失败不影响其他块
        
        降级保护：
        - 禁止写入随机向量，防止污染向量库
        - 降级模式下仅存储到 SQLite，标记为 pending_embedding
        - 磁盘空间不足时预检查并降级
        """
        if not texts:
            return []
        
        chroma_path = getattr(self, 'persist_directory', None)
        if chroma_path is None:
            chroma_path = getattr(config, 'chroma_persist_dir', 'chroma_db')
        has_space, free_mb = check_disk_space(chroma_path)
        
        if not has_space:
            print(f"[警告] 磁盘空间不足（可用 {free_mb}MB），降级到 SQLite 存储")
            
            if sqlite_store:
                try:
                    for i, text in enumerate(texts):
                        meta = metadatas[i] if metadatas and i < len(metadatas) else {}
                        meta["pending_embedding"] = True
                        meta["disk_full_fallback"] = True
                        record = MemoryRecord(
                            text=text,
                            source=meta.get("source", "local"),
                            metadata=meta,
                            is_vectorized=-1
                        )
                        sqlite_store.add(record)
                    print(f"[降级] 已将 {len(texts)} 条记录存储到 SQLite（磁盘空间不足）")
                except Exception as e:
                    print(f"[错误] SQLite 降级存储失败: {e}")
            return []
        
        if self.embedding_function.is_fallback_mode():
            print(f"[警告] EmbeddingService 处于降级模式，禁止写入向量库，仅存储到 SQLite")
            
            if sqlite_store:
                try:
                    for i, text in enumerate(texts):
                        meta = metadatas[i] if metadatas and i < len(metadatas) else {}
                        meta["pending_embedding"] = True
                        record = MemoryRecord(
                            text=text,
                            source=meta.get("source", "local"),
                            metadata=meta,
                            is_vectorized=-1
                        )
                        sqlite_store.add(record)
                    print(f"[降级] 已将 {len(texts)} 条记录存储到 SQLite，标记为 pending_embedding")
                except Exception as e:
                    print(f"[错误] SQLite 降级存储失败: {e}")
            return []
        
        if ids is None:
            ids = [self._generate_deterministic_id(text) for text in texts]
        
        if metadatas is None:
            metadatas = [{
                MemoryTags.TIMESTAMP: datetime.now().isoformat(),
                "type": "memory"
            } for _ in texts]
        else:
            for meta in metadatas:
                if MemoryTags.TIMESTAMP not in meta:
                    meta[MemoryTags.TIMESTAMP] = datetime.now().isoformat()
        
        serialized_metadatas = [_serialize_metadata(meta) for meta in metadatas]
        
        all_ids = []
        total_count = len(texts)
        
        for batch_start in range(0, total_count, self.BATCH_SIZE):
            batch_end = min(batch_start + self.BATCH_SIZE, total_count)
            batch_texts = texts[batch_start:batch_end]
            batch_ids = ids[batch_start:batch_end]
            batch_metadatas = serialized_metadatas[batch_start:batch_end]
            
            try:
                embeddings = self.embedding_function(batch_texts)
                
                with self.lock:
                    if upsert:
                        self.collection.upsert(
                            documents=batch_texts,
                            embeddings=embeddings,
                            metadatas=batch_metadatas,
                            ids=batch_ids
                        )
                    else:
                        self.collection.add(
                            documents=batch_texts,
                            embeddings=embeddings,
                            metadatas=batch_metadatas,
                            ids=batch_ids
                        )
                
                all_ids.extend(batch_ids)
                
            except OSError as e:
                is_disk_full = (
                    e.errno == errno.ENOSPC or
                    "no space left" in str(e).lower() or
                    "disk full" in str(e).lower()
                )
                
                if is_disk_full:
                    print(f"[严重] ChromaDB 磁盘满 (errno={e.errno})，降级到 SQLite 存储")
                    
                    if sqlite_store:
                        try:
                            for text, meta in zip(batch_texts, metadatas[batch_start:batch_end] or []):
                                record = MemoryRecord(
                                    text=text,
                                    metadata=meta,
                                    is_vectorized=-1
                                )
                                sqlite_store.add(record)
                            print(f"[降级] {len(batch_texts)} 条记忆已存入 SQLite（待后续升级）")
                        except Exception as sqlite_error:
                            print(f"[错误] SQLite 降级存储失败: {sqlite_error}")
                    else:
                        print(f"[错误] 无 SQLite 存储，记忆丢失")
                else:
                    raise
                    
            except Exception as e:
                print(f"[错误] ChromaDB 写入批次 {batch_start}-{batch_end} 失败: {e}")
                
                if sqlite_store:
                    try:
                        for text, meta in zip(batch_texts, metadatas[batch_start:batch_end] or []):
                            record = MemoryRecord(
                                text=text,
                                metadata=meta,
                                is_vectorized=0
                            )
                            sqlite_store.add(record)
                        print(f"[降级] {len(batch_texts)} 条记忆已存入 SQLite（待重试）")
                    except Exception:
                        pass
        
        if all_ids:
            self._log.debug(
                "VECTOR_STORE_ADDED",
                count=len(all_ids),
                total=self.collection.count()
            )
        
        return all_ids
    
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
        
        注意：
            ChromaDB 使用内积空间 (hnsw:space: ip)，返回的 distances 是负内积值。
            对于归一化向量，内积 = 余弦相似度，范围 [-1, 1]。
            因此 similarity = 1 - distance，将距离转换为相似度。
        """
        if n_results is None:
            n_results = config.max_retrieve_results
        
        collection_count = self.collection.count()
        if collection_count == 0:
            self._log.info("VECTOR_SEARCH_EMPTY", message="collection is empty")
            return []
        
        query_embedding = self.embedding_function([query])[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, collection_count),
            where=where,
            where_document=where_document,
            include=["documents", "metadatas", "distances"]
        )
        
        formatted_results = []
        if results and results["ids"]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0.0
                similarity = 1 - distance
                
                formatted_results.append({
                    "id": doc_id,
                    "text": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "similarity": similarity
                })
        
        if formatted_results:
            similarities = [r["similarity"] for r in formatted_results]
            self._log.info(
                "VECTOR_SEARCH_RESULTS",
                query=query[:50],
                collection_count=collection_count,
                result_count=len(formatted_results),
                max_similarity=max(similarities) if similarities else 0,
                min_similarity=min(similarities) if similarities else 0
            )
        else:
            self._log.info(
                "VECTOR_SEARCH_EMPTY",
                query=query[:50],
                collection_count=collection_count
            )
        
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
        
        注意：
            ChromaDB 使用内积空间 (hnsw:space: ip)，返回的 distances 是 1 - cosine_similarity。
            因此 similarity = 1 - distance，将距离转换为相似度。
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
                distance = results["distances"][0][i] if results["distances"] else 0.0
                similarity = 1 - distance
                
                formatted_results.append({
                    "id": doc_id,
                    "text": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "similarity": similarity
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
    
    def deduplicate(self, sqlite_store=None, batch_size: int = 500, last_dedup_timestamp: str = None, yield_cpu: bool = True) -> Tuple[int, str]:
        """
        去重 - 基于文本内容（增量处理版）
        
        Args:
            sqlite_store: SQLite存储实例（可选，用于同步删除）
            batch_size: 每批处理的文档数量（默认500）
            last_dedup_timestamp: 上次去重的时间戳（ISO格式），只处理该时间之后的记录
            yield_cpu: 是否在批次间让出 CPU（默认 True）
        
        Returns:
            (移除的重复文档数量, 本次去重的时间戳)
        
        增量去重策略：
        1. 如果提供了 last_dedup_timestamp，只处理该时间之后的记录
        2. 使用 ChromaDB 的 where 过滤按时间范围筛选
        3. 分批处理间让出 CPU，避免阻塞其他操作
        4. 返回本次去重的时间戳，供下次调用使用
        
        时间复杂度：
            - 全量去重: O(n) 内存加载全部文档
            - 增量去重: O(k) 只处理 k 条新记录，k << n
        """
        from datetime import datetime
        
        current_timestamp = datetime.now().isoformat()
        
        where_filter = None
        if last_dedup_timestamp:
            where_filter = {
                "timestamp": {"$gte": last_dedup_timestamp}
            }
        
        try:
            if where_filter:
                results = self.collection.get(
                    where=where_filter,
                    include=["documents", "metadatas"]
                )
            else:
                total_count = self.collection.count()
                if total_count == 0:
                    return 0, current_timestamp
                
                results = self.collection.get(
                    limit=batch_size,
                    include=["documents", "metadatas"]
                )
        except Exception as e:
            print(f"[去重] 获取文档失败: {e}")
            return 0, current_timestamp
        
        if not results["ids"]:
            return 0, current_timestamp
        
        total_duplicates = 0
        seen_texts = {}
        
        if not where_filter:
            try:
                all_results = self.collection.get(
                    include=["documents", "metadatas"]
                )
                if all_results["ids"]:
                    for i, text in enumerate(all_results["documents"]):
                        if text not in seen_texts:
                            seen_texts[text] = all_results["ids"][i]
            except Exception as e:
                print(f"[去重] 获取已有文档失败: {e}")
        
        try:
            check_results = self.collection.get(
                include=["documents", "metadatas"]
            )
            if check_results["ids"]:
                for i, text in enumerate(check_results["documents"]):
                    if text not in seen_texts:
                        seen_texts[text] = check_results["ids"][i]
        except Exception:
            pass
        
        duplicate_ids = []
        
        for i, text in enumerate(results["documents"]):
            doc_id = results["ids"][i]
            if text in seen_texts and seen_texts[text] != doc_id:
                duplicate_ids.append(doc_id)
        
        if duplicate_ids:
            self.collection.delete(ids=duplicate_ids)
            total_duplicates += len(duplicate_ids)
            
            if sqlite_store:
                for dup_id in duplicate_ids:
                    try:
                        sqlite_store.delete_by_vector_id(dup_id)
                    except Exception:
                        pass
        
        if total_duplicates > 0:
            print(f"去重完成：移除 {total_duplicates} 条重复记忆")
        
        return total_duplicates, current_timestamp
    
    def deduplicate_incremental(self, sqlite_store=None, batch_size: int = 100, max_records: int = 1000, yield_cpu: bool = True) -> int:
        """
        增量去重 - 只检查最近添加的 N 条记录
        
        Args:
            sqlite_store: SQLite存储实例（可选，用于同步删除）
            batch_size: 每批处理的文档数量（默认100）
            max_records: 最大检查记录数（默认1000）
            yield_cpu: 是否在批次间让出 CPU（默认 True）
        
        Returns:
            移除的重复文档数量
        
        增量去重策略：
        1. 只检查最近添加的 max_records 条记录
        2. 分批处理，每批 batch_size 条
        3. 批次间让出 CPU，避免阻塞
        4. 适合在后台线程中定期执行
        """
        
        total_count = self.collection.count()
        if total_count == 0:
            return 0
        
        check_count = min(max_records, total_count)
        total_duplicates = 0
        seen_texts = {}
        
        try:
            existing_results = self.collection.get(
                limit=total_count - check_count if total_count > check_count else 0,
                include=["documents"]
            )
            if existing_results["ids"]:
                for text in existing_results["documents"]:
                    if text not in seen_texts:
                        seen_texts[text] = True
        except Exception:
            pass
        
        offset = total_count - check_count if total_count > check_count else 0
        
        while offset < total_count:
            try:
                results = self.collection.get(
                    limit=batch_size,
                    offset=offset,
                    include=["documents", "metadatas"]
                )
                
                if not results["ids"]:
                    break
                
                duplicate_ids = []
                
                for i, text in enumerate(results["documents"]):
                    doc_id = results["ids"][i]
                    if text in seen_texts:
                        duplicate_ids.append(doc_id)
                    else:
                        seen_texts[text] = doc_id
                
                if duplicate_ids:
                    self.collection.delete(ids=duplicate_ids)
                    total_duplicates += len(duplicate_ids)
                    
                    if sqlite_store:
                        for dup_id in duplicate_ids:
                            try:
                                sqlite_store.delete_by_vector_id(dup_id)
                            except Exception:
                                pass
                
                offset += batch_size
                
                if yield_cpu:
                    time.sleep(0.01)
                
            except Exception as e:
                print(f"[增量去重] 批次处理失败 (offset={offset}): {e}")
                break
        
        if total_duplicates > 0:
            print(f"增量去重完成：检查 {check_count} 条，移除 {total_duplicates} 条重复")
        
        return total_duplicates
    
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
    
    def get_ids_batch(self, limit: int = 100, offset: int = 0) -> List[str]:
        """
        分批获取文档ID（增量处理版）
        
        Args:
            limit: 每批返回数量
            offset: 起始偏移量
        
        Returns:
            文档ID列表
        """
        try:
            results = self.collection.get(limit=limit, offset=offset)
            return results.get("ids", [])
        except Exception as e:
            print(f"分批获取文档ID失败: {e}")
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
