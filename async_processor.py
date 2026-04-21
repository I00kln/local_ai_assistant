# async_processor.py
import threading
import time
import queue
from typing import List, Dict, Any
from datetime import datetime
from vector_store import VectorStore, get_vector_store
from config import config

# 废话过滤器
from nonsense_filter import get_nonsense_filter, FilterResult

# SQLite存储
from sqlite_store import get_sqlite_store, MemoryRecord


class AsyncMemoryProcessor:
    """异步记忆处理器：后台收集对话并增量更新 ChromaDB + SQLite"""
    
    DEDUP_INTERVAL = 300
    COMPRESSION_INTERVAL = 600  # 压缩检查间隔（秒）
    FORGET_INTERVAL = 3600  # 遗忘检查间隔（秒）
    FLUSH_INTERVAL = 30  # 定时落盘间隔（秒），确保断电时数据丢失不超过batch_size
    
    def __init__(self, vector_store: VectorStore = None):
        self.vector_store = vector_store or get_vector_store()
        self.pending_queue = queue.Queue()
        self.running = False
        self.thread = None
        
        self.batch_buffer: List[Dict] = []
        self.batch_size = config.batch_update_size
        
        self._last_dedup_time = time.time()
        self._last_compression_time = time.time()
        self._last_forget_time = time.time()
        self._last_flush_time = time.time()  # 新增：上次落盘时间
        self._idle_count = 0
        
        # SQLite存储实例
        self.sqlite = None
        if config.sqlite_enabled:
            try:
                self.sqlite = get_sqlite_store()
            except Exception as e:
                print(f"SQLite初始化失败: {e}")
        
        # 压缩器（延迟初始化）
        self.compressor = None
        
    def start(self):
        """启动后台处理线程"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        
        self._startup_recovery()
        
        print("异步记忆处理器已启动")
    
    def _startup_recovery(self):
        """
        启动补偿机制
        
        检查SQLite中未向量化的记录，重新触发同步
        解决：ChromaDB写入成功但SQLite更新失败导致的数据不一致
        """
        if not self.sqlite:
            return
        
        try:
            unvectorized = self.sqlite.get_unvectorized(limit=50)
            if unvectorized:
                print(f"[启动补偿] 发现 {len(unvectorized)} 条未向量化记录，开始重试...")
                
                for record in unvectorized:
                    if record.is_vectorized == -1:
                        continue
                    
                    try:
                        vector_ids = self.vector_store.add([record.text], [record.metadata or {}])
                        if vector_ids:
                            self.sqlite.update_vector_status(record.id, vector_ids[0], is_vectorized=1)
                            print(f"[启动补偿] 记录 {record.id} 同步成功")
                    except Exception as e:
                        print(f"[启动补偿] 记录 {record.id} 同步失败: {e}")
            
            orphan_vectors = self._check_and_clean_orphan_vectors()
            if orphan_vectors > 0:
                print(f"[启动补偿] 清理了 {orphan_vectors} 个孤立向量")
            
            self._sync_timestamps()
                
        except Exception as e:
            print(f"[启动补偿] 检查失败: {e}")
    
    def _check_and_clean_orphan_vectors(self) -> int:
        """
        检查并清理ChromaDB中的孤立向量
        
        孤立向量：ChromaDB中存在但SQLite中无对应记录
        """
        if not self.sqlite:
            return 0
        
        try:
            all_ids = self.vector_store.get_all_ids()
            if not all_ids:
                return 0
            
            orphan_ids = []
            for vid in all_ids[:100]:
                records = self.sqlite.get_by_vector_id(vid)
                if not records:
                    orphan_ids.append(vid)
            
            if orphan_ids:
                self.vector_store.delete(orphan_ids)
                print(f"[清理孤立向量] 删除了 {len(orphan_ids)} 个孤立向量")
            
            return len(orphan_ids)
        except Exception as e:
            print(f"[清理孤立向量] 失败: {e}")
            return 0
    
    def _sync_timestamps(self):
        """
        同步时间戳
        
        确保SQLite和ChromaDB中的timestamp一致
        """
        if not self.sqlite:
            return
        
        try:
            all_ids = self.vector_store.get_all_ids()
            if not all_ids:
                return
            
            for vid in all_ids[:50]:
                records = self.sqlite.get_by_vector_id(vid)
                if records:
                    record = records[0]
                    chroma_metadata = self.vector_store.get_metadata(vid)
                    if chroma_metadata:
                        sqlite_time = record.created_time
                        chroma_time = chroma_metadata.get("timestamp")
                        if sqlite_time and chroma_time and sqlite_time != chroma_time:
                            self.vector_store.update_metadata(vid, {"timestamp": sqlite_time})
        except Exception as e:
            print(f"[同步时间戳] 失败: {e}")
    
    def stop(self):
        """停止后台处理"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        self._flush_buffer()
        # ChromaDB自动持久化，无需手动保存
        print("异步记忆处理器已停止")
    
    def add_conversation(self, user_input: str, assistant_response: str, metadata: Dict = None):
        """添加一轮对话到处理队列"""
        conv_data = {
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "assistant": assistant_response,
            "metadata": metadata or {}
        }
        self.pending_queue.put(conv_data)
    
    def _process_loop(self):
        """后台处理循环"""
        while self.running:
            try:
                try:
                    conv = self.pending_queue.get(timeout=config.async_update_interval)
                    self._process_conversation(conv)
                    self._idle_count = 0
                except queue.Empty:
                    self._idle_count += 1
                    self._check_and_flush()
                    self._check_and_dedup()
                    self._check_and_compress()  # 新增：压缩检查
                    self._check_and_forget()  # 新增：遗忘检查
                    continue
                
            except Exception as e:
                print(f"异步处理异常: {e}")
    
    def _process_conversation(self, conv: Dict):
        """
        处理单条对话 - 分层存储策略
        
        存储策略：
        - full: 存入SQLite + ChromaDB（有价值记忆）
        - sqlite_only: 仅存SQLite（保持对话完整性，不污染向量空间）
        - discard: 完全丢弃（纯噪音）
        """
        user_input = conv['user']
        assistant_response = conv['assistant']
        source = conv.get('metadata', {}).get('source', 'local')
        
        conversation_memory = f"用户: {user_input}\n助理: {assistant_response}"
        metadata = {
            "type": "conversation",
            "source": source,
            "timestamp": conv["timestamp"]
        }
        
        # 废话过滤检查
        if config.nonsense_filter_enabled:
            filter_result = get_nonsense_filter().filter(user_input, assistant_response)
            
            if filter_result.storage_type == "discard":
                print(f"[废话过滤] 丢弃: {filter_result.reason} (置信度: {filter_result.confidence:.2f})")
                return
            
            if filter_result.storage_type == "sqlite_only":
                print(f"[废话过滤] 仅存SQLite: {filter_result.reason}")
                self._store_to_sqlite_only(conversation_memory, metadata)
                return
        
        # 正常存储：SQLite + ChromaDB
        self.batch_buffer.append({
            "text": conversation_memory,
            "metadata": metadata
        })
        
        if len(self.batch_buffer) >= self.batch_size:
            self._flush_buffer()
    
    def _store_to_sqlite_only(self, text: str, metadata: Dict):
        """
        仅存储到SQLite（不向量化）
        
        用于保持对话完整性，但不污染向量空间
        """
        if not self.sqlite:
            return
        
        try:
            from sqlite_store import MemoryRecord
            record = MemoryRecord(
                text=text,
                source=metadata.get("source", "local"),
                metadata=metadata,
                is_vectorized=-1  # 标记为不向量化
            )
            self.sqlite.add(record)
        except Exception as e:
            print(f"SQLite存储失败: {e}")
    
    def _flush_buffer(self):
        """
        将缓冲区数据原子化写入
        
        流程：SQLite → ChromaDB → 更新状态
        """
        if not self.batch_buffer:
            return
        
        texts = [item["text"] for item in self.batch_buffer]
        metadatas = [item["metadata"] for item in self.batch_buffer]
        
        # 步骤1: 先写入SQLite
        sqlite_ids = []
        if self.sqlite:
            from sqlite_store import MemoryRecord
            for text, meta in zip(texts, metadatas):
                record = MemoryRecord(
                    text=text,
                    source=meta.get("source", "local"),
                    metadata=meta,
                    is_vectorized=0  # 标记为未向量化
                )
                record_id = self.sqlite.add(record)
                sqlite_ids.append(record_id)
        
        # 步骤2: 写入ChromaDB
        try:
            vector_ids = self.vector_store.add(texts, metadatas)
            
            # 步骤3: 更新SQLite的向量化状态
            if self.sqlite and sqlite_ids and vector_ids:
                for sqlite_id, vector_id in zip(sqlite_ids, vector_ids):
                    self.sqlite.update_vector_status(sqlite_id, vector_id, is_vectorized=1)
            
            print(f"原子化写入完成: {len(texts)} 条记忆")
            self._last_flush_time = time.time()  # 更新落盘时间
            
        except Exception as e:
            print(f"ChromaDB写入失败: {e}")
            # 标记为向量化失败，后续可重试
            if self.sqlite and sqlite_ids:
                for sqlite_id in sqlite_ids:
                    self.sqlite.update_vector_status(sqlite_id, "", is_vectorized=-1)
        
        self.batch_buffer.clear()
    
    def _retry_failed_vectorizations(self):
        """重试向量化失败的记录"""
        if not self.sqlite:
            return
        
        try:
            unvectorized = self.sqlite.get_unvectorized(limit=20)
            if not unvectorized:
                return
            
            print(f"[重试] 发现 {len(unvectorized)} 条未向量化记录")
            
            for record in unvectorized:
                try:
                    vector_ids = self.vector_store.add([record.text], [record.metadata or {}])
                    if vector_ids:
                        self.sqlite.update_vector_status(record.id, vector_ids[0], is_vectorized=1)
                        print(f"[重试成功] 记录 {record.id}")
                except Exception as e:
                    print(f"[重试失败] 记录 {record.id}: {e}")
                    
        except Exception as e:
            print(f"重试向量化失败: {e}")
    
    def _check_and_flush(self):
        """
        定时检查并落盘
        
        触发条件：
        1. batch_buffer 已满（在 _process_conversation 中处理）
        2. 超过 FLUSH_INTERVAL 时间未落盘
        
        确保断电时数据丢失不超过 batch_size
        """
        current_time = time.time()
        
        # 条件1: 缓冲区有数据且超过定时落盘间隔
        if len(self.batch_buffer) > 0:
            time_since_last_flush = current_time - self._last_flush_time
            
            if time_since_last_flush >= self.FLUSH_INTERVAL:
                print(f"[定时落盘] 缓冲区 {len(self.batch_buffer)} 条记录，距上次落盘 {time_since_last_flush:.1f} 秒")
                self._flush_buffer()
                self._last_flush_time = current_time
            elif len(self.batch_buffer) >= self.batch_size:
                # 条件2: 缓冲区已满（虽然这应该在 _process_conversation 中处理）
                self._flush_buffer()
                self._last_flush_time = current_time
    
    def _check_and_dedup(self):
        """空闲时检查并执行去重"""
        current_time = time.time()
        
        if self._idle_count < 3:
            return
        
        if current_time - self._last_dedup_time < self.DEDUP_INTERVAL:
            return
        
        if len(self.vector_store) < 10:
            return
        
        try:
            removed = self.vector_store.deduplicate(sqlite_store=self.sqlite)
            if removed > 0:
                pass  # ChromaDB自动持久化
            self._last_dedup_time = current_time
            self._idle_count = 0
        except Exception as e:
            print(f"去重失败: {e}")
    
    def _check_and_compress(self):
        """
        空闲时检查并执行记忆压缩
        
        将向量库中的低频访问记忆压缩后存入SQLite
        """
        if not config.compression_enabled or not self.sqlite:
            return
        
        current_time = time.time()
        
        # 空闲阈值检查
        if self._idle_count < 5:
            return
        
        # 时间间隔检查
        if current_time - self._last_compression_time < self.COMPRESSION_INTERVAL:
            return
        
        # 记忆数量检查
        if len(self.vector_store) < 20:
            return
        
        try:
            print("[后台任务] 开始记忆压缩...")
            
            # 获取长时间未访问的记忆（从SQLite获取）
            unaccessed = self.sqlite.get_unaccessed_memories(
                days=config.memory_decay_days // 2,
                limit=20
            )
            
            if not unaccessed:
                self._last_compression_time = current_time
                return
            
            # 初始化压缩器
            if self.compressor is None:
                try:
                    from llm_client import LlamaClient
                    self.compressor = LlamaClient()
                except Exception as e:
                    print(f"压缩器初始化失败: {e}")
                    return
            
            compressed_count = 0
            for record in unaccessed:
                try:
                    # 检查是否需要压缩
                    if len(record.text) < config.compression_min_length:
                        continue
                    
                    # 使用LLM压缩
                    compressed_text = self._compress_text(record.text)
                    
                    if compressed_text and len(compressed_text) < len(record.text) * 0.6:
                        # 更新SQLite中的压缩文本
                        record.compressed_text = compressed_text
                        self.sqlite.add(record)
                        compressed_count += 1
                        
                except Exception as e:
                    print(f"压缩单条记忆失败: {e}")
            
            if compressed_count > 0:
                print(f"[后台任务] 压缩完成: {compressed_count} 条记忆")
            
            self._last_compression_time = current_time
            self._idle_count = 0
            
        except Exception as e:
            print(f"压缩流程失败: {e}")
    
    def _compress_text(self, text: str) -> str:
        """使用LLM压缩文本"""
        prompt = f"""请将以下对话记录压缩为简洁的摘要，保留所有关键信息。

原始内容：
{text}

压缩要求：
1. 保留所有专有名词、人名、地名、数值
2. 保留因果关系和关键决策
3. 去除冗余和修饰性内容
4. 压缩后长度约为原文的30%-50%

压缩结果："""

        try:
            messages = [
                {"role": "system", "content": "你是一个记忆压缩专家。"},
                {"role": "user", "content": prompt}
            ]
            
            result = self.compressor.chat(messages, max_tokens=500)
            return result.strip() if result else text
            
        except Exception as e:
            print(f"LLM压缩失败: {e}")
            return text
    
    def _check_and_forget(self):
        """
        空闲时检查并执行记忆遗忘
        
        对SQLite中的记忆进行权重衰减和遗忘
        """
        if not config.sqlite_enabled or not self.sqlite:
            return
        
        current_time = time.time()
        
        # 空闲阈值检查
        if self._idle_count < 10:
            return
        
        # 时间间隔检查
        if current_time - self._last_forget_time < self.FORGET_INTERVAL:
            return
        
        try:
            print("[后台任务] 开始记忆衰减与遗忘...")
            
            result = self.sqlite.decay_weights(config.memory_decay_days)
            
            decayed = result.get("decayed", 0)
            forgotten = result.get("forgotten", 0)
            
            if decayed > 0 or forgotten > 0:
                print(f"[后台任务] 衰减: {decayed} 条, 遗忘: {forgotten} 条")
            
            self._last_forget_time = current_time
            self._idle_count = 0
            
        except Exception as e:
            print(f"遗忘流程失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        stats = {
            "pending_queue_size": self.pending_queue.qsize(),
            "batch_buffer_size": len(self.batch_buffer),
            "vector_count": len(self.vector_store),
            "idle_count": self._idle_count
        }
        
        if self.sqlite:
            sqlite_stats = self.sqlite.get_stats()
            stats["sqlite"] = sqlite_stats
        
        return stats
