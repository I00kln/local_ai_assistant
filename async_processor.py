# async_processor.py
import threading
import time
import queue
import hashlib
from typing import List, Dict, Any, Set, Tuple
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
    MAX_BUFFER_AGE = 60  # 缓冲区最大存活时间（秒），强制落盘
    
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
        self._last_flush_time = time.time()
        self._buffer_first_item_time: Optional[float] = None
        self._idle_count = 0
        
        self.sqlite = None
        if config.sqlite_enabled:
            try:
                self.sqlite = get_sqlite_store()
            except Exception as e:
                print(f"SQLite初始化失败: {e}")
        
        self.compressor = None
        self._memory_manager = None
    
    def set_memory_manager(self, memory_manager):
        """设置记忆管理器引用（用于通知写入操作）"""
        self._memory_manager = memory_manager
        
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
                    self._check_memory_flow()
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
        
        if config.nonsense_filter_enabled:
            filter_result = get_nonsense_filter().filter(user_input, assistant_response)
            
            if filter_result.storage_type == "discard":
                print(f"[废话过滤] 丢弃: {filter_result.reason} (置信度: {filter_result.confidence:.2f})")
                return
            
            if filter_result.storage_type == "sqlite_only":
                print(f"[废话过滤] 仅存SQLite: {filter_result.reason}")
                self._store_to_sqlite_only(conversation_memory, metadata)
                return
        
        if len(self.batch_buffer) == 0:
            self._buffer_first_item_time = time.time()
        
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
        
        流程（两阶段提交）：
        1. 检查幂等性（文本哈希）
        2. 写入SQLite（标记为待向量化）
        3. 写入ChromaDB
        4. 更新SQLite状态
        
        失败恢复：
        - 启动时通过 _startup_recovery 重试未向量化的记录
        - 使用文本哈希防止重复写入
        """
        if not self.batch_buffer:
            return
        
        texts = [item["text"] for item in self.batch_buffer]
        metadatas = [item["metadata"] for item in self.batch_buffer]
        
        sqlite_ids = []
        texts_to_vectorize = []
        metadatas_to_vectorize = []
        hash_list = []
        
        if self.sqlite:
            for text, meta in zip(texts, metadatas):
                text_hash = self.sqlite.compute_text_hash(text)
                hash_list.append(text_hash)
                
                exists, existing_id = self.sqlite.exists_by_text_hash(text_hash)
                if exists:
                    record = self.sqlite.get(existing_id)
                    if record and record.is_vectorized == 1:
                        print(f"[幂等检查] 文本已存在且已向量化，跳过: {text[:50]}...")
                        continue
                    sqlite_ids.append((existing_id, text_hash))
                    texts_to_vectorize.append(text)
                    metadatas_to_vectorize.append(meta)
                else:
                    from sqlite_store import MemoryRecord
                    record = MemoryRecord(
                        text=text,
                        source=meta.get("source", "local"),
                        metadata=meta,
                        is_vectorized=0
                    )
                    record_id = self.sqlite.add_with_hash(record, text_hash)
                    sqlite_ids.append((record_id, text_hash))
                    texts_to_vectorize.append(text)
                    metadatas_to_vectorize.append(meta)
        else:
            texts_to_vectorize = texts
            metadatas_to_vectorize = metadatas
        
        if not texts_to_vectorize:
            self.batch_buffer.clear()
            return
        
        try:
            vector_ids = self.vector_store.add(texts_to_vectorize, metadatas_to_vectorize)
            
            if self.sqlite and sqlite_ids and vector_ids:
                for (sqlite_id, text_hash), vector_id in zip(sqlite_ids, vector_ids):
                    self.sqlite.update_vector_status(sqlite_id, vector_id, is_vectorized=1)
            
            print(f"原子化写入完成: {len(texts_to_vectorize)} 条记忆")
            self._last_flush_time = time.time()
            
            if self._memory_manager:
                self._memory_manager.notify_write()
            
        except Exception as e:
            print(f"ChromaDB写入失败: {e}")
            if self.sqlite and sqlite_ids:
                for sqlite_id, text_hash in sqlite_ids:
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
        3. 缓冲区首条记录超过 MAX_BUFFER_AGE（强制落盘）
        
        确保断电时数据丢失不超过 batch_size
        """
        current_time = time.time()
        
        if len(self.batch_buffer) > 0:
            time_since_last_flush = current_time - self._last_flush_time
            
            buffer_age = 0
            if self._buffer_first_item_time:
                buffer_age = current_time - self._buffer_first_item_time
            
            if buffer_age >= self.MAX_BUFFER_AGE:
                print(f"[强制落盘] 缓冲区 {len(self.batch_buffer)} 条记录，已存活 {buffer_age:.1f} 秒")
                self._flush_buffer()
                self._last_flush_time = current_time
                self._buffer_first_item_time = None
            elif time_since_last_flush >= self.FLUSH_INTERVAL:
                print(f"[定时落盘] 缓冲区 {len(self.batch_buffer)} 条记录，距上次落盘 {time_since_last_flush:.1f} 秒")
                self._flush_buffer()
                self._last_flush_time = current_time
                self._buffer_first_item_time = None
            elif len(self.batch_buffer) >= self.batch_size:
                self._flush_buffer()
                self._last_flush_time = current_time
                self._buffer_first_item_time = None
    
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
    
    def _check_memory_flow(self):
        """
        空闲时检查并执行记忆流动（统一入口）
        
        记忆流动机制（按顺序执行）：
        1. L2→L3：不常用、低权重的记忆从向量库移动到SQLite暂存
        2. L3压缩：对L3中的记忆进行压缩
        3. L3→L2：高权重、常用的记忆从SQLite回流到向量库
        4. 待压缩处理：处理之前标记为待压缩的记忆
        5. 遗忘机制：对L3中权重极低的记忆进行遗忘
        
        所有操作共享时间检查，避免重复执行
        """
        if not config.sqlite_enabled or not self.sqlite:
            return
        
        current_time = time.time()
        
        if self._idle_count < 5:
            return
        
        if current_time - self._last_compression_time < self.COMPRESSION_INTERVAL:
            return
        
        try:
            print("[后台任务] 开始记忆流动...")
            
            moved_to_l3 = self._move_l2_to_l3()
            
            compressed = self._compress_l3_memories()
            
            moved_to_l2 = self._move_l3_to_l2()
            
            self._process_pending_compressions()
            
            if self._idle_count >= 10 and current_time - self._last_forget_time >= self.FORGET_INTERVAL:
                result = self.sqlite.decay_weights(config.memory_decay_days)
                decayed = result.get("decayed", 0)
                forgotten = result.get("forgotten", 0)
                if decayed > 0 or forgotten > 0:
                    print(f"[后台任务] 衰减: {decayed}, 遗忘: {forgotten}")
                self._last_forget_time = current_time
            
            total_actions = moved_to_l3 + compressed + moved_to_l2
            if total_actions > 0:
                print(f"[后台任务] L2→L3: {moved_to_l3}, 压缩: {compressed}, L3→L2: {moved_to_l2}")
            
            self._last_compression_time = current_time
            self._idle_count = 0
            
        except Exception as e:
            print(f"记忆流动失败: {e}")
    
    def _move_l2_to_l3(self) -> int:
        """
        将L2(向量库)中不常用、低权重的记忆移动到L3(SQLite)
        
        条件：
        - 长时间未访问
        - 权重较低
        
        流程：
        1. 从SQLite获取已向量化的低权重记忆
        2. 从ChromaDB删除
        3. 更新SQLite记录为未向量化状态
        """
        if not self.sqlite:
            return 0
        
        try:
            low_weight_records = self.sqlite.get_low_weight_memories(
                threshold=config.memory_min_weight * 2,
                limit=20
            )
            
            moved_count = 0
            for record in low_weight_records:
                if not record.vector_id:
                    continue
                
                try:
                    self.vector_store.delete(ids=[record.vector_id])
                    
                    self.sqlite.update_vector_status(record.id, "", is_vectorized=0)
                    
                    if not record.metadata:
                        record.metadata = {}
                    record.metadata["moved_from_l2"] = True
                    record.metadata["moved_time"] = datetime.now().isoformat()
                    self.sqlite.add(record)
                    
                    moved_count += 1
                    
                except Exception as e:
                    print(f"L2→L3移动失败: {e}")
            
            return moved_count
            
        except Exception as e:
            print(f"L2→L3流动失败: {e}")
            return 0
    
    def _should_skip_compression(self, record: MemoryRecord) -> bool:
        """
        检查记忆是否应该跳过压缩
        
        跳过条件：
        1. 已有压缩文本（compressed_text存在）
        2. metadata中标记为preserve=True（受保护）
        3. metadata中标记为compressed=True（已压缩）
        4. metadata中标记为promoted_from_l3=True（从L3回流，已压缩过）
        """
        if record.compressed_text:
            return True
        
        if record.metadata:
            if record.metadata.get("preserve") is True:
                return True
            if record.metadata.get("compressed") is True:
                return True
            if record.metadata.get("promoted_from_l3") is True:
                return True
        
        return False
    
    def _compress_l3_memories(self) -> int:
        """
        压缩L3(SQLite)中的记忆
        
        条件：
        - 长时间未访问
        - 未压缩过
        - 非高密度内容
        - 非受保护记忆
        """
        if not self.sqlite:
            return 0
        
        try:
            unaccessed = self.sqlite.get_unaccessed_memories(
                days=config.memory_decay_days // 2,
                limit=20
            )
            
            if not unaccessed:
                return 0
            
            compressor_available = self._check_compressor_available()
            
            compressed_count = 0
            preserved_count = 0
            skipped_count = 0
            
            for record in unaccessed:
                try:
                    if self._should_skip_compression(record):
                        skipped_count += 1
                        continue
                    
                    if self._is_high_density_content(record.text):
                        self._mark_preserve(record)
                        preserved_count += 1
                        continue
                    
                    if len(record.text) < config.compression_min_length:
                        continue
                    
                    if compressor_available:
                        compressed_text = self._compress_text(record.text)
                        
                        if compressed_text and len(compressed_text) < len(record.text) * 0.6:
                            record.compressed_text = compressed_text
                            if not record.metadata:
                                record.metadata = {}
                            record.metadata["compressed"] = True
                            record.metadata["compressed_time"] = datetime.now().isoformat()
                            self.sqlite.add(record)
                            compressed_count += 1
                    else:
                        self._mark_pending_compression(record)
                        
                except Exception as e:
                    print(f"压缩单条记忆失败: {e}")
            
            if skipped_count > 0:
                print(f"[后台任务] 跳过已压缩/受保护: {skipped_count}")
            
            return compressed_count
            
        except Exception as e:
            print(f"L3压缩失败: {e}")
            return 0
    
    def _move_l3_to_l2(self) -> int:
        """
        将L3(SQLite)中高权重、常用的记忆回流到L2(向量库)
        
        条件：
        - 权重超过阈值
        - 最近有访问
        - 当前未向量化
        
        注意：
        - 如果记忆有compressed_text，标记为已压缩，未来不再压缩
        - 回流后的记忆若再次存入L3，会跳过压缩流程
        """
        if not self.sqlite:
            return 0
        
        try:
            high_weight_records = self.sqlite.get_high_weight_memories(limit=10)
            
            moved_count = 0
            for record in high_weight_records:
                try:
                    text_to_vectorize = record.compressed_text or record.text
                    
                    vector_ids = self.vector_store.add(
                        [text_to_vectorize],
                        [record.metadata or {}]
                    )
                    
                    if vector_ids:
                        self.sqlite.update_vector_status(
                            record.id, 
                            vector_ids[0], 
                            is_vectorized=1
                        )
                        
                        if not record.metadata:
                            record.metadata = {}
                        record.metadata["promoted_to_l2"] = True
                        record.metadata["promoted_time"] = datetime.now().isoformat()
                        
                        if record.compressed_text:
                            record.metadata["promoted_from_l3"] = True
                            record.metadata["compressed"] = True
                        
                        self.sqlite.add(record)
                        
                        moved_count += 1
                        
                except Exception as e:
                    print(f"L3→L2回流失败: {e}")
            
            return moved_count
            
        except Exception as e:
            print(f"L3→L2回流失败: {e}")
            return 0
    
    def _check_compressor_available(self) -> bool:
        """检查压缩器是否可用"""
        if self.compressor is None:
            try:
                from llm_client import LlamaClient
                self.compressor = LlamaClient()
                if not self.compressor.check_connection():
                    self.compressor = None
                    return False
            except Exception as e:
                print(f"压缩器初始化失败: {e}")
                return False
        return self.compressor.check_connection()
    
    def _is_high_density_content(self, text: str) -> bool:
        """检测是否为高信息密度内容"""
        import re
        patterns = config.high_density_patterns.split("|")
        for pattern in patterns:
            if pattern.strip() in text:
                return True
        
        code_patterns = [
            r'```[\s\S]*?```',
            r'``[\s\S]*?``',
            r'`[^`]+`',
            r'def\s+\w+\s*\(',
            r'class\s+\w+',
            r'function\s+\w+',
            r'import\s+\w+',
            r'from\s+\w+\s+import',
        ]
        for pattern in code_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _mark_preserve(self, record: MemoryRecord):
        """标记记忆为保留（不压缩）"""
        if not record.metadata:
            record.metadata = {}
        record.metadata["preserve"] = True
        record.metadata["preserve_reason"] = "high_density_content"
        self.sqlite.add(record)
    
    def _mark_pending_compression(self, record: MemoryRecord):
        """标记记忆为待压缩"""
        if not record.metadata:
            record.metadata = {}
        record.metadata["pending_compression"] = True
        record.metadata["pending_since"] = datetime.now().isoformat()
        self.sqlite.add(record)
    
    def _process_pending_compressions(self):
        """
        处理待压缩的记忆
        
        当LLM可用时，对之前标记为待压缩的记忆进行压缩
        """
        if not self.sqlite:
            return
        
        compressor_available = self._check_compressor_available()
        if not compressor_available:
            return
        
        try:
            pending_records = self.sqlite.get_pending_compressions(limit=10)
            
            if not pending_records:
                return
            
            print(f"[后台任务] 处理 {len(pending_records)} 条待压缩记忆...")
            
            compressed_count = 0
            skipped_count = 0
            for record in pending_records:
                try:
                    if self._should_skip_compression(record):
                        if record.metadata:
                            record.metadata.pop("pending_compression", None)
                            record.metadata.pop("pending_since", None)
                        self.sqlite.add(record)
                        skipped_count += 1
                        continue
                    
                    if self._is_high_density_content(record.text):
                        self._mark_preserve(record)
                        continue
                    
                    if len(record.text) < config.compression_min_length:
                        if record.metadata:
                            record.metadata.pop("pending_compression", None)
                            record.metadata.pop("pending_since", None)
                        self.sqlite.add(record)
                        continue
                    
                    compressed_text = self._compress_text(record.text)
                    
                    if compressed_text and len(compressed_text) < len(record.text) * 0.6:
                        record.compressed_text = compressed_text
                        if not record.metadata:
                            record.metadata = {}
                        record.metadata.pop("pending_compression", None)
                        record.metadata.pop("pending_since", None)
                        record.metadata["compressed"] = True
                        record.metadata["compressed_time"] = datetime.now().isoformat()
                        self.sqlite.add(record)
                        compressed_count += 1
                        
                except Exception as e:
                    print(f"处理待压缩记忆失败: {e}")
            
            if compressed_count > 0 or skipped_count > 0:
                print(f"[后台任务] 待压缩处理: {compressed_count} 条, 跳过: {skipped_count} 条")
                
        except Exception as e:
            print(f"处理待压缩记忆失败: {e}")
    
    def _compress_text(self, text: str) -> str:
        """
        使用LLM压缩文本
        
        处理策略：
        1. 短文本（<2000字符）：直接压缩
        2. 中等文本（2000-6000字符）：直接压缩
        3. 长文本（>6000字符）：分段压缩后合并
        4. 超长文本（>10000字符）：尝试云端压缩，失败则标记不压缩
        """
        if not text:
            return text
        
        text_len = len(text)
        max_input_chars = config.local.max_context // 2
        
        if text_len > 10000:
            return self._compress_long_text(text)
        
        if text_len > 6000:
            return self._compress_chunked_text(text)
        
        return self._compress_single_text(text)
    
    def _compress_single_text(self, text: str) -> str:
        """压缩单段文本"""
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
    
    def _compress_chunked_text(self, text: str) -> str:
        """
        分段压缩长文本
        
        将长文本分成多个段落分别压缩，然后合并
        """
        try:
            chunk_size = 3000
            chunks = []
            
            lines = text.split('\n')
            current_chunk = []
            current_len = 0
            
            for line in lines:
                if current_len + len(line) > chunk_size and current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_len = 0
                current_chunk.append(line)
                current_len += len(line)
            
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
            
            compressed_chunks = []
            for i, chunk in enumerate(chunks):
                compressed = self._compress_single_text(chunk)
                compressed_chunks.append(compressed)
            
            merged = '\n'.join(compressed_chunks)
            
            if len(merged) < len(text) * 0.7:
                return self._compress_single_text(merged)
            
            return merged
            
        except Exception as e:
            print(f"分段压缩失败: {e}")
            return text
    
    def _compress_long_text(self, text: str) -> str:
        """
        处理超长文本
        
        策略：
        1. 尝试云端压缩
        2. 失败则分段压缩
        3. 再失败则标记为不压缩
        """
        try:
            if config.cloud.enabled and self._try_cloud_compression(text):
                return self._cloud_compress_text(text)
            
            chunked_result = self._compress_chunked_text(text)
            if chunked_result and len(chunked_result) < len(text) * 0.8:
                return chunked_result
            
            print(f"超长文本({len(text)}字符)无法压缩，保留原文")
            return text
            
        except Exception as e:
            print(f"超长文本处理失败: {e}")
            return text
    
    def _try_cloud_compression(self, text: str) -> bool:
        """检查是否可以使用云端压缩"""
        if not config.cloud.enabled:
            return False
        if not config.cloud.api_key:
            return False
        return True
    
    def _cloud_compress_text(self, text: str) -> str:
        """使用云端API压缩文本"""
        try:
            from cloud_client import CloudClient
            cloud = CloudClient()
            
            prompt = f"""请将以下内容压缩为简洁的摘要，保留所有关键信息。

原始内容：
{text}

压缩要求：
1. 保留所有专有名词、人名、地名、数值
2. 保留因果关系和关键决策
3. 去除冗余和修饰性内容
4. 压缩后长度约为原文的30%-50%

压缩结果："""

            messages = [
                {"role": "system", "content": "你是一个记忆压缩专家。"},
                {"role": "user", "content": prompt}
            ]
            
            result = cloud.chat(messages)
            return result.strip() if result else text
            
        except Exception as e:
            print(f"云端压缩失败: {e}")
            return text
    
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
