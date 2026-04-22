# async_processor.py
import threading
import time
import queue
import hashlib
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Set, Tuple, Optional
from datetime import datetime, timedelta
from vector_store import VectorStore, get_vector_store
from config import config, get_memory_config
from memory_transaction import get_transaction_coordinator
from event_bus import get_event_bus, EventType
from nonsense_filter import get_nonsense_filter, FilterResult
from sqlite_store import get_sqlite_store, MemoryRecord
from logger import get_logger
from memory_tags import MemoryTags, MemoryTagHelper
from memory_merger import get_merger


class CompressionStrategy(ABC):
    """压缩策略抽象接口"""
    
    @abstractmethod
    def compress(self, text: str) -> Optional[str]:
        """压缩文本，返回 None 表示无法压缩"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查策略是否可用"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称"""
        pass


class LLMCompressionStrategy(CompressionStrategy):
    """LLM 压缩策略 - 带熔断器"""
    
    def __init__(self, mem_config):
        self._mem_config = mem_config
        self._llm_client = None
        self._last_check = 0
        self._failure_count = 0
        self._available = False
        self._log = get_logger()
        
        self._circuit_breaker = {
            "failures": 0,
            "last_failure": 0,
            "open_until": 0,
            "threshold": 5,
            "reset_timeout": 300
        }
        try:
            async_config = mem_config.async_processor
            self._circuit_breaker["threshold"] = getattr(async_config, 'circuit_breaker_threshold', 5)
            self._circuit_breaker["reset_timeout"] = getattr(async_config, 'circuit_breaker_reset_timeout', 300)
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
            return False
        
        check_interval = self._mem_config.async_processor.compressor_check_interval
        if current_time - self._last_check < check_interval:
            return self._available
        
        self._last_check = current_time
        
        if self._llm_client is None:
            try:
                from llm_client import LlamaClient
                self._llm_client = LlamaClient()
            except Exception as e:
                self._record_failure(str(e))
                return False
        
        try:
            llm_timeout = getattr(self._mem_config.async_processor, 'llm_timeout', 30)
            
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
            
            def check_connection():
                return self._llm_client.check_connection()
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(check_connection)
                try:
                    available = future.result(timeout=llm_timeout)
                except FuturesTimeoutError:
                    self._record_failure("Connection check timeout")
                    return False
            
            if available:
                self._available = True
                self._reset_circuit_breaker()
                return True
            else:
                self._record_failure("Connection check returned False")
                return False
                
        except TimeoutError:
            self._record_failure("Connection check timeout")
            return False
        except Exception as e:
            self._record_failure(str(e))
            return False
    
    def _record_failure(self, error: str):
        """记录失败，更新熔断器状态"""
        self._failure_count += 1
        self._available = False
        self._circuit_breaker["failures"] += 1
        self._circuit_breaker["last_failure"] = time.time()
        
        if self._circuit_breaker["failures"] >= self._circuit_breaker["threshold"]:
            self._circuit_breaker["open_until"] = time.time() + self._circuit_breaker["reset_timeout"]
            self._log.warning("CIRCUIT_BREAKER_OPENED",
                              failures=self._circuit_breaker["failures"],
                              reset_in=self._circuit_breaker["reset_timeout"],
                              error=error)
    
    def _reset_circuit_breaker(self):
        """重置熔断器"""
        self._circuit_breaker["failures"] = 0
        self._circuit_breaker["open_until"] = 0
    
    @property
    def name(self) -> str:
        return "llm"
    
    def _compress_text(self, text: str) -> Optional[str]:
        if not text:
            return None
        
        text_len = len(text)
        
        if text_len > 10000:
            return self._compress_long_text(text)
        
        if text_len > 6000:
            return self._compress_chunked_text(text)
        
        return self._compress_single_text(text)
    
    def _compress_single_text(self, text: str) -> Optional[str]:
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
            for chunk in chunks:
                compressed = self._compress_single_text(chunk)
                if compressed:
                    compressed_chunks.append(compressed)
            
            if not compressed_chunks:
                return None
            
            merged = '\n'.join(compressed_chunks)
            
            if len(merged) < len(text) * 0.7:
                return self._compress_single_text(merged)
            
            return merged
            
        except Exception:
            return None
    
    def _compress_long_text(self, text: str) -> Optional[str]:
        try:
            if config.cloud.enabled and config.cloud.api_key:
                return self._cloud_compress_text(text)
            
            chunked_result = self._compress_chunked_text(text)
            if chunked_result and len(chunked_result) < len(text) * 0.8:
                return chunked_result
            
            return None
            
        except Exception:
            return None
    
    def _cloud_compress_text(self, text: str) -> Optional[str]:
        try:
            from cloud_client import CloudClient
            from prompts import get_prompt_manager
            
            cloud = CloudClient()
            prompt_manager = get_prompt_manager()
            
            memory_list = [line.strip() for line in text.split('\n') if line.strip()]
            
            if len(memory_list) > 1:
                messages = prompt_manager.get_compression_prompt(memory_list)
            else:
                messages = [
                    {"role": "system", "content": "你是一个记忆压缩专家。"},
                    {"role": "user", "content": f"""请将以下内容压缩为简洁的摘要，保留所有关键信息。

原始内容：
{text}

压缩要求：
1. 保留所有专有名词、人名、地名、数值
2. 保留因果关系和关键决策
3. 去除冗余和修饰性内容
4. 压缩后长度约为原文的30%-50%

压缩结果："""}
                ]
            
            result = cloud.chat(messages, max_tokens=1000)
            return result.strip() if result else None
            
        except Exception:
            return None


class RuleBasedCompressionStrategy(CompressionStrategy):
    """规则压缩策略（降级方案）"""
    
    def __init__(self, mem_config):
        self._mem_config = mem_config
    
    def compress(self, text: str) -> Optional[str]:
        sentences = re.split(r'[。！？\n]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        key_sentences = []
        key_patterns = self._mem_config.compression.key_patterns.split('|')
        
        for sentence in sentences:
            for pattern in key_patterns:
                if re.search(pattern, sentence):
                    if sentence not in key_sentences:
                        key_sentences.append(sentence)
                    break
        
        target_ratio = self._mem_config.compression.target_ratio + 0.1
        max_segment = self._mem_config.compression.max_segment_length
        
        if key_sentences:
            compressed = '。'.join(key_sentences[:5])
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
    """压缩策略链 - 按优先级尝试多种策略"""
    
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


class AsyncMemoryProcessor:
    """异步记忆处理器：后台收集对话并增量更新 ChromaDB + SQLite"""
    
    def __init__(self, vector_store: VectorStore = None):
        self.vector_store = vector_store or get_vector_store()
        
        max_queue_size = self._get_queue_config().max_queue_size
        self.pending_queue = queue.Queue(maxsize=max_queue_size)
        self._queue_dropped_count = 0
        self._queue_total_count = 0
        
        self.running = False
        self.thread = None
        self._log = get_logger()
        self._mem_config = get_memory_config()
        self._tx_coordinator = get_transaction_coordinator()
        self._event_bus = get_event_bus()
        
        self._buffer_lock = threading.Lock()
        self.batch_buffer: List[Dict] = []
        self.batch_size = self._mem_config.async_processor.batch_size
        
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
                self._tx_coordinator.set_stores(self.sqlite, self.vector_store)
            except Exception as e:
                self._log.error("SQLITE_INIT_FAILED", error=str(e))
        
        self._compression_chain = CompressionStrategyChain([
            LLMCompressionStrategy(self._mem_config),
            RuleBasedCompressionStrategy(self._mem_config)
        ])
        self._compressor_warning_shown = False
        
        self._event_bus.subscribe(EventType.MEMORY_WRITTEN, self._handle_memory_written)
    
    def _get_queue_config(self):
        """获取队列配置"""
        try:
            return self._mem_config.async_processor
        except Exception:
            from config import AsyncProcessorConfig
            return AsyncProcessorConfig()
    
    def _handle_memory_written(self, event):
        """处理记忆写入事件"""
        pass
        
    def start(self):
        """启动后台处理线程"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        
        recovery_thread = threading.Thread(target=self._startup_recovery, daemon=True)
        recovery_thread.start()
        
        compression_thread = threading.Thread(target=self._compression_loop, daemon=True)
        compression_thread.start()
        
        self._log.info("ASYNC_PROCESSOR_STARTED")
    
    def _compression_loop(self):
        """
        独立的压缩处理循环
        
        将压缩和遗忘等重型操作从主处理循环中分离，
        避免阻塞队列处理
        """
        while self.running:
            try:
                time.sleep(self._mem_config.async_processor.compressor_check_interval)
                if not self.running:
                    break
                self._check_memory_flow()
            except Exception as e:
                self._log.error("COMPRESSION_LOOP_ERROR", error=str(e))
    
    def _startup_recovery(self):
        """
        启动补偿机制
        
        检查SQLite中未向量化的记录，重新触发同步
        解决：
        1. ChromaDB写入成功但SQLite更新失败导致的数据不一致
        2. L2→L3迁移中断导致的孤立记录（is_vectorized=2）
        3. L2→L3迁移崩溃导致的迁移中记录（is_vectorized=3）
        
        防重复机制：
        - 添加向量前先搜索是否已存在相同文本（相似度>0.99）
        - 如果存在则更新SQLite状态，不重复添加
        """
        if not self.sqlite:
            return
        
        try:
            migration_interrupted = self.sqlite.get_records_by_vector_status(is_vectorized=3, limit=50)
            if migration_interrupted:
                print(f"[启动补偿] 发现 {len(migration_interrupted)} 条迁移中断记录，开始恢复...")
                for record in migration_interrupted:
                    try:
                        if record.vector_id:
                            try:
                                self.vector_store.delete(ids=[record.vector_id])
                            except Exception:
                                pass
                        self.sqlite.update_vector_status(record.id, "", is_vectorized=0)
                        self._log.info("MIGRATION_INTERRUPTED_RECOVERED", record_id=record.id)
                    except Exception as e:
                        self._log.error("MIGRATION_INTERRUPTED_RECOVERY_FAILED", record_id=record.id, error=str(e))
            
            unvectorized = self.sqlite.get_unvectorized(limit=50)
            if not unvectorized:
                return
            
            print(f"[启动补偿] 发现 {len(unvectorized)} 条未向量化记录，开始检查...")
            
            success_count = 0
            skip_count = 0
            fail_count = 0
            
            for record in unvectorized:
                if record.is_vectorized == -1:
                    skip_count += 1
                    continue
                
                if record.is_vectorized == 2 and record.vector_id:
                    try:
                        self.vector_store.delete(ids=[record.vector_id])
                    except Exception:
                        pass
                
                existing = self.vector_store.search(record.text, n_results=1)
                if existing and existing[0].get("similarity", 0) > 0.99:
                    existing_id = existing[0].get("id")
                    self.sqlite.update_vector_status(record.id, existing_id, is_vectorized=1)
                    skip_count += 1
                    self._log.debug("STARTUP_RECOVERY_SKIP", record_id=record.id, reason="vector_exists")
                    continue
                
                try:
                    vector_ids = self.vector_store.add([record.text], [record.metadata or {}])
                    if vector_ids:
                        self.sqlite.update_vector_status(record.id, vector_ids[0], is_vectorized=1)
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    fail_count += 1
                    self._log.error("STARTUP_RECOVERY_FAILED", record_id=record.id, error=str(e))
            
            if success_count > 0 or skip_count > 0 or fail_count > 0:
                print(f"[启动补偿] 完成: 成功={success_count}, 跳过={skip_count}, 失败={fail_count}")
            
            orphan_vectors = self._check_and_clean_orphan_vectors()
            if orphan_vectors > 0:
                print(f"[启动补偿] 清理了 {orphan_vectors} 个孤立向量")
            
            self._sync_timestamps()
            
            self._retry_failed_vectorizations()
                
        except Exception as e:
            print(f"[启动补偿] 检查失败: {e}")
    
    def _check_and_clean_orphan_vectors(self) -> int:
        """
        检查并清理ChromaDB中的孤立向量
        
        孤立向量：ChromaDB中存在但SQLite中无对应记录
        
        优化策略：
        1. 通过metadata中的sqlite_id快速定位
        2. 分批处理，每批100条
        3. 无sqlite_id的旧数据通过vector_id反查
        """
        if not self.sqlite:
            return 0
        
        try:
            all_ids = self.vector_store.get_all_ids()
            if not all_ids:
                return 0
            
            orphan_ids = []
            batch_size = 100
            
            for i in range(0, len(all_ids), batch_size):
                batch_ids = all_ids[i:i + batch_size]
                
                for vid in batch_ids:
                    metadata = self.vector_store.get_metadata(vid)
                    
                    if metadata and "sqlite_id" in metadata:
                        sqlite_id = metadata["sqlite_id"]
                        record = self.sqlite.get(sqlite_id)
                        if not record:
                            orphan_ids.append(vid)
                    else:
                        record = self.sqlite.get_by_vector_id(vid)
                        if not record:
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
                record = self.sqlite.get_by_vector_id(vid)
                if record:
                    chroma_metadata = self.vector_store.get_metadata(vid)
                    if chroma_metadata:
                        sqlite_time = record.created_time
                        chroma_time = chroma_metadata.get("timestamp")
                        if sqlite_time and chroma_time and sqlite_time != chroma_time:
                            self.vector_store.update_metadata(vid, {"timestamp": sqlite_time})
        except Exception as e:
            print(f"[同步时间戳] 失败: {e}")
    
    def stop(self):
        """
        停止后台处理
        
        优雅关闭流程：
        1. 设置停止标志
        2. 等待队列清空（最多30秒）
        3. 等待后台线程结束（最多10秒）
        4. 最后一次落盘
        5. 等待所有迁移操作完成
        """
        self.running = False
        
        queue_timeout = 30
        start = time.time()
        while not self.pending_queue.empty() and time.time() - start < queue_timeout:
            time.sleep(0.5)
        
        remaining = self.pending_queue.qsize()
        if remaining > 0:
            self._log.warning("STOP_QUEUE_NOT_EMPTY", remaining=remaining)
        
        if self.thread:
            self.thread.join(timeout=10)
            if self.thread.is_alive():
                self._log.warning("STOP_THREAD_TIMEOUT")
        
        self._flush_buffer()
        
        while self._tx_coordinator.is_migration_active():
            time.sleep(0.1)
        
        self._log.info("ASYNC_PROCESSOR_STOPPED", 
                       queue_remaining=remaining,
                       buffer_remaining=len(self.batch_buffer))
    
    def add_conversation(self, user_input: str, assistant_response: str, metadata: Dict = None):
        """
        添加一轮对话到处理队列
        
        背压机制：
        - 队列满时，根据配置策略处理：
          - drop_oldest: 丢弃最旧的记录
          - reject: 拒绝新记录
        - 记录丢弃统计，用于监控
        
        过滤机制：
        - 入口处提前进行nonsense过滤
        - 确保过滤覆盖所有路径
        """
        if config.nonsense_filter_enabled:
            filter_result = get_nonsense_filter().filter(user_input, assistant_response)
            
            from metrics import get_metrics_collector
            metrics = get_metrics_collector()
            
            if filter_result.storage_type == "discard":
                self._log.debug("NONSENSE_FILTERED_DISCARD", 
                              reason=filter_result.reason,
                              confidence=filter_result.confidence)
                metrics.record_filter_result("discard")
                return
            
            metadata = metadata or {}
            metadata["nonsense_filter_result"] = filter_result.storage_type
            
            if filter_result.storage_type == "sqlite_only":
                metrics.record_filter_result("sqlite_only")
                self._log.debug("NONSENSE_FILTERED_SQLITE_ONLY",
                              reason=filter_result.reason)
        
        conv_data = {
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "assistant": assistant_response,
            "metadata": metadata or {}
        }
        
        self._queue_total_count += 1
        
        try:
            self.pending_queue.put_nowait(conv_data)
        except queue.Full:
            queue_config = self._get_queue_config()
            action = getattr(queue_config, 'queue_full_action', 'drop_oldest')
            
            if action == "drop_oldest":
                try:
                    self.pending_queue.get_nowait()
                    self._queue_dropped_count += 1
                    self._log.warning("QUEUE_FULL_DROPPING_OLDEST", 
                                      dropped_count=self._queue_dropped_count,
                                      queue_size=self.pending_queue.qsize())
                    self.pending_queue.put_nowait(conv_data)
                except queue.Empty:
                    self.pending_queue.put_nowait(conv_data)
            else:
                self._queue_dropped_count += 1
                self._log.warning("QUEUE_FULL_REJECTED",
                                  rejected_count=self._queue_dropped_count,
                                  queue_size=self.pending_queue.qsize())
    
    def get_queue_stats(self) -> Dict[str, int]:
        """获取队列统计信息"""
        return {
            "queue_size": self.pending_queue.qsize(),
            "queue_maxsize": self.pending_queue.maxsize,
            "total_enqueued": self._queue_total_count,
            "total_dropped": self._queue_dropped_count,
            "drop_rate": self._queue_dropped_count / max(1, self._queue_total_count)
        }
    
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
        
        注意：nonsense过滤已在add_conversation入口处完成
        """
        user_input = conv['user']
        assistant_response = conv['assistant']
        source = conv.get('metadata', {}).get('source', 'local')
        conv_metadata = conv.get('metadata', {})
        
        conversation_memory = f"用户: {user_input}\n助理: {assistant_response}"
        metadata = {
            "type": "conversation",
            "source": source,
            "timestamp": conv["timestamp"]
        }
        
        nonsense_result = conv_metadata.get("nonsense_filter_result", "normal")
        if nonsense_result == "sqlite_only":
            self._store_to_sqlite_only(conversation_memory, metadata)
            return
        
        should_flush = False
        with self._buffer_lock:
            if len(self.batch_buffer) == 0:
                self._buffer_first_item_time = time.time()
            
            self.batch_buffer.append({
                "text": conversation_memory,
                "metadata": metadata
            })
            
            if len(self.batch_buffer) >= self.batch_size:
                should_flush = True
        
        if should_flush:
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
        将缓冲区数据原子化写入（批量优化版）
        
        流程（两阶段提交）：
        1. 批量检查幂等性（文本哈希）
        2. 批量写入SQLite（标记为待向量化）
        3. 批量写入ChromaDB
        4. 批量更新SQLite状态
        
        性能优化：
        - 使用批量哈希计算
        - 使用批量存在性检查
        - 使用批量插入（add_batch）
        - 使用批量状态更新
        - ID预生成实现幂等性
        
        失败恢复：
        - 启动时通过 _startup_recovery 重试未向量化的记录
        - 使用文本哈希防止重复写入
        - 预生成vector_id确保重试幂等性
        
        线程安全：
        - 使用 _buffer_lock 保护 batch_buffer 的读取和清空
        """
        with self._buffer_lock:
            if not self.batch_buffer:
                return
            
            texts = [item["text"] for item in self.batch_buffer]
            metadatas = [item["metadata"] for item in self.batch_buffer]
            self.batch_buffer.clear()
            self._buffer_first_item_time = None
        
        if not texts:
            return
        
        import uuid
        sqlite_ids = []
        vector_ids = []
        texts_to_vectorize = []
        metadatas_to_vectorize = []
        
        if self.sqlite:
            text_hashes = [self.sqlite.compute_text_hash(text) for text in texts]
            
            existing_map = {}
            for text_hash in text_hashes:
                exists, existing_id = self.sqlite.exists_by_text_hash(text_hash)
                if exists:
                    existing_map[text_hash] = existing_id
            
            new_records = []
            new_record_hashes = []
            
            for i, (text, meta, text_hash) in enumerate(zip(texts, metadatas, text_hashes)):
                if text_hash in existing_map:
                    existing_id = existing_map[text_hash]
                    record = self.sqlite.get(existing_id)
                    if record and record.is_vectorized == 1:
                        self._log.debug("SKIP_DUPLICATE", text_preview=text[:50])
                        continue
                    
                    pre_generated_id = record.vector_id if record and record.vector_id else str(uuid.uuid4())
                    if not record or not record.vector_id:
                        self.sqlite.update_vector_status(existing_id, pre_generated_id, is_vectorized=0)
                    
                    sqlite_ids.append(existing_id)
                    vector_ids.append(pre_generated_id)
                    texts_to_vectorize.append(text)
                    meta_with_id = dict(meta) if meta else {}
                    meta_with_id["sqlite_id"] = existing_id
                    metadatas_to_vectorize.append(meta_with_id)
                else:
                    pre_generated_id = str(uuid.uuid4())
                    from sqlite_store import MemoryRecord
                    record = MemoryRecord(
                        text=text,
                        source=meta.get("source", "local"),
                        metadata=meta,
                        is_vectorized=0,
                        vector_id=pre_generated_id
                    )
                    new_records.append(record)
                    new_record_hashes.append(text_hash)
            
            if new_records:
                new_ids = self.sqlite.add_batch_with_hash(new_records, new_record_hashes)
                for new_id, record in zip(new_ids, new_records):
                    sqlite_ids.append(new_id)
                    vector_ids.append(record.vector_id)
                    texts_to_vectorize.append(record.text)
                    meta_with_id = dict(record.metadata) if record.metadata else {}
                    meta_with_id["sqlite_id"] = new_id
                    metadatas_to_vectorize.append(meta_with_id)
        else:
            vector_ids = [str(uuid.uuid4()) for _ in texts]
            texts_to_vectorize = texts
            metadatas_to_vectorize = metadatas
        
        if not texts_to_vectorize:
            return
        
        try:
            result_ids = self.vector_store.add(texts_to_vectorize, metadatas_to_vectorize, ids=vector_ids)
            
            if self.sqlite and sqlite_ids and result_ids:
                self.sqlite.batch_update_vector_status(
                    [(sid, vid) for sid, vid in zip(sqlite_ids, result_ids)],
                    is_vectorized=1
                )
            
            self._log.info("ATOMIC_WRITE_COMPLETE", count=len(texts_to_vectorize))
            self._last_flush_time = time.time()
            
            self._event_bus.publish(
                EventType.MEMORY_WRITTEN,
                {"count": len(texts_to_vectorize)},
                source="AsyncProcessor"
            )
            
        except Exception as e:
            self._log.error("CHROMADB_WRITE_FAILED", error=str(e))
            
            if self.sqlite and sqlite_ids:
                for sid in sqlite_ids:
                    self.sqlite.update_vector_status(sid, "", is_vectorized=-1)
    
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
        
        线程安全：
        - 使用 _buffer_lock 保护对 batch_buffer 的检查
        """
        current_time = time.time()
        
        should_flush = False
        flush_reason = ""
        
        max_buffer_age = self._mem_config.async_processor.max_buffer_age
        flush_interval = self._mem_config.async_processor.flush_interval
        
        with self._buffer_lock:
            if len(self.batch_buffer) > 0:
                time_since_last_flush = current_time - self._last_flush_time
                
                buffer_age = 0
                if self._buffer_first_item_time:
                    buffer_age = current_time - self._buffer_first_item_time
                
                if buffer_age >= max_buffer_age:
                    should_flush = True
                    flush_reason = f"[强制落盘] 缓冲区 {len(self.batch_buffer)} 条记录，已存活 {buffer_age:.1f} 秒"
                elif time_since_last_flush >= flush_interval:
                    should_flush = True
                    flush_reason = f"[定时落盘] 缓冲区 {len(self.batch_buffer)} 条记录，距上次落盘 {time_since_last_flush:.1f} 秒"
                elif len(self.batch_buffer) >= self.batch_size:
                    should_flush = True
                    flush_reason = f"[缓冲区满] {len(self.batch_buffer)} 条记录"
        
        if should_flush:
            if flush_reason:
                self._log.info("FLUSH_TRIGGERED", reason=flush_reason)
            self._flush_buffer()
            self._last_flush_time = current_time
    
    def _check_and_dedup(self):
        """空闲时检查并执行去重"""
        current_time = time.time()
        
        if self._idle_count < 3:
            return
        
        dedup_interval = self._mem_config.async_processor.dedup_interval
        
        if current_time - self._last_dedup_time < dedup_interval:
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
        4. sqlite_only升级：高价值的sqlite_only记忆升级到向量库
        5. 待压缩处理：处理之前标记为待压缩的记忆
        6. 遗忘机制：对L3中权重极低的记忆进行遗忘
        
        所有操作共享时间检查，避免重复执行
        """
        if not config.sqlite_enabled or not self.sqlite:
            return
        
        current_time = time.time()
        
        if self._idle_count < 5:
            return
        
        compression_interval = self._mem_config.async_processor.compression_interval
        forget_interval = self._mem_config.async_processor.forget_interval
        
        if current_time - self._last_compression_time < compression_interval:
            return
        
        try:
            self._log.debug("MEMORY_FLOW_STARTED")
            
            moved_to_l3 = self._move_l2_to_l3()
            
            compressed = self._compress_l3_memories()
            
            moved_to_l2 = self._move_l3_to_l2()
            
            upgraded = self._upgrade_sqlite_only()
            
            self._process_pending_compressions()
            
            merged = self._check_and_merge_l2()
            
            if self._idle_count >= 10 and current_time - self._last_forget_time >= forget_interval:
                result = self.sqlite.decay_weights(config.memory_decay_days)
                decayed = result.get("decayed", 0)
                forgotten = result.get("forgotten", 0)
                vector_ids_to_delete = result.get("vector_ids_to_delete", [])
                
                if vector_ids_to_delete:
                    try:
                        self.vector_store.delete(ids=vector_ids_to_delete)
                        self._log.info("CHROMADB_SYNC_DELETE", count=len(vector_ids_to_delete))
                    except Exception as e:
                        self._log.error("CHROMADB_DELETE_FAILED", error=str(e))
                
                if decayed > 0 or forgotten > 0:
                    self._log.info("MEMORY_DECAY", decayed=decayed, forgotten=forgotten)
                self._last_forget_time = current_time
            
            total_actions = moved_to_l3 + compressed + moved_to_l2 + merged
            if total_actions > 0:
                self._log.info("MEMORY_FLOW_COMPLETE", 
                               l2_to_l3=moved_to_l3, 
                               compressed=compressed, 
                               l3_to_l2=moved_to_l2,
                               merged=merged)
            
            self._last_compression_time = current_time
            self._idle_count = 0
            
        except Exception as e:
            self._log.error("MEMORY_FLOW_FAILED", error=str(e))
    
    def _move_l2_to_l3(self) -> int:
        """
        将L2(向量库)中不常用、低权重的记忆移动到L3(SQLite)
        
        条件：
        - 长时间未访问
        - 权重较低
        - 不在冷却期内（最近从L3回流的记忆需要等待24小时）
        
        流程（原子化迁移）：
        1. 标记SQLite记录为"正在迁移"（is_vectorized=2）
        2. 从ChromaDB删除向量
        3. 更新SQLite记录为未向量化状态（is_vectorized=0）
        
        恢复机制：
        - 启动时检查 is_vectorized=2 的记录，重新向量化
        
        并发安全：
        - 使用事务协调器的迁移锁保护整个迁移过程
        - 检索操作会等待迁移完成
        
        状态定义：
        - is_vectorized=0: 未向量化(L3)
        - is_vectorized=1: 已向量化(L2)
        - is_vectorized=2: 向量化失败
        - is_vectorized=3: 迁移中（原子性保证）
        """
        if not self.sqlite:
            return 0
        
        cooldown_hours = self._mem_config.memory_flow.cooldown_hours
        
        self._tx_coordinator.begin_migration()
        try:
            threshold = config.memory_min_weight * self._mem_config.memory_flow.l2_move_threshold_multiplier
            
            low_weight_records = self.sqlite.get_low_weight_memories(
                threshold=threshold,
                limit=self._mem_config.memory_flow.max_l2_to_l3_batch
            )
            
            moved_count = 0
            for record in low_weight_records:
                if not record.vector_id:
                    continue
                
                if record.metadata:
                    if MemoryTagHelper.is_in_cooldown_from_l2(record.metadata, cooldown_hours):
                        self._log.debug("L2_TO_L3_COOLDOWN_PROMOTED", 
                                      record_id=record.id)
                        continue
                    
                    if MemoryTagHelper.is_in_cooldown_from_l3(record.metadata, cooldown_hours):
                        self._log.debug("L2_TO_L3_COOLDOWN_MOVED_FROM_L3",
                                      record_id=record.id)
                        continue
                
                try:
                    self.sqlite.update_vector_status(record.id, record.vector_id, is_vectorized=3)
                    
                    self.vector_store.delete(ids=[record.vector_id])
                    
                    self.sqlite.update_vector_status(record.id, "", is_vectorized=0)
                    
                    record.metadata = MemoryTagHelper.mark_moved_to_l3(record.metadata)
                    
                    self.sqlite.add(record)
                    
                    moved_count += 1
                    
                except Exception as e:
                    self._log.error("L2_TO_L3_MOVE_FAILED", record_id=record.id, error=str(e))
                    
                    try:
                        current_record = self.sqlite.get(record.id)
                        if current_record and current_record.is_vectorized == 3:
                            if current_record.vector_id:
                                try:
                                    self.vector_store.delete(ids=[current_record.vector_id])
                                except Exception:
                                    pass
                            self.sqlite.update_vector_status(record.id, "", is_vectorized=0)
                            self._log.info("L2_TO_L3_ROLLBACK", record_id=record.id)
                        else:
                            self.sqlite.update_vector_status(record.id, record.vector_id, is_vectorized=1)
                    except Exception as rollback_e:
                        self._log.error("L2_TO_L3_ROLLBACK_FAILED", record_id=record.id, error=str(rollback_e))
            
            return moved_count
            
        except Exception as e:
            self._log.error("L2_TO_L3_FLOW_FAILED", error=str(e))
            return 0
        finally:
            self._tx_coordinator.end_migration()
    
    def _should_skip_compression(self, record: MemoryRecord) -> bool:
        """
        检查记忆是否应该跳过压缩
        
        跳过条件：
        1. 已有压缩文本（compressed_text存在）
        2. metadata中标记为受保护
        3. metadata中标记为已压缩
        """
        if record.compressed_text:
            return True
        
        if MemoryTagHelper.is_protected(record.metadata):
            return True
        
        if MemoryTagHelper.is_compressed(record.metadata):
            return True
        
        return False
    
    def _compress_l3_memories(self) -> int:
        """
        压缩L3(SQLite)中的记忆
        
        条件：
        - 长时间未访问
        - 权重较低（低于阈值）
        - 未压缩过
        - 非高密度内容
        - 非受保护记忆
        """
        import time
        from metrics import get_metrics_collector
        
        if not self.sqlite:
            return 0
        
        compression_weight_threshold = self._mem_config.memory_flow.l3_promotion_weight_threshold * self.sqlite.MAX_WEIGHT
        
        try:
            unaccessed = self.sqlite.get_unaccessed_memories(
                days=config.memory_decay_days // 2,
                limit=self._mem_config.async_processor.max_pending_compressions
            )
            
            if not unaccessed:
                return 0
            
            compressed_count = 0
            preserved_count = 0
            skipped_count = 0
            metrics = get_metrics_collector()
            
            for record in unaccessed:
                try:
                    if self._should_skip_compression(record):
                        skipped_count += 1
                        continue
                    
                    if record.weight >= compression_weight_threshold:
                        skipped_count += 1
                        continue
                    
                    if self._is_high_density_content(record.text):
                        self._mark_preserve(record)
                        preserved_count += 1
                        continue
                    
                    min_length = self._mem_config.compression.min_length
                    if len(record.text) < min_length:
                        continue
                    
                    target_ratio = self._mem_config.compression.target_ratio
                    original_length = len(record.text)
                    
                    compress_start = time.perf_counter()
                    compressed_text, strategy_name = self._compression_chain.compress(record.text)
                    compress_duration = (time.perf_counter() - compress_start) * 1000
                    
                    if compressed_text and len(compressed_text) < original_length * target_ratio:
                        record.compressed_text = compressed_text
                        record.metadata = MemoryTagHelper.mark_compressed(
                            record.metadata, strategy_name, original_length
                        )
                        
                        if record.vector_id:
                            try:
                                self.vector_store.delete(ids=[record.vector_id])
                                self._log.info("INVALIDATED_L2_AFTER_COMPRESSION", 
                                             record_id=record.id,
                                             vector_id=record.vector_id)
                            except Exception as del_e:
                                self._log.warning("FAILED_TO_INVALIDATE_L2", 
                                                record_id=record.id,
                                                error=str(del_e))
                            record.vector_id = ""
                            record.is_vectorized = 0
                        
                        self.sqlite.add(record)
                        compressed_count += 1
                        metrics.record_compression(True, compress_duration)
                    else:
                        self._mark_pending_compression(record)
                        metrics.record_compression(False)
                        
                except Exception as e:
                    self._log.error("COMPRESSION_SINGLE_FAILED", error=str(e))
                    metrics.record_compression(False)
            
            if skipped_count > 0:
                self._log.debug("COMPRESSION_SKIPPED", count=skipped_count)
            
            return compressed_count
            
        except Exception as e:
            self._log.error("L3_COMPRESSION_FAILED", error=str(e))
            return 0
    
    def _move_l3_to_l2(self) -> int:
        """
        将L3(SQLite)中高权重、常用的记忆回流到L2(向量库)
        
        条件：
        - 权重超过阈值
        - 最近有访问
        - 当前未向量化
        - 不在冷却期内（最近从L2移动过来的需要等待24小时）
        
        注意：
        - 如果记忆有compressed_text，标记为已压缩，未来不再压缩
        - 回流后的记忆若再次存入L3，会跳过压缩流程
        - 回流完成后触发相似记忆合并检测
        
        并发安全：
        - 使用事务协调器的迁移锁保护整个回流过程
        - 回流失败时回滚已写入的向量
        """
        if not self.sqlite:
            return 0
        
        cooldown_hours = self._mem_config.memory_flow.cooldown_hours
        
        self._tx_coordinator.begin_migration()
        try:
            high_weight_records = self.sqlite.get_high_weight_memories(
                limit=self._mem_config.memory_flow.max_l3_to_l2_batch
            )
            
            moved_count = 0
            moved_records = []
            
            for record in high_weight_records:
                try:
                    if record.metadata and record.metadata.get(MemoryTags.MOVED_FROM_L2):
                        moved_time_str = record.metadata.get(MemoryTags.MOVED_FROM_L2_TIME)
                        if moved_time_str:
                            try:
                                moved_time = datetime.fromisoformat(moved_time_str)
                                cooldown_end = moved_time + timedelta(hours=cooldown_hours)
                                if datetime.now() < cooldown_end:
                                    continue
                            except Exception:
                                pass
                    
                    self.sqlite.update_vector_status(record.id, "", is_vectorized=2)
                    
                    text_to_vectorize = record.text
                    
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
                        
                        has_compressed = bool(record.compressed_text)
                        record.metadata = MemoryTagHelper.mark_moved_to_l2(
                            record.metadata, has_compressed
                        )
                        
                        self.sqlite.add(record)
                        
                        moved_records.append({
                            "id": record.id,
                            "text": record.text,
                            "created_time": record.created_time,
                            "weight": record.weight,
                            "access_count": record.access_count,
                            "metadata": record.metadata or {}
                        })
                        
                        moved_count += 1
                    else:
                        self.sqlite.update_vector_status(record.id, "", is_vectorized=0)
                        
                except Exception as e:
                    self._log.error("L3_TO_L2_MOVE_FAILED", record_id=record.id, error=str(e))
                    self.sqlite.update_vector_status(record.id, "", is_vectorized=0)
            
            if moved_records:
                self._trigger_merge_on_backflow(moved_records)
            
            return moved_count
            
        except Exception as e:
            self._log.error("L3_TO_L2_FLOW_FAILED", error=str(e))
            return 0
        finally:
            self._tx_coordinator.end_migration()
    
    def _trigger_merge_on_backflow(self, moved_records: List[Dict]):
        """
        L3→L2 回流时触发合并检测
        
        检测回流记忆之间或与 L2 现有记忆的相似性，
        如果发现相似记忆组则触发合并。
        
        Args:
            moved_records: 回流的记忆列表
        """
        try:
            merger = get_merger()
            
            result = merger.merge_on_l3_to_l2(
                moved_records,
                self.vector_store,
                self.sqlite
            )
            
            merged_count = result.get("merged_count", 0)
            
            if merged_count > 0:
                self._log.info(
                    "L3_TO_L2_MERGE_COMPLETE",
                    merged_count=merged_count,
                    message=result.get("message", "")
                )
                
        except Exception as e:
            self._log.error("L3_TO_L2_MERGE_FAILED", error=str(e))
    
    def _check_and_merge_l2(self) -> int:
        """
        检查并合并 L2（向量库）中的相似记忆
        
        触发条件：
        - 距离上次合并检查超过配置间隔
        - 发现相似记忆组（相似度 > 阈值）
        
        流程：
        1. 获取 L2 中最近的记忆
        2. 检测相似记忆组
        3. 合并相似条目
        4. 更新存储
        
        不影响原有压缩、归档、遗忘逻辑。
        """
        try:
            merger = get_merger()
            
            result = merger.check_and_merge_l2(
                self.vector_store,
                self.sqlite,
                force=False
            )
            
            merged_count = result.get("merged_count", 0)
            
            if merged_count > 0:
                self._log.info(
                    "L2_MERGE_COMPLETE",
                    merged_count=merged_count,
                    groups=result.get("groups", 0)
                )
            
            return merged_count
            
        except Exception as e:
            self._log.error("L2_MERGE_CHECK_FAILED", error=str(e))
            return 0
    
    def _upgrade_sqlite_only(self) -> int:
        """
        将高价值的 sqlite_only 记忆升级到向量库
        
        条件：
        - 权重超过阈值（说明有价值）
        - 最近有访问（说明仍在使用）
        - is_vectorized = -1（sqlite_only 标记）
        
        这些记录最初被判定为低价值，但后续访问表明有价值，
        应该升级到向量库以支持语义检索。
        
        并发安全：
        - 使用事务协调器的迁移锁保护整个升级过程
        """
        if not self.sqlite:
            return 0
        
        self._tx_coordinator.begin_migration()
        try:
            upgradeable_records = self.sqlite.get_upgradeable_sqlite_only(
                limit=self._mem_config.memory_flow.max_l3_to_l2_batch
            )
            
            if not upgradeable_records:
                return 0
            
            upgraded_count = 0
            for record in upgradeable_records:
                try:
                    self.sqlite.update_vector_status(record.id, "", is_vectorized=2)
                    
                    vector_ids = self.vector_store.add(
                        [record.text],
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
                        record.metadata["upgraded_from_sqlite_only"] = True
                        record.metadata["upgraded_time"] = datetime.now().isoformat()
                        self.sqlite.add(record)
                        
                        upgraded_count += 1
                        self._log.info("SQLITE_ONLY_UPGRADED", record_id=record.id)
                    else:
                        self.sqlite.update_vector_status(record.id, "", is_vectorized=-1)
                        
                except Exception as e:
                    self._log.error("SQLITE_ONLY_UPGRADE_FAILED", record_id=record.id, error=str(e))
                    self.sqlite.update_vector_status(record.id, "", is_vectorized=-1)
            
            return upgraded_count
            
        except Exception as e:
            self._log.error("SQLITE_ONLY_UPGRADE_FLOW_FAILED", error=str(e))
            return 0
        finally:
            self._tx_coordinator.end_migration()
    
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
        record.metadata = MemoryTagHelper.mark_preserve(
            record.metadata, "high_density_content"
        )
        self.sqlite.add(record)
    
    def _mark_pending_compression(self, record: MemoryRecord):
        """标记记忆为待压缩"""
        retry_count = 0
        if record.metadata:
            retry_count = record.metadata.get(MemoryTags.RETRY_COUNT, 0)
        record.metadata = MemoryTagHelper.mark_pending_compression(
            record.metadata, retry_count
        )
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
            
            self._log.info("PENDING_COMPRESSION_START", count=len(pending_records))
            
            compressed_count = 0
            skipped_count = 0
            for record in pending_records:
                try:
                    if self._should_skip_compression(record):
                        record.metadata = MemoryTagHelper.clear_pending_compression(record.metadata)
                        self.sqlite.add(record)
                        skipped_count += 1
                        continue
                    
                    if self._is_high_density_content(record.text):
                        self._mark_preserve(record)
                        continue
                    
                    if len(record.text) < self._mem_config.compression.min_length:
                        record.metadata = MemoryTagHelper.clear_pending_compression(record.metadata)
                        self.sqlite.add(record)
                        continue
                    
                    original_length = len(record.text)
                    compressed_text, strategy_name = self._compression_chain.compress(record.text)
                    
                    if compressed_text and len(compressed_text) < original_length * self._mem_config.compression.target_ratio:
                        record.compressed_text = compressed_text
                        record.metadata = MemoryTagHelper.clear_pending_compression(record.metadata)
                        record.metadata = MemoryTagHelper.mark_compressed(
                            record.metadata, strategy_name, original_length
                        )
                        self.sqlite.add(record)
                        compressed_count += 1
                        
                except Exception as e:
                    self._log.error("PENDING_COMPRESSION_FAILED", error=str(e))
            
            if compressed_count > 0 or skipped_count > 0:
                self._log.info("PENDING_COMPRESSION_DONE", 
                              compressed=compressed_count, 
                              skipped=skipped_count)
                
        except Exception as e:
            self._log.error("PENDING_COMPRESSION_ERROR", error=str(e))
    
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
