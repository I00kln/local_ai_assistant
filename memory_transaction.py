import json
import threading
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from logger import get_logger
from memory_tags import MemoryTags


class TransactionState(Enum):
    """事务状态"""
    PENDING = "pending"
    PREPARING = "preparing"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class TransactionRecord:
    """事务记录"""
    transaction_id: str
    operation_type: str
    state: TransactionState
    created_time: datetime
    updated_time: datetime
    data: Dict[str, Any]
    error_message: Optional[str] = None


class TransactionCoordinator:
    """
    事务协调器
    
    实现 SQLite + ChromaDB 的两阶段提交：
    1. 准备阶段：在 SQLite 中记录事务状态（持久化）
    2. 提交阶段：执行 ChromaDB 操作，成功后更新 SQLite 状态
    3. 恢复阶段：启动时检查未完成的事务并重试
    
    原子性保证：
    - 事务状态持久化到 SQLite 的 transactions 表
    - 崩溃后可通过事务表恢复未完成的事务
    - 使用 WAL 模式确保写入原子性
    
    使用方式：
    ```python
    coordinator = TransactionCoordinator(sqlite_store, vector_store)
    
    # 执行事务
    result = coordinator.execute_transaction(
        operation_type="add_memory",
        data={"text": "...", "metadata": {...}},
        prepare_fn=lambda data: sqlite_store.add_pending(data),
        commit_fn=lambda data: vector_store.add([data["text"]], [data["metadata"]])
    )
    ```
    """
    
    _instance: Optional['TransactionCoordinator'] = None
    _lock = threading.Lock()
    
    TRANSACTION_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id TEXT PRIMARY KEY,
            operation_type TEXT NOT NULL,
            state TEXT NOT NULL,
            created_time TEXT NOT NULL,
            updated_time TEXT NOT NULL,
            data TEXT,
            error_message TEXT
        )
    """
    
    TRANSACTION_INDEX_SQL = """
        CREATE INDEX IF NOT EXISTS idx_transactions_state 
        ON transactions(state, updated_time)
    """
    
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
        
        self._initialized = True
        self._log = get_logger()
        self._transactions: Dict[str, TransactionRecord] = {}
        self._sqlite_store = None
        self._vector_store = None
        self._migration_lock = threading.RLock()
        self._migration_active = False
        self._migration_type: Optional[str] = None
        self._tx_table_initialized = False
    
    def set_stores(self, sqlite_store, vector_store):
        """设置存储实例并初始化事务表"""
        self._sqlite_store = sqlite_store
        self._vector_store = vector_store
        self._init_transaction_table()
    
    def _init_transaction_table(self):
        """初始化事务表"""
        if self._tx_table_initialized or not self._sqlite_store:
            return
        
        try:
            with self._sqlite_store._get_write_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(self.TRANSACTION_TABLE_SQL)
                cursor.execute(self.TRANSACTION_INDEX_SQL)
                conn.commit()
                self._tx_table_initialized = True
                self._log.debug("TRANSACTION_TABLE_INITIALIZED")
        except Exception as e:
            self._log.error("TRANSACTION_TABLE_INIT_FAILED", error=str(e))
    
    def _persist_transaction(self, tx_record: TransactionRecord) -> bool:
        """
        持久化事务状态到 SQLite
        
        Args:
            tx_record: 事务记录
        
        Returns:
            是否成功
        """
        if not self._sqlite_store:
            return False
        
        try:
            with self._sqlite_store._get_write_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO transactions 
                    (transaction_id, operation_type, state, created_time, 
                     updated_time, data, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    tx_record.transaction_id,
                    tx_record.operation_type,
                    tx_record.state.value,
                    tx_record.created_time.isoformat(),
                    tx_record.updated_time.isoformat(),
                    json.dumps(tx_record.data, ensure_ascii=False),
                    tx_record.error_message
                ))
                conn.commit()
                return True
        except Exception as e:
            self._log.error("TRANSACTION_PERSIST_FAILED", 
                           transaction_id=tx_record.transaction_id,
                           error=str(e))
            return False
    
    def _update_transaction_state(self, tx_id: str, state: TransactionState, 
                                   error_message: str = None) -> bool:
        """
        更新事务状态
        
        Args:
            tx_id: 事务ID
            state: 新状态
            error_message: 错误信息（可选）
        
        Returns:
            是否成功
        """
        if not self._sqlite_store:
            return False
        
        try:
            with self._sqlite_store._get_write_connection() as conn:
                cursor = conn.cursor()
                if error_message:
                    cursor.execute("""
                        UPDATE transactions 
                        SET state = ?, updated_time = ?, error_message = ?
                        WHERE transaction_id = ?
                    """, (state.value, datetime.now().isoformat(), error_message, tx_id))
                else:
                    cursor.execute("""
                        UPDATE transactions 
                        SET state = ?, updated_time = ?
                        WHERE transaction_id = ?
                    """, (state.value, datetime.now().isoformat(), tx_id))
                conn.commit()
                return True
        except Exception as e:
            self._log.error("TRANSACTION_UPDATE_FAILED", 
                           transaction_id=tx_id,
                           error=str(e))
            return False
    
    def _delete_transaction(self, tx_id: str) -> bool:
        """
        删除事务记录
        
        Args:
            tx_id: 事务ID
        
        Returns:
            是否成功
        """
        if not self._sqlite_store:
            return False
        
        try:
            with self._sqlite_store._get_write_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM transactions WHERE transaction_id = ?", (tx_id,))
                conn.commit()
                return True
        except Exception as e:
            self._log.error("TRANSACTION_DELETE_FAILED", 
                           transaction_id=tx_id,
                           error=str(e))
            return False
    
    def _get_pending_transactions_from_db(self) -> List[TransactionRecord]:
        """
        从数据库获取未完成的事务
        
        Returns:
            未完成的事务列表
        """
        if not self._sqlite_store:
            return []
        
        try:
            conn = self._sqlite_store._get_read_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT transaction_id, operation_type, state, created_time,
                       updated_time, data, error_message
                FROM transactions
                WHERE state IN ('pending', 'preparing')
                ORDER BY created_time ASC
            """)
            
            records = []
            for row in cursor.fetchall():
                try:
                    records.append(TransactionRecord(
                        transaction_id=row[0],
                        operation_type=row[1],
                        state=TransactionState(row[2]),
                        created_time=datetime.fromisoformat(row[3]),
                        updated_time=datetime.fromisoformat(row[4]),
                        data=json.loads(row[5]) if row[5] else {},
                        error_message=row[6]
                    ))
                except (ValueError, json.JSONDecodeError) as e:
                    self._log.error("TRANSACTION_PARSE_FAILED", 
                                   transaction_id=row[0],
                                   error=str(e))
            
            return records
        except Exception as e:
            self._log.error("GET_PENDING_TRANSACTIONS_FAILED", error=str(e))
            return []
    
    def begin_migration(self, migration_type: str = "unknown"):
        """
        开始迁移操作
        
        获取迁移锁，阻止检索操作访问正在迁移的数据
        
        Args:
            migration_type: 迁移类型 (flush_buffer, l2_to_l3, l3_to_l2, merge, etc.)
        """
        self._migration_lock.acquire()
        self._migration_active = True
        self._migration_type = migration_type
        self._log.debug("MIGRATION_STARTED", migration_type=migration_type)
    
    def end_migration(self):
        """
        结束迁移操作
        
        释放迁移锁
        """
        migration_type = self._migration_type
        self._migration_active = False
        self._migration_type = None
        self._migration_lock.release()
        self._log.debug("MIGRATION_ENDED", migration_type=migration_type)
    
    def is_migration_active(self) -> bool:
        """检查是否有迁移操作正在进行"""
        return self._migration_active
    
    def get_migration_type(self) -> Optional[str]:
        """获取当前迁移类型"""
        return self._migration_type
    
    def wait_for_migration(self, timeout: float = 5.0) -> bool:
        """
        等待迁移完成
        
        Args:
            timeout: 超时时间（秒）
        
        Returns:
            是否成功获取锁
        """
        return self._migration_lock.acquire(timeout=timeout)
    
    def release_migration_wait(self):
        """释放迁移等待锁"""
        try:
            self._migration_lock.release()
        except RuntimeError:
            pass
    
    def execute_transaction(
        self,
        operation_type: str,
        data: Dict[str, Any],
        prepare_fn: Callable,
        commit_fn: Callable,
        rollback_fn: Optional[Callable] = None,
        transaction_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        执行事务（带持久化）
        
        两阶段提交流程：
        1. 创建事务记录，状态为 PENDING，持久化到 SQLite
        2. 执行准备阶段，状态更新为 PREPARING，持久化
        3. 执行提交阶段（ChromaDB 写入）
        4. 成功则状态更新为 COMMITTED，持久化
        5. 失败则执行回滚，状态更新为 FAILED 或 ROLLED_BACK
        
        Args:
            operation_type: 操作类型
            data: 操作数据
            prepare_fn: 准备阶段函数
            commit_fn: 提交阶段函数
            rollback_fn: 回滚函数（可选）
            transaction_id: 事务 ID（可选）
        
        Returns:
            {
                "success": bool,
                "transaction_id": str,
                "result": Any,
                "error": str (if failed)
            }
        """
        tx_id = transaction_id or f"{operation_type}_{int(time.time() * 1000)}"
        
        tx_record = TransactionRecord(
            transaction_id=tx_id,
            operation_type=operation_type,
            state=TransactionState.PENDING,
            created_time=datetime.now(),
            updated_time=datetime.now(),
            data=data
        )
        
        self._transactions[tx_id] = tx_record
        
        self._persist_transaction(tx_record)
        
        try:
            tx_record.state = TransactionState.PREPARING
            tx_record.updated_time = datetime.now()
            self._update_transaction_state(tx_id, TransactionState.PREPARING)
            
            prepare_result = prepare_fn(data)
            tx_record.data["prepare_result"] = prepare_result
            
            commit_result = commit_fn(data)
            tx_record.data["commit_result"] = commit_result
            
            tx_record.state = TransactionState.COMMITTED
            tx_record.updated_time = datetime.now()
            self._update_transaction_state(tx_id, TransactionState.COMMITTED)
            
            self._log.debug("TRANSACTION_COMMITTED", 
                           transaction_id=tx_id, 
                           operation=operation_type)
            
            return {
                "success": True,
                "transaction_id": tx_id,
                "result": commit_result
            }
            
        except Exception as e:
            tx_record.state = TransactionState.FAILED
            tx_record.error_message = str(e)
            tx_record.updated_time = datetime.now()
            self._update_transaction_state(tx_id, TransactionState.FAILED, str(e))
            
            self._log.error("TRANSACTION_FAILED",
                           transaction_id=tx_id,
                           operation=operation_type,
                           error=str(e))
            
            if rollback_fn:
                try:
                    rollback_fn(tx_record.data)
                    tx_record.state = TransactionState.ROLLED_BACK
                    self._update_transaction_state(tx_id, TransactionState.ROLLED_BACK)
                except Exception as rb_error:
                    self._log.error("ROLLBACK_FAILED",
                                   transaction_id=tx_id,
                                   error=str(rb_error))
            
            return {
                "success": False,
                "transaction_id": tx_id,
                "error": str(e)
            }
    
    def recover_pending_transactions(self) -> int:
        """
        恢复未完成的事务
        
        检查：
        1. SQLite 事务表中的 PREPARING 状态事务
        2. SQLite 中 is_vectorized=2 的记录（旧兼容）
        
        恢复策略：
        - add_memory 操作：尝试重新向量化
        - 其他操作：标记为需要手动干预
        
        Returns:
            恢复的记录数
        """
        if not self._sqlite_store:
            return 0
        
        recovered = 0
        failed = 0
        
        pending_txs = self._get_pending_transactions_from_db()
        for tx_record in pending_txs:
            self._log.info("TRANSACTION_RECOVERY_PENDING", 
                          transaction_id=tx_record.transaction_id,
                          operation_type=tx_record.operation_type,
                          state=tx_record.state.value)
            
            if tx_record.operation_type == "add_memory":
                recovery_result = self._recover_add_memory_transaction(tx_record)
                if recovery_result:
                    recovered += 1
                else:
                    failed += 1
            elif tx_record.operation_type == "add_memory_batch":
                recovery_result = self._recover_add_memory_batch_transaction(tx_record)
                if recovery_result:
                    recovered += recovery_result
                else:
                    failed += 1
            elif tx_record.operation_type == "add_memory_batch_fallback":
                recovery_result = self._recover_add_memory_batch_transaction(tx_record)
                if recovery_result:
                    recovered += recovery_result
                else:
                    failed += 1
            else:
                self._update_transaction_state(
                    tx_record.transaction_id, 
                    TransactionState.FAILED,
                    "Unknown operation type - requires manual intervention"
                )
                failed += 1
        
        try:
            pending_records = self._sqlite_store.get_unvectorized(limit=50)
            
            for record in pending_records:
                if record.is_vectorized != 2:
                    continue
                
                if record.vector_id:
                    try:
                        self._vector_store.delete(ids=[record.vector_id])
                    except Exception:
                        pass
                
                try:
                    deterministic_id = self._vector_store._generate_deterministic_id(record.text)
                    
                    vector_ids = self._vector_store.add(
                        [record.text],
                        [record.metadata or {}],
                        ids=[deterministic_id]
                    )
                    
                    if vector_ids:
                        self._sqlite_store.update_vector_status(
                            record.id, vector_ids[0], is_vectorized=1
                        )
                        recovered += 1
                        self._log.info("TRANSACTION_RECOVERED", record_id=record.id)
                except Exception as e:
                    self._log.error("TRANSACTION_RECOVERY_FAILED",
                                   record_id=record.id,
                                   error=str(e))
            
            if recovered > 0 or failed > 0:
                self._log.info("TRANSACTION_RECOVERY_COMPLETE", 
                              recovered=recovered, 
                              failed=failed)
            
        except Exception as e:
            self._log.error("TRANSACTION_RECOVERY_ERROR", error=str(e))
        
        return recovered
    
    def _recover_add_memory_transaction(self, tx_record: TransactionRecord) -> bool:
        """
        恢复 add_memory 类型的事务
        
        Args:
            tx_record: 事务记录
        
        Returns:
            是否恢复成功
        """
        data = tx_record.data
        text = data.get("text", "")
        metadata = data.get("metadata", {})
        
        if not text:
            self._update_transaction_state(
                tx_record.transaction_id,
                TransactionState.FAILED,
                "Missing text data"
            )
            return False
        
        try:
            existing = self._vector_store.search(text, n_results=1)
            if existing and existing[0].get("similarity", 0) > 0.99:
                existing_id = existing[0].get("id")
                sqlite_id = metadata.get(MemoryTags.SQLITE_ID)
                if sqlite_id:
                    self._sqlite_store.update_vector_status(
                        sqlite_id, existing_id, is_vectorized=1
                    )
                self._update_transaction_state(
                    tx_record.transaction_id,
                    TransactionState.COMMITTED
                )
                self._log.info("TRANSACTION_RECOVERY_REUSE_VECTOR",
                              transaction_id=tx_record.transaction_id)
                return True
            
            deterministic_id = self._vector_store._generate_deterministic_id(text)
            
            vector_ids = self._vector_store.add([text], [metadata], ids=[deterministic_id])
            
            if vector_ids:
                sqlite_id = metadata.get(MemoryTags.SQLITE_ID)
                if sqlite_id:
                    self._sqlite_store.update_vector_status(
                        sqlite_id, vector_ids[0], is_vectorized=1
                    )
                
                self._update_transaction_state(
                    tx_record.transaction_id,
                    TransactionState.COMMITTED
                )
                self._log.info("TRANSACTION_RECOVERY_SUCCESS",
                              transaction_id=tx_record.transaction_id)
                return True
            else:
                self._update_transaction_state(
                    tx_record.transaction_id,
                    TransactionState.FAILED,
                    "Vector store returned empty result"
                )
                return False
                
        except Exception as e:
            self._update_transaction_state(
                tx_record.transaction_id,
                TransactionState.FAILED,
                f"Recovery failed: {str(e)}"
            )
            self._log.error("TRANSACTION_RECOVERY_EXCEPTION",
                           transaction_id=tx_record.transaction_id,
                           error=str(e))
            return False
    
    def _recover_add_memory_batch_transaction(self, tx_record: TransactionRecord) -> Optional[int]:
        """
        恢复 add_memory_batch 类型的批量事务
        
        恢复策略：
        1. 检查 prepare_result 中是否有待向量化的数据
        2. 尝试批量写入 ChromaDB
        3. 成功后更新 SQLite 状态为 is_vectorized=1
        4. 失败则标记事务失败
        
        Args:
            tx_record: 事务记录
        
        Returns:
            恢复成功的记录数，失败返回 None
        """
        data = tx_record.data
        prepare_result = data.get("prepare_result", {})
        
        texts_to_vectorize = prepare_result.get("texts_to_vectorize", [])
        metadatas_to_vectorize = prepare_result.get("metadatas_to_vectorize", [])
        vector_ids = prepare_result.get("vector_ids", [])
        sqlite_ids = prepare_result.get("sqlite_ids", [])
        
        if not texts_to_vectorize:
            self._update_transaction_state(
                tx_record.transaction_id,
                TransactionState.COMMITTED
            )
            self._log.info("TRANSACTION_RECOVERY_EMPTY_BATCH",
                          transaction_id=tx_record.transaction_id)
            return 0
        
        try:
            result_ids = self._vector_store.add(
                texts_to_vectorize,
                metadatas_to_vectorize,
                ids=vector_ids
            )
            
            if result_ids and len(result_ids) > 0:
                if self._sqlite_store and sqlite_ids:
                    for sid, vid in zip(sqlite_ids, result_ids):
                        if sid and vid:
                            self._sqlite_store.update_vector_status(sid, vid, is_vectorized=1)
                
                self._update_transaction_state(
                    tx_record.transaction_id,
                    TransactionState.COMMITTED
                )
                
                self._log.info("TRANSACTION_BATCH_RECOVERY_SUCCESS",
                              transaction_id=tx_record.transaction_id,
                              count=len(result_ids))
                return len(result_ids)
            else:
                self._update_transaction_state(
                    tx_record.transaction_id,
                    TransactionState.FAILED,
                    "Vector store returned empty result"
                )
                return None
                
        except Exception as e:
            self._update_transaction_state(
                tx_record.transaction_id,
                TransactionState.FAILED,
                f"Batch recovery failed: {str(e)}"
            )
            self._log.error("TRANSACTION_BATCH_RECOVERY_EXCEPTION",
                           transaction_id=tx_record.transaction_id,
                           error=str(e))
            return None
    
    def get_transaction_status(self, transaction_id: str) -> Optional[TransactionRecord]:
        """获取事务状态"""
        return self._transactions.get(transaction_id)
    
    def cleanup_completed_transactions(self, max_age_hours: int = 24):
        """
        清理已完成的事务记录
        
        同时清理内存和数据库中的记录
        
        Args:
            max_age_hours: 最大保留时间（小时）
        """
        cutoff = datetime.now()
        to_remove = []
        
        for tx_id, tx_record in self._transactions.items():
            if tx_record.state in (TransactionState.COMMITTED, 
                                   TransactionState.ROLLED_BACK,
                                   TransactionState.FAILED):
                age_hours = (cutoff - tx_record.created_time).total_seconds() / 3600
                if age_hours > max_age_hours:
                    to_remove.append(tx_id)
        
        for tx_id in to_remove:
            del self._transactions[tx_id]
            self._delete_transaction(tx_id)
        
        if to_remove:
            self._log.debug("TRANSACTION_CLEANUP", count=len(to_remove))
        
        self._cleanup_db_transactions(max_age_hours)
    
    def _cleanup_db_transactions(self, max_age_hours: int):
        """
        清理数据库中已完成的事务记录
        
        Args:
            max_age_hours: 最大保留时间（小时）
        """
        if not self._sqlite_store:
            return
        
        try:
            cutoff_time = datetime.now()
            cutoff_str = cutoff_time.isoformat()
            
            with self._sqlite_store._get_write_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM transactions
                    WHERE state IN ('committed', 'rolled_back', 'failed')
                    AND updated_time < ?
                """, (cutoff_str,))
                deleted = cursor.rowcount
                conn.commit()
                
                if deleted > 0:
                    self._log.debug("DB_TRANSACTION_CLEANUP", count=deleted)
        except Exception as e:
            self._log.error("DB_TRANSACTION_CLEANUP_FAILED", error=str(e))


_transaction_coordinator: Optional[TransactionCoordinator] = None


def get_transaction_coordinator() -> TransactionCoordinator:
    """获取全局事务协调器实例"""
    global _transaction_coordinator
    if _transaction_coordinator is None:
        _transaction_coordinator = TransactionCoordinator()
    return _transaction_coordinator
