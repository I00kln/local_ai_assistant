import json
import threading
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from logger import get_logger, get_trace_id
from memory_tags import MemoryTags


class TransactionState(Enum):
    """事务状态"""
    PENDING = "pending"
    PREPARING = "preparing"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class TransactionPhase(Enum):
    """事务阶段 - 用于精确恢复"""
    INIT = "init"
    PREPARING = "preparing"
    PREPARED = "prepared"
    COMMITTING = "committing"
    COMMITTED = "committed"
    ROLLING_BACK = "rolling_back"
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
    phase: TransactionPhase = TransactionPhase.INIT
    error_message: Optional[str] = None
    sqlite_id: Optional[str] = None
    vector_id: Optional[str] = None
    affected_ids: List[str] = None
    
    def __post_init__(self):
        if self.affected_ids is None:
            self.affected_ids = []


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
            phase TEXT NOT NULL DEFAULT 'init',
            created_time TEXT NOT NULL,
            updated_time TEXT NOT NULL,
            data TEXT,
            error_message TEXT,
            sqlite_id TEXT,
            vector_id TEXT,
            affected_ids TEXT
        )
    """
    
    TRANSACTION_INDEX_SQL = """
        CREATE INDEX IF NOT EXISTS idx_transactions_state 
        ON transactions(state, updated_time)
    """
    
    TRANSACTION_PHASE_MIGRATION_SQL = """
        ALTER TABLE transactions ADD COLUMN phase TEXT NOT NULL DEFAULT 'init'
    """
    
    TRANSACTION_VECTOR_ID_MIGRATION_SQL = """
        ALTER TABLE transactions ADD COLUMN vector_id TEXT
    """
    
    TRANSACTION_SQLITE_ID_MIGRATION_SQL = """
        ALTER TABLE transactions ADD COLUMN sqlite_id TEXT
    """
    
    TRANSACTION_AFFECTED_IDS_MIGRATION_SQL = """
        ALTER TABLE transactions ADD COLUMN affected_ids TEXT
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
        self._scheduler = None
        self._migration_active = False
        self._migration_type: Optional[str] = None
        self._tx_table_initialized = False
    
    def set_stores(self, sqlite_store, vector_store, scheduler=None):
        """设置存储实例并初始化事务表"""
        self._sqlite_store = sqlite_store
        self._vector_store = vector_store
        self._scheduler = scheduler
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
                
                try:
                    cursor.execute(self.TRANSACTION_PHASE_MIGRATION_SQL)
                    self._log.debug("TRANSACTION_TABLE_PHASE_MIGRATED")
                except Exception:
                    pass
                
                try:
                    cursor.execute(self.TRANSACTION_VECTOR_ID_MIGRATION_SQL)
                    self._log.debug("TRANSACTION_TABLE_VECTOR_ID_MIGRATED")
                except Exception:
                    pass
                
                try:
                    cursor.execute(self.TRANSACTION_SQLITE_ID_MIGRATION_SQL)
                    self._log.debug("TRANSACTION_TABLE_SQLITE_ID_MIGRATED")
                except Exception:
                    pass
                
                try:
                    cursor.execute(self.TRANSACTION_AFFECTED_IDS_MIGRATION_SQL)
                    self._log.debug("TRANSACTION_TABLE_AFFECTED_IDS_MIGRATED")
                except Exception:
                    pass
                
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
                    (transaction_id, operation_type, state, phase, created_time, 
                     updated_time, data, error_message, sqlite_id, vector_id, affected_ids)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    tx_record.transaction_id,
                    tx_record.operation_type,
                    tx_record.state.value,
                    tx_record.phase.value,
                    tx_record.created_time.isoformat(),
                    tx_record.updated_time.isoformat(),
                    json.dumps(tx_record.data, ensure_ascii=False),
                    tx_record.error_message,
                    tx_record.sqlite_id,
                    tx_record.vector_id,
                    json.dumps(tx_record.affected_ids, ensure_ascii=False)
                ))
                conn.commit()
                return True
        except Exception as e:
            self._log.error("TRANSACTION_PERSIST_FAILED", 
                           transaction_id=tx_record.transaction_id,
                           error=str(e))
            return False
    
    def _update_transaction_state(self, tx_id: str, state: TransactionState, 
                                   phase: TransactionPhase = None,
                                   error_message: str = None) -> bool:
        """
        更新事务状态
        
        Args:
            tx_id: 事务ID
            state: 新状态
            phase: 新阶段（可选）
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
                    if phase:
                        cursor.execute("""
                            UPDATE transactions 
                            SET state = ?, phase = ?, updated_time = ?, error_message = ?
                            WHERE transaction_id = ?
                        """, (state.value, phase.value, datetime.now().isoformat(), error_message, tx_id))
                    else:
                        cursor.execute("""
                            UPDATE transactions 
                            SET state = ?, updated_time = ?, error_message = ?
                            WHERE transaction_id = ?
                        """, (state.value, datetime.now().isoformat(), error_message, tx_id))
                else:
                    if phase:
                        cursor.execute("""
                            UPDATE transactions 
                            SET state = ?, phase = ?, updated_time = ?
                            WHERE transaction_id = ?
                        """, (state.value, phase.value, datetime.now().isoformat(), tx_id))
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
                SELECT transaction_id, operation_type, state, phase, created_time,
                       updated_time, data, error_message
                FROM transactions
                WHERE state IN ('pending', 'preparing')
                ORDER BY created_time ASC
            """)
            
            records = []
            for row in cursor.fetchall():
                try:
                    phase_str = row[3] if len(row) > 3 else 'init'
                    try:
                        phase = TransactionPhase(phase_str)
                    except ValueError:
                        phase = TransactionPhase.INIT
                    
                    records.append(TransactionRecord(
                        transaction_id=row[0],
                        operation_type=row[1],
                        state=TransactionState(row[2]),
                        phase=phase,
                        created_time=datetime.fromisoformat(row[4]),
                        updated_time=datetime.fromisoformat(row[5]),
                        data=json.loads(row[6]) if row[6] else {},
                        error_message=row[7] if len(row) > 7 else None
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
        
        修复：真正获取 BackgroundScheduler 的写锁，确保读写隔离
        
        Args:
            migration_type: 迁移类型 (flush_buffer, l2_to_l3, l3_to_l2, merge, etc.)
        """
        self._migration_active = True
        self._migration_type = migration_type
        
        if self._scheduler:
            acquired = self._scheduler.acquire_migration_lock(migration_type, timeout=10.0)
            if not acquired:
                self._log.warning("MIGRATION_LOCK_ACQUIRE_FAILED_FALLBACK", migration_type=migration_type)
        
        self._log.debug("MIGRATION_STARTED", migration_type=migration_type)
    
    def end_migration(self):
        """
        结束迁移操作
        
        修复：真正释放 BackgroundScheduler 的写锁
        """
        migration_type = self._migration_type
        self._migration_active = False
        self._migration_type = None
        
        if self._scheduler:
            self._scheduler.release_migration_lock()
        
        self._log.debug("MIGRATION_ENDED", migration_type=migration_type)
    
    def is_migration_active(self) -> bool:
        """检查是否有迁移操作正在进行"""
        if self._scheduler:
            return self._scheduler.is_migration_active()
        return self._migration_active
    
    def get_migration_type(self) -> Optional[str]:
        """获取当前迁移类型"""
        if self._scheduler:
            task = self._scheduler.get_current_task()
            return task.value if task else None
        return self._migration_type
    
    def wait_for_migration(self, timeout: float = 5.0) -> bool:
        """
        等待迁移完成
        
        委托给 BackgroundScheduler 的读锁机制
        
        Args:
            timeout: 超时时间（秒）
        
        Returns:
            是否成功获取锁（迁移是否完成）
        """
        if self._scheduler:
            with self._scheduler.read_lock(timeout=timeout) as lock_context:
                return lock_context.is_locked()
        return True
    
    def release_migration_wait(self):
        """释放迁移等待锁"""
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
        1. 创建事务记录，状态为 PENDING，phase 为 INIT，持久化到 SQLite
        2. 执行准备阶段，状态更新为 PREPARING，phase 为 PREPARING，持久化
        3. 准备完成，phase 更新为 PREPARED，持久化
        4. 执行提交阶段（ChromaDB 写入），phase 为 COMMITTING
        5. ChromaDB 写入成功后，phase 更新为 COMMITTED，持久化
        6. 最后状态更新为 COMMITTED
        
        恢复策略：
        - phase=INIT/PREPARING: 从头开始
        - phase=PREPARED: 重新执行 commit_fn
        - phase=COMMITTING: 检查 ChromaDB 是否已写入，决定重试或标记完成
        
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
        trace_id = get_trace_id()
        tx_id = transaction_id or f"{operation_type}_{int(time.time() * 1000)}"
        
        tx_record = TransactionRecord(
            transaction_id=tx_id,
            operation_type=operation_type,
            state=TransactionState.PENDING,
            phase=TransactionPhase.INIT,
            created_time=datetime.now(),
            updated_time=datetime.now(),
            data={"phase": "init", "trace_id": trace_id, **data}
        )
        
        self._transactions[tx_id] = tx_record
        
        self._persist_transaction(tx_record)
        
        self._log.debug("TRANSACTION_STARTED",
                       trace_id=trace_id,
                       transaction_id=tx_id,
                       operation_type=operation_type)
        
        try:
            tx_record.state = TransactionState.PREPARING
            tx_record.phase = TransactionPhase.PREPARING
            tx_record.updated_time = datetime.now()
            tx_record.data["phase"] = "preparing"
            self._update_transaction_state(tx_id, TransactionState.PREPARING, TransactionPhase.PREPARING)
            
            prepare_result = prepare_fn(data)
            tx_record.data["prepare_result"] = prepare_result
            
            if isinstance(prepare_result, dict):
                tx_record.sqlite_id = prepare_result.get("sqlite_id") or prepare_result.get("id")
            
            tx_record.phase = TransactionPhase.PREPARED
            tx_record.data["phase"] = "prepared"
            self._update_transaction_state(tx_id, TransactionState.PREPARING, TransactionPhase.PREPARED)
            
            self._log.debug("TRANSACTION_PREPARED",
                           trace_id=trace_id,
                           transaction_id=tx_id)
            
            tx_record.phase = TransactionPhase.COMMITTING
            tx_record.data["phase"] = "committing"
            self._update_transaction_state(tx_id, TransactionState.PREPARING, TransactionPhase.COMMITTING)
            
            commit_result = commit_fn(data)
            tx_record.data["commit_result"] = commit_result
            
            if isinstance(commit_result, dict):
                tx_record.vector_id = commit_result.get("vector_id") or commit_result.get("id")
            elif isinstance(commit_result, list) and commit_result:
                first_result = commit_result[0]
                if isinstance(first_result, str):
                    tx_record.vector_id = first_result
                elif isinstance(first_result, dict):
                    tx_record.vector_id = first_result.get("id")
            
            if tx_record.sqlite_id or tx_record.vector_id:
                self._persist_transaction(tx_record)
            
            tx_record.phase = TransactionPhase.COMMITTED
            tx_record.data["phase"] = "committed"
            self._update_transaction_state(tx_id, TransactionState.PREPARING, TransactionPhase.COMMITTED)
            
            tx_record.state = TransactionState.COMMITTED
            tx_record.updated_time = datetime.now()
            self._update_transaction_state(tx_id, TransactionState.COMMITTED, TransactionPhase.COMMITTED)
            
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
            tx_record.phase = TransactionPhase.FAILED
            tx_record.error_message = str(e)
            tx_record.updated_time = datetime.now()
            tx_record.data["phase"] = "failed"
            self._update_transaction_state(tx_id, TransactionState.FAILED, TransactionPhase.FAILED, str(e))
            
            self._log.error("TRANSACTION_FAILED",
                           transaction_id=tx_id,
                           operation=operation_type,
                           error=str(e))
            
            if rollback_fn:
                try:
                    tx_record.phase = TransactionPhase.ROLLING_BACK
                    tx_record.data["phase"] = "rolling_back"
                    tx_record.data["sqlite_id"] = tx_record.sqlite_id
                    tx_record.data["vector_id"] = tx_record.vector_id
                    self._update_transaction_state(tx_id, TransactionState.FAILED, TransactionPhase.ROLLING_BACK)
                    
                    rollback_fn(tx_record.data)
                    
                    tx_record.state = TransactionState.ROLLED_BACK
                    tx_record.phase = TransactionPhase.ROLLED_BACK
                    self._update_transaction_state(tx_id, TransactionState.ROLLED_BACK, TransactionPhase.ROLLED_BACK)
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
        
        根据 phase 精确恢复：
        - phase=INIT/PREPARING: 从头开始执行
        - phase=PREPARED: 跳过 prepare，直接执行 commit
        - phase=COMMITTING: 检查 ChromaDB 是否已写入，决定重试或标记完成
        
        检查：
        1. SQLite 事务表中的 PREPARING 状态事务
        2. SQLite 中 is_vectorized=2 的记录（旧兼容）
        
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
                          state=tx_record.state.value,
                          phase=tx_record.phase.value)
            
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
                    TransactionPhase.FAILED,
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
        
        根据 phase 精确恢复：
        - phase=INIT/PREPARING: 从头开始
        - phase=PREPARED/COMMITTING: 检查 ChromaDB 是否已写入
        
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
                TransactionPhase.FAILED,
                "Missing text data"
            )
            return False
        
        try:
            if tx_record.phase in (TransactionPhase.COMMITTING, TransactionPhase.COMMITTED):
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
                        TransactionState.COMMITTED,
                        TransactionPhase.COMMITTED
                    )
                    self._log.info("TRANSACTION_RECOVERY_ALREADY_COMMITTED",
                                  transaction_id=tx_record.transaction_id)
                    return True
            
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
                    TransactionState.COMMITTED,
                    TransactionPhase.COMMITTED
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
                    TransactionState.COMMITTED,
                    TransactionPhase.COMMITTED
                )
                self._log.info("TRANSACTION_RECOVERY_SUCCESS",
                              transaction_id=tx_record.transaction_id)
                return True
            else:
                self._update_transaction_state(
                    tx_record.transaction_id,
                    TransactionState.FAILED,
                    TransactionPhase.FAILED,
                    "Vector store returned empty result"
                )
                return False
                
        except Exception as e:
            self._update_transaction_state(
                tx_record.transaction_id,
                TransactionState.FAILED,
                TransactionPhase.FAILED,
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
        1. 验证 prepare_result 数据完整性
        2. 检查是否有待向量化的数据
        3. 尝试批量写入 ChromaDB
        4. 成功后更新 SQLite 状态为 is_vectorized=1
        5. 失败则标记事务失败
        
        数据完整性验证：
        - texts_to_vectorize 必须存在且非空
        - metadatas_to_vectorize 数量必须与 texts_to_vectorize 一致
        - vector_ids 数量必须与 texts_to_vectorize 一致
        
        Args:
            tx_record: 事务记录
        
        Returns:
            恢复成功的记录数，失败返回 None
        """
        data = tx_record.data
        prepare_result = data.get("prepare_result", {})
        
        if not prepare_result:
            self._update_transaction_state(
                tx_record.transaction_id,
                TransactionState.FAILED,
                "Missing prepare_result in transaction data"
            )
            self._log.error("TRANSACTION_RECOVERY_MISSING_PREPARE_RESULT",
                           transaction_id=tx_record.transaction_id)
            return None
        
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
        
        if len(texts_to_vectorize) != len(metadatas_to_vectorize):
            self._update_transaction_state(
                tx_record.transaction_id,
                TransactionState.FAILED,
                f"Data mismatch: {len(texts_to_vectorize)} texts vs {len(metadatas_to_vectorize)} metadatas"
            )
            self._log.error("TRANSACTION_RECOVERY_DATA_MISMATCH",
                           transaction_id=tx_record.transaction_id,
                           texts_count=len(texts_to_vectorize),
                           metadatas_count=len(metadatas_to_vectorize))
            return None
        
        if len(texts_to_vectorize) != len(vector_ids):
            self._update_transaction_state(
                tx_record.transaction_id,
                TransactionState.FAILED,
                f"Data mismatch: {len(texts_to_vectorize)} texts vs {len(vector_ids)} vector_ids"
            )
            self._log.error("TRANSACTION_RECOVERY_VECTOR_ID_MISMATCH",
                           transaction_id=tx_record.transaction_id,
                           texts_count=len(texts_to_vectorize),
                           vector_ids_count=len(vector_ids))
            return None
        
        try:
            result_ids = self._vector_store.add(
                texts_to_vectorize,
                metadatas_to_vectorize,
                ids=vector_ids
            )
            
            if result_ids and len(result_ids) > 0:
                if self._sqlite_store and sqlite_ids:
                    matched_count = 0
                    for sid, vid in zip(sqlite_ids, result_ids):
                        if sid and vid:
                            self._sqlite_store.update_vector_status(sid, vid, is_vectorized=1)
                            matched_count += 1
                    
                    if matched_count < len(result_ids):
                        self._log.warning("TRANSACTION_RECOVERY_PARTIAL_MATCH",
                                         transaction_id=tx_record.transaction_id,
                                         matched=matched_count,
                                         total=len(result_ids))
                
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
    
    def cleanup_orphan_vectors(self) -> Dict[str, int]:
        """
        清理孤儿向量数据（双向同步）
        
        设计原则：
        - 对比 SQLite 和 ChromaDB 的 ID 全集
        - 清理向量库中存在但数据库中不存在的孤儿向量
        - 修复数据库中标记为已向量化但向量库中不存在的孤儿记录
        - 记录清理统计信息
        
        Returns:
            {
                "cleaned_vectors": int,  # 清理的孤儿向量数
                "fixed_records": int,    # 修复的孤儿记录数
                "total_vector": int, 
                "total_sqlite": int
            }
        """
        if not self._sqlite_store or not self._vector_store:
            return {"cleaned_vectors": 0, "fixed_records": 0, "error": "stores not available"}
        
        try:
            sqlite_vectorized = {}
            with self._sqlite_store._get_read_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, vector_id FROM memories WHERE is_vectorized = 1")
                for row in cursor.fetchall():
                    sqlite_vectorized[str(row[0])] = row[1]
            
            vector_ids = set()
            try:
                if self._vector_store.collection:
                    all_vectors = self._vector_store.collection.get()
                    vector_ids = set(all_vectors.get("ids", []))
            except Exception as e:
                self._log.error("ORPHAN_CLEANUP_VECTOR_FETCH_FAILED", error=str(e))
                return {"cleaned_vectors": 0, "fixed_records": 0, "error": str(e)}
            
            sqlite_ids = set(sqlite_vectorized.keys())
            
            orphan_vector_ids = vector_ids - sqlite_ids
            
            cleaned_vectors = 0
            if orphan_vector_ids:
                try:
                    self._vector_store.collection.delete(ids=list(orphan_vector_ids))
                    cleaned_vectors = len(orphan_vector_ids)
                    self._log.info("ORPHAN_VECTORS_CLEANED", 
                                  count=cleaned_vectors,
                                  total_vector=len(vector_ids),
                                  total_sqlite=len(sqlite_ids))
                except Exception as e:
                    self._log.error("ORPHAN_VECTORS_DELETE_FAILED", 
                                   count=len(orphan_vector_ids),
                                   error=str(e))
            
            fixed_records = 0
            for sqlite_id, vector_id in sqlite_vectorized.items():
                if vector_id and vector_id not in vector_ids:
                    try:
                        self._sqlite_store.update_vector_status(sqlite_id, "", is_vectorized=0)
                        fixed_records += 1
                        self._log.debug("ORPHAN_RECORD_FIXED",
                                       sqlite_id=sqlite_id,
                                       missing_vector_id=vector_id)
                    except Exception as e:
                        self._log.error("ORPHAN_RECORD_FIX_FAILED",
                                       sqlite_id=sqlite_id,
                                       error=str(e))
            
            if fixed_records > 0:
                self._log.info("ORPHAN_RECORDS_FIXED",
                              count=fixed_records,
                              message="SQLite 记录标记为已向量化但向量库中不存在，已重置为未向量化")
            
            return {
                "cleaned_vectors": cleaned_vectors,
                "fixed_records": fixed_records,
                "total_vector": len(vector_ids),
                "total_sqlite": len(sqlite_ids)
            }
            
        except Exception as e:
            self._log.error("ORPHAN_CLEANUP_FAILED", error=str(e))
            return {"cleaned_vectors": 0, "fixed_records": 0, "error": str(e)}
    
    def schedule_orphan_cleanup(self, interval_hours: int = 24):
        """
        调度定期孤儿清理任务
        
        Args:
            interval_hours: 清理间隔（小时）
        """
        try:
            from background_scheduler import get_background_scheduler, TaskType
            
            scheduler = get_background_scheduler()
            
            def cleanup_task():
                self.cleanup_orphan_vectors()
            
            scheduler.schedule_periodic(
                task_id="orphan_cleanup",
                interval_seconds=interval_hours * 3600,
                fn=cleanup_task,
                task_type=TaskType.MAINTENANCE
            )
            
            self._log.info("ORPHAN_CLEANUP_SCHEDULED", interval_hours=interval_hours)
            
        except Exception as e:
            self._log.error("ORPHAN_CLEANUP_SCHEDULE_FAILED", error=str(e))


_transaction_coordinator: Optional[TransactionCoordinator] = None


def get_transaction_coordinator() -> TransactionCoordinator:
    """获取全局事务协调器实例"""
    global _transaction_coordinator
    if _transaction_coordinator is None:
        _transaction_coordinator = TransactionCoordinator()
    return _transaction_coordinator
