import threading
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from logger import get_logger


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
    1. 准备阶段：在 SQLite 中记录事务状态
    2. 提交阶段：执行 ChromaDB 操作，成功后更新 SQLite 状态
    3. 恢复阶段：启动时检查未完成的事务并重试
    
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
    
    def set_stores(self, sqlite_store, vector_store):
        """设置存储实例"""
        self._sqlite_store = sqlite_store
        self._vector_store = vector_store
    
    def begin_migration(self):
        """
        开始迁移操作
        
        获取迁移锁，阻止检索操作访问正在迁移的数据
        """
        self._migration_lock.acquire()
        self._migration_active = True
        self._log.debug("MIGRATION_STARTED")
    
    def end_migration(self):
        """
        结束迁移操作
        
        释放迁移锁
        """
        self._migration_active = False
        self._migration_lock.release()
        self._log.debug("MIGRATION_ENDED")
    
    def is_migration_active(self) -> bool:
        """检查是否有迁移操作正在进行"""
        return self._migration_active
    
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
        执行事务
        
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
        
        try:
            tx_record.state = TransactionState.PREPARING
            tx_record.updated_time = datetime.now()
            
            prepare_result = prepare_fn(data)
            tx_record.data["prepare_result"] = prepare_result
            
            commit_result = commit_fn(data)
            tx_record.data["commit_result"] = commit_result
            
            tx_record.state = TransactionState.COMMITTED
            tx_record.updated_time = datetime.now()
            
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
            
            self._log.error("TRANSACTION_FAILED",
                           transaction_id=tx_id,
                           operation=operation_type,
                           error=str(e))
            
            if rollback_fn:
                try:
                    rollback_fn(tx_record.data)
                    tx_record.state = TransactionState.ROLLED_BACK
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
        
        检查 SQLite 中 is_vectorized=2 的记录并重新处理
        
        Returns:
            恢复的记录数
        """
        if not self._sqlite_store:
            return 0
        
        recovered = 0
        
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
                    vector_ids = self._vector_store.add(
                        [record.text],
                        [record.metadata or {}]
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
            
            if recovered > 0:
                self._log.info("TRANSACTION_RECOVERY_COMPLETE", count=recovered)
            
        except Exception as e:
            self._log.error("TRANSACTION_RECOVERY_ERROR", error=str(e))
        
        return recovered
    
    def get_transaction_status(self, transaction_id: str) -> Optional[TransactionRecord]:
        """获取事务状态"""
        return self._transactions.get(transaction_id)
    
    def cleanup_completed_transactions(self, max_age_hours: int = 24):
        """
        清理已完成的事务记录
        
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
        
        if to_remove:
            self._log.debug("TRANSACTION_CLEANUP", count=len(to_remove))


_transaction_coordinator: Optional[TransactionCoordinator] = None


def get_transaction_coordinator() -> TransactionCoordinator:
    """获取全局事务协调器实例"""
    global _transaction_coordinator
    if _transaction_coordinator is None:
        _transaction_coordinator = TransactionCoordinator()
    return _transaction_coordinator
