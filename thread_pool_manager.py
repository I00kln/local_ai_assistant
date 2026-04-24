# thread_pool_manager.py
# 统一线程池管理器 - 解决线程爆炸问题

from concurrent.futures import ThreadPoolExecutor, Future
from enum import Enum, auto
import threading
from typing import Callable, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import queue

try:
    from logger import get_logger
except ImportError:
    def get_logger():
        class DummyLogger:
            def info(self, **kwargs): pass
            def warning(self, **kwargs): pass
            def error(self, **kwargs): pass
        return DummyLogger()


class TaskType(Enum):
    """任务类型枚举"""
    IO_BOUND = auto()
    CPU_BOUND = auto()
    UI = auto()


class RejectionPolicy(Enum):
    """拒绝策略"""
    REJECT = "reject"
    DROP_OLDEST = "drop_oldest"
    CALLER_RUNS = "caller_runs"
    BLOCK = "block"


@dataclass
class PoolStats:
    """线程池统计信息"""
    submitted: int = 0
    completed: int = 0
    failed: int = 0
    rejected: int = 0
    active: int = 0
    queue_size: int = 0
    queue_high_watermark: int = 0


class ThreadPoolManager:
    """
    统一线程池管理器
    
    设计原则：
    - SRP: 仅负责线程池生命周期和任务分发
    - DIP: 依赖抽象的 TaskType，不依赖具体实现
    - OCP: 可扩展新的任务类型
    
    特性：
    - 按任务类型隔离线程池
    - 统一监控和统计
    - 优雅关闭支持
    - 与 LifecycleManager 集成
    """
    
    _instance: Optional['ThreadPoolManager'] = None
    _lock = threading.Lock()
    
    DEFAULT_CONFIG: Dict[TaskType, Dict[str, Any]] = {
        TaskType.IO_BOUND: {
            "max_workers": 8,
            "thread_name_prefix": "io_worker",
            "max_queue_size": 100,
            "rejection_policy": RejectionPolicy.CALLER_RUNS,
            "high_watermark_ratio": 0.8
        },
        TaskType.CPU_BOUND: {
            "max_workers": 4,
            "thread_name_prefix": "cpu_worker",
            "max_queue_size": 50,
            "rejection_policy": RejectionPolicy.REJECT,
            "high_watermark_ratio": 0.8
        },
        TaskType.UI: {
            "max_workers": 2,
            "thread_name_prefix": "ui_worker",
            "max_queue_size": 20,
            "rejection_policy": RejectionPolicy.DROP_OLDEST,
            "high_watermark_ratio": 0.8
        },
    }
    
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
        self._pools: Dict[TaskType, ThreadPoolExecutor] = {}
        self._custom_config: Dict[TaskType, Dict[str, Any]] = {}
        self._stats: Dict[TaskType, PoolStats] = {
            t: PoolStats() for t in TaskType
        }
        self._stats_lock = threading.Lock()
        self._shutdown_requested = False
        
        EXECUTOR_PARAMS = {"max_workers", "thread_name_prefix"}
        
        for task_type, config in self.DEFAULT_CONFIG.items():
            executor_config = {k: v for k, v in config.items() if k in EXECUTOR_PARAMS}
            self._pools[task_type] = ThreadPoolExecutor(**executor_config)
            self._custom_config[task_type] = {k: v for k, v in config.items() if k not in EXECUTOR_PARAMS}
        
        self._register_lifecycle()
        
        self._log.info(
            "THREAD_POOL_INITIALIZED",
            io_workers=self.DEFAULT_CONFIG[TaskType.IO_BOUND]["max_workers"],
            cpu_workers=self.DEFAULT_CONFIG[TaskType.CPU_BOUND]["max_workers"],
            ui_workers=self.DEFAULT_CONFIG[TaskType.UI]["max_workers"]
        )
    
    def _register_lifecycle(self):
        """注册到生命周期管理器"""
        try:
            from lifecycle_manager import get_lifecycle_manager, ServicePriority
            lifecycle = get_lifecycle_manager()
            lifecycle.register(
                name="thread_pool_manager",
                cleanup_fn=self.shutdown,
                priority=ServicePriority.CRITICAL,
                timeout=10.0,
                is_running=lambda: not self._shutdown_requested,
                stop_fn=lambda: setattr(self, '_shutdown_requested', True)
            )
        except Exception:
            pass
    
    def submit(
        self, 
        task_type: TaskType, 
        fn: Callable, 
        *args, 
        **kwargs
    ) -> Optional[Future]:
        """
        提交任务到对应线程池
        
        Args:
            task_type: 任务类型
            fn: 执行函数
            *args: 位置参数
            **kwargs: 关键字参数
        
        Returns:
            Future 对象，失败时返回 None
        
        Raises:
            RuntimeError: 线程池已关闭
        """
        if self._shutdown_requested:
            self._log.warning("THREAD_POOL_SHUTDOWN", message="拒绝新任务提交")
            return None
        
        pool = self._pools.get(task_type)
        if pool is None:
            self._log.error("UNKNOWN_TASK_TYPE", task_type=str(task_type))
            return None
        
        config = self._custom_config.get(task_type, {})
        queue_size = self.get_queue_size(task_type)
        max_queue_size = config.get("max_queue_size", 100)
        rejection_policy = config.get("rejection_policy", RejectionPolicy.REJECT)
        high_watermark_ratio = config.get("high_watermark_ratio", 0.8)
        
        with self._stats_lock:
            self._stats[task_type].queue_size = queue_size
            if queue_size > self._stats[task_type].queue_high_watermark:
                self._stats[task_type].queue_high_watermark = queue_size
        
        if queue_size >= max_queue_size:
            if rejection_policy == RejectionPolicy.REJECT:
                with self._stats_lock:
                    self._stats[task_type].rejected += 1
                self._log.warning(
                    "TASK_REJECTED_QUEUE_FULL",
                    task_type=task_type.name,
                    queue_size=queue_size,
                    max_queue_size=max_queue_size
                )
                return None
            
            elif rejection_policy == RejectionPolicy.DROP_OLDEST:
                self._drop_oldest_task(pool)
                self._log.info(
                    "TASK_DROPPED_OLDEST",
                    task_type=task_type.name,
                    queue_size=queue_size
                )
            
            elif rejection_policy == RejectionPolicy.CALLER_RUNS:
                self._log.info(
                    "TASK_CALLER_RUNS",
                    task_type=task_type.name,
                    queue_size=queue_size
                )
                try:
                    fn(*args, **kwargs)
                except Exception as e:
                    with self._stats_lock:
                        self._stats[task_type].failed += 1
                    self._log.error(
                        "CALLER_RUNS_FAILED",
                        task_type=task_type.name,
                        error=str(e)
                    )
                return None
            
            elif rejection_policy == RejectionPolicy.BLOCK:
                pass
        
        if queue_size >= max_queue_size * high_watermark_ratio:
            self._log.warning(
                "QUEUE_HIGH_WATERMARK",
                task_type=task_type.name,
                queue_size=queue_size,
                max_queue_size=max_queue_size,
                ratio=high_watermark_ratio
            )
        
        with self._stats_lock:
            self._stats[task_type].submitted += 1
        
        try:
            future = pool.submit(
                self._wrap_task(task_type, fn), 
                *args, 
                **kwargs
            )
            return future
        except RuntimeError as e:
            if "cannot schedule new futures" in str(e):
                self._log.warning("THREAD_POOL_CLOSED", task_type=task_type.name)
                return None
            raise
    
    def _drop_oldest_task(self, pool: ThreadPoolExecutor) -> bool:
        """
        尝试丢弃最旧的任务
        
        Args:
            pool: 线程池实例
        
        Returns:
            是否成功丢弃
        """
        try:
            if hasattr(pool, '_work_queue'):
                work_queue = pool._work_queue
                if hasattr(work_queue, 'get_nowait'):
                    try:
                        work_queue.get_nowait()
                        return True
                    except queue.Empty:
                        return False
        except Exception:
            pass
        return False
    
    def _wrap_task(self, task_type: TaskType, fn: Callable) -> Callable:
        """
        包装任务以统计执行情况
        
        Args:
            task_type: 任务类型
            fn: 原始函数
        
        Returns:
            包装后的函数
        """
        fn_name = getattr(fn, '__name__', str(fn))
        
        def wrapper(*args, **kwargs):
            with self._stats_lock:
                self._stats[task_type].active += 1
            
            try:
                result = fn(*args, **kwargs)
                with self._stats_lock:
                    self._stats[task_type].completed += 1
                return result
            except Exception as e:
                with self._stats_lock:
                    self._stats[task_type].failed += 1
                self._log.error(
                    "TASK_EXECUTION_FAILED",
                    task_type=task_type.name,
                    function=fn_name,
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise
            finally:
                with self._stats_lock:
                    self._stats[task_type].active -= 1
                
                try:
                    from sqlite_store import get_sqlite_store
                    sqlite_store = get_sqlite_store()
                    if sqlite_store:
                        sqlite_store.close_thread_connection()
                except Exception:
                    pass
        
        return wrapper
    
    def submit_with_callback(
        self,
        task_type: TaskType,
        fn: Callable,
        callback: Callable[[Future], None],
        *args,
        **kwargs
    ) -> Optional[Future]:
        """
        提交任务并设置回调
        
        Args:
            task_type: 任务类型
            fn: 执行函数
            callback: 完成回调
            *args: 位置参数
            **kwargs: 关键字参数
        
        Returns:
            Future 对象
        """
        future = self.submit(task_type, fn, *args, **kwargs)
        if future and callback:
            future.add_done_callback(callback)
        return future
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取线程池统计信息
        
        Returns:
            包含各线程池状态的字典
        """
        with self._stats_lock:
            result = {}
            for task_type in TaskType:
                pool = self._pools.get(task_type)
                stats = self._stats[task_type]
                config = self.DEFAULT_CONFIG[task_type]
                
                result[task_type.name] = {
                    "submitted": stats.submitted,
                    "completed": stats.completed,
                    "failed": stats.failed,
                    "rejected": stats.rejected,
                    "active": stats.active,
                    "queue_size": stats.queue_size,
                    "queue_high_watermark": stats.queue_high_watermark,
                    "max_workers": config["max_workers"],
                    "max_queue_size": config.get("max_queue_size", 100),
                    "rejection_policy": config.get("rejection_policy", RejectionPolicy.REJECT).value,
                }
            
            result["total_submitted"] = sum(s.submitted for s in self._stats.values())
            result["total_completed"] = sum(s.completed for s in self._stats.values())
            result["total_failed"] = sum(s.failed for s in self._stats.values())
            result["total_rejected"] = sum(s.rejected for s in self._stats.values())
            result["shutdown_requested"] = self._shutdown_requested
            
            return result
    
    def get_queue_size(self, task_type: TaskType) -> int:
        """获取指定类型任务队列大小"""
        pool = self._pools.get(task_type)
        if pool and hasattr(pool, '_work_queue'):
            try:
                return pool._work_queue.qsize()
            except Exception:
                pass
        return 0
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        等待所有任务完成
        
        Args:
            timeout: 超时时间（秒）
        
        Returns:
            是否所有任务都已完成
        """
        import time
        start_time = time.time()
        
        while True:
            with self._stats_lock:
                total_active = sum(s.active for s in self._stats.values())
            
            if total_active == 0:
                return True
            
            if timeout and (time.time() - start_time) > timeout:
                return False
            
            time.sleep(0.1)
    
    def shutdown(self, wait: bool = True, timeout: Optional[float] = None):
        """
        优雅关闭所有线程池
        
        Args:
            wait: 是否等待任务完成
            timeout: 超时时间（秒）
        """
        self._shutdown_requested = True
        
        self._log.info(
            "THREAD_POOL_SHUTTING_DOWN",
            wait=wait,
            timeout=timeout
        )
        
        if wait and timeout:
            self.wait_for_completion(timeout)
        
        for task_type, pool in self._pools.items():
            try:
                pool.shutdown(wait=wait)
            except Exception as e:
                self._log.error(
                    "POOL_SHUTDOWN_ERROR",
                    task_type=task_type.name,
                    error=str(e)
                )
        
        stats = self.get_stats()
        self._log.info(
            "THREAD_POOL_SHUTDOWN_COMPLETE",
            total_submitted=stats["total_submitted"],
            total_completed=stats["total_completed"],
            total_failed=stats["total_failed"]
        )
        
        ThreadPoolManager._instance = None
    
    @property
    def is_shutdown(self) -> bool:
        """检查是否已请求关闭"""
        return self._shutdown_requested


def get_thread_pool() -> ThreadPoolManager:
    """获取线程池管理器单例"""
    return ThreadPoolManager()


def submit_io_task(fn: Callable, *args, **kwargs) -> Optional[Future]:
    """
    提交 IO 密集型任务
    
    适用于：数据库操作、网络请求、文件 IO
    
    Args:
        fn: 执行函数
        *args: 位置参数
        **kwargs: 关键字参数
    
    Returns:
        Future 对象
    """
    return get_thread_pool().submit(TaskType.IO_BOUND, fn, *args, **kwargs)


def submit_cpu_task(fn: Callable, *args, **kwargs) -> Optional[Future]:
    """
    提交 CPU 密集型任务
    
    适用于：Embedding 计算、向量相似度、压缩算法
    
    Args:
        fn: 执行函数
        *args: 位置参数
        **kwargs: 关键字参数
    
    Returns:
        Future 对象
    """
    return get_thread_pool().submit(TaskType.CPU_BOUND, fn, *args, **kwargs)


def submit_ui_task(fn: Callable, *args, **kwargs) -> Optional[Future]:
    """
    提交 UI 相关任务
    
    适用于：UI 状态更新、回调处理
    
    Args:
        fn: 执行函数
        *args: 位置参数
        **kwargs: 关键字参数
    
    Returns:
        Future 对象
    """
    return get_thread_pool().submit(TaskType.UI, fn, *args, **kwargs)
