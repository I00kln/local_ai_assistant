#!/usr/bin/env python
# background_scheduler.py
"""
后台任务调度器

解决并发竞态问题：
- 统一协调所有记忆状态变更操作（迁移、压缩、合并、去重、衰减）
- 确保所有操作通过迁移锁保护
- 检索操作等待迁移完成

使用方式：
    scheduler = get_background_scheduler()
    scheduler.start()
    
    # 提交后台任务
    scheduler.submit_task(TaskType.MIGRATION_L2_TO_L3, callback)
    
    # 检索前等待迁移完成
    with scheduler.read_lock():
        results = search(...)
"""

import threading
import time
import queue
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from logger import get_logger


class TaskType(Enum):
    """后台任务类型"""
    MIGRATION_L2_TO_L3 = "migration_l2_to_l3"
    MIGRATION_L3_TO_L2 = "migration_l3_to_l2"
    PRELOAD_L3_TO_L2 = "preload_l3_to_l2"
    COMPRESSION = "compression"
    MERGE = "merge"
    DEDUP = "dedup"
    DECAY = "decay"
    FORGET = "forget"
    UPGRADE_SQLITE_ONLY = "upgrade_sqlite_only"
    FLUSH_BUFFER = "flush_buffer"
    RETRY_VECTORIZATION = "retry_vectorization"


class TaskPriority(Enum):
    """任务优先级"""
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class TaskResult:
    """任务执行结果"""
    task_type: TaskType
    success: bool
    affected_count: int = 0
    error: Optional[str] = None
    duration_ms: float = 0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(order=True)
class PrioritizedTask:
    """优先级任务"""
    priority: int
    task_type: TaskType = field(compare=False)
    callback: Callable = field(compare=False)
    args: tuple = field(compare=False, default=())
    kwargs: dict = field(compare=False, default_factory=dict)
    submitted_at: float = field(compare=False, default_factory=time.time)


class BackgroundTaskScheduler:
    """
    后台任务调度器
    
    功能：
    1. 统一协调所有记忆状态变更操作
    2. 确保操作通过迁移锁保护
    3. 支持任务优先级
    4. 记录任务执行日志
    
    并发安全：
    - 使用迁移锁保护所有写操作
    - 检索操作使用读锁等待迁移完成
    - 任务队列线程安全
    """
    
    _instance: Optional['BackgroundTaskScheduler'] = None
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
        
        self._task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        
        self._migration_lock = threading.RLock()
        self._migration_active = False
        self._current_task: Optional[TaskType] = None
        self._migration_start_time: Optional[float] = None
        
        self._task_stats: Dict[TaskType, Dict[str, int]] = {
            t: {"total": 0, "success": 0, "failed": 0, "affected": 0}
            for t in TaskType
        }
        
        self._last_task_times: Dict[TaskType, float] = {}
        
        self._readers_count = 0
        self._readers_lock = threading.Lock()
        self._read_write_lock = threading.Lock()
        self._write_waiting = threading.Condition(self._read_write_lock)
        self._write_in_progress = False
    
    def start(self):
        """启动调度器"""
        if self._running:
            return
        
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="BackgroundTaskScheduler",
            daemon=True
        )
        self._worker_thread.start()
        self._log.info("BACKGROUND_SCHEDULER_STARTED")
    
    def stop(self):
        """停止调度器"""
        self._running = False
        self._task_queue.put(PrioritizedTask(
            priority=0,
            task_type=TaskType.FLUSH_BUFFER,
            callback=lambda: None
        ))
        
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5)
        
        self._log.info("BACKGROUND_SCHEDULER_STOPPED")
    
    def submit_task(
        self,
        task_type: TaskType,
        callback: Callable,
        args: tuple = (),
        kwargs: Optional[Dict] = None,
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> bool:
        """
        提交后台任务
        
        Args:
            task_type: 任务类型
            callback: 执行函数
            args: 位置参数
            kwargs: 关键字参数
            priority: 优先级
        
        Returns:
            是否成功提交
        """
        if not self._running:
            self._log.warning("SCHEDULER_NOT_RUNNING", task_type=task_type.value)
            return False
        
        task = PrioritizedTask(
            priority=priority.value,
            task_type=task_type,
            callback=callback,
            args=args,
            kwargs=kwargs or {}
        )
        
        try:
            self._task_queue.put(task, block=False)
            self._log.debug("TASK_SUBMITTED", task_type=task_type.value, priority=priority.name)
            return True
        except queue.Full:
            self._log.warning("TASK_QUEUE_FULL", task_type=task_type.value)
            return False
    
    def _worker_loop(self):
        """工作线程循环"""
        while self._running:
            try:
                try:
                    task = self._task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                if task.priority == 0:
                    continue
                
                result = self._execute_task(task)
                
                self._update_stats(result)
                
                self._task_queue.task_done()
                
            except Exception as e:
                self._log.error("WORKER_LOOP_ERROR", error=str(e))
    
    def _execute_task(self, task: PrioritizedTask) -> TaskResult:
        """
        执行任务（带迁移锁保护）
        
        Args:
            task: 任务对象
        
        Returns:
            执行结果
        """
        start_time = time.time()
        task_type = task.task_type
        
        try:
            self._begin_migration(task_type)
        except TimeoutError as e:
            duration_ms = (time.time() - start_time) * 1000
            self._log.error(
                "TASK_MIGRATION_TIMEOUT",
                task_type=task_type.value,
                error=str(e),
                duration_ms=round(duration_ms, 2)
            )
            return TaskResult(
                task_type=task_type,
                success=False,
                error=str(e),
                duration_ms=duration_ms
            )
        
        try:
            result = task.callback(*task.args, **task.kwargs)
            
            affected_count = 0
            if isinstance(result, dict):
                affected_count = result.get("affected_count", result.get("count", 0))
            elif isinstance(result, int):
                affected_count = result
            
            duration_ms = (time.time() - start_time) * 1000
            
            self._log.info(
                "TASK_EXECUTED",
                task_type=task_type.value,
                affected_count=affected_count,
                duration_ms=round(duration_ms, 2)
            )
            
            return TaskResult(
                task_type=task_type,
                success=True,
                affected_count=affected_count,
                duration_ms=duration_ms,
                details={"result": result} if result else {}
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            self._log.error(
                "TASK_FAILED",
                task_type=task_type.value,
                error=str(e),
                duration_ms=round(duration_ms, 2)
            )
            
            return TaskResult(
                task_type=task_type,
                success=False,
                error=str(e),
                duration_ms=duration_ms
            )
            
        finally:
            self._end_migration()
    
    def _begin_migration(self, task_type: TaskType, timeout: float = 30.0):
        """
        开始迁移操作
        
        获取写锁，阻止所有读操作
        
        Args:
            task_type: 任务类型
            timeout: 等待读者完成的最大超时时间（秒），默认 30 秒
        
        Raises:
            TimeoutError: 等待读者超时
        """
        deadline = time.time() + timeout
        
        with self._write_waiting:
            while self._readers_count > 0:
                remaining = deadline - time.time()
                if remaining <= 0:
                    self._log.error(
                        "MIGRATION_WAIT_READERS_TIMEOUT",
                        task_type=task_type.value,
                        readers_count=self._readers_count,
                        timeout=timeout
                    )
                    raise TimeoutError(f"等待读者完成超时 ({timeout}秒)，当前读者数: {self._readers_count}")
                
                self._write_waiting.wait(timeout=min(remaining, 0.5))
            
            self._write_in_progress = True
            self._migration_active = True
            self._current_task = task_type
            self._migration_start_time = time.time()
            
            self._migration_lock.acquire()
            
            self._log.debug(
                "MIGRATION_STARTED",
                task_type=task_type.value,
                queue_size=self._task_queue.qsize()
            )
    
    def _end_migration(self):
        """
        结束迁移操作
        
        释放写锁，允许读操作
        """
        task_type = self._current_task
        duration = time.time() - self._migration_start_time if self._migration_start_time else 0
        
        self._migration_active = False
        self._current_task = None
        self._migration_start_time = None
        
        try:
            self._migration_lock.release()
        except RuntimeError:
            pass
        
        with self._write_waiting:
            self._write_in_progress = False
            self._write_waiting.notify_all()
        
        self._last_task_times[task_type] = time.time()
        
        self._log.debug(
            "MIGRATION_ENDED",
            task_type=task_type.value if task_type else "unknown",
            duration_ms=round(duration * 1000, 2)
        )
    
    def read_lock(self, timeout: float = 5.0):
        """
        获取读锁上下文管理器
        
        用于检索操作，等待迁移完成后执行
        
        Args:
            timeout: 超时时间（秒），默认 5.0 秒
        
        Returns:
            读锁上下文管理器
        """
        return _ReadLockContext(self, timeout)
    
    def acquire_read(self, timeout: float = 5.0) -> bool:
        """
        获取读锁
        
        Args:
            timeout: 超时时间（秒）
        
        Returns:
            是否成功获取读锁
        """
        deadline = time.time() + timeout
        
        with self._write_waiting:
            while self._write_in_progress:
                remaining = deadline - time.time()
                if remaining <= 0:
                    self._log.warning(
                        "READ_LOCK_TIMEOUT",
                        timeout=timeout
                    )
                    return False
                
                self._write_waiting.wait(timeout=min(remaining, 0.1))
            
            self._readers_count += 1
        
        return True
    
    def release_read(self):
        """释放读锁"""
        with self._write_waiting:
            self._readers_count -= 1
            if self._readers_count == 0:
                self._write_waiting.notify_all()
    
    def is_migration_active(self) -> bool:
        """检查是否有迁移操作正在进行"""
        return self._migration_active
    
    def get_current_task(self) -> Optional[TaskType]:
        """获取当前正在执行的任务类型"""
        return self._current_task
    
    def wait_for_migration(self, timeout: float = 5.0) -> bool:
        """
        等待迁移完成
        
        Args:
            timeout: 超时时间（秒）
        
        Returns:
            是否成功等待完成
        """
        start_time = time.time()
        
        while self._migration_active:
            if time.time() - start_time > timeout:
                self._log.warning(
                    "MIGRATION_WAIT_TIMEOUT",
                    current_task=self._current_task.value if self._current_task else None,
                    timeout=timeout
                )
                return False
            time.sleep(0.05)
        
        return True
    
    def _update_stats(self, result: TaskResult):
        """更新任务统计"""
        stats = self._task_stats[result.task_type]
        stats["total"] += 1
        if result.success:
            stats["success"] += 1
            stats["affected"] += result.affected_count
        else:
            stats["failed"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """获取任务统计"""
        return {
            "running": self._running,
            "migration_active": self._migration_active,
            "current_task": self._current_task.value if self._current_task else None,
            "queue_size": self._task_queue.qsize(),
            "task_stats": {
                t.value: stats for t, stats in self._task_stats.items()
            },
            "last_task_times": {
                t.value: ts for t, ts in self._last_task_times.items()
            }
        }
    
    def get_last_task_time(self, task_type: TaskType) -> Optional[float]:
        """获取指定类型任务的最后执行时间"""
        return self._last_task_times.get(task_type)
    
    def should_run_task(self, task_type: TaskType, min_interval: float) -> bool:
        """
        检查是否应该运行任务（基于最小间隔）
        
        Args:
            task_type: 任务类型
            min_interval: 最小间隔（秒）
        
        Returns:
            是否应该运行
        """
        last_time = self._last_task_times.get(task_type, 0)
        return time.time() - last_time >= min_interval


class _ReadLockContext:
    """
    读锁上下文管理器
    
    支持超时机制，超时后可选择降级执行
    """
    
    def __init__(self, scheduler: BackgroundTaskScheduler, timeout: float = 5.0):
        self._scheduler = scheduler
        self._timeout = timeout
        self._acquired = False
    
    def __enter__(self):
        self._acquired = self._scheduler.acquire_read(self._timeout)
        if not self._acquired:
            self._scheduler._log.warning(
                "READ_LOCK_FALLBACK",
                message="读锁获取超时，降级为无锁搜索"
            )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._acquired:
            self._scheduler.release_read()
            self._acquired = False
        return False
    
    def is_locked(self) -> bool:
        """检查是否成功获取读锁"""
        return self._acquired


_scheduler: Optional[BackgroundTaskScheduler] = None


def get_background_scheduler() -> BackgroundTaskScheduler:
    """获取全局后台任务调度器实例"""
    global _scheduler
    if _scheduler is None:
        _scheduler = BackgroundTaskScheduler()
    return _scheduler
