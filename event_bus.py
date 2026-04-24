import threading
import queue
import weakref
import time
from typing import Dict, List, Callable, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from logger import get_logger


class EventType(Enum):
    """事件类型"""
    L1_OVERFLOW = "l1_overflow"
    MEMORY_WRITTEN = "memory_written"
    QUEUE_OVERFLOW = "queue_overflow"
    QUEUE_RECOVERED = "queue_recovered"
    QUEUE_DROPPED = "queue_dropped"
    ORPHAN_CLEANUP_PROGRESS = "orphan_cleanup_progress"
    ORPHAN_CLEANUP_COMPLETE = "orphan_cleanup_complete"
    SHUTDOWN = "shutdown"
    HEARTBEAT = "heartbeat"
    CRITICAL_SERVICE_DOWN = "critical_service_down"


@dataclass
class Event:
    """事件数据"""
    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any]
    source: str


@dataclass
class SubscriberInfo:
    """订阅者信息"""
    handler_ref: Any
    source: str
    subscribed_at: datetime
    call_count: int = 0
    error_count: int = 0
    is_weak_ref: bool = True
    handler_name: str = ""


EventHandler = Callable[[Event], None]


class EventBus:
    """
    事件总线 - 解耦组件间通信
    
    使用发布-订阅模式：
    - 发布者：MemoryManager, AsyncProcessor
    - 订阅者：AsyncProcessor, 监控组件
    
    线程安全：
    - 使用 RLock 保护订阅者列表
    - 事件处理在锁外执行，避免死锁
    - 使用 queue.Queue + 后台线程处理异步事件
    
    内存安全：
    - 使用弱引用存储订阅者
    - 自动清理已销毁的订阅者
    - 订阅者生命周期追踪
    - 定期清理死订阅者
    """
    
    _instance: Optional['EventBus'] = None
    _lock = threading.Lock()
    
    CLEANUP_INTERVAL_SECONDS = 60.0
    
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
        self._subscribers: Dict[EventType, List[SubscriberInfo]] = {}
        self._subscribers_lock = threading.RLock()
        self._event_queue: queue.Queue = queue.Queue(maxsize=1000)
        self._queue_dropped_count = 0
        self._running = True
        self._strong_refs: Set[int] = set()
        self._last_cleanup_time = time.time()
        
        self._worker_thread = threading.Thread(
            target=self._process_loop,
            daemon=True,
            name="EventBusWorker"
        )
        self._worker_thread.start()
        
        self._register_lifecycle()
    
    def _register_lifecycle(self):
        """注册到生命周期管理器"""
        try:
            from lifecycle_manager import get_lifecycle_manager, ServicePriority
            lifecycle = get_lifecycle_manager()
            lifecycle.register(
                name="event_bus",
                cleanup_fn=self.shutdown,
                priority=ServicePriority.HIGH,
                timeout=2.0,
                is_running=lambda: self._running,
                stop_fn=lambda: setattr(self, '_running', False)
            )
        except Exception:
            pass
    
    def _process_loop(self):
        """后台线程处理异步事件队列"""
        while self._running:
            try:
                event = self._event_queue.get(timeout=1.0)
                if event is None:
                    continue
                self.publish(event.event_type, event.data, event.source)
                self._event_queue.task_done()
                
                self._maybe_cleanup()
                
            except queue.Empty:
                self._maybe_cleanup()
                continue
            except Exception as e:
                self._log.error("EVENT_PROCESS_ERROR", error=str(e))
    
    def _maybe_cleanup(self):
        """定期清理死订阅者"""
        now = time.time()
        if now - self._last_cleanup_time >= self.CLEANUP_INTERVAL_SECONDS:
            self._cleanup_all_dead_subscribers()
            self._last_cleanup_time = now
    
    def _cleanup_all_dead_subscribers(self):
        """清理所有事件类型的死订阅者"""
        with self._subscribers_lock:
            for event_type in list(self._subscribers.keys()):
                self._cleanup_dead_subscribers(event_type)
    
    def _create_handler_ref(self, handler: EventHandler) -> tuple:
        """
        创建处理器引用
        
        优先使用弱引用，如果处理器不支持则使用强引用
        
        Returns:
            (ref, is_weak_ref)
        """
        handler_name = getattr(handler, '__name__', str(handler))
        
        try:
            if hasattr(handler, '__self__'):
                return weakref.WeakMethod(handler), True, handler_name
            else:
                ref = weakref.ref(handler)
                return ref, True, handler_name
        except TypeError:
            handler_id = id(handler)
            self._strong_refs.add(handler_id)
            self._log.warning(
                "EVENT_HANDLER_STRONG_REF",
                handler=handler_name,
                note="处理器不支持弱引用，使用强引用可能导致内存延迟释放"
            )
            return handler, False, handler_name
    
    def _get_handler(self, ref: Any, is_weak_ref: bool) -> Optional[EventHandler]:
        """从引用中获取处理器"""
        if is_weak_ref:
            if isinstance(ref, weakref.ref):
                handler = ref()
                return handler if handler is not None else None
            return None
        else:
            return ref if callable(ref) else None
    
    def subscribe(self, event_type: EventType, handler: EventHandler, source: str = "unknown"):
        """
        订阅事件
        
        Args:
            event_type: 事件类型
            handler: 事件处理函数
            source: 订阅者来源（用于追踪）
        """
        with self._subscribers_lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            
            handler_ref, is_weak_ref, handler_name = self._create_handler_ref(handler)
            
            for info in self._subscribers[event_type]:
                existing = self._get_handler(info.handler_ref, info.is_weak_ref)
                if existing == handler:
                    return
            
            info = SubscriberInfo(
                handler_ref=handler_ref,
                source=source,
                subscribed_at=datetime.now(),
                is_weak_ref=is_weak_ref,
                handler_name=handler_name
            )
            self._subscribers[event_type].append(info)
            
            self._log.debug("EVENT_SUBSCRIBED", 
                           event_type=event_type.value, 
                           handler=handler_name,
                           source=source,
                           is_weak_ref=is_weak_ref)
    
    def unsubscribe(self, event_type: EventType, handler: EventHandler):
        """
        取消订阅
        
        Args:
            event_type: 事件类型
            handler: 事件处理函数
        """
        with self._subscribers_lock:
            if event_type in self._subscribers:
                to_remove = []
                for i, info in enumerate(self._subscribers[event_type]):
                    existing = self._get_handler(info.handler_ref, info.is_weak_ref)
                    if existing is None or existing == handler:
                        to_remove.append(i)
                
                for i in reversed(to_remove):
                    self._subscribers[event_type].pop(i)
    
    def _cleanup_dead_subscribers(self, event_type: EventType):
        """清理已销毁的订阅者"""
        if event_type not in self._subscribers:
            return
        
        to_remove = []
        for i, info in enumerate(self._subscribers[event_type]):
            handler = self._get_handler(info.handler_ref, info.is_weak_ref)
            if handler is None:
                to_remove.append(i)
        
        for i in reversed(to_remove):
            removed = self._subscribers[event_type].pop(i)
            self._log.debug("SUBSCRIBER_CLEANED_UP",
                           event_type=event_type.value,
                           source=removed.source,
                           handler=removed.handler_name)
    
    def publish(self, event_type: EventType, data: Dict[str, Any], source: str = "unknown"):
        """
        发布事件
        
        Args:
            event_type: 事件类型
            data: 事件数据
            source: 事件来源
        """
        event = Event(
            event_type=event_type,
            timestamp=datetime.now(),
            data=data,
            source=source
        )
        
        handlers = []
        with self._subscribers_lock:
            self._cleanup_dead_subscribers(event_type)
            
            for info in self._subscribers.get(event_type, []):
                handler = self._get_handler(info.handler_ref, info.is_weak_ref)
                if handler:
                    handlers.append((handler, info))
        
        for handler, info in handlers:
            try:
                handler(event)
                info.call_count += 1
            except Exception as e:
                info.error_count += 1
                self._log.error("EVENT_HANDLER_ERROR",
                               event_type=event_type.value,
                               handler=info.handler_name,
                               source=info.source,
                               error=str(e))
        
        self._log.debug("EVENT_PUBLISHED",
                       event_type=event_type.value,
                       source=source,
                       subscriber_count=len(handlers))
    
    def publish_async(self, event_type: EventType, data: Dict[str, Any], source: str = "unknown"):
        """
        异步发布事件（不阻塞发布者）
        
        事件会被放入队列，由后台线程异步处理
        
        队列满时丢弃策略：
        - 队列满时丢弃新事件，防止内存膨胀
        - 记录丢弃计数，供监控使用
        
        Args:
            event_type: 事件类型
            data: 事件数据
            source: 事件来源
        """
        event = Event(
            event_type=event_type,
            timestamp=datetime.now(),
            data=data,
            source=source
        )
        try:
            self._event_queue.put_nowait(event)
        except queue.Full:
            self._queue_dropped_count += 1
            self._log.warning(
                "EVENT_QUEUE_FULL",
                dropped_count=self._queue_dropped_count,
                event_type=event_type.value
            )
    
    def get_queue_size(self) -> int:
        """获取待处理事件队列大小"""
        return self._event_queue.qsize()
    
    def shutdown(self):
        """关闭事件总线（优雅退出）"""
        self._running = False
        self._event_queue.put(None)
        self._worker_thread.join(timeout=5.0)
    
    def get_subscriber_count(self, event_type: EventType) -> int:
        """获取事件订阅者数量"""
        with self._subscribers_lock:
            self._cleanup_dead_subscribers(event_type)
            return len(self._subscribers.get(event_type, []))
    
    def get_subscriber_stats(self) -> Dict[str, Any]:
        """
        获取订阅者统计信息
        
        Returns:
            各事件的订阅者统计
        """
        stats = {}
        with self._subscribers_lock:
            for event_type, infos in self._subscribers.items():
                self._cleanup_dead_subscribers(event_type)
                stats[event_type.value] = {
                    "count": len(infos),
                    "subscribers": [
                        {
                            "source": info.source,
                            "subscribed_at": info.subscribed_at.isoformat(),
                            "call_count": info.call_count,
                            "error_count": info.error_count
                        }
                        for info in infos
                    ]
                }
        return stats


_event_bus: Optional[EventBus] = None
_event_bus_lock = threading.Lock()


def get_event_bus() -> EventBus:
    """获取全局事件总线实例（线程安全单例）"""
    global _event_bus
    if _event_bus is None:
        with _event_bus_lock:
            if _event_bus is None:
                _event_bus = EventBus()
    return _event_bus
