import threading
import queue
import weakref
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from logger import get_logger


class EventType(Enum):
    """事件类型"""
    L1_OVERFLOW = "l1_overflow"
    MEMORY_WRITTEN = "memory_written"
    MEMORY_COMPRESSED = "memory_compressed"
    MEMORY_MOVED = "memory_moved"
    MEMORY_FORGOTTEN = "memory_forgotten"
    SEARCH_COMPLETED = "search_completed"
    BACKFILL_COMPLETED = "backfill_completed"


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
    """
    
    _instance: Optional['EventBus'] = None
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
        self._subscribers: Dict[EventType, List[SubscriberInfo]] = {}
        self._subscribers_lock = threading.RLock()
        self._event_queue: queue.Queue = queue.Queue()
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._process_loop,
            daemon=True,
            name="EventBusWorker"
        )
        self._worker_thread.start()
    
    def _process_loop(self):
        """后台线程处理异步事件队列"""
        while self._running:
            try:
                event = self._event_queue.get(timeout=1.0)
                if event is None:
                    continue
                self.publish(event.event_type, event.data, event.source)
                self._event_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self._log.error("EVENT_PROCESS_ERROR", error=str(e))
    
    def _create_handler_ref(self, handler: EventHandler) -> Any:
        """
        创建处理器引用
        
        优先使用弱引用，如果处理器不支持则使用强引用
        """
        try:
            if hasattr(handler, '__self__'):
                return weakref.WeakMethod(handler)
            else:
                return weakref.ref(handler)
        except TypeError:
            return handler
    
    def _get_handler(self, ref: Any) -> Optional[EventHandler]:
        """从引用中获取处理器"""
        if isinstance(ref, weakref.ref):
            handler = ref()
            return handler if handler is not None else None
        elif callable(ref):
            return ref
        return None
    
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
            
            handler_ref = self._create_handler_ref(handler)
            
            for info in self._subscribers[event_type]:
                existing = self._get_handler(info.handler_ref)
                if existing == handler:
                    return
            
            info = SubscriberInfo(
                handler_ref=handler_ref,
                source=source,
                subscribed_at=datetime.now()
            )
            self._subscribers[event_type].append(info)
            
            self._log.debug("EVENT_SUBSCRIBED", 
                           event_type=event_type.value, 
                           handler=getattr(handler, '__name__', str(handler)),
                           source=source)
    
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
                    existing = self._get_handler(info.handler_ref)
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
            handler = self._get_handler(info.handler_ref)
            if handler is None:
                to_remove.append(i)
        
        for i in reversed(to_remove):
            removed = self._subscribers[event_type].pop(i)
            self._log.debug("SUBSCRIBER_CLEANED_UP",
                           event_type=event_type.value,
                           source=removed.source)
    
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
                handler = self._get_handler(info.handler_ref)
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
                               handler=getattr(handler, '__name__', str(handler)),
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
        self._event_queue.put(event)
    
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
