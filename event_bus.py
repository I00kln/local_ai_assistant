import threading
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
        self._subscribers: Dict[EventType, List[EventHandler]] = {}
        self._subscribers_lock = threading.RLock()
        self._event_queue: List[Event] = []
        self._queue_lock = threading.Lock()
    
    def subscribe(self, event_type: EventType, handler: EventHandler):
        """
        订阅事件
        
        Args:
            event_type: 事件类型
            handler: 事件处理函数
        """
        with self._subscribers_lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            if handler not in self._subscribers[event_type]:
                self._subscribers[event_type].append(handler)
                self._log.debug("EVENT_SUBSCRIBED", 
                               event_type=event_type.value, 
                               handler=handler.__name__)
    
    def unsubscribe(self, event_type: EventType, handler: EventHandler):
        """
        取消订阅
        
        Args:
            event_type: 事件类型
            handler: 事件处理函数
        """
        with self._subscribers_lock:
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(handler)
                except ValueError:
                    pass
    
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
            handlers = self._subscribers.get(event_type, []).copy()
        
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                self._log.error("EVENT_HANDLER_ERROR",
                               event_type=event_type.value,
                               handler=handler.__name__,
                               error=str(e))
        
        self._log.debug("EVENT_PUBLISHED",
                       event_type=event_type.value,
                       source=source,
                       subscriber_count=len(handlers))
    
    def publish_async(self, event_type: EventType, data: Dict[str, Any], source: str = "unknown"):
        """
        异步发布事件（不阻塞发布者）
        
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
        
        with self._queue_lock:
            self._event_queue.append(event)
    
    def process_queued_events(self):
        """处理队列中的异步事件"""
        events = []
        with self._queue_lock:
            events = self._event_queue.copy()
            self._event_queue.clear()
        
        for event in events:
            self.publish(event.event_type, event.data, event.source)
    
    def get_subscriber_count(self, event_type: EventType) -> int:
        """获取事件订阅者数量"""
        with self._subscribers_lock:
            return len(self._subscribers.get(event_type, []))


_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """获取全局事件总线实例"""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus
