import logging
import os
import uuid
from typing import Optional, Dict, Any
from logging.handlers import TimedRotatingFileHandler
from contextvars import ContextVar


_trace_id: ContextVar[str] = ContextVar('trace_id', default='')
_span_id: ContextVar[str] = ContextVar('span_id', default='')
_trace_context: ContextVar[Dict[str, Any]] = ContextVar('trace_context', default={})


def generate_trace_id() -> str:
    """生成新的 TraceID"""
    return uuid.uuid4().hex[:16]


def generate_span_id() -> str:
    """生成新的 SpanID"""
    return uuid.uuid4().hex[:8]


def get_trace_id() -> str:
    """获取当前 TraceID，如果不存在则创建"""
    tid = _trace_id.get()
    if not tid:
        tid = generate_trace_id()
        _trace_id.set(tid)
    return tid


def set_trace_id(trace_id: str) -> str:
    """设置 TraceID"""
    _trace_id.set(trace_id)
    return trace_id


def get_span_id() -> str:
    """获取当前 SpanID"""
    return _span_id.get()


def set_span_id(span_id: str = None) -> str:
    """设置 SpanID"""
    if span_id is None:
        span_id = generate_span_id()
    _span_id.set(span_id)
    return span_id


def get_trace_context() -> Dict[str, Any]:
    """获取追踪上下文"""
    return _trace_context.get()


def set_trace_context(key: str, value: Any):
    """设置追踪上下文字段"""
    ctx = _trace_context.get().copy()
    ctx[key] = value
    _trace_context.set(ctx)


def clear_trace_context():
    """清除追踪上下文"""
    _trace_id.set('')
    _span_id.set('')
    _trace_context.set({})


class TraceContext:
    """
    追踪上下文管理器
    
    Usage:
        with TraceContext("operation_name"):
            logger.info("EVENT", key=value)
    """
    
    def __init__(self, operation: str = None, parent_trace_id: str = None):
        self.operation = operation
        self.parent_trace_id = parent_trace_id
        self._old_trace_id = None
        self._old_span_id = None
        self._old_context = None
    
    def __enter__(self):
        self._old_trace_id = _trace_id.get()
        self._old_span_id = _span_id.get()
        self._old_context = _trace_context.get().copy()
        
        if self.parent_trace_id:
            _trace_id.set(self.parent_trace_id)
        else:
            _trace_id.set(generate_trace_id())
        
        _span_id.set(generate_span_id())
        _trace_context.set({"operation": self.operation} if self.operation else {})
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        _trace_id.set(self._old_trace_id)
        _span_id.set(self._old_span_id)
        _trace_context.set(self._old_context)
        return False


class StructuredLogger:
    """
    结构化日志记录器
    
    支持日志分级：
    - DEBUG: 调试信息
    - INFO: 常规信息
    - WARNING: 警告信息
    - ERROR: 错误信息
    - CRITICAL: 严重错误
    
    支持结构化输出：
    - trace_id: 追踪ID（跨组件链路追踪）
    - span_id: SpanID（子操作标识）
    - event: 事件名称
    - 关键字段: 如 userId, recordId 等
    - 决策路径: 如 primaryAttempt, fallbackUsed
    
    日志轮转：
    - 使用 TimedRotatingFileHandler 实现按天轮转
    - 自动处理跨天日志切分
    - 保留30天历史日志
    """
    
    _instance: Optional['StructuredLogger'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.logger = logging.getLogger('MemorySystem')
        self.logger.setLevel(logging.DEBUG)
        
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, 'memory.log')
        
        file_handler = TimedRotatingFileHandler(
            filename=log_file,
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.suffix = "%Y%m%d"
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _format_message(self, event: str, **kwargs) -> str:
        """格式化结构化日志消息"""
        trace_id = get_trace_id()
        span_id = get_span_id()
        trace_context = get_trace_context()
        
        parts = []
        
        if trace_id:
            parts.append(f"[{trace_id}]")
        
        if span_id:
            parts.append(f"[{span_id}]")
        
        parts.append(f"[{event}]")
        
        for key, value in trace_context.items():
            if key not in kwargs:
                parts.append(f"{key}={value}")
        
        for key, value in kwargs.items():
            parts.append(f"{key}={value}")
        
        return " | ".join(parts)
    
    def debug(self, event: str, **kwargs):
        """调试级别日志"""
        self.logger.debug(self._format_message(event, **kwargs))
    
    def info(self, event: str, **kwargs):
        """信息级别日志"""
        self.logger.info(self._format_message(event, **kwargs))
    
    def warning(self, event: str, **kwargs):
        """警告级别日志"""
        self.logger.warning(self._format_message(event, **kwargs))
    
    def error(self, event: str, **kwargs):
        """错误级别日志"""
        self.logger.error(self._format_message(event, **kwargs))
    
    def critical(self, event: str, **kwargs):
        """严重错误级别日志"""
        self.logger.critical(self._format_message(event, **kwargs))
    
    def with_trace(self, trace_id: str = None, **context) -> 'TraceLogger':
        """
        创建带追踪上下文的日志器
        
        Args:
            trace_id: 指定 TraceID（可选）
            **context: 上下文字段
        
        Returns:
            TraceLogger 实例
        """
        return TraceLogger(self, trace_id, context)
    
    def set_level(self, level: str):
        """
        动态调整日志级别
        
        Args:
            level: 日志级别名称 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        
        new_level = level_map.get(level.upper())
        if new_level is None:
            self.logger.warning(f"无效的日志级别: {level}")
            return
        
        self.logger.setLevel(new_level)
        
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(new_level)
        
        self.info("LOG_LEVEL_CHANGED", new_level=level.upper())
    
    def get_level(self) -> str:
        """获取当前日志级别"""
        level_map = {
            logging.DEBUG: "DEBUG",
            logging.INFO: "INFO",
            logging.WARNING: "WARNING",
            logging.ERROR: "ERROR",
            logging.CRITICAL: "CRITICAL",
        }
        return level_map.get(self.logger.level, "UNKNOWN")
    
    def set_console_level(self, level: str):
        """
        单独设置控制台日志级别
        
        Args:
            level: 日志级别名称
        """
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        
        new_level = level_map.get(level.upper())
        if new_level is None:
            return
        
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(new_level)
        
        self.info("CONSOLE_LOG_LEVEL_CHANGED", new_level=level.upper())


class TraceLogger:
    """
    带追踪上下文的日志器
    
    自动在所有日志中添加 TraceID 和上下文字段
    """
    
    def __init__(self, logger: StructuredLogger, trace_id: str = None, context: Dict[str, Any] = None):
        self._logger = logger
        self._trace_id = trace_id or generate_trace_id()
        self._context = context or {}
        self._span_id = generate_span_id()
    
    def _log(self, level: str, event: str, **kwargs):
        """内部日志方法"""
        old_trace_id = _trace_id.get()
        old_span_id = _span_id.get()
        old_context = _trace_context.get().copy()
        
        try:
            _trace_id.set(self._trace_id)
            _span_id.set(self._span_id)
            
            ctx = self._context.copy()
            ctx.update(_trace_context.get())
            _trace_context.set(ctx)
            
            method = getattr(self._logger, level)
            method(event, **kwargs)
        finally:
            _trace_id.set(old_trace_id)
            _span_id.set(old_span_id)
            _trace_context.set(old_context)
    
    def debug(self, event: str, **kwargs):
        self._log('debug', event, **kwargs)
    
    def info(self, event: str, **kwargs):
        self._log('info', event, **kwargs)
    
    def warning(self, event: str, **kwargs):
        self._log('warning', event, **kwargs)
    
    def error(self, event: str, **kwargs):
        self._log('error', event, **kwargs)
    
    def critical(self, event: str, **kwargs):
        self._log('critical', event, **kwargs)
    
    @property
    def trace_id(self) -> str:
        return self._trace_id
    
    @property
    def span_id(self) -> str:
        return self._span_id


_logger: Optional[StructuredLogger] = None


def get_logger() -> StructuredLogger:
    """获取全局日志实例"""
    global _logger
    if _logger is None:
        _logger = StructuredLogger()
    return _logger
