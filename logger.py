import logging
import os
from datetime import datetime
from typing import Optional
from config import config


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
    - event: 事件名称
    - 关键字段: 如 userId, recordId 等
    - 决策路径: 如 primaryAttempt, fallbackUsed
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
        
        log_file = os.path.join(log_dir, f'memory_{datetime.now().strftime("%Y%m%d")}.log')
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
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
        parts = [f"[{event}]"]
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


_logger: Optional[StructuredLogger] = None


def get_logger() -> StructuredLogger:
    """获取全局日志实例"""
    global _logger
    if _logger is None:
        _logger = StructuredLogger()
    return _logger
