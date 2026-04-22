import logging
import os
from datetime import datetime
from typing import Optional
from logging.handlers import TimedRotatingFileHandler
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


_logger: Optional[StructuredLogger] = None


def get_logger() -> StructuredLogger:
    """获取全局日志实例"""
    global _logger
    if _logger is None:
        _logger = StructuredLogger()
    return _logger
