"""
生命周期管理器

集中管理所有服务的启动和关闭，解决以下问题：
1. 硬编码清理：每增加一个模块都要手动维护 _on_close
2. Join 阻塞：多个线程都要 join 5秒，导致窗口关闭卡顿
3. 清理顺序：确保依赖关系正确

使用方式：
1. 服务启动时注册清理函数：lifecycle.register_cleanup("vector_store", vector_store.close)
2. 应用关闭时调用：lifecycle.shutdown()
"""
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from logger import get_logger


class ServicePriority(Enum):
    """服务关闭优先级（数字越大越先关闭）"""
    CRITICAL = 100
    HIGH = 80
    NORMAL = 50
    LOW = 20
    BACKGROUND = 10


@dataclass
class ServiceInfo:
    """服务信息"""
    name: str
    cleanup_fn: Callable
    priority: ServicePriority
    timeout: float = 5.0
    is_running: Callable[[], bool] = None
    stop_fn: Callable = None
    force_stop_fn: Callable = None
    thread: threading.Thread = None


class LifecycleManager:
    """
    生命周期管理器
    
    功能：
    - 集中注册服务清理函数
    - 按优先级顺序关闭服务
    - 并行关闭独立服务
    - 超时保护，避免无限等待
    - 发布 SHUTDOWN 事件通知所有服务
    """
    
    _instance: Optional['LifecycleManager'] = None
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
        self._services: Dict[str, ServiceInfo] = {}
        self._shutdown_hooks: List[Callable] = []
        self._is_shutting_down = False
        self._shutdown_timeout = 10.0
    
    def register(
        self, 
        name: str, 
        cleanup_fn: Callable,
        priority: ServicePriority = ServicePriority.NORMAL,
        timeout: float = 5.0,
        is_running: Callable[[], bool] = None,
        stop_fn: Callable = None,
        force_stop_fn: Callable = None,
        thread: threading.Thread = None
    ):
        """
        注册服务清理函数
        
        Args:
            name: 服务名称
            cleanup_fn: 清理函数
            priority: 关闭优先级（数字越大越先关闭）
            timeout: 清理超时时间（秒）
            is_running: 检查服务是否在运行的函数
            stop_fn: 停止服务的函数（在 cleanup_fn 之前调用）
            force_stop_fn: 强制停止服务的函数（超时后调用）
            thread: 服务的主线程引用（用于强制中断）
        """
        self._services[name] = ServiceInfo(
            name=name,
            cleanup_fn=cleanup_fn,
            priority=priority,
            timeout=timeout,
            is_running=is_running,
            stop_fn=stop_fn,
            force_stop_fn=force_stop_fn,
            thread=thread
        )
        self._log.debug("SERVICE_REGISTERED", name=name, priority=priority.value)
    
    def register_shutdown_hook(self, hook: Callable):
        """
        注册关闭钩子
        
        钩子会在 shutdown 开始时被调用
        """
        self._shutdown_hooks.append(hook)
    
    def unregister(self, name: str):
        """取消注册服务"""
        if name in self._services:
            del self._services[name]
    
    def is_shutting_down(self) -> bool:
        """检查是否正在关闭"""
        return self._is_shutting_down
    
    def shutdown(self, timeout: float = None):
        """
        关闭所有服务
        
        Args:
            timeout: 总超时时间（秒），默认 10 秒
        """
        if self._is_shutting_down:
            return
        
        self._is_shutting_down = True
        total_timeout = timeout or self._shutdown_timeout
        start_time = time.time()
        
        self._log.info("SHUTDOWN_STARTED", total_timeout=total_timeout)
        
        for hook in self._shutdown_hooks:
            try:
                hook()
            except Exception as e:
                self._log.warning("SHUTDOWN_HOOK_FAILED", error=str(e))
        
        sorted_services = sorted(
            self._services.values(),
            key=lambda s: s.priority.value,
            reverse=True
        )
        
        for service in sorted_services:
            elapsed = time.time() - start_time
            remaining = total_timeout - elapsed
            
            if remaining <= 0:
                self._log.warning("SHUTDOWN_TIMEOUT_EXCEEDED", remaining_services=[s.name for s in sorted_services])
                break
            
            service_timeout = min(service.timeout, remaining)
            self._shutdown_service(service, service_timeout)
        
        total_time = time.time() - start_time
        self._log.info("SHUTDOWN_COMPLETE", duration=round(total_time, 2))
    
    def _shutdown_service(self, service: ServiceInfo, timeout: float):
        """
        关闭单个服务
        
        Args:
            service: 服务信息
            timeout: 超时时间
        """
        self._log.debug("SERVICE_SHUTDOWN_START", name=service.name)
        
        if service.stop_fn:
            try:
                service.stop_fn()
            except Exception as e:
                self._log.warning("SERVICE_STOP_FAILED", name=service.name, error=str(e))
        
        if service.is_running and service.is_running():
            deadline = time.time() + timeout
            while time.time() < deadline and service.is_running():
                time.sleep(0.1)
            
            if service.is_running():
                self._log.warning("SERVICE_STOP_TIMEOUT", name=service.name, timeout=timeout)
                
                if service.force_stop_fn:
                    try:
                        self._log.info("SERVICE_FORCE_STOP", name=service.name)
                        service.force_stop_fn()
                        
                        force_deadline = time.time() + 2.0
                        while time.time() < force_deadline and service.is_running():
                            time.sleep(0.1)
                        
                        if service.is_running():
                            self._log.error(
                                "SERVICE_FORCE_STOP_FAILED",
                                name=service.name,
                                note="服务无法被强制停止，可能需要手动干预"
                            )
                    except Exception as e:
                        self._log.error(
                            "SERVICE_FORCE_STOP_ERROR",
                            name=service.name,
                            error=str(e)
                        )
        
        if service.cleanup_fn:
            try:
                service.cleanup_fn()
            except Exception as e:
                self._log.warning("SERVICE_CLEANUP_FAILED", name=service.name, error=str(e))
        
        self._log.debug("SERVICE_SHUTDOWN_COMPLETE", name=service.name)
    
    def get_status(self) -> Dict:
        """获取所有服务状态"""
        return {
            "is_shutting_down": self._is_shutting_down,
            "services": {
                name: {
                    "priority": info.priority.value,
                    "timeout": info.timeout,
                    "is_running": info.is_running() if info.is_running else None
                }
                for name, info in self._services.items()
            }
        }


def get_lifecycle_manager() -> LifecycleManager:
    """获取生命周期管理器单例"""
    return LifecycleManager()
