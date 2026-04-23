import threading
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from logger import get_logger


class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """组件健康状态"""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    checked_at: datetime


class HealthChecker:
    """
    健康检查器
    
    检查各组件状态：
    - VectorStore: 向量存储连接、文档数量
    - SQLiteStore: 数据库连接、记录数量
    - MemoryManager: L1/L2/L3 记忆数量
    - AsyncProcessor: 处理队列状态、压缩器可用性
    """
    
    _instance: Optional['HealthChecker'] = None
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
        self._components: Dict[str, Any] = {}
    
    def register_component(self, name: str, component: Any):
        """注册组件"""
        self._components[name] = component
        self._log.debug("HEALTH_COMPONENT_REGISTERED", component=name)
    
    def check_all(self) -> Dict[str, ComponentHealth]:
        """检查所有组件健康状态"""
        results = {}
        
        for name, component in self._components.items():
            try:
                health = self._check_component(name, component)
                results[name] = health
            except Exception as e:
                results[name] = ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"检查失败: {str(e)}",
                    details={"error": str(e)},
                    checked_at=datetime.now()
                )
        
        return results
    
    def _check_component(self, name: str, component: Any) -> ComponentHealth:
        """检查单个组件"""
        checked_at = datetime.now()
        
        if name == "vector_store":
            return self._check_vector_store(component, checked_at)
        elif name == "sqlite_store":
            return self._check_sqlite_store(component, checked_at)
        elif name == "memory_manager":
            return self._check_memory_manager(component, checked_at)
        elif name == "async_processor":
            return self._check_async_processor(component, checked_at)
        else:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNKNOWN,
                message="未知组件类型",
                details={},
                checked_at=checked_at
            )
    
    def _check_vector_store(self, store: Any, checked_at: datetime) -> ComponentHealth:
        """检查向量存储"""
        try:
            count = len(store)
            status = HealthStatus.HEALTHY if count >= 0 else HealthStatus.UNHEALTHY
            return ComponentHealth(
                name="vector_store",
                status=status,
                message=f"向量库正常，当前文档数: {count}",
                details={"document_count": count},
                checked_at=checked_at
            )
        except Exception as e:
            return ComponentHealth(
                name="vector_store",
                status=HealthStatus.UNHEALTHY,
                message=f"向量库异常: {str(e)}",
                details={"error": str(e)},
                checked_at=checked_at
            )
    
    def _check_sqlite_store(self, store: Any, checked_at: datetime) -> ComponentHealth:
        """检查 SQLite 存储"""
        try:
            stats = store.get_stats()
            total = stats.get("total_records", 0)
            status = HealthStatus.HEALTHY if total >= 0 else HealthStatus.UNHEALTHY
            return ComponentHealth(
                name="sqlite_store",
                status=status,
                message=f"数据库正常，当前记录数: {total}",
                details=stats,
                checked_at=checked_at
            )
        except Exception as e:
            return ComponentHealth(
                name="sqlite_store",
                status=HealthStatus.UNHEALTHY,
                message=f"数据库异常: {str(e)}",
                details={"error": str(e)},
                checked_at=checked_at
            )
    
    def _check_memory_manager(self, manager: Any, checked_at: datetime) -> ComponentHealth:
        """检查记忆管理器"""
        try:
            l1_count = len(manager.conversation_history)
            l2_count = len(manager.vector_store)
            stats = manager.stats.copy()
            
            status = HealthStatus.HEALTHY
            return ComponentHealth(
                name="memory_manager",
                status=status,
                message=f"L1: {l1_count}, L2: {l2_count}",
                details={
                    "l1_count": l1_count,
                    "l2_count": l2_count,
                    "stats": stats
                },
                checked_at=checked_at
            )
        except Exception as e:
            return ComponentHealth(
                name="memory_manager",
                status=HealthStatus.UNHEALTHY,
                message=f"记忆管理器异常: {str(e)}",
                details={"error": str(e)},
                checked_at=checked_at
            )
    
    def _check_async_processor(self, processor: Any, checked_at: datetime) -> ComponentHealth:
        """检查异步处理器"""
        try:
            queue_size = processor.pending_queue.qsize()
            running = processor.running
            compressor_available = processor._compressor_available
            
            if not running:
                status = HealthStatus.UNHEALTHY
                message = "异步处理器未运行"
            elif queue_size > 100:
                status = HealthStatus.DEGRADED
                message = f"队列积压: {queue_size}"
            elif not compressor_available:
                status = HealthStatus.DEGRADED
                message = "压缩器不可用"
            else:
                status = HealthStatus.HEALTHY
                message = "异步处理器正常"
            
            return ComponentHealth(
                name="async_processor",
                status=status,
                message=message,
                details={
                    "running": running,
                    "queue_size": queue_size,
                    "compressor_available": compressor_available,
                    "buffer_size": len(processor.batch_buffer)
                },
                checked_at=checked_at
            )
        except Exception as e:
            return ComponentHealth(
                name="async_processor",
                status=HealthStatus.UNHEALTHY,
                message=f"异步处理器异常: {str(e)}",
                details={"error": str(e)},
                checked_at=checked_at
            )
    
    def get_summary(self) -> Dict[str, Any]:
        """获取健康状态摘要"""
        results = self.check_all()
        
        healthy = sum(1 for h in results.values() if h.status == HealthStatus.HEALTHY)
        degraded = sum(1 for h in results.values() if h.status == HealthStatus.DEGRADED)
        unhealthy = sum(1 for h in results.values() if h.status == HealthStatus.UNHEALTHY)
        
        if unhealthy > 0:
            overall = HealthStatus.UNHEALTHY
        elif degraded > 0:
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY
        
        return {
            "overall_status": overall.value,
            "healthy_count": healthy,
            "degraded_count": degraded,
            "unhealthy_count": unhealthy,
            "components": {name: {
                "status": health.status.value,
                "message": health.message,
                "details": health.details
            } for name, health in results.items()},
            "checked_at": datetime.now().isoformat()
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        获取运行时指标
        
        整合：
        - 组件健康状态
        - 记忆数量统计
        - 队列状态
        - 性能指标（来自 metrics.py）
        """
        from metrics import get_metrics_collector
        
        results = self.check_all()
        
        memory_counts = {"l1": 0, "l2": 0, "l3": 0}
        queue_stats = {}
        
        if "memory_manager" in results:
            details = results["memory_manager"].details
            memory_counts["l1"] = details.get("l1_count", 0)
            memory_counts["l2"] = details.get("l2_count", 0)
        
        if "sqlite_store" in results:
            details = results["sqlite_store"].details
            memory_counts["l3"] = details.get("total_records", 0)
        
        if "async_processor" in results:
            details = results["async_processor"].details
            queue_stats = {
                "queue_size": details.get("queue_size", 0),
                "buffer_size": details.get("buffer_size", 0),
                "running": details.get("running", False),
            }
        
        if "vector_store" in results:
            details = results["vector_store"].details
            memory_counts["l2"] = details.get("document_count", memory_counts["l2"])
        
        metrics_collector = get_metrics_collector()
        performance_metrics = metrics_collector.get_metrics()
        
        return {
            "memory": memory_counts,
            "queue": queue_stats,
            "health": {
                "overall": self._get_overall_status(results),
                "components": {name: h.status.value for name, h in results.items()},
            },
            "performance": performance_metrics,
            "checked_at": datetime.now().isoformat(),
        }
    
    def _get_overall_status(self, results: Dict[str, ComponentHealth]) -> str:
        """获取整体健康状态"""
        unhealthy = sum(1 for h in results.values() if h.status == HealthStatus.UNHEALTHY)
        degraded = sum(1 for h in results.values() if h.status == HealthStatus.DEGRADED)
        
        if unhealthy > 0:
            return "unhealthy"
        elif degraded > 0:
            return "degraded"
        else:
            return "healthy"
    
    def get_metrics_summary(self) -> str:
        """获取指标摘要文本"""
        from metrics import get_metrics_collector
        
        metrics = self.get_metrics()
        collector = get_metrics_collector()
        
        lines = [
            "=== 系统状态监控 ===",
            "",
            "【记忆统计】",
            f"  L1 内存层: {metrics['memory']['l1']} 条",
            f"  L2 向量库: {metrics['memory']['l2']} 条",
            f"  L3 数据库: {metrics['memory']['l3']} 条",
            "",
            "【队列状态】",
            f"  待处理: {metrics['queue'].get('queue_size', 0)} 条",
            f"  缓冲区: {metrics['queue'].get('buffer_size', 0)} 条",
            f"  运行中: {'是' if metrics['queue'].get('running', False) else '否'}",
            "",
            "【健康状态】",
            f"  整体: {metrics['health']['overall']}",
        ]
        
        for name, status in metrics['health']['components'].items():
            lines.append(f"  {name}: {status}")
        
        lines.append("")
        lines.append(collector.get_summary())
        
        return "\n".join(lines)


_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """获取全局健康检查器实例"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker
