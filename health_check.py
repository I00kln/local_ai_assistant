import threading
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from logger import get_logger


class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class DegradationLevel(Enum):
    """降级级别"""
    NONE = "none"
    MINIMAL = "minimal"
    PARTIAL = "partial"
    SEVERE = "severe"
    CRITICAL = "critical"


@dataclass
class ComponentHealth:
    """组件健康状态"""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    checked_at: datetime
    degradation_level: DegradationLevel = DegradationLevel.NONE
    fallback_active: bool = False
    dependencies: List[str] = field(default_factory=list)


@dataclass
class DependencyInfo:
    """依赖信息"""
    name: str
    component: str
    critical: bool
    status: HealthStatus


DEPENDENCY_GRAPH = {
    "vector_store": [
        DependencyInfo("embedding_service", "embedding", True, HealthStatus.UNKNOWN),
        DependencyInfo("chromadb", "chromadb", True, HealthStatus.UNKNOWN),
    ],
    "sqlite_store": [
        DependencyInfo("sqlite", "sqlite", True, HealthStatus.UNKNOWN),
        DependencyInfo("disk_space", "system", True, HealthStatus.UNKNOWN),
    ],
    "async_processor": [
        DependencyInfo("vector_store", "vector_store", True, HealthStatus.UNKNOWN),
        DependencyInfo("sqlite_store", "sqlite_store", True, HealthStatus.UNKNOWN),
        DependencyInfo("llm_client", "llm", False, HealthStatus.UNKNOWN),
    ],
    "memory_manager": [
        DependencyInfo("vector_store", "vector_store", True, HealthStatus.UNKNOWN),
        DependencyInfo("sqlite_store", "sqlite_store", False, HealthStatus.UNKNOWN),
    ],
}


class HealthChecker:
    """
    健康检查器
    
    检查各组件状态：
    - VectorStore: 向量存储连接、文档数量
    - SQLiteStore: 数据库连接、记录数量
    - MemoryManager: L1/L2/L3 记忆数量
    - AsyncProcessor: 处理队列状态、压缩器可用性
    
    增强：
    - 依赖关系检查
    - 降级状态追踪
    - 降级状态机
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
        self._degradation_state: Dict[str, DegradationLevel] = {}
        self._fallback_status: Dict[str, bool] = {}
    
    def register_component(self, name: str, component: Any):
        """注册组件"""
        self._components[name] = component
        self._degradation_state[name] = DegradationLevel.NONE
        self._fallback_status[name] = False
        self._log.debug("HEALTH_COMPONENT_REGISTERED", component=name)
    
    def set_degradation(self, component: str, level: DegradationLevel, fallback: bool = False):
        """
        设置组件降级状态
        
        Args:
            component: 组件名称
            level: 降级级别
            fallback: 是否启用了降级方案
        """
        self._degradation_state[component] = level
        self._fallback_status[component] = fallback
        self._log.info("DEGRADATION_STATE_CHANGED", 
                      component=component, 
                      level=level.value,
                      fallback=fallback)
    
    def get_degradation_state(self) -> Dict[str, Any]:
        """获取所有组件的降级状态"""
        return {
            name: {
                "level": level.value,
                "fallback_active": self._fallback_status.get(name, False)
            }
            for name, level in self._degradation_state.items()
        }
    
    def check_all(self) -> Dict[str, ComponentHealth]:
        """检查所有组件健康状态"""
        results = {}
        
        for name, component in self._components.items():
            try:
                health = self._check_component(name, component)
                health.degradation_level = self._degradation_state.get(name, DegradationLevel.NONE)
                health.fallback_active = self._fallback_status.get(name, False)
                health.dependencies = [d.name for d in DEPENDENCY_GRAPH.get(name, [])]
                results[name] = health
            except Exception as e:
                results[name] = ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"检查失败: {str(e)}",
                    details={"error": str(e)},
                    checked_at=datetime.now(),
                    degradation_level=self._degradation_state.get(name, DegradationLevel.CRITICAL)
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


class DegradationStateMachine:
    """
    降级状态机
    
    管理组件降级状态转换：
    - NONE → MINIMAL → PARTIAL → SEVERE → CRITICAL
    - 支持自动恢复和手动恢复
    """
    
    TRANSITIONS = {
        DegradationLevel.NONE: [DegradationLevel.MINIMAL],
        DegradationLevel.MINIMAL: [DegradationLevel.NONE, DegradationLevel.PARTIAL],
        DegradationLevel.PARTIAL: [DegradationLevel.MINIMAL, DegradationLevel.SEVERE],
        DegradationLevel.SEVERE: [DegradationLevel.PARTIAL, DegradationLevel.CRITICAL],
        DegradationLevel.CRITICAL: [DegradationLevel.SEVERE],
    }
    
    RECOVERY_PATH = {
        DegradationLevel.CRITICAL: DegradationLevel.SEVERE,
        DegradationLevel.SEVERE: DegradationLevel.PARTIAL,
        DegradationLevel.PARTIAL: DegradationLevel.MINIMAL,
        DegradationLevel.MINIMAL: DegradationLevel.NONE,
    }
    
    def __init__(self):
        self._states: Dict[str, DegradationLevel] = {}
        self._recovery_attempts: Dict[str, int] = {}
        self._max_recovery_attempts = 3
    
    def get_state(self, component: str) -> DegradationLevel:
        """获取组件当前降级状态"""
        return self._states.get(component, DegradationLevel.NONE)
    
    def degrade(self, component: str) -> DegradationLevel:
        """
        降级组件状态
        
        Returns:
            新的降级级别
        """
        current = self._states.get(component, DegradationLevel.NONE)
        
        if current == DegradationLevel.CRITICAL:
            return current
        
        transitions = self.TRANSITIONS.get(current, [])
        if len(transitions) > 1:
            new_state = transitions[1]
        elif transitions:
            new_state = transitions[0]
        else:
            new_state = current
        
        self._states[component] = new_state
        self._recovery_attempts[component] = 0
        return new_state
    
    def recover(self, component: str) -> DegradationLevel:
        """
        尝试恢复组件状态
        
        Returns:
            新的降级级别
        """
        current = self._states.get(component, DegradationLevel.NONE)
        
        if current == DegradationLevel.NONE:
            return current
        
        attempts = self._recovery_attempts.get(component, 0) + 1
        self._recovery_attempts[component] = attempts
        
        if attempts >= self._max_recovery_attempts:
            new_state = self.RECOVERY_PATH.get(current, DegradationLevel.NONE)
            self._states[component] = new_state
            self._recovery_attempts[component] = 0
            return new_state
        
        return current
    
    def force_recover(self, component: str) -> DegradationLevel:
        """强制恢复到上一级"""
        current = self._states.get(component, DegradationLevel.NONE)
        new_state = self.RECOVERY_PATH.get(current, DegradationLevel.NONE)
        self._states[component] = new_state
        self._recovery_attempts[component] = 0
        return new_state


class HealthEndpoint:
    """
    健康检查端点
    
    提供 HTTP 风格的健康检查 API：
    - /health: 基本健康状态
    - /health/ready: 就绪探针
    - /health/live: 存活探针
    - /health/degradation: 降级状态
    - /health/dependencies: 依赖关系
    """
    
    def __init__(self, health_checker: HealthChecker):
        self._checker = health_checker
        self._state_machine = DegradationStateMachine()
        self._log = get_logger()
    
    def health(self) -> Dict[str, Any]:
        """
        基本健康状态
        
        Returns:
            整体健康状态和各组件状态
        """
        return self._checker.get_summary()
    
    def ready(self) -> Dict[str, Any]:
        """
        就绪探针
        
        检查系统是否准备好接收请求
        
        Returns:
            就绪状态和原因
        """
        results = self._checker.check_all()
        
        critical_unhealthy = []
        for name, health in results.items():
            deps = DEPENDENCY_GRAPH.get(name, [])
            for dep in deps:
                if dep.critical and health.status == HealthStatus.UNHEALTHY:
                    critical_unhealthy.append(name)
                    break
        
        ready = len(critical_unhealthy) == 0
        
        return {
            "ready": ready,
            "reason": "所有关键组件正常" if ready else f"关键组件异常: {critical_unhealthy}",
            "checked_at": datetime.now().isoformat()
        }
    
    def live(self) -> Dict[str, Any]:
        """
        存活探针
        
        检查进程是否存活
        
        Returns:
            存活状态
        """
        return {
            "alive": True,
            "checked_at": datetime.now().isoformat()
        }
    
    def degradation(self) -> Dict[str, Any]:
        """
        降级状态
        
        Returns:
            各组件的降级状态
        """
        return {
            "states": self._checker.get_degradation_state(),
            "state_machine": {
                component: self._state_machine.get_state(component).value
                for component in self._checker._components.keys()
            },
            "checked_at": datetime.now().isoformat()
        }
    
    def dependencies(self) -> Dict[str, Any]:
        """
        依赖关系
        
        Returns:
            组件依赖关系图
        """
        return {
            "graph": {
                component: [
                    {"name": dep.name, "critical": dep.critical}
                    for dep in deps
                ]
                for component, deps in DEPENDENCY_GRAPH.items()
            },
            "checked_at": datetime.now().isoformat()
        }
    
    def trigger_degradation(self, component: str) -> Dict[str, Any]:
        """
        触发组件降级
        
        Args:
            component: 组件名称
        
        Returns:
            新的降级状态
        """
        new_level = self._state_machine.degrade(component)
        self._checker.set_degradation(component, new_level, fallback=True)
        
        return {
            "component": component,
            "new_level": new_level.value,
            "checked_at": datetime.now().isoformat()
        }
    
    def trigger_recovery(self, component: str, force: bool = False) -> Dict[str, Any]:
        """
        触发组件恢复
        
        Args:
            component: 组件名称
            force: 是否强制恢复
        
        Returns:
            新的降级状态
        """
        if force:
            new_level = self._state_machine.force_recover(component)
        else:
            new_level = self._state_machine.recover(component)
        
        fallback_active = new_level != DegradationLevel.NONE
        self._checker.set_degradation(component, new_level, fallback_active)
        
        return {
            "component": component,
            "new_level": new_level.value,
            "force": force,
            "checked_at": datetime.now().isoformat()
        }


_health_checker: Optional[HealthChecker] = None
_health_endpoint: Optional[HealthEndpoint] = None


def get_health_checker() -> HealthChecker:
    """获取全局健康检查器实例"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


def get_health_endpoint() -> HealthEndpoint:
    """获取全局健康端点实例"""
    global _health_endpoint
    if _health_endpoint is None:
        _health_endpoint = HealthEndpoint(get_health_checker())
    return _health_endpoint
