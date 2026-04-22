import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from collections import deque
from datetime import datetime


@dataclass
class MetricsCollector:
    """
    轻量级指标收集器
    
    特性：
    - 线程安全
    - 内存安全（固定大小队列）
    - 低开销（无外部依赖）
    
    指标类型：
    - 计数器：累计值
    - 直方图：延迟分布（P50/P95/P99）
    - 仪表盘：当前值
    """
    
    _instance: Optional['MetricsCollector'] = None
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
        self._lock = threading.RLock()
        
        self._retrieval_latencies: deque = deque(maxlen=1000)
        self._embedding_latencies: deque = deque(maxlen=500)
        self._compression_latencies: deque = deque(maxlen=200)
        self._queue_wait_times: deque = deque(maxlen=500)
        
        self._compression_attempts: int = 0
        self._compression_successes: int = 0
        self._compression_failures: int = 0
        
        self._filter_total: int = 0
        self._filter_discarded: int = 0
        self._filter_sqlite_only: int = 0
        
        self._retrieval_total: int = 0
        self._retrieval_l1_hits: int = 0
        self._retrieval_l2_hits: int = 0
        self._retrieval_l3_hits: int = 0
        
        self._start_time: float = time.time()
        self._last_reset: float = time.time()
    
    def record_retrieval_latency(self, duration_ms: float):
        """记录检索延迟"""
        with self._lock:
            self._retrieval_latencies.append(duration_ms)
            self._retrieval_total += 1
    
    def record_retrieval_hits(self, l1: int, l2: int, l3: int):
        """记录各层检索命中数"""
        with self._lock:
            self._retrieval_l1_hits += l1
            self._retrieval_l2_hits += l2
            self._retrieval_l3_hits += l3
    
    def record_embedding_latency(self, duration_ms: float):
        """记录嵌入计算延迟"""
        with self._lock:
            self._embedding_latencies.append(duration_ms)
    
    def record_compression(self, success: bool, duration_ms: float = 0):
        """记录压缩结果"""
        with self._lock:
            self._compression_attempts += 1
            if success:
                self._compression_successes += 1
                if duration_ms > 0:
                    self._compression_latencies.append(duration_ms)
            else:
                self._compression_failures += 1
    
    def record_filter_result(self, storage_type: str):
        """记录过滤结果"""
        with self._lock:
            self._filter_total += 1
            if storage_type == "discard":
                self._filter_discarded += 1
            elif storage_type == "sqlite_only":
                self._filter_sqlite_only += 1
    
    def record_queue_wait_time(self, wait_ms: float):
        """记录队列等待时间"""
        with self._lock:
            self._queue_wait_times.append(wait_ms)
    
    def _calculate_percentiles(self, data: deque) -> Dict[str, float]:
        """计算百分位数"""
        if not data:
            return {"p50": 0, "p95": 0, "p99": 0, "avg": 0, "max": 0}
        
        sorted_data = sorted(data)
        n = len(sorted_data)
        
        return {
            "p50": sorted_data[n // 2],
            "p95": sorted_data[int(n * 0.95)] if n >= 20 else sorted_data[-1],
            "p99": sorted_data[int(n * 0.99)] if n >= 100 else sorted_data[-1],
            "avg": sum(sorted_data) / n,
            "max": sorted_data[-1],
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取所有指标"""
        with self._lock:
            uptime_seconds = time.time() - self._start_time
            
            return {
                "uptime_seconds": round(uptime_seconds, 1),
                "uptime_human": self._format_uptime(uptime_seconds),
                
                "retrieval": {
                    "total_requests": self._retrieval_total,
                    "latency_ms": self._calculate_percentiles(self._retrieval_latencies),
                    "hits": {
                        "l1": self._retrieval_l1_hits,
                        "l2": self._retrieval_l2_hits,
                        "l3": self._retrieval_l3_hits,
                    },
                    "hit_rate": {
                        "l1": self._retrieval_l1_hits / max(1, self._retrieval_total),
                        "l2": self._retrieval_l2_hits / max(1, self._retrieval_total),
                        "l3": self._retrieval_l3_hits / max(1, self._retrieval_total),
                    },
                },
                
                "compression": {
                    "attempts": self._compression_attempts,
                    "successes": self._compression_successes,
                    "failures": self._compression_failures,
                    "success_rate": self._compression_successes / max(1, self._compression_attempts),
                    "latency_ms": self._calculate_percentiles(self._compression_latencies),
                },
                
                "filter": {
                    "total": self._filter_total,
                    "discarded": self._filter_discarded,
                    "sqlite_only": self._filter_sqlite_only,
                    "discard_rate": self._filter_discarded / max(1, self._filter_total),
                    "sqlite_only_rate": self._filter_sqlite_only / max(1, self._filter_total),
                },
                
                "embedding": {
                    "latency_ms": self._calculate_percentiles(self._embedding_latencies),
                },
                
                "queue": {
                    "wait_time_ms": self._calculate_percentiles(self._queue_wait_times),
                },
            }
    
    def get_summary(self) -> str:
        """获取指标摘要文本"""
        metrics = self.get_metrics()
        
        lines = [
            f"=== 系统指标 (运行时间: {metrics['uptime_human']}) ===",
            "",
            "【检索性能】",
            f"  总请求数: {metrics['retrieval']['total_requests']}",
            f"  延迟 P50: {metrics['retrieval']['latency_ms']['p50']:.1f}ms",
            f"  延迟 P95: {metrics['retrieval']['latency_ms']['p95']:.1f}ms",
            f"  延迟 P99: {metrics['retrieval']['latency_ms']['p99']:.1f}ms",
            f"  L1命中率: {metrics['retrieval']['hit_rate']['l1']*100:.1f}%",
            f"  L2命中率: {metrics['retrieval']['hit_rate']['l2']*100:.1f}%",
            f"  L3命中率: {metrics['retrieval']['hit_rate']['l3']*100:.1f}%",
            "",
            "【压缩统计】",
            f"  成功率: {metrics['compression']['success_rate']*100:.1f}%",
            f"  成功/失败: {metrics['compression']['successes']}/{metrics['compression']['failures']}",
            f"  延迟 P50: {metrics['compression']['latency_ms']['p50']:.1f}ms",
            "",
            "【过滤统计】",
            f"  丢弃率: {metrics['filter']['discard_rate']*100:.1f}%",
            f"  仅SQLite率: {metrics['filter']['sqlite_only_rate']*100:.1f}%",
            f"  已过滤: {metrics['filter']['total']} 条",
            "",
            "【嵌入计算】",
            f"  延迟 P50: {metrics['embedding']['latency_ms']['p50']:.1f}ms",
            f"  延迟 P95: {metrics['embedding']['latency_ms']['p95']:.1f}ms",
        ]
        
        return "\n".join(lines)
    
    def _format_uptime(self, seconds: float) -> str:
        """格式化运行时间"""
        if seconds < 60:
            return f"{seconds:.0f}秒"
        elif seconds < 3600:
            return f"{seconds/60:.1f}分钟"
        elif seconds < 86400:
            return f"{seconds/3600:.1f}小时"
        else:
            return f"{seconds/86400:.1f}天"
    
    def reset(self):
        """重置所有指标"""
        with self._lock:
            self._retrieval_latencies.clear()
            self._embedding_latencies.clear()
            self._compression_latencies.clear()
            self._queue_wait_times.clear()
            
            self._compression_attempts = 0
            self._compression_successes = 0
            self._compression_failures = 0
            
            self._filter_total = 0
            self._filter_discarded = 0
            self._filter_sqlite_only = 0
            
            self._retrieval_total = 0
            self._retrieval_l1_hits = 0
            self._retrieval_l2_hits = 0
            self._retrieval_l3_hits = 0
            
            self._last_reset = time.time()


class MetricsTimer:
    """计时上下文管理器"""
    
    def __init__(self, collector: MetricsCollector, metric_type: str):
        self._collector = collector
        self._metric_type = metric_type
        self._start_time = 0
    
    def __enter__(self):
        self._start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        duration_ms = (time.perf_counter() - self._start_time) * 1000
        
        if self._metric_type == "retrieval":
            self._collector.record_retrieval_latency(duration_ms)
        elif self._metric_type == "embedding":
            self._collector.record_embedding_latency(duration_ms)
        elif self._metric_type == "compression":
            pass


_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """获取全局指标收集器"""
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector


def time_retrieval():
    """检索计时上下文管理器"""
    return MetricsTimer(get_metrics_collector(), "retrieval")


def time_embedding():
    """嵌入计算计时上下文管理器"""
    return MetricsTimer(get_metrics_collector(), "embedding")
