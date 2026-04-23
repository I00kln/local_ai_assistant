#!/usr/bin/env python
# system_health_checklist.py
# 系统健康检查清单
"""
检查项：
1. 关键进程状态（llama.cpp, ChromaDB 持久化目录可写）
2. 队列积压阈值（pending_queue_size < 100）
3. 压缩成功率（> 80%）
4. 检索延迟 P95（< 200ms）
5. 磁盘剩余空间（> 10%）
6. SQLite 数据库完整性（PRAGMA integrity_check）

运行方式：
    python system_health_checklist.py
"""

import os
import sys
import time
import shutil
import sqlite3
import threading
from datetime import datetime
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum


class CheckStatus(Enum):
    """检查状态"""
    PASS = "✅ PASS"
    WARN = "⚠️ WARN"
    FAIL = "❌ FAIL"
    SKIP = "⏭️ SKIP"
    ERROR = "🔴 ERROR"


@dataclass
class CheckResult:
    """检查结果"""
    name: str
    status: CheckStatus
    value: Any
    threshold: Any
    message: str
    details: Dict[str, Any]


class SystemHealthChecklist:
    """
    系统健康检查清单
    
    检查项：
    1. 关键进程状态
    2. 队列积压阈值
    3. 压缩成功率
    4. 检索延迟 P95
    5. 磁盘剩余空间
    6. SQLite 数据库完整性
    """
    
    def __init__(self):
        self.results: List[CheckResult] = []
        self.start_time = time.time()
        
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.chroma_dir = os.path.join(self.project_root, "chroma_db")
        self.sqlite_db_path = os.path.join(self.project_root, "memory.db")
        
        self._init_modules()
    
    def _init_modules(self):
        """初始化模块引用"""
        self.config = None
        self.metrics = None
        self.sqlite_store = None
        self.vector_store = None
        self.async_processor = None
        
        try:
            from config import config
            self.config = config
            self.chroma_dir = getattr(config, 'chroma_persist_dir', self.chroma_dir)
        except Exception:
            pass
        
        try:
            from metrics import get_metrics_collector
            self.metrics = get_metrics_collector()
        except Exception:
            pass
        
        try:
            from sqlite_store import get_sqlite_store
            self.sqlite_store = get_sqlite_store()
        except Exception:
            pass
        
        try:
            from vector_store import get_vector_store
            self.vector_store = get_vector_store()
        except Exception:
            pass
        
        try:
            from async_processor import AsyncMemoryProcessor
            if AsyncMemoryProcessor._instance:
                self.async_processor = AsyncMemoryProcessor._instance
        except Exception:
            pass
    
    def run_all_checks(self) -> Dict[str, Any]:
        """运行所有检查"""
        print("=" * 60)
        print("系统健康检查清单")
        print("=" * 60)
        print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 60)
        
        self._check_1_critical_processes()
        self._check_2_queue_backlog()
        self._check_3_compression_success_rate()
        self._check_4_retrieval_latency_p95()
        self._check_5_disk_space()
        self._check_6_sqlite_integrity()
        
        return self._generate_report()
    
    def _check_1_critical_processes(self):
        """
        检查 1: 关键进程状态
        
        检查项：
        - llama.cpp 进程（如果启用）
        - ChromaDB 持久化目录可写
        - ONNX 模型可用性
        """
        print("\n[1/6] 关键进程状态检查")
        print("-" * 40)
        
        llama_status = CheckStatus.SKIP
        llama_message = "llama.cpp 未启用"
        
        if self.config:
            try:
                if hasattr(self.config, 'llm_enabled') and self.config.llm_enabled:
                    from llm_client import LlamaClient
                    client = LlamaClient()
                    if client.check_connection():
                        llama_status = CheckStatus.PASS
                        llama_message = "llama.cpp 进程运行正常"
                    else:
                        llama_status = CheckStatus.WARN
                        llama_message = "llama.cpp 进程未响应（可能使用云端）"
            except Exception as e:
                llama_status = CheckStatus.WARN
                llama_message = f"llama.cpp 检查失败: {str(e)[:50]}"
        
        print(f"  - llama.cpp: {llama_status.value} - {llama_message}")
        
        chroma_status = CheckStatus.FAIL
        chroma_message = "ChromaDB 目录不存在"
        chroma_writable = False
        
        try:
            if os.path.exists(self.chroma_dir):
                test_file = os.path.join(self.chroma_dir, ".write_test")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                chroma_writable = True
                chroma_status = CheckStatus.PASS
                chroma_message = "ChromaDB 目录可写"
            else:
                os.makedirs(self.chroma_dir, exist_ok=True)
                test_file = os.path.join(self.chroma_dir, ".write_test")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                chroma_writable = True
                chroma_status = CheckStatus.PASS
                chroma_message = "ChromaDB 目录已创建且可写"
        except PermissionError:
            chroma_status = CheckStatus.FAIL
            chroma_message = "ChromaDB 目录无写入权限"
        except Exception as e:
            chroma_status = CheckStatus.ERROR
            chroma_message = f"ChromaDB 检查失败: {str(e)[:50]}"
        
        print(f"  - ChromaDB 目录: {chroma_status.value} - {chroma_message}")
        
        onnx_status = CheckStatus.SKIP
        onnx_message = "ONNX 模型检查跳过"
        onnx_available = False
        
        try:
            from embedding_service import get_embedding_service
            service = get_embedding_service()
            if service.is_available:
                onnx_status = CheckStatus.PASS
                onnx_message = "ONNX 模型加载成功"
                onnx_available = True
            elif service.is_fallback:
                onnx_status = CheckStatus.WARN
                onnx_message = "ONNX 模型处于降级模式（使用随机向量）"
            else:
                onnx_status = CheckStatus.FAIL
                onnx_message = "ONNX 模型不可用"
        except Exception as e:
            onnx_status = CheckStatus.WARN
            onnx_message = f"ONNX 检查失败: {str(e)[:50]}"
        
        print(f"  - ONNX 模型: {onnx_status.value} - {onnx_message}")
        
        overall = CheckStatus.PASS
        if chroma_status in [CheckStatus.FAIL, CheckStatus.ERROR]:
            overall = CheckStatus.FAIL
        elif llama_status == CheckStatus.WARN or onnx_status == CheckStatus.WARN:
            overall = CheckStatus.WARN
        
        self.results.append(CheckResult(
            name="关键进程状态",
            status=overall,
            value={
                "llama_status": llama_status.value,
                "chroma_writable": chroma_writable,
                "onnx_available": onnx_available
            },
            threshold="所有关键进程可用",
            message=f"llama: {llama_message}; ChromaDB: {chroma_message}; ONNX: {onnx_message}",
            details={}
        ))
    
    def _check_2_queue_backlog(self):
        """
        检查 2: 队列积压阈值
        
        阈值: pending_queue_size < 100
        """
        print("\n[2/6] 队列积压阈值检查")
        print("-" * 40)
        
        queue_size = 0
        queue_status = CheckStatus.SKIP
        queue_message = "异步处理器未初始化"
        
        if self.async_processor:
            try:
                queue_size = self.async_processor.pending_queue.qsize()
                max_queue = 100
                
                if queue_size < max_queue:
                    queue_status = CheckStatus.PASS
                    queue_message = f"队列积压正常: {queue_size} < {max_queue}"
                elif queue_size < max_queue * 2:
                    queue_status = CheckStatus.WARN
                    queue_message = f"队列积压警告: {queue_size} >= {max_queue}"
                else:
                    queue_status = CheckStatus.FAIL
                    queue_message = f"队列积压严重: {queue_size} >= {max_queue * 2}"
            except Exception as e:
                queue_status = CheckStatus.ERROR
                queue_message = f"队列检查失败: {str(e)[:50]}"
        
        print(f"  - 队列大小: {queue_size}")
        print(f"  - 状态: {queue_status.value} - {queue_message}")
        
        self.results.append(CheckResult(
            name="队列积压阈值",
            status=queue_status,
            value=queue_size,
            threshold="< 100",
            message=queue_message,
            details={"max_threshold": 100}
        ))
    
    def _check_3_compression_success_rate(self):
        """
        检查 3: 压缩成功率
        
        阈值: > 80%
        """
        print("\n[3/6] 压缩成功率检查")
        print("-" * 40)
        
        success_rate = 0.0
        total = 0
        success = 0
        compression_status = CheckStatus.SKIP
        compression_message = "无压缩统计数据"
        
        if self.metrics:
            try:
                stats = self.metrics.get_metrics()
                compression_stats = stats.get("compression", {})
                total = compression_stats.get("attempts", 0)
                success = compression_stats.get("successes", 0)
                
                if total > 0:
                    success_rate = (success / total) * 100
                    threshold = 80.0
                    
                    if success_rate >= threshold:
                        compression_status = CheckStatus.PASS
                        compression_message = f"压缩成功率正常: {success_rate:.1f}% >= {threshold}%"
                    elif success_rate >= threshold * 0.5:
                        compression_status = CheckStatus.WARN
                        compression_message = f"压缩成功率偏低: {success_rate:.1f}% < {threshold}%"
                    else:
                        compression_status = CheckStatus.FAIL
                        compression_message = f"压缩成功率过低: {success_rate:.1f}%"
            except Exception as e:
                compression_status = CheckStatus.ERROR
                compression_message = f"压缩统计获取失败: {str(e)[:50]}"
        
        print(f"  - 压缩尝试: {total}")
        print(f"  - 成功次数: {success}")
        print(f"  - 成功率: {success_rate:.1f}%")
        print(f"  - 状态: {compression_status.value} - {compression_message}")
        
        self.results.append(CheckResult(
            name="压缩成功率",
            status=compression_status,
            value=f"{success_rate:.1f}%",
            threshold="> 80%",
            message=compression_message,
            details={"total": total, "success": success}
        ))
    
    def _check_4_retrieval_latency_p95(self):
        """
        检查 4: 检索延迟 P95
        
        阈值: < 200ms
        """
        print("\n[4/6] 检索延迟 P95 检查")
        print("-" * 40)
        
        p95_latency = 0.0
        latency_status = CheckStatus.SKIP
        latency_message = "无检索延迟数据"
        
        if self.metrics:
            try:
                stats = self.metrics.get_metrics()
                retrieval_stats = stats.get("retrieval", {})
                latency_stats = retrieval_stats.get("latency_ms", {})
                p95_latency = latency_stats.get("p95", 0)
                
                if p95_latency > 0:
                    threshold = 200.0
                    
                    if p95_latency < threshold:
                        latency_status = CheckStatus.PASS
                        latency_message = f"P95 延迟正常: {p95_latency:.1f}ms < {threshold}ms"
                    elif p95_latency < threshold * 2:
                        latency_status = CheckStatus.WARN
                        latency_message = f"P95 延迟偏高: {p95_latency:.1f}ms >= {threshold}ms"
                    else:
                        latency_status = CheckStatus.FAIL
                        latency_message = f"P95 延迟过高: {p95_latency:.1f}ms >= {threshold * 2}ms"
                else:
                    latency_status = CheckStatus.SKIP
                    latency_message = "暂无检索延迟数据"
            except Exception as e:
                latency_status = CheckStatus.ERROR
                latency_message = f"延迟统计获取失败: {str(e)[:50]}"
        
        print(f"  - P95 延迟: {p95_latency:.1f}ms")
        print(f"  - 状态: {latency_status.value} - {latency_message}")
        
        self.results.append(CheckResult(
            name="检索延迟 P95",
            status=latency_status,
            value=f"{p95_latency:.1f}ms",
            threshold="< 200ms",
            message=latency_message,
            details={}
        ))
    
    def _check_5_disk_space(self):
        """
        检查 5: 磁盘剩余空间
        
        阈值: > 10%
        """
        print("\n[5/6] 磁盘剩余空间检查")
        print("-" * 40)
        
        disk_status = CheckStatus.ERROR
        disk_message = "磁盘检查失败"
        free_percent = 0.0
        total_gb = 0
        free_gb = 0
        
        try:
            disk_usage = shutil.disk_usage(self.project_root)
            total_gb = disk_usage.total / (1024 ** 3)
            free_gb = disk_usage.free / (1024 ** 3)
            free_percent = (disk_usage.free / disk_usage.total) * 100
            
            threshold = 10.0
            
            if free_percent >= threshold:
                disk_status = CheckStatus.PASS
                disk_message = f"磁盘空间充足: {free_percent:.1f}% >= {threshold}%"
            elif free_percent >= threshold * 0.5:
                disk_status = CheckStatus.WARN
                disk_message = f"磁盘空间偏低: {free_percent:.1f}% < {threshold}%"
            else:
                disk_status = CheckStatus.FAIL
                disk_message = f"磁盘空间不足: {free_percent:.1f}%"
        except Exception as e:
            disk_status = CheckStatus.ERROR
            disk_message = f"磁盘检查失败: {str(e)[:50]}"
        
        print(f"  - 总空间: {total_gb:.1f} GB")
        print(f"  - 剩余空间: {free_gb:.1f} GB ({free_percent:.1f}%)")
        print(f"  - 状态: {disk_status.value} - {disk_message}")
        
        self.results.append(CheckResult(
            name="磁盘剩余空间",
            status=disk_status,
            value=f"{free_percent:.1f}%",
            threshold="> 10%",
            message=disk_message,
            details={"total_gb": f"{total_gb:.1f}", "free_gb": f"{free_gb:.1f}"}
        ))
    
    def _check_6_sqlite_integrity(self):
        """
        检查 6: SQLite 数据库完整性
        
        使用 PRAGMA integrity_check
        """
        print("\n[6/6] SQLite 数据库完整性检查")
        print("-" * 40)
        
        integrity_status = CheckStatus.SKIP
        integrity_message = "SQLite 数据库不存在"
        integrity_result = "N/A"
        
        if os.path.exists(self.sqlite_db_path):
            try:
                conn = sqlite3.connect(self.sqlite_db_path)
                cursor = conn.cursor()
                cursor.execute("PRAGMA integrity_check")
                integrity_result = cursor.fetchone()[0]
                conn.close()
                
                if integrity_result == "ok":
                    integrity_status = CheckStatus.PASS
                    integrity_message = "数据库完整性检查通过"
                else:
                    integrity_status = CheckStatus.FAIL
                    integrity_message = f"数据库完整性问题: {integrity_result[:50]}"
            except Exception as e:
                integrity_status = CheckStatus.ERROR
                integrity_message = f"完整性检查失败: {str(e)[:50]}"
        
        print(f"  - 数据库路径: {self.sqlite_db_path}")
        print(f"  - 完整性结果: {integrity_result}")
        print(f"  - 状态: {integrity_status.value} - {integrity_message}")
        
        self.results.append(CheckResult(
            name="SQLite 数据库完整性",
            status=integrity_status,
            value=integrity_result,
            threshold="ok",
            message=integrity_message,
            details={"db_path": self.sqlite_db_path}
        ))
    
    def _generate_report(self) -> Dict[str, Any]:
        """生成报告"""
        elapsed = time.time() - self.start_time
        
        pass_count = sum(1 for r in self.results if r.status == CheckStatus.PASS)
        warn_count = sum(1 for r in self.results if r.status == CheckStatus.WARN)
        fail_count = sum(1 for r in self.results if r.status == CheckStatus.FAIL)
        skip_count = sum(1 for r in self.results if r.status == CheckStatus.SKIP)
        error_count = sum(1 for r in self.results if r.status == CheckStatus.ERROR)
        
        if fail_count > 0 or error_count > 0:
            overall_status = CheckStatus.FAIL
        elif warn_count > 0:
            overall_status = CheckStatus.WARN
        elif skip_count == len(self.results):
            overall_status = CheckStatus.SKIP
        else:
            overall_status = CheckStatus.PASS
        
        print("\n" + "=" * 60)
        print("检查结果汇总")
        print("=" * 60)
        print(f"  通过: {pass_count}")
        print(f"  警告: {warn_count}")
        print(f"  失败: {fail_count}")
        print(f"  跳过: {skip_count}")
        print(f"  错误: {error_count}")
        print(f"  总体状态: {overall_status.value}")
        print(f"  检查耗时: {elapsed:.2f}s")
        print("=" * 60)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status.value,
            "summary": {
                "pass": pass_count,
                "warn": warn_count,
                "fail": fail_count,
                "skip": skip_count,
                "error": error_count
            },
            "elapsed_seconds": round(elapsed, 2),
            "checks": [
                {
                    "name": r.name,
                    "status": r.status.value,
                    "value": r.value,
                    "threshold": r.threshold,
                    "message": r.message,
                    "details": r.details
                }
                for r in self.results
            ]
        }
        
        report_path = os.path.join(self.project_root, "health_check_report.json")
        try:
            import json
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"\n报告已保存: {report_path}")
        except Exception as e:
            print(f"\n报告保存失败: {e}")
        
        return report


def main():
    """主入口"""
    checklist = SystemHealthChecklist()
    report = checklist.run_all_checks()
    
    if report["overall_status"] in ["❌ FAIL", "🔴 ERROR"]:
        sys.exit(1)
    elif report["overall_status"] == "⚠️ WARN":
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
