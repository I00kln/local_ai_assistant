"""
记忆归档模块

功能：
- 冷热数据分离
- 长期记忆归档
- 数据导出与恢复
- 不影响原有记忆流转机制

使用方式：
- 独立运行，可选启用
- 通过配置设置归档策略
- 手动触发或定时执行
"""

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from memory_tags import MemoryTags


@dataclass
class ArchiveConfig:
    """归档配置"""
    enabled: bool = False
    archive_days: int = 180
    archive_weight_threshold: float = 0.3
    export_path: str = "archive"
    keep_metadata: bool = True
    compress_export: bool = True
    max_archive_size: int = 10000


@dataclass
class ArchiveStats:
    """归档统计"""
    total_archived: int = 0
    total_exported: int = 0
    last_archive_time: Optional[str] = None
    archive_size_bytes: int = 0


class MemoryArchiver:
    """
    记忆归档器
    
    独立模块，不影响原有记忆流转：
    - L1 → L2 → L3 → 压缩 → 遗忘（原有流程不变）
    - 归档是额外的冷数据管理
    
    归档策略：
    1. 时间阈值：超过 N 天的记忆
    2. 权重阈值：权重低于 M 的记忆
    3. 访问频率：长期未访问的记忆
    4. 保护机制：重要记忆不归档
    """
    
    _instance: Optional['MemoryArchiver'] = None
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
        self.config = ArchiveConfig()
        self.stats = ArchiveStats()
        self._archive_lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        self._init_archive_storage()
    
    def _init_archive_storage(self):
        """初始化归档存储"""
        archive_path = Path(self.config.export_path)
        archive_path.mkdir(parents=True, exist_ok=True)
        
        archive_db_path = archive_path / "archive.db"
        self.archive_db_path = str(archive_db_path)
        
        if not archive_db_path.exists():
            self._create_archive_database()
    
    def _create_archive_database(self):
        """创建归档数据库"""
        with sqlite3.connect(self.archive_db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS archived_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_id INTEGER,
                    text TEXT NOT NULL,
                    compressed_text TEXT,
                    source TEXT,
                    weight REAL,
                    access_count INTEGER,
                    original_created_time TEXT,
                    archived_time TEXT,
                    metadata TEXT,
                    archive_reason TEXT
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_archived_time 
                ON archived_memories(archived_time DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_original_id 
                ON archived_memories(original_id)
            """)
            
            conn.commit()
    
    def configure(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def get_archive_candidates(self, sqlite_store) -> List[Dict]:
        """
        获取归档候选记录
        
        条件：
        - 超过归档天数
        - 权重低于阈值
        - 非重要记忆
        - 非活跃记忆
        """
        threshold_time = (datetime.now() - timedelta(days=self.config.archive_days)).isoformat()
        
        candidates = []
        
        try:
            with sqlite_store._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute(f"""
                    SELECT id, text, compressed_text, source, weight, 
                           access_count, created_time, metadata
                    FROM memories
                    WHERE created_time < ?
                    AND weight < ?
                    AND is_archived = 0
                    AND (metadata IS NULL OR metadata NOT LIKE ?)
                    AND (metadata IS NULL OR metadata NOT LIKE ?)
                    ORDER BY weight ASC, created_time ASC
                    LIMIT ?
                """, (
                    threshold_time,
                    self.config.archive_weight_threshold,
                    f'%"{MemoryTags.IMPORTANT}"%',
                    f'%"{MemoryTags.PROTECTED}"%',
                    self.config.max_archive_size
                ))
                
                for row in cursor.fetchall():
                    candidates.append({
                        "id": row[0],
                        "text": row[1],
                        "compressed_text": row[2],
                        "source": row[3],
                        "weight": row[4],
                        "access_count": row[5],
                        "created_time": row[6],
                        "metadata": json.loads(row[7]) if row[7] else {}
                    })
        
        except Exception as e:
            print(f"[归档] 获取候选记录失败: {e}")
        
        return candidates
    
    def archive_memories(self, sqlite_store, vector_store=None) -> Dict[str, int]:
        """
        执行归档操作
        
        Args:
            sqlite_store: SQLite 存储实例
            vector_store: 向量存储实例（可选，用于删除向量）
        
        Returns:
            {"archived": 归档数量, "errors": 错误数量}
        """
        if not self.config.enabled:
            return {"archived": 0, "errors": 0, "message": "归档未启用"}
        
        with self._archive_lock:
            candidates = self.get_archive_candidates(sqlite_store)
            
            if not candidates:
                return {"archived": 0, "errors": 0, "message": "无归档候选"}
            
            archived_count = 0
            error_count = 0
            vector_ids_to_delete = []
            
            try:
                with sqlite3.connect(self.archive_db_path) as archive_conn:
                    archive_cursor = archive_conn.cursor()
                    
                    for record in candidates:
                        try:
                            archive_cursor.execute("""
                                INSERT INTO archived_memories 
                                (original_id, text, compressed_text, source, weight,
                                 access_count, original_created_time, archived_time, 
                                 metadata, archive_reason)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                record["id"],
                                record["text"],
                                record.get("compressed_text"),
                                record.get("source"),
                                record.get("weight"),
                                record.get("access_count"),
                                record.get("created_time"),
                                datetime.now().isoformat(),
                                json.dumps(record.get("metadata", {}), ensure_ascii=False),
                                "auto_archive"
                            ))
                            
                            with sqlite_store._get_connection() as conn:
                                cursor = conn.cursor()
                                cursor.execute("""
                                    SELECT vector_id FROM memories WHERE id = ?
                                """, (record["id"],))
                                result = cursor.fetchone()
                                if result and result[0]:
                                    vector_ids_to_delete.append(result[0])
                            
                            archived_count += 1
                            
                        except Exception as e:
                            print(f"[归档] 归档记录 {record['id']} 失败: {e}")
                            error_count += 1
                    
                    archive_conn.commit()
                
                if vector_ids_to_delete and vector_store:
                    try:
                        vector_store.delete(ids=vector_ids_to_delete)
                    except Exception as e:
                        print(f"[归档] 删除向量失败: {e}")
                
                with sqlite_store.lock:
                    with sqlite_store._get_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                            UPDATE memories 
                            SET is_archived = 1,
                                vector_id = NULL
                            WHERE id IN ({})
                        """.format(",".join(str(c["id"]) for c in candidates[:archived_count])))
                        conn.commit()
                
                self.stats.total_archived += archived_count
                self.stats.last_archive_time = datetime.now().isoformat()
                
                print(f"[归档] 完成: 归档 {archived_count} 条, 错误 {error_count} 条")
                
                return {"archived": archived_count, "errors": error_count}
                
            except Exception as e:
                print(f"[归档] 归档操作失败: {e}")
                return {"archived": 0, "errors": 1, "message": str(e)}
    
    def export_archive(self, output_path: str = None) -> Dict[str, Any]:
        """
        导出归档数据
        
        Args:
            output_path: 导出路径（默认为 archive/export_时间戳.json）
        
        Returns:
            {"exported": 导出数量, "path": 导出路径}
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.config.export_path, f"export_{timestamp}.json")
        
        try:
            with sqlite3.connect(self.archive_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM archived_memories ORDER BY archived_time DESC")
                rows = cursor.fetchall()
            
            export_data = {
                "export_time": datetime.now().isoformat(),
                "total_records": len(rows),
                "records": []
            }
            
            for row in rows:
                export_data["records"].append({
                    "id": row[0],
                    "original_id": row[1],
                    "text": row[2],
                    "compressed_text": row[3],
                    "source": row[4],
                    "weight": row[5],
                    "access_count": row[6],
                    "original_created_time": row[7],
                    "archived_time": row[8],
                    "metadata": json.loads(row[9]) if row[9] else {},
                    "archive_reason": row[10]
                })
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            self.stats.total_exported += len(rows)
            
            print(f"[归档] 导出完成: {len(rows)} 条记录 → {output_path}")
            
            return {"exported": len(rows), "path": output_path}
            
        except Exception as e:
            print(f"[归档] 导出失败: {e}")
            return {"exported": 0, "error": str(e)}
    
    def import_archive(self, input_path: str, sqlite_store) -> Dict[str, int]:
        """
        导入归档数据（恢复记忆）
        
        Args:
            input_path: 导入文件路径
            sqlite_store: SQLite 存储实例
        
        Returns:
            {"imported": 导入数量, "errors": 错误数量}
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            records = import_data.get("records", [])
            imported_count = 0
            error_count = 0
            
            for record in records:
                try:
                    from sqlite_store import MemoryRecord
                    
                    memory_record = MemoryRecord(
                        text=record["text"],
                        compressed_text=record.get("compressed_text"),
                        source=record.get("source", "archive"),
                        weight=record.get("weight", 0.5),
                        metadata=record.get("metadata", {})
                    )
                    memory_record.created_time = record.get("original_created_time")
                    
                    sqlite_store.add(memory_record)
                    imported_count += 1
                    
                except Exception as e:
                    print(f"[归档] 导入记录失败: {e}")
                    error_count += 1
            
            print(f"[归档] 导入完成: {imported_count} 条, 错误 {error_count} 条")
            
            return {"imported": imported_count, "errors": error_count}
            
        except Exception as e:
            print(f"[归档] 导入失败: {e}")
            return {"imported": 0, "errors": 1, "error": str(e)}
    
    def get_archive_stats(self) -> Dict[str, Any]:
        """获取归档统计"""
        stats = {
            "enabled": self.config.enabled,
            "archive_days": self.config.archive_days,
            "weight_threshold": self.config.archive_weight_threshold,
            "total_archived": self.stats.total_archived,
            "total_exported": self.stats.total_exported,
            "last_archive_time": self.stats.last_archive_time,
        }
        
        try:
            if os.path.exists(self.archive_db_path):
                stats["archive_size_bytes"] = os.path.getsize(self.archive_db_path)
                
                with sqlite3.connect(self.archive_db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM archived_memories")
                    stats["archive_count"] = cursor.fetchone()[0]
        except Exception:
            stats["archive_count"] = 0
        
        return stats
    
    def start_auto_archive(self, interval_hours: int = 24):
        """
        启动自动归档线程
        
        Args:
            interval_hours: 归档间隔（小时）
        """
        if self._running:
            return
        
        self._running = True
        
        def archive_loop():
            from sqlite_store import get_sqlite_store
            from vector_store import get_vector_store
            
            sqlite = get_sqlite_store()
            vector = get_vector_store()
            
            while self._running:
                try:
                    self.archive_memories(sqlite, vector)
                    
                except Exception as e:
                    print(f"[归档] 自动归档失败: {e}")
                
                for _ in range(interval_hours * 3600):
                    if not self._running:
                        break
                    time.sleep(1)
        
        self._thread = threading.Thread(target=archive_loop, daemon=True)
        self._thread.start()
        print(f"[归档] 自动归档已启动，间隔 {interval_hours} 小时")
    
    def stop_auto_archive(self):
        """停止自动归档"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        print("[归档] 自动归档已停止")
    
    def clear_archive(self):
        """清空归档数据"""
        try:
            if os.path.exists(self.archive_db_path):
                os.remove(self.archive_db_path)
            self._create_archive_database()
            self.stats = ArchiveStats()
            print("[归档] 归档数据已清空")
        except Exception as e:
            print(f"[归档] 清空失败: {e}")


_archiver: Optional[MemoryArchiver] = None


def get_archiver() -> MemoryArchiver:
    """获取全局归档器实例"""
    global _archiver
    if _archiver is None:
        _archiver = MemoryArchiver()
    return _archiver
