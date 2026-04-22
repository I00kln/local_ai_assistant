"""
记忆合并模块

功能：
- 检测相似记忆
- 合并相似条目
- 触发条件判断

触发时机：
1. L2 相似条目过多时（> 阈值）
2. L3→L2 回流时检测相似记忆

压缩时机：
1. L2→L3 迁移时
2. L2 中 token 过长时

原有压缩、归档、遗忘逻辑不变，此模块作为补充。
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from memory_tags import MemoryTags, MemoryTagHelper
from logger import get_logger


@dataclass
class MergeConfig:
    """合并配置"""
    enabled: bool = True
    similarity_threshold: float = 0.92
    min_group_size: int = 3
    max_merged_length: int = 500
    max_merge_batch: int = 20
    merge_interval_hours: int = 6


@dataclass
class MergeStats:
    """合并统计"""
    total_merged: int = 0
    total_groups: int = 0
    last_merge_time: Optional[str] = None
    by_trigger: Dict[str, int] = field(default_factory=dict)


class MemoryMerger:
    """
    记忆合并器
    
    功能：
    - 检测相似记忆组
    - 合并相似条目
    - 控制合并时机
    
    触发条件：
    - L2 相似条目 > 阈值
    - L3→L2 回流时
    
    不影响原有压缩、归档、遗忘逻辑。
    """
    
    _instance: Optional['MemoryMerger'] = None
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
        self.config = MergeConfig()
        self.stats = MergeStats()
        self._merge_lock = threading.RLock()
        self._embedding_service = None
        self._log = get_logger()
        self._last_merge_check = 0
    
    def configure(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def _ensure_embedding_service(self):
        """确保 EmbeddingService 已初始化"""
        if self._embedding_service is None:
            from embedding_service import get_embedding_service
            self._embedding_service = get_embedding_service()
    
    def should_trigger_merge(
        self, 
        memories: List[Dict], 
        trigger: str = "l2_check"
    ) -> Tuple[bool, str]:
        """
        判断是否应该触发合并
        
        Args:
            memories: 待检查的记忆列表
            trigger: 触发来源 ("l2_check" | "l3_to_l2" | "manual")
        
        Returns:
            (是否触发, 原因)
        """
        if not self.config.enabled:
            return False, "合并功能未启用"
        
        if len(memories) < self.config.min_group_size:
            return False, f"记忆数量不足（{len(memories)} < {self.config.min_group_size}）"
        
        similar_groups = self._find_similar_groups(memories)
        
        large_groups = [g for g in similar_groups if len(g) >= self.config.min_group_size]
        
        if large_groups:
            return True, f"发现 {len(large_groups)} 组相似记忆（共 {sum(len(g) for g in large_groups)} 条）"
        
        return False, "未发现需要合并的相似记忆组"
    
    def _find_similar_groups(self, memories: List[Dict]) -> List[List[Dict]]:
        """
        使用向量相似度聚类相似记忆
        
        Args:
            memories: 记忆列表
        
        Returns:
            相似记忆分组列表
        """
        if len(memories) < 2:
            return [[m] for m in memories]
        
        self._ensure_embedding_service()
        
        if not self._embedding_service.is_available:
            return [[m] for m in memories]
        
        vectors = []
        valid_memories = []
        
        for m in memories:
            text = m.get("text", "")
            if text:
                try:
                    vec = self._embedding_service.embed_single(text)
                    vectors.append(vec)
                    valid_memories.append(m)
                except Exception:
                    continue
        
        if len(vectors) < 2:
            return [[m] for m in valid_memories]
        
        vectors = np.array(vectors)
        similarity_matrix = np.dot(vectors, vectors.T)
        
        groups = []
        used = set()
        
        for i in range(len(valid_memories)):
            if i in used:
                continue
            
            group = [valid_memories[i]]
            used.add(i)
            
            for j in range(i + 1, len(valid_memories)):
                if j in used:
                    continue
                if similarity_matrix[i][j] >= self.config.similarity_threshold:
                    group.append(valid_memories[j])
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    def merge_memories(
        self, 
        memories: List[Dict], 
        trigger: str = "manual"
    ) -> Dict[str, Any]:
        """
        合并相似记忆
        
        Args:
            memories: 待合并的记忆列表
            trigger: 触发来源
        
        Returns:
            {
                "merged_count": 合并数量,
                "groups": 分组数,
                "merged_memories": 合并后的记忆列表,
                "deleted_ids": 需要删除的记忆ID列表
            }
        """
        if not self.config.enabled:
            return {"merged_count": 0, "message": "合并功能未启用"}
        
        with self._merge_lock:
            similar_groups = self._find_similar_groups(memories)
            
            large_groups = [g for g in similar_groups if len(g) >= self.config.min_group_size]
            
            if not large_groups:
                return {"merged_count": 0, "message": "未发现需要合并的相似记忆组"}
            
            merged_memories = []
            deleted_ids = []
            merged_count = 0
            
            for group in large_groups[:self.config.max_merge_batch]:
                merged = self._merge_group(group)
                if merged:
                    merged_memories.append(merged)
                    deleted_ids.extend([m.get("id") for m in group[1:] if m.get("id")])
                    merged_count += len(group) - 1
            
            self.stats.total_merged += merged_count
            self.stats.total_groups += len(large_groups)
            self.stats.last_merge_time = datetime.now().isoformat()
            self.stats.by_trigger[trigger] = self.stats.by_trigger.get(trigger, 0) + merged_count
            
            self._log.info(
                "MEMORY_MERGE_COMPLETE",
                merged_count=merged_count,
                groups=len(large_groups),
                trigger=trigger
            )
            
            return {
                "merged_count": merged_count,
                "groups": len(large_groups),
                "merged_memories": merged_memories,
                "deleted_ids": deleted_ids
            }
    
    def _merge_group(self, group: List[Dict]) -> Optional[Dict]:
        """
        合并一组相似记忆
        
        策略：
        1. 保留最早的创建时间
        2. 合并权重（取最大值）
        3. 智能合并文本
        4. 合并元数据
        """
        if len(group) == 1:
            return group[0]
        
        sorted_group = sorted(group, key=lambda x: x.get("created_time", ""))
        
        merged_text = self._smart_merge_texts([m.get("text", "") for m in sorted_group])
        
        if not merged_text:
            return None
        
        all_metadata = {}
        for m in sorted_group:
            metadata = m.get("metadata", {})
            if isinstance(metadata, dict):
                all_metadata.update(metadata)
        
        all_metadata[MemoryTags.MERGED] = True
        all_metadata["merged_count"] = len(group)
        
        merged = {
            "text": merged_text,
            "created_time": sorted_group[0].get("created_time"),
            "weight": max(m.get("weight", 1.0) for m in group),
            "access_count": sum(m.get("access_count", 0) for m in group),
            "metadata": all_metadata,
            "source": "merged",
            "original_ids": [m.get("id") for m in group if m.get("id")]
        }
        
        return merged
    
    def _smart_merge_texts(self, texts: List[str]) -> str:
        """
        智能合并文本
        
        策略：
        1. 找到公共部分
        2. 合并差异部分
        3. 控制总长度
        """
        if not texts:
            return ""
        
        if len(texts) == 1:
            return texts[0]
        
        base = texts[0]
        additions = []
        
        for text in texts[1:]:
            diff = self._extract_difference(base, text)
            if diff and len(diff) > 2:
                additions.append(diff)
        
        merged = base
        seen_additions = set()
        
        for add in additions:
            add_key = add[:50]
            if add_key not in seen_additions:
                if len(merged) + len(add) + 10 <= self.config.max_merged_length:
                    merged += f" | {add}"
                    seen_additions.add(add_key)
        
        if len(merged) > self.config.max_merged_length:
            merged = merged[:self.config.max_merged_length - 3] + "..."
        
        return merged
    
    def _extract_difference(self, base: str, text: str) -> str:
        """提取两个文本的差异部分"""
        import difflib
        
        if not base or not text:
            return text or base
        
        matcher = difflib.SequenceMatcher(None, base, text)
        diffs = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag in ('replace', 'insert'):
                diff_text = text[j1:j2]
                if diff_text and not diff_text.isspace():
                    diffs.append(diff_text)
        
        return ' '.join(diffs).strip()
    
    def check_and_merge_l2(
        self, 
        vector_store, 
        sqlite_store,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        检查 L2（向量库）中的相似记忆并合并
        
        触发条件：
        - 距离上次检查超过配置间隔
        - 或强制执行
        
        Args:
            vector_store: 向量存储实例
            sqlite_store: SQLite 存储实例
            force: 是否强制执行
        
        Returns:
            合并结果
        """
        import time
        
        current_time = time.time()
        interval_seconds = self.config.merge_interval_hours * 3600
        
        if not force and current_time - self._last_merge_check < interval_seconds:
            return {"merged_count": 0, "message": "未到合并检查间隔"}
        
        self._last_merge_check = current_time
        
        try:
            recent_memories = sqlite_store.get_recent_memories(
                limit=100,
                vectorized_only=True
            )
            
            if not recent_memories:
                return {"merged_count": 0, "message": "L2 中无记忆"}
            
            memories = [
                {
                    "id": m.id,
                    "text": m.compressed_text or m.text,
                    "created_time": m.created_time,
                    "weight": m.weight,
                    "access_count": m.access_count,
                    "metadata": m.metadata or {}
                }
                for m in recent_memories
            ]
            
            should_merge, reason = self.should_trigger_merge(memories, "l2_check")
            
            if not should_merge:
                return {"merged_count": 0, "message": reason}
            
            result = self.merge_memories(memories, "l2_check")
            
            if result.get("merged_count", 0) > 0:
                self._apply_merge_result(
                    result, 
                    vector_store, 
                    sqlite_store
                )
            
            return result
            
        except Exception as e:
            self._log.error("L2_MERGE_CHECK_FAILED", error=str(e))
            return {"merged_count": 0, "error": str(e)}
    
    def merge_on_l3_to_l2(
        self, 
        memories: List[Dict],
        vector_store,
        sqlite_store
    ) -> Dict[str, Any]:
        """
        L3→L2 回流时触发合并检测
        
        Args:
            memories: 回流的记忆列表
            vector_store: 向量存储实例
            sqlite_store: SQLite 存储实例
        
        Returns:
            合并结果
        """
        if not self.config.enabled or len(memories) < self.config.min_group_size:
            return {"merged_count": 0, "message": "无需合并"}
        
        should_merge, reason = self.should_trigger_merge(memories, "l3_to_l2")
        
        if not should_merge:
            return {"merged_count": 0, "message": reason}
        
        result = self.merge_memories(memories, "l3_to_l2")
        
        if result.get("merged_count", 0) > 0:
            self._apply_merge_result(
                result, 
                vector_store, 
                sqlite_store
            )
        
        return result
    
    def _apply_merge_result(
        self, 
        result: Dict[str, Any],
        vector_store,
        sqlite_store
    ):
        """
        应用合并结果到存储
        
        Args:
            result: 合并结果
            vector_store: 向量存储实例
            sqlite_store: SQLite 存储实例
        """
        merged_memories = result.get("merged_memories", [])
        deleted_ids = result.get("deleted_ids", [])
        
        if deleted_ids:
            try:
                vector_store.delete(ids=deleted_ids)
                self._log.info("MERGE_DELETE_VECTORS", count=len(deleted_ids))
            except Exception as e:
                self._log.error("MERGE_DELETE_VECTORS_FAILED", error=str(e))
        
        for merged in merged_memories:
            try:
                original_ids = merged.get("original_ids", [])
                if original_ids:
                    primary_id = original_ids[0]
                    
                    existing = sqlite_store.get(primary_id)
                    if existing:
                        existing.text = merged["text"]
                        existing.metadata = merged.get("metadata", {})
                        sqlite_store.add(existing)
                    
                    other_ids = original_ids[1:]
                    for other_id in other_ids:
                        sqlite_store.delete_memory(other_id)
                        
            except Exception as e:
                self._log.error("APPLY_MERGE_FAILED", error=str(e))
    
    def get_stats(self) -> Dict[str, Any]:
        """获取合并统计"""
        return {
            "enabled": self.config.enabled,
            "similarity_threshold": self.config.similarity_threshold,
            "min_group_size": self.config.min_group_size,
            "total_merged": self.stats.total_merged,
            "total_groups": self.stats.total_groups,
            "last_merge_time": self.stats.last_merge_time,
            "by_trigger": dict(self.stats.by_trigger)
        }
    
    def reset_stats(self):
        """重置统计"""
        self.stats = MergeStats()


_merger: Optional[MemoryMerger] = None


def get_merger() -> MemoryMerger:
    """获取全局合并器实例"""
    global _merger
    if _merger is None:
        _merger = MemoryMerger()
    return _merger
