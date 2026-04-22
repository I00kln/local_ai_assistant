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

安全特性：
- 向量库同步：合并时同步清理 ChromaDB 向量
- 元数据深度合并：保留所有非冲突键值对
- 事务保护：确保 SQLite 操作原子性
- 冲突检测：检测时间/地点等实体冲突
"""

import threading
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Set
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
    short_text_threshold: int = 20
    long_text_threshold: int = 200
    short_text_similarity_boost: float = 0.03
    long_text_similarity_penalty: float = 0.04


@dataclass
class MergeStats:
    """合并统计"""
    total_merged: int = 0
    total_groups: int = 0
    total_conflicts: int = 0
    last_merge_time: Optional[str] = None
    by_trigger: Dict[str, int] = field(default_factory=dict)


class MergeConflictError(Exception):
    """合并冲突错误"""
    def __init__(self, code: str, message: str, details: Dict = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


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
    
    安全特性：
    - 向量库同步清理
    - 元数据深度合并
    - 事务保护
    - 冲突检测
    
    不影响原有压缩、归档、遗忘逻辑。
    """
    
    _instance: Optional['MemoryMerger'] = None
    _lock = threading.Lock()
    
    TIME_PATTERNS = [
        r'明天',
        r'后天',
        r'大后天',
        r'昨天',
        r'前天',
        r'\d+月\d+[日号]',
        r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日号]?',
        r'下周[一二三四五六日天]',
        r'这周[一二三四五六日天]',
        r'[上中下]午',
        r'[早中晚]上?',
        r'\d+[点时](\d+分?)?',
    ]
    
    LOCATION_PATTERNS = [
        r'去[北京上海广州深圳杭州成都武汉西安南京重庆天津苏州]',
        r'在[北京上海广州深圳杭州成都武汉西安南京重庆天津苏州]',
        r'到[北京上海广州深圳杭州成都武汉西安南京重庆天津苏州]',
        r'回[北京上海广州深圳杭州成都武汉西安南京重庆天津苏州]',
    ]
    
    NUMBER_PATTERNS = [
        r'\d+块',
        r'\d+元',
        r'\d+万',
        r'\d+千',
        r'\d+百',
        r'\d+个',
        r'\d+次',
    ]
    
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
        self._tx_coordinator = None
    
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
    
    def _ensure_tx_coordinator(self):
        """确保事务协调器已初始化"""
        if self._tx_coordinator is None:
            from memory_transaction import get_transaction_coordinator
            self._tx_coordinator = get_transaction_coordinator()
    
    def _get_dynamic_threshold(self, text: str) -> float:
        """
        根据文本长度动态调整相似度阈值
        
        策略：
        - 短文本（<20字）：阈值调高，防止误杀
        - 长文本（>200字）：阈值放宽，语义重合更难
        """
        base_threshold = self.config.similarity_threshold
        text_len = len(text)
        
        if text_len < self.config.short_text_threshold:
            return min(0.98, base_threshold + self.config.short_text_similarity_boost)
        elif text_len > self.config.long_text_threshold:
            return max(0.85, base_threshold - self.config.long_text_similarity_penalty)
        else:
            return base_threshold
    
    def _detect_conflicts(self, texts: List[str]) -> Tuple[bool, str]:
        """
        检测文本间的冲突
        
        检测：
        - 时间冲突（"明天去北京" vs "后天去北京"）
        - 地点冲突（"去北京" vs "去上海"）
        - 数值冲突（"100块" vs "200块"）
        
        Returns:
            (是否有冲突, 冲突描述)
        """
        if len(texts) < 2:
            return False, ""
        
        time_entities = []
        location_entities = []
        number_entities = []
        
        for text in texts:
            time_matches = set()
            for pattern in self.TIME_PATTERNS:
                matches = re.findall(pattern, text)
                time_matches.update(matches)
            time_entities.append(time_matches)
            
            location_matches = set()
            for pattern in self.LOCATION_PATTERNS:
                matches = re.findall(pattern, text)
                location_matches.update(matches)
            location_entities.append(location_matches)
            
            number_matches = set()
            for pattern in self.NUMBER_PATTERNS:
                matches = re.findall(pattern, text)
                number_matches.update(matches)
            number_entities.append(number_matches)
        
        all_times = [t for ts in time_entities for t in ts]
        all_locations = [l for ls in location_entities for l in ls]
        all_numbers = [n for ns in number_entities for n in ns]
        
        if len(all_times) > 1 and len(set(all_times)) > 1:
            return True, f"时间冲突: {', '.join(set(all_times))}"
        
        if len(all_locations) > 1 and len(set(all_locations)) > 1:
            return True, f"地点冲突: {', '.join(set(all_locations))}"
        
        if len(all_numbers) > 1 and len(set(all_numbers)) > 1:
            return True, f"数值冲突: {', '.join(set(all_numbers))}"
        
        return False, ""
    
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
            
            text_i = valid_memories[i].get("text", "")
            threshold_i = self._get_dynamic_threshold(text_i)
            
            for j in range(i + 1, len(valid_memories)):
                if j in used:
                    continue
                
                text_j = valid_memories[j].get("text", "")
                threshold_j = self._get_dynamic_threshold(text_j)
                
                effective_threshold = max(threshold_i, threshold_j)
                
                if similarity_matrix[i][j] >= effective_threshold:
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
                "deleted_ids": 需要删除的记忆ID列表,
                "conflicts": 冲突数量
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
            conflict_count = 0
            
            for group in large_groups[:self.config.max_merge_batch]:
                texts = [m.get("text", "") for m in group]
                has_conflict, conflict_desc = self._detect_conflicts(texts)
                
                if has_conflict:
                    conflict_count += 1
                    self._log.info(
                        "MERGE_CONFLICT_DETECTED",
                        conflict=conflict_desc,
                        texts=texts[:3]
                    )
                    continue
                
                merged = self._merge_group(group)
                if merged:
                    merged_memories.append(merged)
                    deleted_ids.extend([m.get("id") for m in group[1:] if m.get("id")])
                    merged_count += len(group) - 1
            
            self.stats.total_merged += merged_count
            self.stats.total_groups += len(large_groups)
            self.stats.total_conflicts += conflict_count
            self.stats.last_merge_time = datetime.now().isoformat()
            self.stats.by_trigger[trigger] = self.stats.by_trigger.get(trigger, 0) + merged_count
            
            self._log.info(
                "MEMORY_MERGE_COMPLETE",
                merged_count=merged_count,
                groups=len(large_groups),
                conflicts=conflict_count,
                trigger=trigger
            )
            
            return {
                "merged_count": merged_count,
                "groups": len(large_groups),
                "merged_memories": merged_memories,
                "deleted_ids": deleted_ids,
                "conflicts": conflict_count
            }
    
    def _merge_group(self, group: List[Dict]) -> Optional[Dict]:
        """
        合并一组相似记忆
        
        策略：
        1. 保留最早的创建时间
        2. 合并权重（取最大值）
        3. 智能合并文本
        4. 深度合并元数据
        """
        if len(group) == 1:
            return group[0]
        
        sorted_group = sorted(group, key=lambda x: x.get("created_time", ""))
        
        merged_text = self._smart_merge_texts([m.get("text", "") for m in sorted_group])
        
        if not merged_text:
            return None
        
        merged_metadata = self._deep_merge_metadata(
            [m.get("metadata", {}) for m in sorted_group]
        )
        
        merged_metadata[MemoryTags.MERGED] = True
        merged_metadata["merged_count"] = len(group)
        merged_metadata["merged_from_ids"] = [m.get("id") for m in sorted_group if m.get("id")]
        
        merged = {
            "text": merged_text,
            "created_time": sorted_group[0].get("created_time"),
            "weight": max(m.get("weight", 1.0) for m in group),
            "access_count": sum(m.get("access_count", 0) for m in group),
            "metadata": merged_metadata,
            "source": "merged",
            "original_ids": [m.get("id") for m in group if m.get("id")],
            "vector_ids_to_delete": []
        }
        
        for m in sorted_group[1:]:
            if m.get("vector_id"):
                merged["vector_ids_to_delete"].append(m["vector_id"])
        
        return merged
    
    def _deep_merge_metadata(self, metadata_list: List[Dict]) -> Dict:
        """
        深度合并多个元数据字典
        
        策略：
        1. 保留所有非冲突的键值对
        2. 冲突时保留第一个出现的值
        3. 数组类型合并去重
        4. 记录合并来源
        """
        if not metadata_list:
            return {}
        
        if len(metadata_list) == 1:
            return dict(metadata_list[0])
        
        merged = {}
        seen_keys = {}
        
        for i, metadata in enumerate(metadata_list):
            if not isinstance(metadata, dict):
                continue
            
            for key, value in metadata.items():
                if key in seen_keys:
                    if isinstance(merged[key], list) and isinstance(value, list):
                        combined = merged[key] + value
                        merged[key] = list(set(str(x) for x in combined))
                    elif isinstance(merged[key], list):
                        merged[key] = list(set(str(x) for x in merged[key] + [str(value)]))
                    elif isinstance(value, list):
                        merged[key] = list(set(str(x) for x in [str(merged[key])] + value))
                    else:
                        pass
                else:
                    seen_keys[key] = i
                    if isinstance(value, list):
                        merged[key] = list(set(str(x) for x in value))
                    else:
                        merged[key] = value
        
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
                    "metadata": m.metadata or {},
                    "vector_id": m.vector_id
                }
                for m in recent_memories
            ]
            
            should_merge, reason = self.should_trigger_merge(memories, "l2_check")
            
            if not should_merge:
                return {"merged_count": 0, "message": reason}
            
            result = self.merge_memories(memories, "l2_check")
            
            if result.get("merged_count", 0) > 0:
                self._apply_merge_result_atomic(
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
            self._apply_merge_result_atomic(
                result, 
                vector_store, 
                sqlite_store
            )
        
        return result
    
    def _apply_merge_result_atomic(
        self, 
        result: Dict[str, Any],
        vector_store,
        sqlite_store
    ):
        """
        原子化应用合并结果到存储
        
        流程：
        1. 获取所有待删除记录的 vector_id
        2. 更新主记录（标记需要重新向量化）
        3. 删除旧记录
        4. 同步清理向量库
        
        事务保护：
        - 使用 TransactionCoordinator 确保原子性
        - 失败时回滚已完成的操作
        
        Args:
            result: 合并结果
            vector_store: 向量存储实例
            sqlite_store: SQLite 存储实例
        """
        self._ensure_tx_coordinator()
        
        merged_memories = result.get("merged_memories", [])
        
        if not merged_memories:
            return
        
        self._tx_coordinator.begin_migration()
        
        try:
            for merged in merged_memories:
                original_ids = merged.get("original_ids", [])
                if not original_ids:
                    continue
                
                primary_id = original_ids[0]
                other_ids = original_ids[1:]
                
                vector_ids_to_delete = []
                
                for oid in other_ids:
                    rec = sqlite_store.get(oid)
                    if rec and rec.vector_id:
                        vector_ids_to_delete.append(rec.vector_id)
                
                vector_ids_to_delete.extend(merged.get("vector_ids_to_delete", []))
                
                existing = sqlite_store.get(primary_id)
                if existing:
                    existing.text = merged["text"]
                    
                    existing.metadata = self._deep_merge_metadata([
                        existing.metadata or {},
                        merged.get("metadata", {})
                    ])
                    existing.metadata["merged_at"] = datetime.now().isoformat()
                    
                    existing.is_vectorized = 0
                    existing.vector_id = ""
                    
                    sqlite_store.add(existing)
                    
                    self._log.info(
                        "MERGE_UPDATE_PRIMARY",
                        primary_id=primary_id,
                        new_text_length=len(merged["text"]),
                        needs_revectorization=True
                    )
                
                for oid in other_ids:
                    sqlite_store.delete_memory(oid)
                
                if vector_ids_to_delete:
                    try:
                        vector_store.delete(ids=vector_ids_to_delete)
                        self._log.info(
                            "MERGE_DELETE_VECTORS",
                            count=len(vector_ids_to_delete),
                            primary_id=primary_id
                        )
                    except Exception as e:
                        self._log.error(
                            "MERGE_DELETE_VECTORS_FAILED",
                            error=str(e),
                            vector_ids=vector_ids_to_delete
                        )
                        
        except Exception as e:
            self._log.error("APPLY_MERGE_FAILED", error=str(e))
            raise
        finally:
            self._tx_coordinator.end_migration()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取合并统计"""
        return {
            "enabled": self.config.enabled,
            "similarity_threshold": self.config.similarity_threshold,
            "min_group_size": self.config.min_group_size,
            "total_merged": self.stats.total_merged,
            "total_groups": self.stats.total_groups,
            "total_conflicts": self.stats.total_conflicts,
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
