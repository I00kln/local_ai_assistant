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
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from memory_tags import MemoryTags, MemoryTagHelper
from logger import get_logger
from models import MemoryRecord


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
    max_memories: int = 100
    use_ann: bool = True
    ann_threshold: int = 20


@dataclass
class ConflictRecord:
    """冲突记录"""
    texts: List[str] = field(default_factory=list)
    conflict_type: str = ""
    conflict_detail: str = ""
    timestamp: str = ""


@dataclass
class MergeStats:
    """合并统计"""
    total_merged: int = 0
    total_groups: int = 0
    total_conflicts: int = 0
    conflict_details: List[ConflictRecord] = field(default_factory=list)
    last_merge_time: Optional[str] = None
    by_trigger: Dict[str, int] = field(default_factory=dict)
    
    def add_conflict(self, texts: List[str], conflict_type: str, conflict_detail: str):
        """添加冲突记录"""
        self.total_conflicts += 1
        self.conflict_details.append(ConflictRecord(
            texts=texts[:3],
            conflict_type=conflict_type,
            conflict_detail=conflict_detail,
            timestamp=datetime.now().isoformat()
        ))
        if len(self.conflict_details) > 100:
            self.conflict_details = self.conflict_details[-100:]


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
        self._vector_store = None
    
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
    
    def _ensure_vector_store(self):
        """确保 VectorStore 已初始化"""
        if self._vector_store is None:
            from vector_store import get_vector_store
            self._vector_store = get_vector_store()
    
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
        - 语义冲突（"不喜欢甜食" vs "喜欢蛋糕"）
        
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
        
        semantic_conflict = self._detect_semantic_conflicts(texts)
        if semantic_conflict:
            return True, semantic_conflict
        
        return False, ""
    
    def _detect_semantic_conflicts(self, texts: List[str]) -> str:
        """
        检测语义冲突
        
        检测否定词对：
        - "不喜欢" vs "喜欢"
        - "没去" vs "去"
        - "不是" vs "是"
        
        Returns:
            冲突描述，无冲突返回空字符串
        """
        NEGATION_PAIRS = [
            (r"不(.{1,4})", r"\1"),
            (r"没(.{1,4})", r"\1"),
            (r"非(.{1,4})", r"\1"),
            (r"无(.{1,4})", r"\1"),
        ]
        
        for neg_pattern, pos_pattern in NEGATION_PAIRS:
            neg_matches = set()
            pos_matches = set()
            
            for text in texts:
                neg = re.findall(neg_pattern, text)
                neg_matches.update(n.strip() for n in neg if n.strip())
                
                pos = re.findall(pos_pattern, text)
                pos_matches.update(p.strip() for p in pos if p.strip())
            
            conflicts = neg_matches & pos_matches
            if conflicts:
                conflict_words = list(conflicts)[:3]
                return f"语义冲突: 否定词对 [{', '.join(conflict_words)}]"
        
        return ""
    
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
        
        similar_groups = self._find_similar_groups(
            memories,
            max_memories=self.config.max_memories,
            use_ann=self.config.use_ann
        )
        
        large_groups = [g for g in similar_groups if len(g) >= self.config.min_group_size]
        
        if large_groups:
            return True, f"发现 {len(large_groups)} 组相似记忆（共 {sum(len(g) for g in large_groups)} 条）"
        
        return False, "未发现需要合并的相似记忆组"
    
    def _find_similar_groups(
        self, 
        memories: List[Dict],
        max_memories: int = None,
        use_ann: bool = None
    ) -> List[List[Dict]]:
        """
        使用向量相似度聚类相似记忆（优化版）
        
        Args:
            memories: 记忆列表
            max_memories: 最大处理数量，超出则截断（默认使用配置）
            use_ann: 是否使用近似最近邻（默认使用配置）
        
        Returns:
            相似记忆分组列表
        
        时间复杂度：
            - 优化前: O(n²) 全量相似度矩阵计算
            - 优化后 (ANN): O(n * k) k为每个记忆的近邻数
            - 优化后 (截断): O(max_memories²)
        """
        if max_memories is None:
            max_memories = self.config.max_memories
        if use_ann is None:
            use_ann = self.config.use_ann
        if len(memories) < 2:
            return [[m] for m in memories]
        
        if len(memories) > max_memories:
            memories = sorted(memories, key=lambda m: m.get("weight", 1.0), reverse=True)[:max_memories]
            self._log.info("SIMILAR_GROUPS_TRUNCATED", 
                          original=len(memories), 
                          limited=max_memories)
        
        self._ensure_embedding_service()
        
        if not self._embedding_service.is_available:
            return [[m] for m in memories]
        
        if use_ann and len(memories) > self.config.ann_threshold:
            return self._find_similar_groups_ann(memories)
        
        return self._find_similar_groups_exact(memories)
    
    def _find_similar_groups_ann(self, memories: List[Dict]) -> List[List[Dict]]:
        """
        使用近似最近邻方法聚类相似记忆
        
        策略：对每个记忆，使用向量搜索找到相似记忆
        时间复杂度: O(n * k) 其中 k 是每个记忆的近邻数
        """
        groups = []
        used_ids = set()
        
        for i, memory in enumerate(memories):
            memory_id = memory.get("id", i)
            if memory_id in used_ids:
                continue
            
            text = memory.get("text", "")
            if not text:
                continue
            
            group = [memory]
            used_ids.add(memory_id)
            
            threshold = self._get_dynamic_threshold(text)
            
            try:
                similar_memories = self._find_similar_via_embedding(text, threshold)
                
                for similar in similar_memories:
                    similar_id = similar.get("id")
                    if similar_id and similar_id not in used_ids:
                        for m in memories:
                            if m.get("id") == similar_id:
                                group.append(m)
                                used_ids.add(similar_id)
                                break
            except Exception:
                pass
            
            groups.append(group)
        
        return groups
    
    def _find_similar_via_embedding(self, text: str, threshold: float) -> List[Dict]:
        """
        通过嵌入向量查找相似记忆
        
        使用向量存储的搜索功能，避免全量矩阵计算
        """
        try:
            self._ensure_vector_store()
            if self._vector_store is None:
                return []
            
            results = self._vector_store.search(text, n_results=10)
            
            similar = []
            for r in results:
                if r.get("similarity", 0) >= threshold:
                    similar.append({
                        "id": r.get("id"),
                        "text": r.get("text"),
                        "similarity": r.get("similarity")
                    })
            
            return similar[1:]
        except Exception:
            return []
    
    def _find_similar_groups_exact(self, memories: List[Dict]) -> List[List[Dict]]:
        """
        精确计算相似度矩阵（用于小规模数据）
        
        优化：
        - 批量嵌入计算，减少单次调用开销
        - 使用 NumPy 向量化操作
        - 时间复杂度: O(n²) 但常数因子更小
        """
        valid_memories = []
        texts = []
        
        for m in memories:
            text = m.get("text", "")
            if text:
                texts.append(text)
                valid_memories.append(m)
        
        if len(texts) < 2:
            return [[m] for m in valid_memories]
        
        try:
            vectors = self._embedding_service.embed(texts, use_cache=False)
            vectors = np.array(vectors)
        except Exception as e:
            self._log.warning("BATCH_EMBED_FAILED", error=str(e))
            return [[m] for m in valid_memories]
        
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = vectors / norms
        similarity_matrix = np.dot(normalized, normalized.T)
        
        thresholds = np.array([
            self._get_dynamic_threshold(m.get("text", ""))
            for m in valid_memories
        ])
        
        groups = []
        used = set()
        
        for i in range(len(valid_memories)):
            if i in used:
                continue
            
            group = [valid_memories[i]]
            used.add(i)
            
            threshold_i = thresholds[i]
            
            for j in range(i + 1, len(valid_memories)):
                if j in used:
                    continue
                
                effective_threshold = max(threshold_i, thresholds[j])
                
                if similarity_matrix[i, j] >= effective_threshold:
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
            
            for group in large_groups[:self.config.max_merge_batch]:
                texts = [m.get("text", "") for m in group]
                has_conflict, conflict_desc = self._detect_conflicts(texts)
                
                if has_conflict:
                    conflict_type = "unknown"
                    if "时间冲突" in conflict_desc:
                        conflict_type = "time"
                    elif "地点冲突" in conflict_desc:
                        conflict_type = "location"
                    elif "数值冲突" in conflict_desc:
                        conflict_type = "number"
                    elif "语义冲突" in conflict_desc:
                        conflict_type = "semantic"
                    
                    self.stats.add_conflict(texts, conflict_type, conflict_desc)
                    self._log.info(
                        "MERGE_CONFLICT_DETECTED",
                        conflict_type=conflict_type,
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
            self.stats.last_merge_time = datetime.now().isoformat()
            self.stats.by_trigger[trigger] = self.stats.by_trigger.get(trigger, 0) + merged_count
            
            self._log.info(
                "MEMORY_MERGE_COMPLETE",
                merged_count=merged_count,
                groups=len(large_groups),
                conflicts=self.stats.total_conflicts,
                trigger=trigger
            )
            
            return {
                "merged_count": merged_count,
                "groups": len(large_groups),
                "merged_memories": merged_memories,
                "deleted_ids": deleted_ids,
                "conflicts": self.stats.total_conflicts
            }
    
    def _merge_group(self, group: List[Dict]) -> Optional[Dict]:
        """
        合并一组相似记忆
        
        策略：
        1. 过滤 protected/high_density 记忆（不参与合并但保留）
        2. 剩余可合并记录不足2条时跳过
        3. 按优先级排序选择主记录
        4. 合并权重（取最大值）
        5. 智能合并文本
        6. 深度合并元数据
        7. 清除压缩状态（合并后需重新压缩）
        8. 重新标记 semantic_tag
        
        优先级规则：
        - protected: 100 (不参与合并，从组中过滤)
        - forgotten: 0 (最低，可被合并/删除)
        - important: 80
        - high_density: 70 (不参与合并，从组中过滤)
        - preserve: 60
        - compressed: 40
        - 普通: 20
        """
        if len(group) == 1:
            return group[0]
        
        skip_memories = [
            m for m in group 
            if MemoryTagHelper.should_skip_merge(m.get("metadata", {}))
        ]
        mergeable = [
            m for m in group 
            if not MemoryTagHelper.should_skip_merge(m.get("metadata", {}))
        ]
        
        if skip_memories:
            self._log.info(
                "MERGE_SKIP_PROTECTED",
                skip_count=len(skip_memories),
                mergeable_count=len(mergeable),
                group_size=len(group),
                reasons=[
                    "protected" if m.get("metadata", {}).get(MemoryTags.PROTECTED) 
                    else "high_density"
                    for m in skip_memories
                ]
            )
        
        if len(mergeable) < 2:
            return None
        
        def get_sort_key(m):
            priority = MemoryTagHelper.get_merge_priority(m.get("metadata", {}))
            created_time = m.get("created_time", "9999-99-99T99:99:99")
            return (-priority, created_time)
        
        sorted_group = sorted(mergeable, key=get_sort_key)
        
        primary = sorted_group[0]
        primary_priority = MemoryTagHelper.get_merge_priority(primary.get("metadata", {}))
        
        self._log.debug(
            "MERGE_PRIMARY_SELECTED",
            primary_id=primary.get("id"),
            priority=primary_priority,
            mergeable_count=len(mergeable),
            skipped_count=len(skip_memories)
        )
        
        merged_text = self._smart_merge_texts([m.get("text", "") for m in sorted_group])
        
        if not merged_text:
            return None
        
        merged_metadata = self._deep_merge_metadata(
            [m.get("metadata", {}) for m in sorted_group]
        )
        
        merged_metadata[MemoryTags.MERGED] = True
        merged_metadata[MemoryTags.MERGED_COUNT] = len(mergeable)
        merged_metadata[MemoryTags.MERGED_FROM_IDS] = [m.get("id") for m in sorted_group if m.get("id")]
        merged_metadata[MemoryTags.MERGED_FROM_PRIORITIES] = [
            MemoryTagHelper.get_merge_priority(m.get("metadata", {})) 
            for m in sorted_group
        ]
        
        compression_keys = [
            MemoryTags.COMPRESSED,
            MemoryTags.COMPRESSED_TIME,
            MemoryTags.COMPRESSED_STRATEGY,
            MemoryTags.ORIGINAL_LENGTH,
            MemoryTags.HAS_COMPRESSED_VERSION,
            MemoryTags.PENDING_COMPRESSION,
            MemoryTags.PENDING_SINCE,
            MemoryTags.RETRY_COUNT
        ]
        for key in compression_keys:
            merged_metadata.pop(key, None)
        
        merged_metadata[MemoryTags.NEEDS_RECOMPRESSION] = True
        
        if primary_priority >= 60:
            for tag in [MemoryTags.IMPORTANT, MemoryTags.PRESERVE]:
                if primary.get("metadata", {}).get(tag) is True:
                    merged_metadata[tag] = True
                    merged_metadata[f"{tag}_inherited"] = True
        
        has_forgotten_member = any(
            m.get("metadata", {}).get(MemoryTags.FORGOTTEN) is True
            for m in sorted_group
        )
        if has_forgotten_member:
            merged_metadata[MemoryTags.HAD_FORGOTTEN_MEMBER] = True
        
        try:
            from tag_classifier import get_tag_classifier
            tagger = get_tag_classifier()
            new_tag = tagger.tag_memory(merged_text)
            merged_metadata[MemoryTags.SEMANTIC_TAG] = json.dumps(new_tag.to_dict(), ensure_ascii=False)
        except Exception:
            pass
        
        merged = {
            "text": merged_text,
            "created_time": primary.get("created_time"),
            "weight": max(m.get("weight", 1.0) for m in group),
            "access_count": sum(m.get("access_count", 0) for m in group),
            "metadata": merged_metadata,
            "source": "merged",
            "original_ids": [m.get("id") for m in sorted_group if m.get("id")],
            "vector_ids_to_delete": [],
            "compressed_text": None,
            "primary_priority": primary_priority
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
        2. 时间戳字段冲突时保留最新值
        3. 其他标量冲突时保留第一个出现的值
        4. 数组类型合并去重
        5. 记录合并来源
        6. semantic_tag 使用专门的合并策略
        """
        if not metadata_list:
            return {}
        
        if len(metadata_list) == 1:
            return dict(metadata_list[0])
        
        TIMESTAMP_KEYS = {
            "timestamp", "created_time", "updated_time", "merged_at",
            "last_access_time", "compressed_time", "forgotten_time"
        }
        
        merged = {}
        seen_keys = {}
        semantic_tags_to_merge = []
        
        for i, metadata in enumerate(metadata_list):
            if not isinstance(metadata, dict):
                continue
            
            if MemoryTags.SEMANTIC_TAG in metadata:
                try:
                    from tag_classifier import MemoryTag
                    tag = MemoryTag.from_dict(metadata[MemoryTags.SEMANTIC_TAG])
                    semantic_tags_to_merge.append(tag)
                except Exception:
                    pass
            
            for key, value in metadata.items():
                if key == MemoryTags.SEMANTIC_TAG:
                    continue
                
                if key in seen_keys:
                    if key in TIMESTAMP_KEYS:
                        if isinstance(value, str) and isinstance(merged[key], str):
                            try:
                                if value > merged[key]:
                                    merged[key] = value
                            except Exception:
                                pass
                    elif isinstance(merged[key], list) and isinstance(value, list):
                        combined = merged[key] + value
                        merged[key] = list(set(str(x) for x in combined))
                    elif isinstance(merged[key], list):
                        merged[key] = list(set(str(x) for x in merged[key] + [str(value)]))
                    elif isinstance(value, list):
                        merged[key] = list(set(str(x) for x in [str(merged[key])] + value))
                else:
                    seen_keys[key] = i
                    if isinstance(value, list):
                        merged[key] = list(set(str(x) for x in value))
                    else:
                        merged[key] = value
        
        if semantic_tags_to_merge:
            try:
                from tag_classifier import get_tag_classifier
                tagger = get_tag_classifier()
                merged_tag = tagger.merge_tags(semantic_tags_to_merge)
                merged[MemoryTags.SEMANTIC_TAG] = json.dumps(merged_tag.to_dict(), ensure_ascii=False)
            except Exception:
                if semantic_tags_to_merge:
                    merged[MemoryTags.SEMANTIC_TAG] = json.dumps(semantic_tags_to_merge[0].to_dict(), ensure_ascii=False)
        
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
        1. 备份主记录原始状态（用于回滚）
        2. 收集所有待删除记录的 vector_id
        3. 更新主记录（标记需要重新向量化）
        4. 删除旧记录
        5. 同步清理向量库（失败不影响一致性，孤立向量会在启动补偿中清理）
        
        事务保护：
        - 使用 TransactionCoordinator 确保原子性
        - 失败时回滚已完成的操作
        - 向量删除失败不触发回滚（孤立向量可被启动补偿清理）
        
        Args:
            result: 合并结果
            vector_store: 向量存储实例
            sqlite_store: SQLite 存储实例
        """
        self._ensure_tx_coordinator()
        
        merged_memories = result.get("merged_memories", [])
        
        if not merged_memories:
            return
        
        self._tx_coordinator.begin_migration("merge")
        
        try:
            for merged in merged_memories:
                original_ids = merged.get("original_ids", [])
                if not original_ids:
                    continue
                
                primary_id = original_ids[0]
                other_ids = original_ids[1:]
                
                primary_backup = sqlite_store.get(primary_id)
                
                vector_ids_to_delete = []
                records_to_delete_backup = []
                
                for oid in other_ids:
                    rec = sqlite_store.get(oid)
                    if rec:
                        records_to_delete_backup.append({
                            "id": rec.id,
                            "text": rec.text,
                            "metadata": dict(rec.metadata) if rec.metadata else {},
                            "vector_id": rec.vector_id
                        })
                        if rec.vector_id:
                            vector_ids_to_delete.append(rec.vector_id)
                
                vector_ids_to_delete.extend(merged.get("vector_ids_to_delete", []))
                
                merge_failed = False
                
                try:
                    existing = sqlite_store.get(primary_id)
                    if existing:
                        existing.text = merged["text"]
                        
                        existing.metadata = self._deep_merge_metadata([
                            existing.metadata or {},
                            merged.get("metadata", {})
                        ])
                        existing.metadata[MemoryTags.MERGED_AT] = datetime.now().isoformat()
                        
                        primary_priority = merged.get("primary_priority", 20)
                        existing.metadata[MemoryTags.MERGED_PRIMARY_PRIORITY] = primary_priority
                        
                        existing.compressed_text = None
                        
                        compression_keys = [
                            MemoryTags.COMPRESSED,
                            MemoryTags.COMPRESSED_TIME,
                            MemoryTags.COMPRESSED_STRATEGY,
                            MemoryTags.ORIGINAL_LENGTH,
                            MemoryTags.HAS_COMPRESSED_VERSION,
                            MemoryTags.PENDING_COMPRESSION,
                            MemoryTags.PENDING_SINCE,
                            MemoryTags.RETRY_COUNT
                        ]
                        for key in compression_keys:
                            existing.metadata.pop(key, None)
                        
                        existing.metadata[MemoryTags.NEEDS_REVECTORIZATION] = True
                        existing.metadata[MemoryTags.NEEDS_RECOMPRESSION] = True
                        
                        existing.is_vectorized = 0
                        existing.vector_id = ""
                        
                        sqlite_store.add(existing)
                        
                        self._log.info(
                            "MERGE_UPDATE_PRIMARY",
                            primary_id=primary_id,
                            new_text_length=len(merged["text"]),
                            needs_revectorization=True,
                            needs_recompression=True,
                            primary_priority=primary_priority
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
                            self._log.warning(
                                "MERGE_DELETE_VECTORS_FAILED",
                                error=str(e),
                                vector_ids=vector_ids_to_delete,
                                note="孤立向量将在启动补偿中清理"
                            )
                            
                except Exception as e:
                    merge_failed = True
                    self._log.error(
                        "MERGE_OPERATION_FAILED",
                        primary_id=primary_id,
                        error=str(e)
                    )
                    
                    if primary_backup:
                        try:
                            sqlite_store.add(primary_backup)
                            self._log.info("MERGE_ROLLBACK_PRIMARY", primary_id=primary_id)
                        except Exception as rb_e:
                            self._log.error(
                                "MERGE_ROLLBACK_FAILED",
                                primary_id=primary_id,
                                error=str(rb_e)
                            )
                    
                    for rec_backup in records_to_delete_backup:
                        try:
                            restored = sqlite_store.get(rec_backup["id"])
                            if not restored:
                                restored = MemoryRecord(
                                    text=rec_backup["text"],
                                    metadata=rec_backup["metadata"],
                                    vector_id=rec_backup["vector_id"],
                                    is_vectorized=1 if rec_backup["vector_id"] else 0
                                )
                                restored.id = rec_backup["id"]
                                sqlite_store.add(restored)
                                self._log.info(
                                    "MERGE_ROLLBACK_DELETED_RECORD",
                                    record_id=rec_backup["id"]
                                )
                        except Exception as rb_e:
                            self._log.error(
                                "MERGE_ROLLBACK_RECORD_FAILED",
                                record_id=rec_backup["id"],
                                error=str(rb_e)
                            )
                    
                    raise
                        
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
