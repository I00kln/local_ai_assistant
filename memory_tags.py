# memory_tags.py
# 记忆标签统一管理
from typing import Optional
from datetime import datetime, timedelta


class MemoryConstants:
    """
    记忆系统常量定义
    
    魔法数字说明：
    - L3_COOLDOWN_MULTIPLIER: L3回流冷却期倍数
      从L3回流后的冷却期 = 原冷却期 * 此倍数
      值为2表示回流后冷却期翻倍，防止频繁回流
    - KEYWORD_USER_WEIGHT_MULTIPLIER: 用户消息关键词权重倍数
      用户消息中的关键词权重 = 原权重 * 此倍数
      值为2.0表示用户消息权重是助理消息的2倍
    - KEYWORD_COVERAGE_MULTIPLIER: 关键词覆盖率计算倍数
      用于计算覆盖率时的分母调整
    - L3_SEARCH_RESULT_MULTIPLIER: L3搜索结果倍数
      L3搜索返回数量 = top_k * 此倍数
      值为2表示返回2倍结果用于后续过滤
    - NUMBER_WEIGHT_SCORE: 数值权重分数
      每个数值对信息密度的贡献分数
    - DENSITY_NORMALIZATION_MULTIPLIER: 密度归一化倍数
      用于将密度值归一化到0-1范围
    """
    
    L3_COOLDOWN_MULTIPLIER = 2
    KEYWORD_USER_WEIGHT_MULTIPLIER = 2.0
    KEYWORD_COVERAGE_MULTIPLIER = 2
    L3_SEARCH_RESULT_MULTIPLIER = 2
    NUMBER_WEIGHT_SCORE = 2.0
    DENSITY_NORMALIZATION_MULTIPLIER = 2.0


class MemoryTags:
    """记忆元数据标签常量"""
    
    # 压缩相关标签
    COMPRESSED = "compressed"
    COMPRESSED_TIME = "compressed_time"
    COMPRESSED_STRATEGY = "compression_strategy"
    ORIGINAL_LENGTH = "original_length"
    HAS_COMPRESSED_VERSION = "has_compressed_version"
    
    # 待压缩标签
    PENDING_COMPRESSION = "pending_compression"
    PENDING_SINCE = "pending_since"
    RETRY_COUNT = "retry_count"
    
    # 保护相关标签
    PRESERVE = "preserve"
    PRESERVE_REASON = "preserve_reason"
    PROTECTED = "protected"
    PROTECTED_REASON = "protected_reason"
    IMPORTANT = "important"
    
    # 流转相关标签
    MOVED_FROM_L2 = "moved_from_l2"
    MOVED_FROM_L2_TIME = "moved_time"
    MOVED_FROM_L3 = "moved_from_l3"
    MOVED_FROM_L3_TIME = "moved_from_l3_time"
    PROMOTED_TO_L2 = "promoted_to_l2"
    PROMOTED_TIME = "promoted_time"
    PROMOTED_FROM_L3 = "promoted_from_l3"
    
    # 高密度内容标签
    HIGH_DENSITY = "high_density"
    
    # 遗忘相关标签
    FORGOTTEN = "forgotten"
    FORGOTTEN_TIME = "forgotten_time"
    FORGOTTEN_REASON = "forgotten_reason"
    ARCHIVED = "archived"
    ARCHIVED_TIME = "archived_time"
    
    # 合并相关标签
    MERGED = "merged"
    MERGED_AT = "merged_at"
    MERGED_COUNT = "merged_count"
    MERGED_FROM_IDS = "merged_from_ids"
    MERGED_FROM_PRIORITIES = "merged_from_priorities"
    MERGED_PRIMARY_PRIORITY = "merged_primary_priority"
    HAD_FORGOTTEN_MEMBER = "had_forgotten_member"
    NEEDS_RECOMPRESSION = "needs_recompression"
    NEEDS_REVECTORIZATION = "needs_revectorization"
    
    # 语义标签相关
    SEMANTIC_TAG = "semantic_tag"
    TAG_CORRECTED_AT = "tag_corrected_at"
    TAGS = "tags"
    
    # 向量化相关
    SQLITE_ID = "sqlite_id"
    IS_SQLITE_ONLY = "is_sqlite_only"
    
    # 过滤相关
    NONSENSE_FILTER_RESULT = "nonsense_filter_result"
    
    # 升级相关
    UPGRADED_FROM_SQLITE_ONLY = "upgraded_from_sqlite_only"
    UPGRADED_TIME = "upgraded_time"
    
    # 高密度原因
    HIGH_DENSITY_REASON = "high_density_reason"
    
    # 来源标签
    SOURCE_L1 = "L1"
    SOURCE_L2 = "L2"
    SOURCE_L3 = "L3"
    
    # 时间戳
    TIMESTAMP = "timestamp"

class MemoryTagHelper:
    """记忆标签操作辅助类"""
    
    @staticmethod
    def mark_compressed(metadata: dict, strategy: str, original_length: int) -> dict:
        """标记记忆为已压缩"""
        if metadata is None:
            metadata = {}
        metadata[MemoryTags.COMPRESSED] = True
        metadata[MemoryTags.COMPRESSED_TIME] = datetime.now().isoformat()
        metadata[MemoryTags.COMPRESSED_STRATEGY] = strategy
        metadata[MemoryTags.ORIGINAL_LENGTH] = original_length
        return metadata
    
    @staticmethod
    def mark_preserve(metadata: dict, reason: str) -> dict:
        """标记记忆为保留（不压缩）"""
        if metadata is None:
            metadata = {}
        metadata[MemoryTags.PRESERVE] = True
        metadata[MemoryTags.PRESERVE_REASON] = reason
        return metadata
    
    @staticmethod
    def mark_protected(metadata: dict, reason: str) -> dict:
        """标记记忆为受保护（不归档）"""
        if metadata is None:
            metadata = {}
        metadata[MemoryTags.PROTECTED] = True
        metadata[MemoryTags.PROTECTED_REASON] = reason
        return metadata
    
    @staticmethod
    def mark_moved_to_l3(metadata: dict) -> dict:
        """标记记忆从L2移动到L3"""
        if metadata is None:
            metadata = {}

        metadata[MemoryTags.MOVED_FROM_L2] = True
        metadata[MemoryTags.MOVED_FROM_L2_TIME] = datetime.now().isoformat()
        if MemoryTags.MOVED_FROM_L3 in metadata:
            del metadata[MemoryTags.MOVED_FROM_L3]
        if MemoryTags.MOVED_FROM_L3_TIME in metadata:
            del metadata[MemoryTags.MOVED_FROM_L3_TIME]
        return metadata
    
    @staticmethod
    def mark_moved_to_l2(metadata: dict, has_compressed: bool = False) -> dict:
        """标记记忆从L3回流到L2"""
        if metadata is None:
            metadata = {}

        metadata[MemoryTags.PROMOTED_TO_L2] = True
        metadata[MemoryTags.PROMOTED_TIME] = datetime.now().isoformat()
        metadata[MemoryTags.MOVED_FROM_L3] = True
        metadata[MemoryTags.MOVED_FROM_L3_TIME] = datetime.now().isoformat()
        if MemoryTags.MOVED_FROM_L2 in metadata:
            del metadata[MemoryTags.MOVED_FROM_L2]
        if MemoryTags.MOVED_FROM_L2_TIME in metadata:
            del metadata[MemoryTags.MOVED_FROM_L2_TIME]
        if has_compressed:
            metadata[MemoryTags.HAS_COMPRESSED_VERSION] = True
        return metadata
    
    @staticmethod
    def mark_pending_compression(metadata: dict, retry_count: int = 0) -> dict:
        """标记记忆为待压缩"""
        if metadata is None:
            metadata = {}

        metadata[MemoryTags.PENDING_COMPRESSION] = True
        metadata[MemoryTags.PENDING_SINCE] = datetime.now().isoformat()
        metadata[MemoryTags.RETRY_COUNT] = retry_count
        return metadata
    
    @staticmethod
    def clear_pending_compression(metadata: dict) -> dict:
        """清除待压缩标记"""
        if metadata is None:
            return {}
        metadata.pop(MemoryTags.PENDING_COMPRESSION, None)
        metadata.pop(MemoryTags.PENDING_SINCE, None)
        return metadata
    
    @staticmethod
    def is_compressed(metadata: dict) -> bool:
        """检查是否已压缩"""
        if metadata is None:
            return False
        return metadata.get(MemoryTags.COMPRESSED) is True
    
    @staticmethod
    def is_protected(metadata: dict) -> bool:
        """检查是否受保护"""
        if metadata is None:
            return False
        return (
            metadata.get(MemoryTags.PROTECTED) is True or
            metadata.get(MemoryTags.IMPORTANT) is True or
            metadata.get(MemoryTags.PRESERVE) is True
        )
    
    @staticmethod
    def get_merge_priority(metadata: dict) -> int:
        """
        获取记忆合并优先级
        
        优先级规则：
        - protected: 100 (最高，不参与合并)
        - forgotten: 0 (最低，可被合并/删除)
        - important: 80
        - high_density: 70 (高密度内容不合并)
        - preserve: 60
        - compressed: 40
        - 普通: 20
        
        合并时：
        1. protected/high_density 记忆不参与合并
        2. forgotten 记忆优先被合并/删除
        3. 高优先级记忆作为主记录保留
        4. 同优先级按时间排序
        
        Args:
            metadata: 记忆元数据
        
        Returns:
            优先级数值（越高越优先）
        """
        if metadata is None:
            return 20
        
        if metadata.get(MemoryTags.PROTECTED) is True:
            return 100
        
        if metadata.get(MemoryTags.FORGOTTEN) is True:
            return 0
        
        if metadata.get(MemoryTags.IMPORTANT) is True:
            return 80
        
        if metadata.get(MemoryTags.HIGH_DENSITY) is True:
            return 70
        
        if metadata.get(MemoryTags.PRESERVE) is True:
            return 60
        
        if metadata.get(MemoryTags.COMPRESSED) is True:
            return 40
        
        return 20
    
    @staticmethod
    def should_skip_merge(metadata: dict) -> bool:
        """
        检查记忆是否应该跳过合并
        
        protected 和 high_density 记忆不应该被合并或删除
        
        Args:
            metadata: 记忆元数据
        
        Returns:
            是否跳过合并
        """
        if metadata is None:
            return False
        
        if metadata.get(MemoryTags.PROTECTED) is True:
            return True
        
        if metadata.get(MemoryTags.HIGH_DENSITY) is True:
            return True
        
        return False
    
    @staticmethod
    def is_forgotten(metadata: dict) -> bool:
        """
        检查是否已被用户标记遗忘
        
        Args:
            metadata: 记忆元数据
        
        Returns:
            是否已遗忘
        """
        if metadata is None:
            return False
        return metadata.get(MemoryTags.FORGOTTEN) is True
    
    @staticmethod
    def mark_forgotten(metadata: dict, reason: str = "user_request") -> dict:
        """
        标记记忆为遗忘状态
        
        遗忘的记忆：
        - 不参与检索
        - 优先被合并/删除
        - 保留原始数据（可恢复）
        
        Args:
            metadata: 记忆元数据
            reason: 遗忘原因
        
        Returns:
            更新后的元数据
        """
        if metadata is None:
            metadata = {}

        metadata[MemoryTags.FORGOTTEN] = True
        metadata[MemoryTags.FORGOTTEN_TIME] = datetime.now().isoformat()
        metadata[MemoryTags.FORGOTTEN_REASON] = reason
        
        return metadata
    
    @staticmethod
    def unmark_forgotten(metadata: dict) -> dict:
        """
        取消遗忘标记（恢复记忆）
        
        Args:
            metadata: 记忆元数据
        
        Returns:
            更新后的元数据
        """
        if metadata is None:
            return {}
        
        metadata.pop(MemoryTags.FORGOTTEN, None)
        metadata.pop(MemoryTags.FORGOTTEN_TIME, None)
        metadata.pop(MemoryTags.FORGOTTEN_REASON, None)
        
        return metadata
    
    @staticmethod
    def mark_high_density(metadata: dict, reason: str = "auto_detected") -> dict:
        """
        标记为高密度内容
        
        高密度内容：
        - 不被压缩
        - 不参与合并
        - 保持原始内容
        
        Args:
            metadata: 记忆元数据
            reason: 标记原因
        
        Returns:
            更新后的元数据
        """
        if metadata is None:
            metadata = {}
        
        metadata[MemoryTags.HIGH_DENSITY] = True
        metadata[MemoryTags.HIGH_DENSITY_REASON] = reason
        
        return metadata
    
    @staticmethod
    def is_pending_compression(metadata: dict) -> bool:
        """检查是否待压缩"""
        if metadata is None:
            return False
        return metadata.get(MemoryTags.PENDING_COMPRESSION) is True
    
    @staticmethod
    def is_in_cooldown_from_l2(metadata: dict, cooldown_hours: int) -> bool:
        """检查是否在从L2移动后的冷却期内"""
        if metadata is None:
            return False
        if metadata.get(MemoryTags.PROMOTED_TO_L2):

            promoted_time_str = metadata.get(MemoryTags.PROMOTED_TIME)
            if promoted_time_str:
                try:
                    promoted_time = datetime.fromisoformat(promoted_time_str)
                    cooldown_end = promoted_time + timedelta(hours=cooldown_hours)
                    if datetime.now() < cooldown_end:
                        return True
                except Exception:
                    pass
        return False
    
    @staticmethod
    def is_in_cooldown_from_l3(metadata: dict, cooldown_hours: int) -> bool:
        """检查是否在从L3回流后的冷却期内"""
        if metadata is None:
            return False
        if metadata.get(MemoryTags.MOVED_FROM_L3):

            moved_time_str = metadata.get(MemoryTags.MOVED_FROM_L3_TIME)
            if moved_time_str:
                try:
                    moved_time = datetime.fromisoformat(moved_time_str)
                    cooldown_end = moved_time + timedelta(hours=cooldown_hours * MemoryConstants.L3_COOLDOWN_MULTIPLIER)
                    if datetime.now() < cooldown_end:
                        return True
                except Exception:
                    pass
        return False
    
    @staticmethod
    def mark_merged(metadata: dict, merged_count: int, merged_from_ids: list,
                    merged_from_priorities: list = None, needs_recompression: bool = True) -> dict:
        """
        标记记忆为已合并
        
        Args:
            metadata: 记忆元数据
            merged_count: 合并的记忆数量
            merged_from_ids: 被合并的记忆ID列表
            merged_from_priorities: 被合并记忆的优先级列表
            needs_recompression: 是否需要重新压缩
        
        Returns:
            更新后的元数据
        """
        if metadata is None:
            metadata = {}

        metadata[MemoryTags.MERGED] = True
        metadata[MemoryTags.MERGED_AT] = datetime.now().isoformat()
        metadata[MemoryTags.MERGED_COUNT] = merged_count
        metadata[MemoryTags.MERGED_FROM_IDS] = merged_from_ids
        
        if merged_from_priorities:
            metadata[MemoryTags.MERGED_FROM_PRIORITIES] = merged_from_priorities
        
        if needs_recompression:
            metadata[MemoryTags.NEEDS_RECOMPRESSION] = True
        
        return metadata
    
    @staticmethod
    def mark_needs_revectorization(metadata: dict, reason: str = "merged") -> dict:
        """
        标记需要重新向量化
        
        Args:
            metadata: 记忆元数据
            reason: 原因
        
        Returns:
            更新后的元数据
        """
        if metadata is None:
            metadata = {}
        
        metadata[MemoryTags.NEEDS_REVECTORIZATION] = True
        return metadata
    
    @staticmethod
    def mark_needs_recompression(metadata: dict, reason: str = "merged") -> dict:
        """
        标记需要重新压缩
        
        Args:
            metadata: 记忆元数据
            reason: 原因
        
        Returns:
            更新后的元数据
        """
        if metadata is None:
            metadata = {}
        
        metadata[MemoryTags.NEEDS_RECOMPRESSION] = True
        return metadata
    
    @staticmethod
    def mark_upgraded(metadata: dict, from_type: str = "sqlite_only") -> dict:
        """
        标记记忆升级来源
        
        Args:
            metadata: 记忆元数据
            from_type: 升级来源类型
        
        Returns:
            更新后的元数据
        """
        if metadata is None:
            metadata = {}

        metadata[MemoryTags.UPGRADED_FROM_SQLITE_ONLY] = True
        metadata[MemoryTags.UPGRADED_TIME] = datetime.now().isoformat()
        return metadata
    
    @staticmethod
    def set_semantic_tag(metadata: dict, tag_dict: dict) -> dict:
        """
        设置语义标签
        
        Args:
            metadata: 记忆元数据
            tag_dict: 语义标签字典
        
        Returns:
            更新后的元数据
        """
        if metadata is None:
            metadata = {}
        
        metadata[MemoryTags.SEMANTIC_TAG] = tag_dict
        return metadata
    
    @staticmethod
    def get_semantic_tag(metadata: dict) -> dict:
        """
        获取语义标签
        
        Args:
            metadata: 记忆元数据
        
        Returns:
            语义标签字典，不存在则返回空字典
        """
        if metadata is None:
            return {}
        return metadata.get(MemoryTags.SEMANTIC_TAG, {})
    
    @staticmethod
    def mark_tag_corrected(metadata: dict) -> dict:
        """
        标记标签已修正
        
        Args:
            metadata: 记忆元数据
        
        Returns:
            更新后的元数据
        """
        if metadata is None:
            metadata = {}

        metadata[MemoryTags.TAG_CORRECTED_AT] = datetime.now().isoformat()
        return metadata
    
    @staticmethod
    def set_nonsense_filter_result(metadata: dict, result: str) -> dict:
        """
        设置垃圾过滤结果
        
        Args:
            metadata: 记忆元数据
            result: 过滤结果 (normal/nonsense)
        
        Returns:
            更新后的元数据
        """
        if metadata is None:
            metadata = {}
        
        metadata[MemoryTags.NONSENSE_FILTER_RESULT] = result
        return metadata
    
    @staticmethod
    def get_nonsense_filter_result(metadata: dict, default: str = "normal") -> str:
        """
        获取垃圾过滤结果
        
        Args:
            metadata: 记忆元数据
            default: 默认值
        
        Returns:
            过滤结果
        """
        if metadata is None:
            return default
        return metadata.get(MemoryTags.NONSENSE_FILTER_RESULT, default)
    
    @staticmethod
    def set_sqlite_id(metadata: dict, sqlite_id: int) -> dict:
        """
        设置 SQLite ID
        
        Args:
            metadata: 记忆元数据
            sqlite_id: SQLite 记录 ID
        
        Returns:
            更新后的元数据
        """
        if metadata is None:
            metadata = {}
        
        metadata[MemoryTags.SQLITE_ID] = sqlite_id
        return metadata
    
    @staticmethod
    def get_sqlite_id(metadata: dict) -> Optional[int]:
        """
        获取 SQLite ID
        
        Args:
            metadata: 记忆元数据
        
        Returns:
            SQLite ID，不存在则返回 None
        """
        if metadata is None:
            return None
        return metadata.get(MemoryTags.SQLITE_ID)
    
    @staticmethod
    def is_sqlite_only(metadata: dict) -> bool:
        """
        检查是否仅存储在 SQLite（未向量化）
        
        Args:
            metadata: 记忆元数据
        
        Returns:
            是否仅 SQLite 存储
        """
        if metadata is None:
            return False
        return metadata.get(MemoryTags.IS_SQLITE_ONLY, False) is True
    
    @staticmethod
    def clear_compression_state(metadata: dict) -> dict:
        """
        清除压缩相关状态
        
        Args:
            metadata: 记忆元数据
        
        Returns:
            更新后的元数据
        """
        if metadata is None:
            return {}
        
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
            metadata.pop(key, None)
        
        return metadata
