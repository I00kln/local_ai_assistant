# memory_tags.py
# 记忆标签统一管理

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
    
    # 来源标签
    SOURCE_L1 = "L1"
    SOURCE_L2 = "L2"
    SOURCE_L3 = "L3"


class MemoryTagHelper:
    """记忆标签操作辅助类"""
    
    @staticmethod
    def mark_compressed(metadata: dict, strategy: str, original_length: int) -> dict:
        """标记记忆为已压缩"""
        if metadata is None:
            metadata = {}
        from datetime import datetime
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
        from datetime import datetime
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
        from datetime import datetime
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
        from datetime import datetime
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
            from datetime import datetime, timedelta
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
            from datetime import datetime, timedelta
            moved_time_str = metadata.get(MemoryTags.MOVED_FROM_L3_TIME)
            if moved_time_str:
                try:
                    moved_time = datetime.fromisoformat(moved_time_str)
                    cooldown_end = moved_time + timedelta(hours=cooldown_hours * 2)
                    if datetime.now() < cooldown_end:
                        return True
                except Exception:
                    pass
        return False
