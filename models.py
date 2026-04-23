# models.py
# 统一数据模型定义
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class MessageSource(Enum):
    """消息来源"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    CLOUD = "cloud"
    LOCAL = "local"


class MemorySource(Enum):
    """记忆来源"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    LOCAL = "local"
    CLOUD = "cloud"


class StorageType(Enum):
    """存储类型"""
    FULL = "full"
    SQLITE_ONLY = "sqlite_only"
    DISCARD = "discard"


class UIState(Enum):
    """UI状态枚举"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    PROCESSING_LOCAL = "processing_local"
    PROCESSING_CLOUD = "processing_cloud"
    PROCESSING_HYBRID = "processing_hybrid"
    BACKGROUND_COMPRESSING = "background_compressing"
    BACKGROUND_DEDUPING = "background_deduping"
    ERROR = "error"


@dataclass
class ConversationMetadata:
    """
    对话元数据结构
    
    用于记录对话的上下文信息，支持混合检索
    """
    source: str = "local"
    timestamp: str = ""
    context_used: int = 0
    memories_retrieved: int = 0
    chunk_id: str = ""
    session_id: str = ""
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.chunk_id:
            self.chunk_id = f"chunk_{int(datetime.now().timestamp() * 1000)}"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationMetadata":
        """从字典创建"""
        return cls(
            source=data.get("source", "local"),
            timestamp=data.get("timestamp", ""),
            context_used=data.get("context_used", 0),
            memories_retrieved=data.get("memories_retrieved", 0),
            chunk_id=data.get("chunk_id", ""),
            session_id=data.get("session_id", ""),
            tags=data.get("tags", [])
        )


@dataclass
class MemoryMetadata:
    """
    记忆元数据结构
    
    用于向量库和数据库的记忆存储
    """
    source: str = "user"
    timestamp: str = ""
    memory_type: str = "conversation"
    chunk_id: str = ""
    session_id: str = ""
    tags: List[str] = field(default_factory=list)
    importance: float = 1.0
    storage_type: str = "full"
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.chunk_id:
            self.chunk_id = f"mem_{int(datetime.now().timestamp() * 1000)}"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryMetadata":
        """从字典创建"""
        if isinstance(data, cls):
            return data
        
        return cls(
            source=data.get("source", "user"),
            timestamp=data.get("timestamp", ""),
            memory_type=data.get("memory_type", "conversation"),
            chunk_id=data.get("chunk_id", ""),
            session_id=data.get("session_id", ""),
            tags=data.get("tags", []),
            importance=data.get("importance", 1.0),
            storage_type=data.get("storage_type", "full")
        )


@dataclass
class MemoryRecord:
    """
    记忆记录结构
    
    核心领域模型，用于 L2/L3 存储层
    """
    id: Optional[int] = None
    text: str = ""
    compressed_text: Optional[str] = None
    source: str = "user"
    weight: float = 1.0
    access_count: int = 0
    last_access_time: Optional[str] = None
    created_time: Optional[str] = None
    metadata: Dict[str, Any] = None
    vector_id: Optional[str] = None
    is_vectorized: int = 0
    bm25_score: float = 0.0
    
    def __post_init__(self):
        if self.created_time is None:
            self.created_time = datetime.now().isoformat()
        if self.last_access_time is None:
            self.last_access_time = self.created_time
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ConversationRecord:
    """
    会话记录结构
    
    用于 SQLite 存储会话历史
    """
    id: Optional[int] = None
    session_id: str = ""
    user_input: str = ""
    assistant_response: str = ""
    source: str = "local"
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.session_id:
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "user": self.user_input,
            "assistant": self.assistant_response,
            "source": self.source,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], session_id: str = "") -> "ConversationRecord":
        """从字典创建"""
        return cls(
            session_id=session_id,
            user_input=data.get("user", ""),
            assistant_response=data.get("assistant", ""),
            source=data.get("source", "local"),
            timestamp=data.get("timestamp", ""),
            metadata=data.get("metadata", {})
        )
