# models.py
# 统一数据模型定义
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
import uuid


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
            self.chunk_id = f"chunk_{uuid.uuid4().hex[:12]}"
    
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
            self.chunk_id = f"mem_{uuid.uuid4().hex[:12]}"
    
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
    is_important: bool = False
    session_id: str = ""
    content_hash: str = ""
    
    def __post_init__(self):
        if self.created_time is None:
            self.created_time = datetime.now().isoformat()
        if self.last_access_time is None:
            self.last_access_time = self.created_time
        if self.metadata is None:
            self.metadata = {}
        if not self.session_id:
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not self.content_hash and self.text:
            import hashlib
            self.content_hash = hashlib.md5(self.text.encode('utf-8')).hexdigest()[:16]


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


class SessionStatus(Enum):
    """会话状态枚举"""
    ACTIVE = "active"
    ENDED = "ended"
    ARCHIVED = "archived"


@dataclass
class SessionRecord:
    """
    会话记录结构
    
    用于持久化会话状态，支持会话恢复
    """
    id: Optional[int] = None
    session_id: str = ""
    status: str = SessionStatus.ACTIVE.value
    created_time: str = ""
    updated_time: str = ""
    message_count: int = 0
    ui_state_snapshot: Dict[str, Any] = field(default_factory=dict)
    context_snapshot: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.session_id:
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not self.created_time:
            self.created_time = datetime.now().isoformat()
        if not self.updated_time:
            self.updated_time = self.created_time
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "session_id": self.session_id,
            "status": self.status,
            "created_time": self.created_time,
            "updated_time": self.updated_time,
            "message_count": self.message_count,
            "ui_state_snapshot": self.ui_state_snapshot,
            "context_snapshot": self.context_snapshot,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionRecord":
        """从字典创建"""
        return cls(
            id=data.get("id"),
            session_id=data.get("session_id", ""),
            status=data.get("status", SessionStatus.ACTIVE.value),
            created_time=data.get("created_time", ""),
            updated_time=data.get("updated_time", ""),
            message_count=data.get("message_count", 0),
            ui_state_snapshot=data.get("ui_state_snapshot", {}),
            context_snapshot=data.get("context_snapshot", {}),
            metadata=data.get("metadata", {})
        )


@dataclass
class L1Message:
    """
    L1 内存消息结构
    
    用于 L1 缓存层，支持去重和临时 ID
    """
    temp_id: str = ""
    content: str = ""
    role: str = "user"
    timestamp: str = ""
    session_id: str = ""
    is_important: bool = False
    content_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    sqlite_id: Optional[int] = None
    vector_id: Optional[str] = None
    
    def __post_init__(self):
        if not self.temp_id:
            self.temp_id = f"temp_{uuid.uuid4().hex[:12]}"
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.session_id:
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not self.content_hash and self.content:
            import hashlib
            self.content_hash = hashlib.md5(self.content.encode('utf-8')).hexdigest()[:16]
    
    def get_dedup_key(self) -> str:
        """获取去重键"""
        if self.sqlite_id:
            return f"sqlite_{self.sqlite_id}"
        return f"temp_{self.temp_id}"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "temp_id": self.temp_id,
            "content": self.content,
            "role": self.role,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "is_important": self.is_important,
            "content_hash": self.content_hash,
            "metadata": self.metadata,
            "sqlite_id": self.sqlite_id,
            "vector_id": self.vector_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "L1Message":
        """从字典创建"""
        return cls(
            temp_id=data.get("temp_id", ""),
            content=data.get("content", ""),
            role=data.get("role", "user"),
            timestamp=data.get("timestamp", ""),
            session_id=data.get("session_id", ""),
            is_important=data.get("is_important", False),
            content_hash=data.get("content_hash", ""),
            metadata=data.get("metadata", {}),
            sqlite_id=data.get("sqlite_id"),
            vector_id=data.get("vector_id")
        )
