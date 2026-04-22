# 🏗️ MemoryMind 系统架构设计文档

> **文档版本**：v1.0  
> **最后更新**：2026年4月22日  
> **适用版本**：MemoryMind v0.2.0

---

## 目录

1. [设计目标与原则](#1-设计目标与原则)
2. [系统分层架构](#2-系统分层架构)
3. [核心模块详述](#3-核心模块详述)
4. [关键流程时序图](#4-关键流程时序图)
5. [数据模型](#5-数据模型)
6. [并发模型](#6-并发模型)
7. [扩展点设计](#7-扩展点设计)
8. [部署架构](#8-部署架构)

---

## 1. 设计目标与原则

### 1.1 核心设计目标

| 目标 | 描述 | 优先级 |
|------|------|--------|
| 🏠 **本地优先** | 核心功能完全本地运行，无需网络连接 | P0 |
| 🔒 **隐私保护** | 敏感数据不离开本地，用户拥有完全控制权 | P0 |
| 🧠 **长期记忆** | 实现跨越会话的记忆持久化和智能检索 | P0 |
| 📈 **渐进式增强** | 基础功能本地可用，云端能力按需接入 | P1 |
| ⚡ **低延迟响应** | 检索延迟控制在 50ms 以内，用户体验流畅 | P1 |
| 🔄 **自动优化** | 自动压缩、去重、遗忘，减少人工维护 | P2 |

### 1.2 遵循的设计原则

#### 分层解耦原则

```
设计理念：上层依赖下层，下层不依赖上层
层间通过接口通信，实现细节对上层透明

表示层 ──▶ 业务逻辑层 ──▶ 记忆管理层 ──▶ 存储层
              单向依赖，无循环引用
```

#### 单一职责原则（SRP）

| 模块 | 单一职责 |
|------|----------|
| MemoryManager | 记忆检索与存储协调 |
| VectorStore | 向量存储与相似度计算 |
| SQLiteStore | 关系数据持久化 |
| NonsenseFilter | 内容质量评估与过滤 |
| ContextBuilder | 上下文组装与长度控制 |
| AsyncProcessor | 后台异步任务调度 |

#### 开闭原则（OCP）

```
扩展开放：
├─ 新增存储后端：实现 StorageInterface
├─ 新增过滤规则：继承 FilterStrategy
├─ 新增 LLM 服务：继承 LLMClient
└─ 新增压缩策略：继承 CompressionStrategy

修改封闭：
├─ 核心模块接口稳定
├─ 配置驱动行为变化
└─ 事件机制解耦模块
```

#### 依赖倒置原则（DIP）

```
高层模块依赖抽象接口，而非具体实现
MemoryManager 依赖 VectorStore 接口，而非 ChromaDB 具体实现
```

### 1.3 架构约束

| 约束类型 | 约束内容 | 理由 |
|----------|----------|------|
| 单向依赖 | 模块间禁止循环引用 | 避免耦合地狱 |
| 纯净领域 | 业务逻辑不依赖框架 | 可测试、可移植 |
| 接口隔离 | 不设计上帝接口 | 职责清晰 |
| 迪米特法则 | 禁止链式调用超过一级 | 降低耦合 |

---

## 2. 系统分层架构

### 2.1 四层架构总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                          表示层 (Presentation Layer)                  │
│  组件：ChatWindow, UIStateManager                                    │
│  职责：用户交互、界面渲染、状态展示                                    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        业务逻辑层 (Business Layer)                   │
│  组件：ContextBuilder, HybridClient, LLMClient, CloudClient         │
│  职责：上下文构建、LLM调用、云端协同、响应处理                         │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        记忆管理层 (Memory Layer)                     │
│  组件：MemoryManager, AsyncProcessor, NonsenseFilter,               │
│        SensitiveFilter, TransactionCoordinator                      │
│  职责：记忆检索、存储协调、压缩、过滤、事务管理                         │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          存储层 (Storage Layer)                      │
│  组件：VectorStore, SQLiteStore, EmbeddingService                   │
│  职责：数据持久化、向量计算、缓存管理                                  │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 各层详细说明

#### 表示层（Presentation Layer）

```
ChatWindow 职责：
• 渲染对话界面（用户/助理/云端分色显示）
• 收集用户输入
• 显示系统状态（记忆数、处理状态）
• 提供控制按钮（清空、保存、监控）

状态管理：
• conversation_history: List[Dict]  当前会话历史
• _initialized: bool                初始化状态
• _closing: bool                    关闭状态

依赖：
• UIStateManager（状态展示）
• MemoryManager（记忆操作）
• HybridClient（LLM调用）
• AsyncProcessor（后台处理）
```

#### 业务逻辑层（Business Layer）

```
ContextBuilder 职责：
• 多级记忆检索（L1→L2→L3）
• 上下文长度控制（Token预算）
• 记忆格式化与排序

策略：
• 本地模式：高阈值（0.90），少量记忆
• 云端模式：低阈值（0.70），大量记忆
• 混合模式：中等阈值，中等记忆量

HybridClient 职责：
• 协调本地/云端 LLM 调用
• 实现混合模式（本地压缩+云端推理）
• 故障降级处理
```

#### 记忆管理层（Memory Layer）

```
MemoryManager 职责：
• 统一记忆检索接口（L1→L2→L3）
• 记忆存储协调
• L3→L2 回填机制
• 权重管理

状态：
• conversation_history: List[Dict]  L1 内存缓存
• _write_version: int               版本号（并发控制）
• stats: Dict                       统计信息

AsyncProcessor 职责：
• 后台异步处理队列
• 记忆压缩调度
• 记忆去重
• 定期维护任务

队列策略：
• max_queue_size: 1000
• queue_full_action: drop_oldest
• batch_size: 5

NonsenseFilter 职责：
• 三层废话过滤（规则→密度→向量）
• 存储策略决策（full/sqlite_only/discard）
• 保护有价值短文本

过滤层级：
• Layer1: 规则过滤（~1ms）
• Layer2: 密度评分（~5ms）
• Layer3: 向量匹配（~10ms）
```

#### 存储层（Storage Layer）

```
VectorStore 职责：
• ChromaDB 向量存储封装
• 向量相似度检索
• 元数据过滤

特点：
• 自动持久化到磁盘
• 支持增量更新
• 内置去重

SQLiteStore 职责：
• SQLite 关系数据存储
• 全文搜索（FTS）
• 权重管理
• 事务支持

特点：
• WAL 模式并发读
• 连接池管理
• 自动备份恢复

EmbeddingService 职责：
• ONNX 模型加载与管理
• 文本嵌入计算
• LRU 缓存

特点：
• 单例模式
• 延迟加载
• 降级模式（随机向量）
```

### 2.3 层间交互协议

```
表示层 ──▶ 业务逻辑层
────────────────────
请求：build_context(user_input, conversation_history, mode)
响应：(memory_context, processed_input, retrieved_memories, has_memories)

请求：chat(messages, mode)
响应：response_text

业务逻辑层 ──▶ 记忆管理层
────────────────────
请求：search(query, top_k, threshold, include_l3)
响应：List[MemorySearchResult]

请求：add_conversation(user_input, assistant_response, metadata)
响应：None（异步处理）

记忆管理层 ──▶ 存储层
────────────────────
请求：add(texts, metadatas)  [VectorStore]
响应：List[vector_id]

请求：add_memory(record)  [SQLiteStore]
响应：memory_id
```

---

## 3. 核心模块详述

### 3.1 MemoryManager（三层记忆管理器）

#### 模块职责

1. 统一记忆检索接口（L1→L2→L3 多级检索）
2. L1 内存缓存管理（当前会话历史）
3. L3→L2 回填机制（冷数据激活）
4. 检索结果合并与排序
5. 权重更新协调
6. 并发控制（版本号检测）

#### 对外接口

```python
class MemoryManager:
    def add_conversation(
        self, 
        user_input: str, 
        assistant_response: str, 
        metadata: Dict = None
    ) -> None:
        """添加对话到 L1 内存层"""
        pass
    
    def search(
        self, 
        query: str, 
        top_k: int = None, 
        include_l3: bool = True, 
        threshold: float = None,
        include_l1: bool = True
    ) -> List[MemorySearchResult]:
        """统一搜索接口 - L1→L2→L3"""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        pass
```

#### 内部状态管理

```python
class MemoryManager:
    def __init__(self):
        # L1 内存缓存
        self.conversation_history: List[Dict] = []
        self.max_l1_size = 25
        
        # 并发控制
        self.lock = threading.RLock()
        self._read_lock = threading.Lock()
        self._write_version = 0
        self._last_sync_time = time.time()
        
        # 统计信息
        self.stats = {
            "l1_hits": 0,
            "l2_hits": 0,
            "l3_hits": 0,
            "l3_backfills": 0,
            "compressions": 0,
            "forgotten": 0,
            "retry_reads": 0
        }
```

#### 依赖关系

```
MemoryManager
    ├──▶ VectorStore (L2 向量存储)
    │       └──▶ ChromaDB
    │       └──▶ EmbeddingService
    ├──▶ SQLiteStore (L3 数据库存储)
    │       └──▶ SQLite
    ├──▶ EventBus (事件发布订阅)
    └──▶ TransactionCoordinator (事务协调)
```

#### 关键数据结构

```python
@dataclass
class MemorySearchResult:
    """记忆搜索结果"""
    text: str                    # 记忆文本
    source: str                  # 来源层：L1, L2, L3
    similarity: float            # 相似度分数
    weight: float                # 记忆权重
    metadata: Dict[str, Any]     # 元数据
```

---

### 3.2 AsyncProcessor（异步记忆处理器）

#### 模块职责

1. 后台异步处理队列管理
2. 记忆压缩调度（LLM/规则）
3. 记忆去重（内容哈希）
4. 定期维护任务（遗忘、归档、清理）
5. 熔断器保护
6. 队列溢出处理

#### 对外接口

```python
class AsyncProcessor:
    def enqueue(self, task: ProcessingTask) -> bool:
        """将任务加入处理队列"""
        pass
    
    def start(self) -> None:
        """启动后台处理线程"""
        pass
    
    def stop(self) -> None:
        """停止后台处理线程"""
        pass
    
    def force_flush(self) -> None:
        """强制刷新所有缓冲区"""
        pass
```

#### 内部状态管理

```python
class AsyncProcessor:
    def __init__(self):
        # 处理队列
        self._queue: queue.Queue = queue.Queue(maxsize=1000)
        self._batch_buffer: List[ProcessingTask] = []
        
        # 线程管理
        self._worker_thread: threading.Thread
        self._running: bool = False
        
        # 定时任务
        self._dedup_interval: int = 300      # 去重间隔
        self._compression_interval: int = 600 # 压缩间隔
        self._forget_interval: int = 3600    # 遗忘间隔
        self._flush_interval: int = 30       # 刷新间隔
        
        # 熔断器
        self._circuit_breaker = {
            "failures": 0,
            "threshold": 5,
            "open_until": 0,
            "reset_timeout": 300
        }
```

#### 关键数据结构

```python
@dataclass
class ProcessingTask:
    """处理任务"""
    task_type: str               # 任务类型：add, compress, forget
    data: Dict[str, Any]         # 任务数据
    priority: int = 0            # 优先级
    created_time: float = 0      # 创建时间

class CompressionStrategy(ABC):
    """压缩策略抽象接口"""
    
    @abstractmethod
    def compress(self, text: str) -> Optional[str]:
        """压缩文本，返回 None 表示无法压缩"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查策略是否可用"""
        pass
```

---

### 3.3 ContextBuilder（上下文构建器）

#### 模块职责

1. 多级记忆检索协调（L1→L2→L3）
2. Token 预算分配
3. 上下文长度控制
4. 记忆格式化与排序
5. 不同模式的阈值调整

#### 对外接口

```python
class ContextBuilder:
    def build_context(
        self, 
        user_input: str, 
        conversation_history: List[Dict] = None,
        mode: str = "local"
    ) -> Tuple[str, str, List[Dict], bool]:
        """
        构建完整的对话上下文
        
        Args:
            user_input: 用户输入
            conversation_history: 对话历史
            mode: "local" | "cloud_only" | "hybrid"
        
        Returns:
            memory_context: 格式化后的记忆字符串
            processed_user_input: 处理后的用户输入
            retrieved_memories: 检索到的记忆列表
            has_memories: 是否检索到有效记忆
        """
        pass
```

#### 内部状态管理

```python
class ContextBuilder:
    def __init__(self, memory_manager):
        self.memory = memory_manager
        
        # Token 预算
        self.max_context = 8192           # 最大上下文
        self.max_output = 3000            # 最大输出
        self.max_memory_tokens = 5000     # 最大记忆 Token
        self.system_reserve = 200         # 系统提示预留
        
        # 检索配置
        self.l1_min_results = 2           # L1 最小结果数
        self.l2_default_threshold = 0.90  # 默认阈值
```

---

### 3.4 VectorStore（ChromaDB 封装）

#### 模块职责

1. ChromaDB 向量存储封装
2. 向量相似度检索
3. 元数据过滤查询
4. 自动持久化
5. 内置去重

#### 对外接口

```python
class VectorStore:
    def add(
        self, 
        texts: List[str], 
        metadatas: List[Dict] = None,
        ids: List[str] = None
    ) -> List[str]:
        """添加向量"""
        pass
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        where: Dict = None
    ) -> List[SearchResult]:
        """搜索相似向量"""
        pass
    
    def delete(self, ids: List[str]) -> bool:
        """删除向量"""
        pass
    
    def count(self) -> int:
        """获取向量数量"""
        pass
```

#### 内部状态管理

```python
class VectorStore:
    def __init__(self):
        # ChromaDB 配置
        self.persist_directory: str = "chroma_db"
        self.collection_name: str = "memories"
        
        # 嵌入函数
        self._embedding_function: ONNXEmbeddingFunction
        
        # ChromaDB 客户端
        self._client: chromadb.Client
        self._collection: chromadb.Collection
        
        # 线程安全
        self._lock = threading.Lock()
```

---

### 3.5 SQLiteStore（SQLite 封装）

#### 模块职责

1. SQLite 关系数据存储
2. 全文搜索（FTS）
3. 权重管理与衰减
4. 事务支持
5. 连接池管理
6. 自动备份恢复

#### 对外接口

```python
class SQLiteStore:
    def add_memory(self, record: MemoryRecord) -> int:
        """添加记忆"""
        pass
    
    def search_memories(
        self, 
        query: str, 
        limit: int = 10,
        min_weight: float = 0.0
    ) -> List[MemoryRecord]:
        """搜索记忆（全文搜索）"""
        pass
    
    def update_weight(self, memory_id: int, weight: float) -> bool:
        """更新权重"""
        pass
    
    def decay_weights(self, batch_size: int = 1000) -> int:
        """权重衰减"""
        pass
    
    def forget_old_memories(
        self, 
        min_weight: float = 0.3, 
        max_age_days: int = 30
    ) -> int:
        """遗忘过期记忆"""
        pass
```

#### 关键数据结构

```python
@dataclass
class MemoryRecord:
    """记忆记录结构"""
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
```

---

### 3.6 NonsenseFilter（废话过滤器）

#### 模块职责

1. 三层废话过滤（规则→密度→向量）
2. 存储策略决策（full/sqlite_only/discard）
3. 保护有价值短文本
4. 高密度内容识别

#### 对外接口

```python
class NonsenseFilter:
    def filter(self, text: str, context: Dict = None) -> FilterResult:
        """过滤文本"""
        pass
    
    def is_nonsense(self, text: str) -> Tuple[bool, str]:
        """快速判断是否为废话"""
        pass
```

#### 关键数据结构

```python
@dataclass
class FilterResult:
    """过滤结果"""
    is_nonsense: bool            # 是否为废话
    reason: str                  # 原因
    confidence: float            # 置信度（0-1）
    storage_type: str = "full"   # 存储类型
    metadata: Dict = None        # 额外元数据

class StorageType(Enum):
    """存储类型"""
    FULL = "full"                # 存向量库 + SQLite
    SQLITE_ONLY = "sqlite_only"  # 仅存 SQLite
    DISCARD = "discard"          # 完全丢弃
```

---

## 4. 关键流程时序图

### 4.1 用户对话处理流程

```
用户          ChatWindow       ContextBuilder      MemoryManager      LLMClient
 │                │                  │                  │                │
 │  输入问题      │                  │                  │                │
 │───────────────▶│                  │                  │                │
 │                │  build_context() │                  │                │
 │                │─────────────────▶│                  │                │
 │                │                  │  search(query)   │                │
 │                │                  │─────────────────▶│                │
 │                │                  │                  │  L1/L2/L3检索  │
 │                │                  │◀─────────────────│                │
 │                │                  │  构建上下文      │                │
 │                │◀─────────────────│                  │                │
 │                │  chat(messages)  │                  │                │
 │                │─────────────────────────────────────────────────────▶│
 │                │                  │                  │     LLM推理    │
 │                │◀─────────────────────────────────────────────────────│
 │                │  显示响应        │                  │                │
 │  显示结果      │◀─────────────────│                  │                │
 │◀───────────────│  异步存储        │                  │                │
 │                │─────────────────────────────────────▶│                │
```

### 4.2 记忆检索流程

```
MemoryManager     L1内存层       VectorStore      SQLiteStore
     │               │               │               │
     │  search(query)│               │               │
     │  检查迁移锁   │               │               │
     │  记录版本号   │               │               │
     │               │               │               │
     │  L1关键词匹配 │               │               │
     │──────────────▶│               │               │
     │◀──────────────│               │               │
     │               │               │               │
     │  [结果不足]   │               │               │
     │  L2向量检索   │               │               │
     │─────────────────────────────▶│               │
     │◀─────────────────────────────│               │
     │               │               │               │
     │  [结果不足]   │               │               │
     │  L3全文搜索   │               │               │
     │─────────────────────────────────────────────▶│
     │◀─────────────────────────────────────────────│
     │               │               │               │
     │  合并结果     │               │               │
     │  去重/排序/过滤阈值            │               │
     │               │               │               │
     │  [版本号变化]重试             │               │
     │               │               │               │
     │  返回最终结果 │               │               │
```

### 4.3 记忆压缩与流动流程

```
AsyncProcessor   CompressionStrategy   VectorStore   SQLiteStore
     │                   │                  │             │
     │  定时触发压缩     │                  │             │
     │  获取待压缩记忆   │                  │             │
     │──────────────────────────────────────────────────▶│
     │◀──────────────────────────────────────────────────│
     │                   │                  │             │
     │  对每条记忆:      │                  │             │
     │  compress(text)   │                  │             │
     │──────────────▶│   │                  │             │
     │               │ LLM压缩             │             │
     │◀──────────────│   │                  │             │
     │  [失败]熔断器检查│                  │             │
     │  ──▶ 降级到规则压缩                │             │
     │                   │                  │             │
     │  更新压缩后的记忆 │                  │             │
     │─────────────────────────────────────────────────▶│
     │                   │                  │             │
     │  定时触发L2→L3流动                  │             │
     │  获取低权重记忆   │                  │             │
     │─────────────────────────────────────▶│             │
     │◀─────────────────────────────────────│             │
     │                   │                  │             │
     │  开始事务         │                  │             │
     │  写入SQLite       │                  │             │
     │─────────────────────────────────────────────────▶│
     │  删除VectorStore  │                  │             │
     │─────────────────────────────────────▶│             │
     │  提交事务         │                  │             │
```

### 4.4 启动恢复流程

```
ChatWindow    MemoryManager    SQLiteStore    VectorStore    TransactionCoordinator
     │              │               │              │                 │
     │  启动初始化  │               │              │                 │
     │              │  初始化       │              │                 │
     │              │  检查数据库完整性            │                 │
     │              │──────────────▶│              │                 │
     │              │  [损坏]尝试备份恢复          │                 │
     │              │──────────────▶│              │                 │
     │              │◀──────────────│              │                 │
     │              │               │              │                 │
     │              │  恢复未完成事务              │                 │
     │              │─────────────────────────────────────────────▶│
     │              │               │              │  扫描事务表    │
     │              │               │              │  重试PENDING事务│
     │              │◀─────────────────────────────────────────────│
     │              │               │              │                 │
     │              │  清理孤立向量 │              │                 │
     │              │─────────────────────────────▶│                 │
     │              │               │  对比SQLite与ChromaDB         │
     │              │               │  删除孤立向量│                 │
     │              │◀─────────────────────────────│                 │
     │              │               │              │                 │
     │  初始化完成  │               │              │                 │
     │◀─────────────│               │              │                 │
```

---

## 5. 数据模型

### 5.1 MemoryRecord 字段说明

| 字段名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `id` | int | 否 | None | 主键，自增 |
| `text` | str | 是 | "" | 原始记忆文本 |
| `compressed_text` | str | 否 | None | 压缩后的文本 |
| `source` | str | 是 | "user" | 来源：user/assistant/local/cloud |
| `weight` | float | 是 | 1.0 | 权重（0.3-5.0） |
| `access_count` | int | 是 | 0 | 访问次数 |
| `last_access_time` | str | 否 | None | 最后访问时间（ISO格式） |
| `created_time` | str | 否 | None | 创建时间（ISO格式） |
| `metadata` | Dict | 否 | {} | 元数据（JSON） |
| `vector_id` | str | 否 | None | 向量库ID |
| `is_vectorized` | int | 是 | 0 | 是否已向量化（0/1） |

### 5.2 各存储层的数据格式

#### L1 内存层

```python
{
    "user": "用户输入文本",
    "assistant": "助理回复文本",
    "timestamp": "2026-04-22T10:30:00",
    "metadata": {
        "source": "local",
        "session_id": "20260422_103000",
        "tags": []
    }
}
```

#### L2 向量库层（ChromaDB）

```python
{
    "id": "mem_1713765600000",
    "text": "记忆文本内容",
    "embedding": [0.123, 0.456, ...],  # 512维向量
    "metadata": {
        "source": "user",
        "timestamp": "2026-04-22T10:30:00",
        "memory_type": "conversation",
        "importance": 1.0,
        "storage_type": "full"
    }
}
```

#### L3 数据库层（SQLite）

```sql
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    compressed_text TEXT,
    source TEXT DEFAULT 'user',
    weight REAL DEFAULT 1.0,
    access_count INTEGER DEFAULT 0,
    last_access_time TEXT,
    created_time TEXT,
    metadata TEXT,  -- JSON 格式
    vector_id TEXT,
    is_vectorized INTEGER DEFAULT 0
);

-- 全文搜索索引
CREATE VIRTUAL TABLE memories_fts USING fts5(
    text, 
    compressed_text,
    content='memories',
    content_rowid='id'
);
```

### 5.3 元数据结构规范

```python
@dataclass
class MemoryMetadata:
    """记忆元数据结构"""
    
    source: str = "user"              # 来源：user/assistant/local/cloud
    timestamp: str = ""               # 时间戳（ISO格式）
    memory_type: str = "conversation" # 类型：conversation/fact/preference
    chunk_id: str = ""                # 分块ID
    session_id: str = ""              # 会话ID
    tags: List[str] = field(default_factory=list)  # 标签列表
    importance: float = 1.0           # 重要性（0-5）
    storage_type: str = "full"        # 存储类型：full/sqlite_only
```

### 5.4 数据流转规则

```
新对话产生
    │
    ▼
┌─────────────────┐
│  废话过滤        │
│  • discard      │──▶ 完全丢弃
│  • sqlite_only  │──▶ 仅存 L3
│  • full         │──▶ 存 L2 + L3
└─────────────────┘
    │
    ▼ [full]
┌─────────────────┐
│  异步处理队列    │
│  • 去重检测      │
│  • LLM 压缩     │
│  • 权重初始化    │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  写入 L2 + L3   │
│  • 事务保证      │
│  • 向量计算      │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  定期维护        │
│  • 权重衰减      │──▶ weight *= 0.95 (每日)
│  • 访问提升      │──▶ weight *= 1.2 (命中时)
│  • 自动遗忘      │──▶ weight < 0.3 时删除
│  • L2→L3 归档   │──▶ 低权重记忆迁移到 L3
│  • L3→L2 回填   │──▶ 命中的 L3 记忆提升到 L2
└─────────────────┘
```

---

## 6. 并发模型

### 6.1 线程模型说明

```
主线程（UI线程）
├── Tkinter 事件循环
├── 用户输入处理
├── 界面更新
└── 状态展示

工作线程 1（LLM调用）
├── 本地 LLM 请求
├── 云端 AI 请求
└── 响应处理

工作线程 2（异步处理）
├── 压缩任务
├── 去重任务
├── 存储任务
└── 定时维护

工作线程 3（后台维护）
├── 权重衰减
├── 记忆遗忘
├── L2↔L3 流动
└── 孤立向量清理
```

### 6.2 锁策略

```
SQLiteStore
├── _write_lock: threading.Lock
│   └── 保护写操作，确保原子性
├── _local: threading.local
│   └── 每线程独立读连接，无需锁
└── WAL 模式
    └── 读操作可并发执行

VectorStore
├── _lock: threading.Lock
│   └── 保护 ChromaDB 操作
└── 嵌入计算
    └── 通过 EmbeddingService 单例保证线程安全

MemoryManager
├── lock: threading.RLock
│   └── 保护 conversation_history
├── _read_lock: threading.Lock
│   └── 检索时阻塞迁移
└── _write_version: int
    └── 版本号检测并发修改

AsyncProcessor
├── queue.Queue (线程安全)
│   └── 内置锁保护
└── _running: bool
    └── 控制线程启停

TransactionCoordinator
├── _migration_lock: threading.RLock
│   └── 保护迁移操作
└── _migration_active: bool
    └── 标记迁移状态
```

### 6.3 队列机制

```
AsyncProcessor 队列配置
├── max_queue_size: 1000
│   └── 最大队列容量
├── queue_full_action: "drop_oldest"
│   └── 溢出策略：丢弃最旧任务
├── batch_size: 5
│   └── 批量处理大小
└── flush_interval: 30s
    └── 定期刷新间隔

队列处理流程
    │
    ▼
┌─────────────────┐
│  任务入队        │
│  queue.put()    │
│  [阻塞/丢弃]    │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  批量取出        │
│  batch = []     │
│  for i in 5:    │
│    batch.append │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  并行处理        │
│  for task:      │
│    process()    │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  结果收集        │
│  成功：提交      │
│  失败：重试/丢弃 │
└─────────────────┘
```

### 6.4 并发安全保证

```
并发安全保证：

1. 检索一致性
   - 版本号检测并发修改
   - 检测到修改时自动重试（最多3次）
   - 迁移时阻塞检索

2. 写入原子性
   - 两阶段提交保证跨存储原子性
   - SQLite WAL 模式保证单库原子性
   - 事务状态持久化支持崩溃恢复

3. 队列安全
   - queue.Queue 内置线程安全
   - 溢出策略防止内存溢出
   - 批量处理提高吞吐

4. 资源隔离
   - 每线程独立 SQLite 读连接
   - 单例模式共享嵌入服务
   - 锁粒度最小化
```

---

## 7. 扩展点设计

### 7.1 如何添加新的存储后端

```python
from abc import ABC, abstractmethod

class StorageBackend(ABC):
    """存储后端抽象接口"""
    
    @abstractmethod
    def add(self, texts: List[str], metadatas: List[Dict]) -> List[str]:
        """添加数据"""
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int) -> List[Dict]:
        """搜索数据"""
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> bool:
        """删除数据"""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """获取数量"""
        pass


class MilvusBackend(StorageBackend):
    """Milvus 向量数据库后端示例"""
    
    def __init__(self, host: str, port: int, collection: str):
        from pymilvus import connections, Collection
        connections.connect(host=host, port=port)
        self._collection = Collection(collection)
    
    def add(self, texts: List[str], metadatas: List[Dict]) -> List[str]:
        embeddings = self._compute_embeddings(texts)
        ids = [str(uuid.uuid4()) for _ in texts]
        self._collection.insert([ids, embeddings, texts])
        return ids
    
    def search(self, query: str, top_k: int) -> List[Dict]:
        embedding = self._compute_embeddings([query])[0]
        results = self._collection.search(
            data=[embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE"},
            limit=top_k
        )
        return self._format_results(results)
    
    def delete(self, ids: List[str]) -> bool:
        self._collection.delete(f"id in {ids}")
        return True
    
    def count(self) -> int:
        return self._collection.num_entities


# 注册新后端
class VectorStore:
    BACKENDS = {
        "chromadb": ChromaDBBackend,
        "milvus": MilvusBackend,
        "pinecone": PineconeBackend,
    }
    
    def __init__(self, backend: str = "chromadb", **kwargs):
        backend_class = self.BACKENDS.get(backend)
        if not backend_class:
            raise ValueError(f"Unknown backend: {backend}")
        self._backend = backend_class(**kwargs)
```

### 7.2 如何添加新的过滤规则

```python
from abc import ABC, abstractmethod

class FilterRule(ABC):
    """过滤规则抽象接口"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """规则名称"""
        pass
    
    @property
    @abstractmethod
    def priority(self) -> int:
        """优先级（越小越先执行）"""
        pass
    
    @abstractmethod
    def apply(self, context: FilterContext) -> FilterResult:
        """应用规则"""
        pass


class EmojiOnlyFilter(FilterRule):
    """表情符号过滤器示例"""
    
    @property
    def name(self) -> str:
        return "emoji_only"
    
    @property
    def priority(self) -> int:
        return 10
    
    def apply(self, context: FilterContext) -> FilterResult:
        import re
        
        text = context.text.strip()
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "]+"
        )
        
        cleaned = emoji_pattern.sub("", text)
        
        if not cleaned.strip():
            return FilterResult(
                is_nonsense=True,
                reason="仅包含表情符号",
                confidence=0.95,
                storage_type="sqlite_only"
            )
        
        return FilterResult(
            is_nonsense=False,
            reason="",
            confidence=0.0,
            storage_type="full"
        )


# 注册新规则
class NonsenseFilter:
    def __init__(self):
        self._rules: List[FilterRule] = []
        self._register_default_rules()
    
    def register_rule(self, rule: FilterRule):
        """注册过滤规则"""
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority)
```

### 7.3 如何接入新的 LLM 服务

```python
from abc import ABC, abstractmethod

class LLMClient(ABC):
    """LLM 客户端抽象接口"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """客户端名称"""
        pass
    
    @abstractmethod
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: int = 2048,
        temperature: float = 0.7
    ) -> str:
        """发送对话请求"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查是否可用"""
        pass
    
    @abstractmethod
    def get_context_limit(self) -> int:
        """获取上下文限制"""
        pass


class ClaudeClient(LLMClient):
    """Claude API 客户端示例"""
    
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        self.api_key = api_key
        self.model = model
        self._client = None
        self._init_client()
    
    @property
    def name(self) -> str:
        return "claude"
    
    def _init_client(self):
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            print("请安装 anthropic 库: pip install anthropic")
    
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: int = 2048,
        temperature: float = 0.7
    ) -> str:
        if not self._client:
            raise RuntimeError("Claude 客户端未初始化")
        
        system_prompt = ""
        chat_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                chat_messages.append(msg)
        
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=chat_messages
        )
        
        return response.content[0].text
    
    def is_available(self) -> bool:
        return self._client is not None
    
    def get_context_limit(self) -> int:
        return 200000  # Claude 3 Opus 支持 200K 上下文


# 注册新客户端
class CloudClientFactory:
    CLIENTS = {
        "openai": OpenAIClient,
        "gemini": GeminiClient,
        "glm": GLMClient,
        "claude": ClaudeClient,
    }
    
    @classmethod
    def create(cls, provider: str, **kwargs) -> LLMClient:
        client_class = cls.CLIENTS.get(provider)
        if not client_class:
            raise ValueError(f"Unknown provider: {provider}")
        return client_class(**kwargs)
```

### 7.4 扩展点总结

| 扩展点 | 抽象接口 | 具体实现 |
|--------|----------|----------|
| 存储后端 | `StorageBackend` | ChromaDB, Milvus, Pinecone |
| 过滤规则 | `FilterRule` | EmojiOnlyFilter, SingleWordFilter |
| LLM 服务 | `LLMClient` | OpenAI, Gemini, GLM, Claude |
| 压缩策略 | `CompressionStrategy` | LLMCompression, RuleCompression |

---

## 8. 部署架构

### 8.1 单机部署拓扑

```
┌─────────────────────────────────────────────────────────────────────┐
│                        单机部署架构                                   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         用户工作站                                    │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    MemoryMind 应用                            │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │   │
│  │  │ ChatWindow  │  │   Memory    │  │    LLM      │         │   │
│  │  │   (Tkinter) │  │  Manager    │  │   Client    │         │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘         │   │
│  │                                                              │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │   │
│  │  │ VectorStore │  │ SQLiteStore │  │  Embedding  │         │   │
│  │  │ (ChromaDB)  │  │  (SQLite)   │  │  Service    │         │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    llama.cpp Server                          │   │
│  │  ┌─────────────────────────────────────────────────────┐   │   │
│  │  │  Local LLM Model (GGUF format)                      │   │   │
│  │  │  • Port: 8080                                       │   │   │
│  │  │  • API: /v1/chat/completions                        │   │   │
│  │  └─────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      数据存储                                │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐              │   │
│  │  │ memory.db │  │ chroma_db/│  │ models/   │              │   │
│  │  │  (SQLite) │  │ (Vectors) │  │  (ONNX)   │              │   │
│  │  │   ~150MB  │  │  ~120MB   │  │   ~50MB   │              │   │
│  │  └───────────┘  └───────────┘  └───────────┘              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

                    │
                    │ HTTPS (可选)
                    ▼

         ┌─────────────────────┐
         │    云端 AI 服务      │
         │  (OpenAI/Gemini/GLM) │
         │    按需调用          │
         └─────────────────────┘
```

### 8.2 组件通信方式

```
┌─────────────────────────────────────────────────────────────────────┐
│                        组件通信方式                                   │
└─────────────────────────────────────────────────────────────────────┘

内部通信（同一进程）
├── 函数调用
│   └── 同步，低延迟（<1ms）
│
├── 事件总线（EventBus）
│   └── 发布订阅模式，解耦模块
│
└── 队列（Queue）
    └── 异步处理，线程安全


本地通信（同一机器）
├── HTTP REST API
│   └── llama.cpp Server
│   └── localhost:8080
│   └── 超时：120s
│
└── 文件系统
    └── SQLite 数据库
    └── ChromaDB 向量库
    └── ONNX 模型文件


外部通信（网络）
├── HTTPS REST API
│   └── OpenAI API
│   └── Gemini API
│   └── GLM API
│   └── 超时：60s
│
└── 认证方式
    └── API Key
    └── Bearer Token
```

### 8.3 资源规划建议

#### 最低配置

| 资源 | 最低要求 | 说明 |
|------|----------|------|
| CPU | 4 核 | 支持并发处理 |
| 内存 | 8 GB | 运行 LLM + 向量计算 |
| 存储 | 10 GB | 模型 + 数据库 + 日志 |
| 操作系统 | Windows 10+ / macOS 10.15+ / Linux | 跨平台支持 |

#### 推荐配置

| 资源 | 推荐配置 | 说明 |
|------|----------|------|
| CPU | 8 核+ | 更快的向量计算 |
| 内存 | 16 GB+ | 支持更大模型 |
| 存储 | 50 GB SSD | 更快的 I/O |
| GPU | 可选 | 加速 LLM 推理 |

#### 资源占用估算

```
组件资源占用（运行时）：

MemoryMind 应用
├── 基础内存：~150 MB
├── 嵌入缓存：~50 MB (1000条)
├── L1 缓存：~2 MB (25条对话)
└── 工作内存：~100 MB (处理时)

llama.cpp Server
├── 模型内存：2-8 GB (取决于模型大小)
├── 上下文：~500 MB (8K 上下文)
└── 推理峰值：+1-2 GB

数据存储
├── SQLite：~150 MB (10,000条记忆)
├── ChromaDB：~120 MB (10,000条向量)
└── ONNX 模型：~50 MB (固定)

总计
├── 最低：~4 GB (小模型)
├── 推荐：~8 GB (中等模型)
└── 高配：~16 GB (大模型)
```

### 8.4 部署检查清单

```
部署前检查：

□ 环境准备
  □ Python 3.9+ 已安装
  □ 虚拟环境已创建
  □ 依赖已安装（requirements.txt）

□ 模型准备
  □ llama.cpp 已下载
  □ GGUF 模型已下载
  □ ONNX 嵌入模型已下载

□ 配置检查
  □ .env 文件已创建
  □ API Key 已配置（如使用云端）
  □ 路径配置正确

□ 服务启动
  □ llama.cpp 服务已启动（端口 8080）
  □ MemoryMind 应用已启动
  □ 连接测试通过

□ 功能验证
  □ 对话功能正常
  □ 记忆存储正常
  □ 记忆检索正常
  □ 云端切换正常（如启用）
```

---

## 附录

### A. 配置参数速查表

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `LOCAL_ENABLED` | `true` | 是否启用本地 LLM |
| `LOCAL_API_URL` | `http://localhost:8080/completion` | llama.cpp API 地址 |
| `LOCAL_MAX_CONTEXT` | `8192` | 本地 LLM 上下文长度 |
| `CLOUD_ENABLED` | `false` | 是否启用云端 AI |
| `CLOUD_PROVIDER` | `gemini` | 云端提供商 |
| `MEMORY_DECAY_DAYS` | `30` | 记忆遗忘周期 |
| `MEMORY_MIN_WEIGHT` | `0.3` | 最小记忆权重 |
| `COMPRESSION_MIN_LENGTH` | `100` | 压缩最小长度 |
| `MAX_RETRIEVE_RESULTS` | `5` | 最大检索结果数 |
| `SIMILARITY_THRESHOLD` | `0.90` | 相似度阈值 |

### B. 错误码定义

| 错误码 | 说明 | 处理建议 |
|--------|------|----------|
| `E001` | LLM 连接失败 | 检查 llama.cpp 服务状态 |
| `E002` | 嵌入模型加载失败 | 检查模型文件路径 |
| `E003` | 数据库损坏 | 尝试从备份恢复 |
| `E004` | 向量库写入失败 | 检查磁盘空间 |
| `E005` | 云端 API 调用失败 | 检查网络和 API Key |

### C. 性能调优建议

```
检索性能优化：
1. 增加嵌入缓存大小（cache_max_size）
2. 调整检索阈值（降低精度换取速度）
3. 限制 L3 检索（include_l3=False）

写入性能优化：
1. 增加批量处理大小（batch_size）
2. 增加刷新间隔（flush_interval）
3. 禁用实时压缩（手动触发）

内存优化：
1. 减少 L1 缓存大小（max_l1_size）
2. 减少嵌入缓存大小
3. 使用更小的嵌入维度
```

---

<p align="center">
  <strong>MemoryMind System Architecture v1.0</strong><br>
  <em>构建可靠的本地AI记忆系统</em>
</p>
