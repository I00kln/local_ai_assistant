# 🏗️ MemoryMind 系统架构设计文档

> **文档版本**：v1.1  
> **最后更新**：2026年4月23日  
> **适用版本**：MemoryMind v0.3.0

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
| 🛡️ **并发安全** | 多线程环境下数据一致性保证 | P0 |
| 🩺 **可观测性** | 健康检查、性能指标、优雅关闭 | P1 |

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
| BackgroundScheduler | 后台任务协调与并发控制 |
| LifecycleManager | 服务生命周期管理 |
| ThreadPoolManager | 线程池统一管理 |
| HealthChecker | 组件健康状态监控 |
| MetricsCollector | 性能指标收集统计 |
| MemoryMerger | 相似记忆检测与合并 |
| TransactionCoordinator | 跨存储事务协调 |

#### 开闭原则（OCP）

```
扩展开放：
├─ 新增存储后端：实现 StorageInterface
├─ 新增过滤规则：继承 FilterStrategy
├─ 新增 LLM 服务：继承 LLMClient
├─ 新增压缩策略：继承 CompressionStrategy
└─ 新增任务类型：扩展 TaskType 枚举

修改封闭：
├─ 核心模块接口稳定
├─ 配置驱动行为变化
└─ 事件机制解耦模块
```

#### 依赖倒置原则（DIP）

```
高层模块依赖抽象接口，而非具体实现
MemoryManager 依赖 VectorStore 接口，而非 ChromaDB 具体实现
AsyncProcessor 依赖 CompressionStrategy 接口，而非具体压缩实现
```

### 1.3 架构约束

| 约束类型 | 约束内容 | 理由 |
|----------|----------|------|
| 单向依赖 | 模块间禁止循环引用 | 避免耦合地狱 |
| 纯净领域 | 业务逻辑不依赖框架 | 可测试、可移植 |
| 接口隔离 | 不设计上帝接口 | 职责清晰 |
| 迪米特法则 | 禁止链式调用超过一级 | 降低耦合 |
| 并发安全 | 所有共享状态必须加锁保护 | 数据一致性 |
| 资源限制 | 线程数、队列大小有上限 | 防止资源耗尽 |

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
│  组件：ContextBuilder, LLMClient, CloudClient                       │
│  职责：上下文构建、LLM调用、云端协同、响应处理                         │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        记忆管理层 (Memory Layer)                     │
│  组件：MemoryManager, AsyncProcessor, NonsenseFilter,               │
│        SensitiveFilter, TransactionCoordinator, MemoryMerger,       │
│        BackgroundScheduler, MemoryTags                              │
│  职责：记忆检索、存储协调、压缩、过滤、事务管理、合并                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          存储层 (Storage Layer)                      │
│  组件：VectorStore, SQLiteStore, EmbeddingService                   │
│  职责：数据持久化、向量计算、缓存管理                                  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        基础设施层 (Infrastructure Layer)              │
│  组件：EventBus, Logger, Metrics, HealthChecker,                    │
│        LifecycleManager, ThreadPoolManager                          │
│  职责：事件分发、日志记录、指标收集、健康检查、生命周期管理             │
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
• LLMClient（LLM调用）
• AsyncProcessor（后台处理）
```

#### 业务逻辑层（Business Layer）

```
ContextBuilder 职责：
• 多级记忆检索（L1→L2→L3）
• 上下文长度控制（Token预算）
• 记忆格式化与排序
• RRF 分数融合

策略：
• 本地模式：高阈值（0.90），少量记忆
• 云端模式：低阈值（0.70），大量记忆
• 混合模式：中等阈值，中等记忆量

LLMClient 职责：
• 本地 LLM 调用（llama.cpp）
• 双接口适配（OpenAI格式 + completion格式）
• 连接健康检查
• 超时控制

CloudClient 职责：
• 云端 AI 调用（OpenAI/Gemini/GLM）
• 故障降级处理
• API Key 管理
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

BackgroundScheduler 职责：
• 统一协调所有记忆状态变更操作
• 任务优先级队列
• 迁移锁保护（读写锁）
• 读锁超时降级

任务类型：
• MIGRATION_L2_TO_L3: L2→L3 迁移
• MIGRATION_L3_TO_L2: L3→L2 回填
• COMPRESSION: 记忆压缩
• MERGE: 记忆合并
• DEDUP: 去重
• DECAY: 权重衰减
• FORGET: 遗忘

NonsenseFilter 职责：
• 三层废话过滤（规则→密度→向量）
• 存储策略决策（full/sqlite_only/discard）
• 保护有价值短文本

过滤层级：
• Layer1: 规则过滤（~1ms）
• Layer2: 密度评分（~5ms）
• Layer3: 向量匹配（~10ms）

MemoryMerger 职责：
• 相似记忆检测（向量相似度 > 0.92）
• 冲突检测（时间/地点/数值）
• 元数据深度合并
• 向量库同步清理

TransactionCoordinator 职责：
• 两阶段提交协议
• 事务状态持久化
• 崩溃恢复机制
• 事务阶段追踪
```

#### 存储层（Storage Layer）

```
VectorStore 职责：
• ChromaDB 向量存储封装
• 向量相似度检索
• 元数据过滤
• 自动持久化
• 内置去重

特点：
• 自动持久化到磁盘
• 支持增量更新
• 线程安全

SQLiteStore 职责：
• SQLite 关系数据存储
• 全文搜索（FTS）
• 权重管理
• 事务支持
• 连接池管理

特点：
• WAL 模式并发读
• 每线程独立读连接
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

#### 基础设施层（Infrastructure Layer）

```
EventBus 职责：
• 发布-订阅模式解耦组件
• 异步事件处理
• 弱引用订阅者管理
• 自动清理死订阅者

事件类型：
• L1_OVERFLOW: L1 缓存溢出
• MEMORY_WRITTEN: 记忆写入
• SHUTDOWN: 系统关闭
• HEARTBEAT: 心跳
• CRITICAL_SERVICE_DOWN: 关键服务下线

Logger 职责：
• 结构化日志记录
• 事件名称标识
• 关键字段记录
• 决策路径追踪

MetricsCollector 职责：
• 延迟分布统计（P50/P95/P99）
• 检索命中率统计
• 压缩成功率统计
• 百分位数缓存

HealthChecker 职责：
• 组件健康状态检查
• 依赖关系检查
• 降级状态追踪
• 降级状态机

LifecycleManager 职责：
• 集中注册服务清理函数
• 按优先级顺序关闭服务
• 超时保护
• 发布 SHUTDOWN 事件

ThreadPoolManager 职责：
• 按任务类型隔离线程池
• 拒绝策略保护
• 统一监控统计
• 优雅关闭支持
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

记忆管理层 ──▶ 基础设施层
────────────────────
请求：submit_task(task_type, callback, priority)  [BackgroundScheduler]
响应：bool（是否成功提交）

请求：publish(event_type, data)  [EventBus]
响应：None

请求：record_retrieval_latency(duration_ms)  [MetricsCollector]
响应：None
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
    ├──▶ TransactionCoordinator (事务协调)
    ├──▶ BackgroundScheduler (后台调度)
    └──▶ MetricsCollector (指标收集)
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
    combined_score: float = 0.0  # 综合分数
    rrf_score: float = 0.0       # RRF 分数
    rank: int = 0                # 排名
```

---

### 3.2 BackgroundScheduler（后台任务调度器）

#### 模块职责

1. 统一协调所有记忆状态变更操作
2. 任务优先级队列管理
3. 迁移锁保护（读写锁）
4. 读锁超时降级
5. 任务执行统计

#### 对外接口

```python
class BackgroundTaskScheduler:
    def start(self) -> None:
        """启动调度器"""
        pass
    
    def stop(self) -> None:
        """停止调度器"""
        pass
    
    def submit_task(
        self,
        task_type: TaskType,
        callback: Callable,
        args: tuple = (),
        kwargs: Dict = None,
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> bool:
        """提交后台任务"""
        pass
    
    def read_lock(self, timeout: float = 5.0):
        """获取读锁上下文管理器"""
        pass
    
    def is_migration_active(self) -> bool:
        """检查是否有迁移操作正在进行"""
        pass
```

#### 内部状态管理

```python
class BackgroundTaskScheduler:
    def __init__(self):
        # 任务队列
        self._task_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=1000)
        self._running = False
        self._worker_thread: threading.Thread = None
        
        # 迁移锁（读写锁）
        self._migration_lock = threading.RLock()
        self._migration_active = False
        self._current_task: TaskType = None
        
        # 读写锁状态
        self._readers_count = 0
        self._readers_lock = threading.Lock()
        self._read_write_lock = threading.Lock()
        self._write_waiting = threading.Condition(self._read_write_lock)
        self._write_in_progress = False
        self._pending_writers = 0
        
        # 任务统计
        self._task_stats: Dict[TaskType, Dict[str, int]] = {}
```

#### 任务类型定义

```python
class TaskType(Enum):
    """后台任务类型"""
    MIGRATION_L2_TO_L3 = "migration_l2_to_l3"
    MIGRATION_L3_TO_L2 = "migration_l3_to_l2"
    PRELOAD_L3_TO_L2 = "preload_l3_to_l2"
    COMPRESSION = "compression"
    MERGE = "merge"
    DEDUP = "dedup"
    DECAY = "decay"
    FORGET = "forget"
    UPGRADE_SQLITE_ONLY = "upgrade_sqlite_only"
    FLUSH_BUFFER = "flush_buffer"
    RETRY_VECTORIZATION = "retry_vectorization"


class TaskPriority(Enum):
    """任务优先级"""
    HIGH = 1
    NORMAL = 2
    LOW = 3
```

---

### 3.3 LifecycleManager（生命周期管理器）

#### 模块职责

1. 集中注册服务清理函数
2. 按优先级顺序关闭服务
3. 超时保护，避免无限等待
4. 发布 SHUTDOWN 事件通知
5. 强制停止支持

#### 对外接口

```python
class LifecycleManager:
    def register(
        self,
        name: str,
        cleanup_fn: Callable,
        priority: ServicePriority = ServicePriority.NORMAL,
        timeout: float = 5.0,
        is_running: Callable[[], bool] = None,
        stop_fn: Callable = None,
        force_stop_fn: Callable = None,
        thread: threading.Thread = None
    ):
        """注册服务清理函数"""
        pass
    
    def shutdown(self, timeout: float = None):
        """关闭所有服务"""
        pass
    
    def is_shutting_down(self) -> bool:
        """检查是否正在关闭"""
        pass
```

#### 服务优先级定义

```python
class ServicePriority(Enum):
    """服务关闭优先级（数字越大越先关闭）"""
    CRITICAL = 100    # 线程池管理器
    HIGH = 80         # 事件总线、异步处理器
    NORMAL = 50       # 记忆管理器、向量存储
    LOW = 20          # SQLite 存储
    BACKGROUND = 10   # 健康检查
```

---

### 3.4 ThreadPoolManager（线程池管理器）

#### 模块职责

1. 按任务类型隔离线程池
2. 拒绝策略保护
3. 统一监控和统计
4. 优雅关闭支持

#### 对外接口

```python
class ThreadPoolManager:
    def submit(
        self,
        task_type: TaskType,
        fn: Callable,
        *args,
        **kwargs
    ) -> Optional[Future]:
        """提交任务到对应线程池"""
        pass
    
    def shutdown(self):
        """关闭所有线程池"""
        pass
    
    def get_stats(self) -> Dict[TaskType, PoolStats]:
        """获取线程池统计信息"""
        pass
```

#### 线程池配置

```python
DEFAULT_CONFIG = {
    TaskType.IO_BOUND: {
        "max_workers": 8,
        "thread_name_prefix": "io_worker",
        "max_queue_size": 100,
        "rejection_policy": RejectionPolicy.CALLER_RUNS,
    },
    TaskType.CPU_BOUND: {
        "max_workers": 4,
        "thread_name_prefix": "cpu_worker",
        "max_queue_size": 50,
        "rejection_policy": RejectionPolicy.REJECT,
    },
    TaskType.UI: {
        "max_workers": 2,
        "thread_name_prefix": "ui_worker",
        "max_queue_size": 20,
        "rejection_policy": RejectionPolicy.DROP_OLDEST,
    },
}
```

---

### 3.5 TransactionCoordinator（事务协调器）

#### 模块职责

1. 两阶段提交协议实现
2. 事务状态持久化
3. 崩溃恢复机制
4. 事务阶段追踪

#### 对外接口

```python
class TransactionCoordinator:
    def execute_transaction(
        self,
        operation_type: str,
        data: Dict[str, Any],
        prepare_fn: Callable,
        commit_fn: Callable
    ) -> bool:
        """执行两阶段提交事务"""
        pass
    
    def recover_pending_transactions(self):
        """恢复未完成的事务"""
        pass
```

#### 事务状态定义

```python
class TransactionState(Enum):
    """事务状态"""
    PENDING = "pending"
    PREPARING = "preparing"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class TransactionPhase(Enum):
    """事务阶段 - 用于精确恢复"""
    INIT = "init"
    PREPARING = "preparing"
    PREPARED = "prepared"
    COMMITTING = "committing"
    COMMITTED = "committed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"
```

---

### 3.6 MemoryMerger（记忆合并器）

#### 模块职责

1. 相似记忆检测（向量相似度）
2. 冲突检测（时间/地点/数值）
3. 元数据深度合并
4. 向量库同步清理

#### 对外接口

```python
class MemoryMerger:
    def find_similar_groups(
        self,
        memories: List[MemoryRecord],
        threshold: float = 0.92
    ) -> List[List[MemoryRecord]]:
        """查找相似记忆组"""
        pass
    
    def merge_group(
        self,
        group: List[MemoryRecord]
    ) -> Tuple[MemoryRecord, List[int]]:
        """合并一组相似记忆"""
        pass
    
    def check_conflicts(
        self,
        texts: List[str]
    ) -> Tuple[bool, str]:
        """检测冲突"""
        pass
```

#### 冲突检测模式

```python
TIME_PATTERNS = [
    r'明天', r'后天', r'昨天', r'前天',
    r'\d+月\d+[日号]',
    r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日号]?',
]

LOCATION_PATTERNS = [
    r'去[北京上海广州深圳...]',
    r'在[北京上海广州深圳...]',
    r'到[北京上海广州深圳...]',
]

NUMBER_PATTERNS = [
    r'\d+块', r'\d+元', r'\d+万',
    r'\d+千', r'\d+百', r'\d+个',
]
```

---

### 3.7 CompressionStrategies（压缩策略模块）

#### 模块职责

1. 策略抽象接口定义
2. LLM 压缩策略（带熔断器）
3. 规则压缩策略
4. 策略链（责任链模式）
5. 长文本分块压缩

#### 策略接口

```python
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
    
    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称"""
        pass


class CompressionStrategyChain:
    """压缩策略链"""
    
    def __init__(self, strategies: List[CompressionStrategy]):
        self._strategies = strategies
    
    def compress(self, text: str) -> Optional[str]:
        """依次尝试各策略"""
        for strategy in self._strategies:
            if strategy.is_available():
                result = strategy.compress(text)
                if result:
                    return result
        return None
```

#### 熔断器状态机

```
         ┌─────────────────────────────────────────┐
         │                                         │
         ▼                                         │
    ┌─────────┐    失败次数 >= 阈值    ┌─────────┐ │
    │ CLOSED  │ ────────────────────▶ │  OPEN   │ │
    │ (正常)   │                       │ (熔断)   │ │
    └─────────┘                       └─────────┘ │
         ▲                                 │       │
         │                                 │       │
         │           重置时间后            │       │
         │     ┌───────────────────────┐   │       │
         └────│   HALF_OPEN (半开)     │◀──┘       │
              │   允许探测请求          │           │
              └───────────────────────┘           │
                          │                       │
                          │ 探测成功              │ 探测失败
                          ▼                       ▼
                    返回 CLOSED              返回 OPEN
```

---

### 3.8 HealthChecker（健康检查器）

#### 模块职责

1. 组件健康状态检查
2. 依赖关系检查
3. 降级状态追踪
4. 降级状态机

#### 对外接口

```python
class HealthChecker:
    def register_component(self, name: str, component: Any):
        """注册组件"""
        pass
    
    def check_all(self) -> Dict[str, ComponentHealth]:
        """检查所有组件健康状态"""
        pass
    
    def set_degradation(
        self,
        component: str,
        level: DegradationLevel,
        fallback: bool = False
    ):
        """设置组件降级状态"""
        pass
```

#### 健康状态定义

```python
class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class DegradationLevel(Enum):
    """降级级别"""
    NONE = "none"
    MINIMAL = "minimal"
    PARTIAL = "partial"
    SEVERE = "severe"
    CRITICAL = "critical"
```

---

### 3.9 MetricsCollector（性能指标收集器）

#### 模块职责

1. 延迟分布统计（P50/P95/P99）
2. 检索命中率统计
3. 压缩成功率统计
4. 过滤结果统计
5. 百分位数缓存

#### 对外接口

```python
class MetricsCollector:
    def record_retrieval_latency(self, duration_ms: float):
        """记录检索延迟"""
        pass
    
    def record_retrieval_hits(self, l1: int, l2: int, l3: int):
        """记录各层检索命中数"""
        pass
    
    def record_compression(self, success: bool, duration_ms: float = 0):
        """记录压缩结果"""
        pass
    
    def get_percentiles(self, metric_type: str) -> Dict[str, float]:
        """获取百分位数"""
        pass
```

---

### 3.10 MemoryTags（记忆标签系统）

#### 模块职责

1. 压缩相关标签管理
2. 流转相关标签管理
3. 保护相关标签管理
4. 遗忘相关标签管理
5. 合并相关标签管理

#### 标签常量定义

```python
class MemoryTags:
    """记忆元数据标签常量"""
    
    # 压缩相关
    COMPRESSED = "compressed"
    COMPRESSED_TIME = "compressed_time"
    COMPRESSED_STRATEGY = "compression_strategy"
    
    # 流转相关
    MOVED_FROM_L2 = "moved_from_l2"
    PROMOTED_TO_L2 = "promoted_to_l2"
    PROMOTED_TIME = "promoted_time"
    
    # 保护相关
    PRESERVE = "preserve"
    PROTECTED = "protected"
    IMPORTANT = "important"
    
    # 遗忘相关
    FORGOTTEN = "forgotten"
    ARCHIVED = "archived"
    
    # 合并相关
    MERGED = "merged"
    MERGED_AT = "merged_at"
    MERGED_COUNT = "merged_count"
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
 │                │                  │  scheduler.read_lock()            │
 │                │                  │─────────────────▶│                │
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

### 4.2 记忆检索流程（带调度器保护）

```
MemoryManager     Scheduler       L1内存层       VectorStore      SQLiteStore
     │               │               │               │               │
     │  search(query)│               │               │               │
     │  read_lock()  │               │               │               │
     │──────────────▶│               │               │               │
     │  [等待写操作完成]             │               │               │
     │◀──────────────│               │               │               │
     │               │               │               │               │
     │  L1关键词匹配 │               │               │               │
     │─────────────────────────────▶│               │               │
     │◀─────────────────────────────│               │               │
     │               │               │               │               │
     │  [结果不足]   │               │               │               │
     │  L2向量检索   │               │               │               │
     │─────────────────────────────────────────────▶│               │
     │◀─────────────────────────────────────────────│               │
     │               │               │               │               │
     │  [结果不足]   │               │               │               │
     │  L3全文搜索   │               │               │               │
     │─────────────────────────────────────────────────────────────▶│
     │◀─────────────────────────────────────────────────────────────│
     │               │               │               │               │
     │  合并结果     │               │               │               │
     │  RRF分数融合  │               │               │               │
     │               │               │               │               │
     │  release_read()               │               │               │
     │──────────────▶│               │               │               │
     │               │               │               │               │
     │  返回最终结果 │               │               │               │
```

### 4.3 后台任务提交流程

```
AsyncProcessor    Scheduler        WorkerThread     VectorStore    SQLiteStore
     │               │                 │                │              │
     │  submit_task( │                 │                │              │
     │    COMPRESSION)                 │                │              │
     │──────────────▶│                 │                │              │
     │               │  任务入队       │                │              │
     │               │────────────────▶│                │              │
     │               │                 │                │              │
     │               │                 │  _begin_migration()           │
     │               │                 │  [获取写锁]    │              │
     │               │                 │────────────────┼──────────────│
     │               │                 │                │              │
     │               │                 │  执行压缩回调  │              │
     │               │                 │────────────────┼──────────────│
     │               │                 │                │              │
     │               │                 │  _end_migration()             │
     │               │                 │  [释放写锁]    │              │
     │               │                 │────────────────┼──────────────│
     │               │                 │                │              │
     │               │  返回结果       │                │              │
     │◀──────────────│◀────────────────│                │              │
```

### 4.4 两阶段提交流程

```
TransactionCoordinator   SQLiteStore    VectorStore    TransactionTable
         │                    │              │               │
         │  execute_transaction()            │               │
         │                    │              │               │
         │  Phase 1: PREPARE  │              │               │
         │  ───────────────────────────────────────────────▶│
         │  记录事务状态(PENDING)            │               │
         │                    │              │               │
         │  执行准备操作      │              │               │
         │──────────────────▶│              │               │
         │  写入SQLite        │              │               │
         │                    │              │               │
         │  更新事务状态(PREPARED)           │               │
         │  ───────────────────────────────────────────────▶│
         │                    │              │               │
         │  Phase 2: COMMIT   │              │               │
         │  执行ChromaDB操作  │              │               │
         │─────────────────────────────────▶│               │
         │                    │              │               │
         │  [成功] 更新事务状态(COMMITTED)   │               │
         │  ───────────────────────────────────────────────▶│
         │                    │              │               │
         │  [失败] 回滚SQLite │              │               │
         │──────────────────▶│              │               │
         │  更新事务状态(ROLLED_BACK)        │               │
         │  ───────────────────────────────────────────────▶│
```

### 4.5 优雅关闭流程

```
ChatWindow    LifecycleManager    ThreadPoolManager    EventBus    SQLiteStore
     │               │                    │                │            │
     │  关闭窗口     │                    │                │            │
     │──────────────▶│                    │                │            │
     │               │  shutdown()        │                │            │
     │               │                    │                │            │
     │               │  发布SHUTDOWN事件  │                │            │
     │               │───────────────────────────────────▶│            │
     │               │                    │                │            │
     │               │  [CRITICAL] 关闭线程池             │            │
     │               │───────────────────▶│                │            │
     │               │  等待任务完成      │                │            │
     │               │◀───────────────────│                │            │
     │               │                    │                │            │
     │               │  [HIGH] 关闭EventBus               │            │
     │               │───────────────────────────────────▶│            │
     │               │◀───────────────────────────────────│            │
     │               │                    │                │            │
     │               │  [LOW] 关闭SQLite  │                │            │
     │               │─────────────────────────────────────────────────▶│
     │               │◀─────────────────────────────────────────────────│
     │               │                    │                │            │
     │  关闭完成     │                    │                │            │
     │◀──────────────│                    │                │            │
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
| `bm25_score` | float | 是 | 0.0 | BM25 检索分数 |
| `is_important` | bool | 是 | False | 是否重要 |
| `session_id` | str | 是 | "" | 会话ID |
| `content_hash` | str | 是 | "" | 内容哈希（去重用） |

### 5.2 各存储层的数据格式

#### L1 内存层

```python
{
    "user": "用户输入文本",
    "assistant": "助理回复文本",
    "timestamp": "2026-04-23T10:30:00",
    "metadata": {
        "source": "local",
        "session_id": "20260423_103000",
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
        "timestamp": "2026-04-23T10:30:00",
        "memory_type": "conversation",
        "importance": 1.0,
        "storage_type": "full",
        "sqlite_id": 123,
        "compressed": False
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
    is_vectorized INTEGER DEFAULT 0,
    bm25_score REAL DEFAULT 0.0,
    is_important INTEGER DEFAULT 0,
    session_id TEXT,
    content_hash TEXT
);

-- 全文搜索索引
CREATE VIRTUAL TABLE memories_fts USING fts5(
    text, 
    compressed_text,
    content='memories',
    content_rowid='id'
);

-- 事务表
CREATE TABLE transactions (
    transaction_id TEXT PRIMARY KEY,
    operation_type TEXT NOT NULL,
    state TEXT NOT NULL,
    phase TEXT NOT NULL DEFAULT 'init',
    created_time TEXT NOT NULL,
    updated_time TEXT NOT NULL,
    data TEXT,
    error_message TEXT
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
    
    # 压缩相关
    compressed: bool = False
    compression_strategy: str = ""
    
    # 流转相关
    moved_from_l2: bool = False
    promoted_to_l2: bool = False
    
    # 保护相关
    preserve: bool = False
    protected: bool = False
    
    # 遗忘相关
    forgotten: bool = False
    archived: bool = False
    
    # 合并相关
    merged: bool = False
    merged_count: int = 0
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
│  调度器提交任务  │
│  (通过队列)      │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  异步处理        │
│  • 去重检测      │
│  • LLM 压缩     │
│  • 权重初始化    │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  两阶段提交      │
│  写入 L2 + L3   │
│  事务保证        │
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
│  • 记忆合并      │──▶ 相似记忆合并
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

工作线程 1（LLM调用）[IO_BOUND]
├── 本地 LLM 请求
├── 云端 AI 请求
└── 响应处理

工作线程 2（后台调度）[独立线程]
├── 任务队列处理
├── 迁移锁管理
└── 任务执行

工作线程 3-6（IO任务）[IO_BOUND 线程池]
├── 数据库操作
├── 向量计算
├── 文件读写
└── 网络请求

工作线程 7-10（CPU任务）[CPU_BOUND 线程池]
├── 嵌入计算
├── 相似度计算
└── 压缩处理
```

### 6.2 锁策略

```
BackgroundScheduler（读写锁）
├── _migration_lock: threading.RLock
│   └── 保护迁移操作
├── _read_write_lock: threading.Lock
│   └── 读写互斥基础锁
├── _write_waiting: threading.Condition
│   └── 写等待条件变量
├── _readers_count: int
│   └── 当前读者数量
└── _write_in_progress: bool
    └── 写操作进行中标志

读写锁协议：
├── 读操作：获取读锁，允许多读并发
├── 写操作：获取写锁，阻塞所有读写
├── 读锁超时：降级为 L1 搜索
└── 写等待超时：抛出 TimeoutError

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
└── 调度器读锁
    └── 保护检索操作

ThreadPoolManager
├── _stats_lock: threading.Lock
│   └── 保护统计信息
└── queue.Queue (线程安全)
    └── 内置锁保护
```

### 6.3 队列机制

```
BackgroundScheduler 任务队列
├── max_queue_size: 1000
│   └── 最大队列容量
├── 优先级队列: PriorityQueue
│   └── 按优先级排序
└── 任务类型
    ├── HIGH: 迁移任务
    ├── NORMAL: 压缩/合并/去重
    └── LOW: 衰减/遗忘

AsyncProcessor 处理队列
├── max_queue_size: 1000
│   └── 最大队列容量
├── queue_full_action: "drop_oldest"
│   └── 溢出策略：丢弃最旧任务
├── batch_size: 5
│   └── 批量处理大小
└── flush_interval: 30s
    └── 定期刷新间隔

ThreadPoolManager 任务队列
├── IO_BOUND: max_queue_size = 100
├── CPU_BOUND: max_queue_size = 50
├── UI: max_queue_size = 20
└── 拒绝策略
    ├── REJECT: 拒绝新任务
    ├── DROP_OLDEST: 丢弃最旧任务
    └── CALLER_RUNS: 调用者线程执行
```

### 6.4 并发安全保证

```
并发安全保证：

1. 检索一致性
   - 调度器读锁保护
   - 读锁超时时降级为 L1 搜索
   - 迁移时阻塞检索

2. 写入原子性
   - 两阶段提交保证跨存储原子性
   - SQLite WAL 模式保证单库原子性
   - 事务状态持久化支持崩溃恢复

3. 任务串行化
   - 所有写操作通过调度器提交
   - 任务队列串行化执行
   - 迁移锁保护写操作

4. 资源隔离
   - 每线程独立 SQLite 读连接
   - 单例模式共享嵌入服务
   - 线程池按任务类型隔离

5. 优雅关闭
   - 生命周期管理器控制关闭顺序
   - 超时保护避免无限等待
   - SHUTDOWN 事件通知所有服务
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

### 7.2 如何添加新的压缩策略

```python
from compression_strategies import CompressionStrategy

class KeywordExtractionStrategy(CompressionStrategy):
    """关键词提取压缩策略示例"""
    
    def __init__(self, mem_config):
        self._mem_config = mem_config
        self._log = get_logger()
    
    @property
    def name(self) -> str:
        return "keyword_extraction"
    
    def compress(self, text: str) -> Optional[str]:
        """提取关键词作为压缩结果"""
        if not text or len(text) < 50:
            return None
        
        import re
        keywords = []
        
        patterns = [
            r'\d+月\d+[日号]',
            r'\d{4}年',
            r'[北京上海广州深圳杭州]',
            r'设置|配置|修改|添加|删除|创建',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            keywords.extend(matches)
        
        if keywords:
            return ' '.join(set(keywords))
        
        return None
    
    def is_available(self) -> bool:
        return True


# 注册新策略
compression_chain = CompressionStrategyChain([
    LLMCompressionStrategy(mem_config),
    KeywordExtractionStrategy(mem_config),
    RuleBasedCompressionStrategy(mem_config),
])
```

### 7.3 如何添加新的后台任务类型

```python
from background_scheduler import TaskType, TaskPriority

# 1. 扩展任务类型枚举
class ExtendedTaskType(TaskType):
    """扩展任务类型"""
    CUSTOM_CLEANUP = "custom_cleanup"
    DATA_EXPORT = "data_export"
    SYNC_CLOUD = "sync_cloud"


# 2. 实现任务处理函数
def handle_custom_cleanup():
    """自定义清理任务"""
    # 实现清理逻辑
    pass


# 3. 提交任务
scheduler = get_background_scheduler()
scheduler.submit_task(
    task_type=ExtendedTaskType.CUSTOM_CLEANUP,
    callback=handle_custom_cleanup,
    priority=TaskPriority.LOW
)
```

### 7.4 扩展点总结

| 扩展点 | 抽象接口 | 具体实现 |
|--------|----------|----------|
| 存储后端 | `StorageBackend` | ChromaDB, Milvus, Pinecone |
| 压缩策略 | `CompressionStrategy` | LLMCompression, RuleBased, KeywordExtraction |
| 过滤规则 | `FilterRule` | EmojiOnlyFilter, SingleWordFilter |
| LLM 服务 | `LLMClient` | OpenAI, Gemini, GLM, Claude |
| 后台任务 | `TaskType` 枚举 | 自定义任务类型 |

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
│  │  │ Scheduler   │  │ Lifecycle   │  │ ThreadPool  │         │   │
│  │  │             │  │ Manager     │  │ Manager     │         │   │
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
│  │  │   ~200MB  │  │  ~150MB   │  │   ~50MB   │              │   │
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
├── 任务队列（BackgroundScheduler）
│   └── 异步任务，优先级调度
│
└── 线程池（ThreadPoolManager）
    └── 任务提交，异步执行


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
├── 线程池：~20 MB
└── 工作内存：~100 MB (处理时)

llama.cpp Server
├── 模型内存：2-8 GB (取决于模型大小)
├── 上下文：~500 MB (8K 上下文)
└── 推理峰值：+1-2 GB

数据存储
├── SQLite：~200 MB (10,000条记忆)
├── ChromaDB：~150 MB (10,000条向量)
└── ONNX 模型：~50 MB (固定)

线程数量
├── 主线程：1
├── 后台调度线程：1
├── IO 线程池：8
├── CPU 线程池：4
├── UI 线程池：2
└── 总计：~16 线程

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
  □ 优雅关闭正常
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
| `MAX_QUEUE_SIZE` | `1000` | 任务队列最大容量 |
| `CIRCUIT_BREAKER_THRESHOLD` | `5` | 熔断器失败阈值 |
| `CIRCUIT_BREAKER_RESET_TIMEOUT` | `300` | 熔断器重置时间（秒） |

### B. 错误码定义

| 错误码 | 说明 | 处理建议 |
|--------|------|----------|
| `E001` | LLM 连接失败 | 检查 llama.cpp 服务状态 |
| `E002` | 嵌入模型加载失败 | 检查模型文件路径 |
| `E003` | 数据库损坏 | 尝试从备份恢复 |
| `E004` | 向量库写入失败 | 检查磁盘空间 |
| `E005` | 云端 API 调用失败 | 检查网络和 API Key |
| `E006` | 读锁获取超时 | 检查是否有长时间迁移操作 |
| `E007` | 写锁获取超时 | 检查是否有长时间检索操作 |
| `E008` | 任务队列已满 | 检查后台处理是否正常 |

### C. 性能调优建议

```
检索性能优化：
1. 增加嵌入缓存大小（cache_max_size）
2. 调整检索阈值（降低精度换取速度）
3. 限制 L3 检索（include_l3=False）
4. 使用 RRF 分数融合替代简单乘法

写入性能优化：
1. 增加批量处理大小（batch_size）
2. 增加刷新间隔（flush_interval）
3. 禁用实时压缩（手动触发）
4. 调整任务队列大小

并发性能优化：
1. 调整线程池大小
2. 调整读写锁超时时间
3. 调整任务优先级

内存优化：
1. 减少 L1 缓存大小（max_l1_size）
2. 减少嵌入缓存大小
3. 使用更小的嵌入维度
```

---

<p align="center">
  <strong>MemoryMind System Architecture v1.1</strong><br>
  <em>构建可靠的本地AI记忆系统</em>
</p>
