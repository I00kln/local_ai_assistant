# config.py
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class LocalConfig:
    """本地AI配置"""
    enabled: bool = True
    api_url: str = "http://localhost:8080/completion"
    max_context: int = 8192
    temperature: float = 0.7
    max_output_tokens: int = 3000


@dataclass
class CloudConfig:
    """云端AI配置"""
    enabled: bool = False
    provider: str = "gemini"
    api_key: str = ""
    model: str = "gemini-2.5-flash"
    base_url: Optional[str] = None
    max_context: int = 100000
    max_retrieve_results: int = 20


@dataclass
class AsyncProcessorConfig:
    """异步处理器配置"""
    dedup_interval: int = 300
    compression_interval: int = 600
    forget_interval: int = 3600
    flush_interval: int = 30
    max_buffer_age: int = 60
    batch_size: int = 5
    max_pending_compressions: int = 50
    compressor_check_interval: int = 300
    compressor_failure_threshold: int = 5
    max_queue_size: int = 1000
    queue_full_action: str = "drop_oldest"
    llm_timeout: int = 30
    circuit_breaker_threshold: int = 5
    circuit_breaker_reset_timeout: int = 300
    dedup_batch_size: int = 500
    orphan_cleanup_batch_size: int = 100


@dataclass
class MemoryFlowConfig:
    """记忆流动配置"""
    cooldown_hours: int = 24
    l2_move_threshold_multiplier: float = 2.0
    l3_promotion_weight_threshold: float = 0.8
    max_l2_to_l3_batch: int = 20
    max_l3_to_l2_batch: int = 10


@dataclass
class CompressionConfig:
    """压缩配置"""
    min_length: int = 100
    target_ratio: float = 0.6
    max_segment_length: int = 2000
    fallback_enabled: bool = True
    key_patterns: str = r'\d+|[A-Z][a-z]+|设置|配置|修改|添加|删除|创建|因为|所以|如果|那么|但是'


@dataclass
class PrivacyConfig:
    """隐私与安全配置"""
    sensitive_filter_enabled: bool = True
    local_encryption_enabled: bool = False
    excluded_keywords: str = ""
    log_sensitive_detection: bool = False
    https_only: bool = True
    verify_ssl: bool = True
    api_key_min_length: int = 20
    mask_in_logs: bool = True


@dataclass
class SQLitePoolConfig:
    """SQLite 连接池配置"""
    max_pool_size: int = 10
    connection_timeout: int = 3600
    cleanup_interval: int = 300
    busy_timeout: int = 30000
    cache_size: int = -64000


@dataclass
class DecayConfig:
    """权重衰减配置"""
    batch_size: int = 1000
    decay_rate: float = 0.95
    min_weight_threshold: float = 0.3
    max_weight: float = 5.0
    weight_boost_on_access: float = 1.2
    forget_age_days: int = 30


@dataclass
class RetrievalConfig:
    """检索配置"""
    max_retrieve_results: int = 5
    similarity_threshold: float = 0.90
    l1_min_results: int = 2
    l2_lower_threshold: float = 0.80
    cloud_l1_threshold: float = 0.70
    cloud_l2_threshold: float = 0.75
    cloud_l3_threshold: float = 0.70
    local_l1_threshold: float = 0.75
    local_l2_threshold: float = 0.80
    local_l3_threshold: float = 0.75
    source_weight_l1: float = 1.2
    source_weight_l2: float = 1.0
    source_weight_l3: float = 0.8
    diversity_threshold: float = 0.95
    recall_multiplier: int = 3
    high_quality_threshold: float = 0.95


@dataclass
class VectorStoreConfig:
    """向量存储配置"""
    onnx_model_path: str = "models/bge_onnx_model"
    embedding_dimension: int = 512
    chroma_persist_dir: str = "chroma_db"
    chroma_collection_name: str = "memories"


@dataclass
class SQLiteStoreConfig:
    """SQLite 存储配置"""
    db_path: str = "memory.db"
    enabled: bool = True
    encryption_enabled: bool = False
    encryption_key_env: str = "SQLITE_ENCRYPTION_KEY"


@dataclass
class SecurityConfig:
    """安全配置"""
    https_only: bool = True
    verify_ssl: bool = True
    api_key_min_length: int = 20
    sensitive_filter_enabled: bool = True
    log_sensitive_detection: bool = True
    excluded_keywords: str = ""
    mask_in_logs: bool = True


@dataclass
class NonsenseFilterConfig:
    """废话过滤配置"""
    enabled: bool = True
    db_path: str = "nonsense_library.json"


@dataclass
class HighDensityConfig:
    """高密度内容配置"""
    patterns: str = "流程图|时序图|架构图|状态机|数据流|ER图|类图|部署图|网络拓扑|API接口|数据库设计|系统设计|技术方案"
    preserve_ratio: float = 1.0


@dataclass
class ContextConfig:
    """上下文构建配置"""
    max_memory_tokens: int = 5000
    system_prompt_reserve: int = 200


class Config:
    """
    全局配置管理器
    
    整合所有配置：AI、记忆系统、存储、检索等
    """
    
    _instance: Optional['Config'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        
        self.local = LocalConfig()
        self.cloud = CloudConfig()
        
        self.async_processor = AsyncProcessorConfig()
        self.memory_flow = MemoryFlowConfig()
        self.compression = CompressionConfig()
        self.privacy = PrivacyConfig()
        self.sqlite_pool = SQLitePoolConfig()
        self.decay = DecayConfig()
        self.retrieval = RetrievalConfig()
        self.vector_store = VectorStoreConfig()
        self.sqlite_store = SQLiteStoreConfig()
        self.nonsense_filter = NonsenseFilterConfig()
        self.high_density = HighDensityConfig()
        self.context = ContextConfig()
        
        # 本地压缩模式使用的提示词（仅压缩检索记忆，不包含用户原文）
        self.system_prompt: str = """你是一个记忆压缩助手。

【核心任务】
压缩下方提供的【相关内容】，保留核心信息和关键细节，去除修饰和冗余。确保未参与对话的第三方阅读后能理解事情的全貌和前因后果。

【压缩规则】
1. 保留所有专有名词、人名、地名、数值、日期等关键信息
2. 保留因果关系和关键决策过程
3. 去除重复内容和修饰性语言
4. 压缩后长度约为原文的 30%-50%
5. 不要添加任何原文没有的内容

【输出要求】
直接返回压缩结果，不要添加任何解释或说明。"""

        # 仅本地模式使用的提示词（本地LLM直接回答用户问题）
        self.local_assistant_prompt: str = """你是一个 helpful AI 助手。

【任务说明】
1. 用户会提供【相关内容】和【原文】
2. 【相关内容】包含之前对话的检索记忆，可能与当前问题相关
3. 你需要结合【相关内容】（如果相关）回答用户的【原文】问题

【回答要求】
1. 提供完整、详细、有帮助的回答
2. 如果【相关内容】与问题相关，可以参考其中的信息
3. 如果【相关内容】与问题无关，直接基于你的知识回答
4. 保持自然、友好的对话风格"""
        
        self._load_from_env()
    
    def _load_from_env(self):
        """从环境变量加载所有配置"""
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        
        self._load_local_config()
        self._load_cloud_config()
        self._load_async_config()
        self._load_flow_config()
        self._load_compression_config()
        self._load_privacy_config()
        self._load_sqlite_pool_config()
        self._load_decay_config()
        self._load_retrieval_config()
        self._load_vector_store_config()
        self._load_sqlite_store_config()
        self._load_nonsense_filter_config()
        self._load_high_density_config()
        self._load_context_config()
    
    def _load_local_config(self):
        """加载本地AI配置"""
        self.local.enabled = os.environ.get("LOCAL_ENABLED", "true").lower() == "true"
        self.local.api_url = os.environ.get("LOCAL_API_URL", "http://localhost:8080/completion")
        self.local.max_context = int(os.environ.get("LOCAL_MAX_CONTEXT", "8192"))
        self.local.temperature = float(os.environ.get("LOCAL_TEMPERATURE", "0.7"))
        self.local.max_output_tokens = int(os.environ.get("LOCAL_MAX_OUTPUT_TOKENS", "3000"))
    
    def _load_cloud_config(self):
        """加载云端AI配置"""
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        glm_key = os.environ.get("GLM_API_KEY", "")
        cloud_provider = os.environ.get("CLOUD_PROVIDER", "gemini").lower()
        cloud_enabled = os.environ.get("CLOUD_ENABLED", "false").lower() == "true"
        
        if cloud_provider == "openai":
            self.cloud = CloudConfig(
                enabled=cloud_enabled,
                provider="openai",
                api_key=openai_key,
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                base_url=os.environ.get("OPENAI_BASE_URL"),
                max_context=int(os.environ.get("CLOUD_MAX_CONTEXT", "100000")),
                max_retrieve_results=int(os.environ.get("CLOUD_MAX_RETRIEVE_RESULTS", "20"))
            )
            if not openai_key:
                print("[配置] 警告: CLOUD_PROVIDER=openai 但 OPENAI_API_KEY 为空")
        elif cloud_provider == "gemini":
            self.cloud = CloudConfig(
                enabled=cloud_enabled,
                provider="gemini",
                api_key=gemini_key,
                model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
                base_url=os.environ.get("GEMINI_BASE_URL"),
                max_context=int(os.environ.get("CLOUD_MAX_CONTEXT", "100000")),
                max_retrieve_results=int(os.environ.get("CLOUD_MAX_RETRIEVE_RESULTS", "20"))
            )
            if not gemini_key:
                print("[配置] 警告: CLOUD_PROVIDER=gemini 但 GEMINI_API_KEY 为空")
        elif cloud_provider == "glm":
            self.cloud = CloudConfig(
                enabled=cloud_enabled,
                provider="glm",
                api_key=glm_key,
                model=os.environ.get("GLM_MODEL", "glm-4-flash"),
                base_url=os.environ.get("GLM_BASE_URL"),
                max_context=int(os.environ.get("CLOUD_MAX_CONTEXT", "128000")),
                max_retrieve_results=int(os.environ.get("CLOUD_MAX_RETRIEVE_RESULTS", "20"))
            )
            if not glm_key:
                print("[配置] 警告: CLOUD_PROVIDER=glm 但 GLM_API_KEY 为空")
        else:
            print(f"[配置] 警告: 未知的 CLOUD_PROVIDER: {cloud_provider}")
    
    def _load_async_config(self):
        """加载异步处理器配置"""
        self.async_processor.dedup_interval = int(os.environ.get("MEMORY_DEDUP_INTERVAL", "300"))
        self.async_processor.compression_interval = int(os.environ.get("MEMORY_COMPRESSION_INTERVAL", "600"))
        self.async_processor.forget_interval = int(os.environ.get("MEMORY_FORGET_INTERVAL", "3600"))
        self.async_processor.flush_interval = int(os.environ.get("MEMORY_FLUSH_INTERVAL", "30"))
        self.async_processor.max_buffer_age = int(os.environ.get("MEMORY_MAX_BUFFER_AGE", "60"))
        self.async_processor.batch_size = int(os.environ.get("BATCH_UPDATE_SIZE", "5"))
        self.async_processor.max_pending_compressions = int(os.environ.get("MAX_PENDING_COMPRESSIONS", "50"))
        self.async_processor.compressor_check_interval = int(os.environ.get("COMPRESSOR_CHECK_INTERVAL", "300"))
        self.async_processor.compressor_failure_threshold = int(os.environ.get("COMPRESSOR_FAILURE_THRESHOLD", "5"))
        self.async_processor.max_queue_size = int(os.environ.get("MAX_QUEUE_SIZE", "1000"))
        self.async_processor.queue_full_action = os.environ.get("QUEUE_FULL_ACTION", "drop_oldest")
        self.async_processor.llm_timeout = int(os.environ.get("LLM_TIMEOUT", "30"))
        self.async_processor.circuit_breaker_threshold = int(os.environ.get("CIRCUIT_BREAKER_THRESHOLD", "5"))
        self.async_processor.circuit_breaker_reset_timeout = int(os.environ.get("CIRCUIT_BREAKER_RESET_TIMEOUT", "300"))
    
    def _load_flow_config(self):
        """加载记忆流动配置"""
        self.memory_flow.cooldown_hours = int(os.environ.get("MEMORY_FLOW_COOLDOWN_HOURS", "24"))
        self.memory_flow.l2_move_threshold_multiplier = float(os.environ.get("L2_MOVE_THRESHOLD_MULTIPLIER", "2.0"))
        self.memory_flow.l3_promotion_weight_threshold = float(os.environ.get("L3_PROMOTION_WEIGHT_THRESHOLD", "0.8"))
        self.memory_flow.max_l2_to_l3_batch = int(os.environ.get("MAX_L2_TO_L3_BATCH", "20"))
        self.memory_flow.max_l3_to_l2_batch = int(os.environ.get("MAX_L3_TO_L2_BATCH", "10"))
    
    def _load_compression_config(self):
        """加载压缩配置"""
        self.compression.min_length = int(os.environ.get("COMPRESSION_MIN_LENGTH", "100"))
        self.compression.target_ratio = float(os.environ.get("COMPRESSION_TARGET_RATIO", "0.6"))
        self.compression.max_segment_length = int(os.environ.get("COMPRESSION_MAX_SEGMENT_LENGTH", "2000"))
        self.compression.fallback_enabled = os.environ.get("COMPRESSION_FALLBACK_ENABLED", "true").lower() == "true"
        self.compression.key_patterns = os.environ.get("COMPRESSION_KEY_PATTERNS", 
            r'\d+|[A-Z][a-z]+|设置|配置|修改|添加|删除|创建|因为|所以|如果|那么|但是')
    
    def _load_privacy_config(self):
        """加载隐私与安全配置"""
        self.privacy.sensitive_filter_enabled = os.environ.get("SENSITIVE_FILTER_ENABLED", "true").lower() == "true"
        self.privacy.local_encryption_enabled = os.environ.get("LOCAL_ENCRYPTION_ENABLED", "false").lower() == "true"
        self.privacy.excluded_keywords = os.environ.get("PRIVACY_EXCLUDED_KEYWORDS", "")
        self.privacy.log_sensitive_detection = os.environ.get("LOG_SENSITIVE_DETECTION", "false").lower() == "true"
        self.privacy.https_only = os.environ.get("HTTPS_ONLY", "true").lower() == "true"
        self.privacy.verify_ssl = os.environ.get("VERIFY_SSL", "true").lower() == "true"
        self.privacy.api_key_min_length = int(os.environ.get("API_KEY_MIN_LENGTH", "20"))
        self.privacy.mask_in_logs = os.environ.get("MASK_IN_LOGS", "true").lower() == "true"
    
    def _load_sqlite_pool_config(self):
        """加载SQLite连接池配置"""
        self.sqlite_pool.max_pool_size = int(os.environ.get("SQLITE_MAX_POOL_SIZE", "10"))
        self.sqlite_pool.connection_timeout = int(os.environ.get("SQLITE_CONNECTION_TIMEOUT", "3600"))
        self.sqlite_pool.cleanup_interval = int(os.environ.get("SQLITE_CLEANUP_INTERVAL", "300"))
        self.sqlite_pool.busy_timeout = int(os.environ.get("SQLITE_BUSY_TIMEOUT", "30000"))
        self.sqlite_pool.cache_size = int(os.environ.get("SQLITE_CACHE_SIZE", "-64000"))
    
    def _load_decay_config(self):
        """加载权重衰减配置"""
        self.decay.batch_size = int(os.environ.get("DECAY_BATCH_SIZE", "1000"))
        self.decay.decay_rate = float(os.environ.get("MEMORY_WEIGHT_DECAY", "0.95"))
        self.decay.min_weight_threshold = float(os.environ.get("MEMORY_MIN_WEIGHT", "0.3"))
        self.decay.max_weight = float(os.environ.get("MEMORY_MAX_WEIGHT", "5.0"))
        self.decay.weight_boost_on_access = float(os.environ.get("MEMORY_WEIGHT_BOOST", "1.2"))
        self.decay.forget_age_days = int(os.environ.get("MEMORY_DECAY_DAYS", "30"))
    
    def _load_retrieval_config(self):
        """加载检索配置"""
        self.retrieval.max_retrieve_results = int(os.environ.get("MAX_RETRIEVE_RESULTS", "5"))
        self.retrieval.similarity_threshold = float(os.environ.get("SIMILARITY_THRESHOLD", "0.90"))
        self.retrieval.l1_min_results = int(os.environ.get("L1_MIN_RESULTS", "2"))
        self.retrieval.l2_lower_threshold = float(os.environ.get("L2_LOWER_THRESHOLD", "0.80"))
        self.retrieval.cloud_l1_threshold = float(os.environ.get("CLOUD_L1_THRESHOLD", "0.85"))
        self.retrieval.cloud_l2_threshold = float(os.environ.get("CLOUD_L2_THRESHOLD", "0.75"))
        self.retrieval.cloud_l3_threshold = float(os.environ.get("CLOUD_L3_THRESHOLD", "0.70"))
        self.retrieval.local_l1_threshold = float(os.environ.get("LOCAL_L1_THRESHOLD", "0.90"))
        self.retrieval.local_l2_threshold = float(os.environ.get("LOCAL_L2_THRESHOLD", "0.80"))
        self.retrieval.local_l3_threshold = float(os.environ.get("LOCAL_L3_THRESHOLD", "0.75"))
    
    def _load_vector_store_config(self):
        """加载向量存储配置"""
        self.vector_store.onnx_model_path = os.environ.get("ONNX_MODEL_PATH", "models/bge_onnx_model")
        self.vector_store.embedding_dimension = int(os.environ.get("EMBEDDING_DIMENSION", "512"))
        self.vector_store.chroma_persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "chroma_db")
        self.vector_store.chroma_collection_name = os.environ.get("CHROMA_COLLECTION_NAME", "memories")
    
    def _load_sqlite_store_config(self):
        """加载SQLite存储配置"""
        self.sqlite_store.db_path = os.environ.get("SQLITE_DB_PATH", "memory.db")
        self.sqlite_store.enabled = os.environ.get("SQLITE_ENABLED", "true").lower() == "true"
    
    def _load_nonsense_filter_config(self):
        """加载废话过滤配置"""
        self.nonsense_filter.enabled = os.environ.get("NONSENSE_FILTER_ENABLED", "true").lower() == "true"
        self.nonsense_filter.db_path = os.environ.get("NONSENSE_DB_PATH", "nonsense_library.json")
    
    def _load_high_density_config(self):
        """加载高密度内容配置"""
        self.high_density.patterns = os.environ.get("HIGH_DENSITY_PATTERNS", 
            "流程图|时序图|架构图|状态机|数据流|ER图|类图|部署图|网络拓扑|API接口|数据库设计|系统设计|技术方案")
        self.high_density.preserve_ratio = float(os.environ.get("HIGH_DENSITY_PRESERVE_RATIO", "1.0"))
    
    def _load_context_config(self):
        """加载上下文构建配置"""
        self.context.max_memory_tokens = int(os.environ.get("MAX_MEMORY_TOKENS", "5000"))
        self.context.system_prompt_reserve = int(os.environ.get("SYSTEM_PROMPT_RESERVE", "200"))
    
    @property
    def max_retrieve_results(self) -> int:
        return self.retrieval.max_retrieve_results
    
    @property
    def similarity_threshold(self) -> float:
        return self.retrieval.similarity_threshold
    
    @property
    def l1_min_results(self) -> int:
        return self.retrieval.l1_min_results
    
    @property
    def l2_lower_threshold(self) -> float:
        return self.retrieval.l2_lower_threshold
    
    @property
    def cloud_l1_threshold(self) -> float:
        return self.retrieval.cloud_l1_threshold
    
    @property
    def cloud_l2_threshold(self) -> float:
        return self.retrieval.cloud_l2_threshold
    
    @property
    def cloud_l3_threshold(self) -> float:
        return self.retrieval.cloud_l3_threshold
    
    @property
    def local_l1_threshold(self) -> float:
        return self.retrieval.local_l1_threshold
    
    @property
    def local_l2_threshold(self) -> float:
        return self.retrieval.local_l2_threshold
    
    @property
    def local_l3_threshold(self) -> float:
        return self.retrieval.local_l3_threshold
    
    @property
    def source_weight_l1(self) -> float:
        return self.retrieval.source_weight_l1
    
    @property
    def source_weight_l2(self) -> float:
        return self.retrieval.source_weight_l2
    
    @property
    def source_weight_l3(self) -> float:
        return self.retrieval.source_weight_l3
    
    @property
    def diversity_threshold(self) -> float:
        return self.retrieval.diversity_threshold
    
    @property
    def high_density_patterns(self) -> str:
        return self.high_density.patterns
    
    @property
    def high_density_preserve_ratio(self) -> float:
        return self.high_density.preserve_ratio
    
    @property
    def onnx_model_path(self) -> str:
        return self.vector_store.onnx_model_path
    
    @property
    def embedding_dimension(self) -> int:
        return self.vector_store.embedding_dimension
    
    @property
    def chroma_persist_dir(self) -> str:
        return self.vector_store.chroma_persist_dir
    
    @property
    def chroma_collection_name(self) -> str:
        return self.vector_store.chroma_collection_name
    
    @property
    def sqlite_enabled(self) -> bool:
        return self.sqlite_store.enabled
    
    @property
    def sqlite_db_path(self) -> str:
        return self.sqlite_store.db_path
    
    @property
    def memory_decay_days(self) -> int:
        return self.decay.forget_age_days
    
    @property
    def memory_min_weight(self) -> float:
        return self.decay.min_weight_threshold
    
    @property
    def memory_max_weight(self) -> float:
        return self.decay.max_weight
    
    @property
    def memory_weight_boost(self) -> float:
        return self.decay.weight_boost_on_access
    
    @property
    def memory_weight_decay(self) -> float:
        return self.decay.decay_rate
    
    @property
    def compression_enabled(self) -> bool:
        return self.compression.fallback_enabled
    
    @property
    def compression_min_length(self) -> int:
        return self.compression.min_length
    
    @property
    def compression_idle_threshold(self) -> int:
        return self.async_processor.compression_interval
    
    @property
    def async_update_interval(self) -> int:
        return self.async_processor.flush_interval
    
    @property
    def batch_update_size(self) -> int:
        return self.async_processor.batch_size
    
    @property
    def nonsense_filter_enabled(self) -> bool:
        return self.nonsense_filter.enabled
    
    @property
    def nonsense_db_path(self) -> str:
        return self.nonsense_filter.db_path
    
    @property
    def max_memory_tokens(self) -> int:
        return self.context.max_memory_tokens
    
    @property
    def system_prompt_reserve(self) -> int:
        return self.context.system_prompt_reserve


config = Config()


def get_memory_config() -> Config:
    """获取全局记忆配置实例（兼容旧接口）"""
    return config
