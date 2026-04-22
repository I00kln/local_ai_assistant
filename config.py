# config.py
import os
from dataclasses import dataclass, field
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
class Config:
    # 本地AI配置
    local: LocalConfig = field(default_factory=LocalConfig)
    
    # 云端AI配置
    cloud: CloudConfig = field(default_factory=CloudConfig)
    
    # 检索配置
    max_retrieve_results: int = 5
    similarity_threshold: float = 0.90
    l1_min_results: int = 2
    l2_lower_threshold: float = 0.80
    
    # 云端专用检索配置（用户可调整）
    cloud_l1_threshold: float = 0.85
    cloud_l2_threshold: float = 0.75
    cloud_l3_threshold: float = 0.70
    
    # 本地压缩专用检索配置
    local_l1_threshold: float = 0.90
    local_l2_threshold: float = 0.80
    local_l3_threshold: float = 0.75
    
    # 高信息密度内容保护
    high_density_patterns: str = "流程图|时序图|架构图|状态机|数据流|ER图|类图|部署图|网络拓扑|API接口|数据库设计|系统设计|技术方案"
    high_density_preserve_ratio: float = 1.0
    
    # ONNX 嵌入模型配置
    onnx_model_path: str = "models/bge_onnx_model"
    embedding_dimension: int = 512
    
    # ChromaDB 配置（L2向量库）
    chroma_persist_dir: str = "chroma_db"
    chroma_collection_name: str = "memories"
    
    # SQLite 配置（L3长期存储）
    sqlite_db_path: str = "memory.db"
    sqlite_enabled: bool = True
    
    # 记忆管理配置
    memory_decay_days: int = 30  # 超过此天数的记忆开始衰减
    memory_min_weight: float = 0.3  # 低于此权重的记忆将被遗忘
    memory_max_weight: float = 5.0  # 最大权重上限
    memory_weight_boost: float = 1.2  # 访问时权重提升比例
    memory_weight_decay: float = 0.95  # 权重衰减比例
    
    # 压缩配置
    compression_enabled: bool = True
    compression_min_length: int = 100  # 最小压缩长度
    compression_idle_threshold: int = 300  # 空闲秒数阈值后开始压缩
    
    # 异步处理配置
    async_update_interval: int = 15
    batch_update_size: int = 5
    
    # 废话过滤配置
    nonsense_filter_enabled: bool = True
    nonsense_db_path: str = "nonsense_library.json"
    
    # 上下文构建配置
    max_memory_tokens: int = 5000
    system_prompt_reserve: int = 200
    
    system_prompt: str = """你是一个文本摘要助手。

【核心任务】
压缩内容，保留核心信息和关键细节，去除修饰和冗余。确保未参与对话的第三方阅读后能理解事情的全貌和前因后果。禁止省略细节中的因果关系。

【记忆使用规则】
1. 下方提供的【相关内容】包含之前的对话内容，结合原文进行压缩,不要写成一句话概括。
2. 如果相关内容与原文完全无关，可以忽略那部分。若无原文,则直接对相关内容进行压缩。
3. 不要提及"根据相关内容"或"原文"等表述；不要作为问题回答，只返回压缩结果。

【回答要求补充】
1. 当你进行总结时，遵循'无损压缩'原则。摘要长度需为[原文+相关内容]的 25%-40%,不超过3000token,且必须包含原文中出现的所有专有名词和数值。不得使用'等'、'之类'、'以及其他'等概括性虚词来省略具体内容。
3. 如果相关内容全部与原文无关，仅做文本精简，不改变原意，不补充任何原文没有的内容"""

    def load_local_config(self):
        """从环境变量加载本地AI配置"""
        local_enabled = os.environ.get("LOCAL_ENABLED", "true").lower() == "true"
        local_api_url = os.environ.get("LOCAL_API_URL", "http://localhost:8080/completion")

        self.local = LocalConfig(
            enabled=local_enabled,
            api_url=local_api_url,
            max_context=int(os.environ.get("LOCAL_MAX_CONTEXT", "8192")),
            temperature=float(os.environ.get("LOCAL_TEMPERATURE", "0.7")),
            max_output_tokens=int(os.environ.get("LOCAL_MAX_OUTPUT_TOKENS", "3000"))
        )

        # 加载检索配置
        self.max_retrieve_results = int(os.environ.get("MAX_RETRIEVE_RESULTS", "5"))
        self.similarity_threshold = float(os.environ.get("SIMILARITY_THRESHOLD", "0.90"))
        self.l1_min_results = int(os.environ.get("L1_MIN_RESULTS", "2"))
        self.l2_lower_threshold = float(os.environ.get("L2_LOWER_THRESHOLD", "0.80"))
        
        # 加载云端专用检索配置
        self.cloud_l1_threshold = float(os.environ.get("CLOUD_L1_THRESHOLD", "0.85"))
        self.cloud_l2_threshold = float(os.environ.get("CLOUD_L2_THRESHOLD", "0.75"))
        self.cloud_l3_threshold = float(os.environ.get("CLOUD_L3_THRESHOLD", "0.70"))
        
        # 加载本地压缩专用检索配置
        self.local_l1_threshold = float(os.environ.get("LOCAL_L1_THRESHOLD", "0.90"))
        self.local_l2_threshold = float(os.environ.get("LOCAL_L2_THRESHOLD", "0.80"))
        self.local_l3_threshold = float(os.environ.get("LOCAL_L3_THRESHOLD", "0.75"))
        
        # 加载高信息密度内容保护配置
        self.high_density_patterns = os.environ.get("HIGH_DENSITY_PATTERNS", 
            "流程图|时序图|架构图|状态机|数据流|ER图|类图|部署图|网络拓扑|API接口|数据库设计|系统设计|技术方案")
        self.high_density_preserve_ratio = float(os.environ.get("HIGH_DENSITY_PRESERVE_RATIO", "1.0"))
        
        # 加载模型配置
        self.onnx_model_path = os.environ.get("ONNX_MODEL_PATH", "models/bge_onnx_model")
        self.embedding_dimension = int(os.environ.get("EMBEDDING_DIMENSION", "512"))

        # 加载废话过滤器配置
        self.nonsense_filter_enabled = os.environ.get("NONSENSE_FILTER_ENABLED", "true").lower() == "true"
        self.nonsense_db_path = os.environ.get("NONSENSE_DB_PATH", "nonsense_library.json")
        
        # 加载SQLite配置
        self.sqlite_enabled = os.environ.get("SQLITE_ENABLED", "true").lower() == "true"
        self.sqlite_db_path = os.environ.get("SQLITE_DB_PATH", "memory.db")
        
        # 加载ChromaDB配置
        self.chroma_persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "chroma_db")
        self.chroma_collection_name = os.environ.get("CHROMA_COLLECTION_NAME", "memories")
        
        # 加载记忆管理配置
        self.memory_decay_days = int(os.environ.get("MEMORY_DECAY_DAYS", "30"))
        self.memory_min_weight = float(os.environ.get("MEMORY_MIN_WEIGHT", "0.3"))
        self.memory_max_weight = float(os.environ.get("MEMORY_MAX_WEIGHT", "5.0"))
        self.memory_weight_boost = float(os.environ.get("MEMORY_WEIGHT_BOOST", "1.2"))
        self.memory_weight_decay = float(os.environ.get("MEMORY_WEIGHT_DECAY", "0.95"))
        
        # 加载压缩配置
        self.compression_enabled = os.environ.get("COMPRESSION_ENABLED", "true").lower() == "true"
        self.compression_min_length = int(os.environ.get("COMPRESSION_MIN_LENGTH", "100"))
        self.compression_idle_threshold = int(os.environ.get("COMPRESSION_IDLE_THRESHOLD", "300"))
        
        # 加载异步处理配置
        self.async_update_interval = int(os.environ.get("ASYNC_UPDATE_INTERVAL", "15"))
        self.batch_update_size = int(os.environ.get("BATCH_UPDATE_SIZE", "5"))
        
        # 加载上下文构建配置
        self.max_memory_tokens = int(os.environ.get("MAX_MEMORY_TOKENS", "5000"))
        self.system_prompt_reserve = int(os.environ.get("SYSTEM_PROMPT_RESERVE", "200"))

    def load_cloud_config(self):
        """从环境变量加载云端配置"""
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        cloud_provider = os.environ.get("CLOUD_PROVIDER", "gemini")
        cloud_enabled = os.environ.get("CLOUD_ENABLED", "false").lower() == "true"
        cloud_max_context = int(os.environ.get("CLOUD_MAX_CONTEXT", "100000"))
        cloud_max_retrieve = int(os.environ.get("CLOUD_MAX_RETRIEVE_RESULTS", "20"))
        
        if openai_key and cloud_provider == "openai":
            self.cloud = CloudConfig(
                enabled=cloud_enabled,
                provider="openai",
                api_key=openai_key,
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                base_url=os.environ.get("OPENAI_BASE_URL"),
                max_context=cloud_max_context,
                max_retrieve_results=cloud_max_retrieve
            )
        elif gemini_key and cloud_provider == "gemini":
            self.cloud = CloudConfig(
                enabled=cloud_enabled,
                provider="gemini",
                api_key=gemini_key,
                model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
                base_url=os.environ.get("GEMINI_BASE_URL"),
                max_context=cloud_max_context,
                max_retrieve_results=cloud_max_retrieve
            )

# 全局配置实例
config = Config()
config.load_local_config()
config.load_cloud_config()
