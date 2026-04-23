# tag_classifier.py
# 独立标签分类系统 - 非阻塞、异步处理
"""
标签系统架构：
1. 规则预标记器：落盘时快速预标记（<5ms）
2. LLM 分类器：压缩时异步细化（复用调用）
3. 标签合并器：合并记忆时处理标签一致性
4. 用户修正：支持手动修正标签

设计原则：
- 独立模块，不修改现有核心逻辑
- 异步处理，不阻塞主流程
- 分层策略：规则 → LLM → 用户修正
"""

import re
import threading
import queue
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import Counter
import json

from memory_tags import MemoryTags


class TagCategory(Enum):
    """记忆分类标签"""
    WORK = "work"
    LIFE = "life"
    STUDY = "study"
    HEALTH = "health"
    FINANCE = "finance"
    TECH = "tech"
    ENTERTAINMENT = "entertainment"
    TRAVEL = "travel"
    RELATIONSHIP = "relationship"
    OTHER = "other"


class TagImportance(Enum):
    """重要性标签"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TagSentiment(Enum):
    """情感标签"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class TagSource(Enum):
    """标签来源"""
    RULE = "rule"
    LLM = "llm"
    USER = "user"
    MERGED = "merged"


@dataclass
class MemoryTag:
    """记忆标签数据结构"""
    category: str = "other"
    importance: str = "medium"
    sentiment: str = "neutral"
    entities: List[str] = field(default_factory=list)
    time_sensitive: bool = False
    topics: List[str] = field(default_factory=list)
    
    source: str = "rule"
    confidence: float = 0.0
    created_at: str = ""
    updated_at: str = ""
    
    user_corrected: bool = False
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryTag":
        if not data:
            return cls()
        return cls(
            category=data.get("category", "other"),
            importance=data.get("importance", "medium"),
            sentiment=data.get("sentiment", "neutral"),
            entities=data.get("entities", []),
            time_sensitive=data.get("time_sensitive", False),
            topics=data.get("topics", []),
            source=data.get("source", "rule"),
            confidence=data.get("confidence", 0.0),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            user_corrected=data.get("user_corrected", False)
        )


@dataclass
class TagConfig:
    """标签系统配置"""
    enabled: bool = True
    rule_tagging_enabled: bool = True
    llm_tagging_enabled: bool = True
    async_llm_tagging: bool = True
    min_confidence: float = 0.6
    max_entities: int = 10
    max_topics: int = 5


@dataclass
class TagStats:
    """标签统计"""
    total_tagged: int = 0
    rule_tagged: int = 0
    llm_tagged: int = 0
    user_corrected: int = 0
    category_distribution: Dict[str, int] = field(default_factory=dict)


class RuleBasedTagger:
    """
    规则预标记器
    
    基于关键词、正则、模板进行快速预标记
    延迟 <5ms，不影响主流程
    """
    
    CATEGORY_KEYWORDS = {
        TagCategory.WORK.value: [
            "工作", "会议", "项目", "报告", "任务", "deadline", "加班",
            "同事", "老板", "客户", "邮件", "ppt", "演示", "汇报",
            "work", "meeting", "project", "report", "deadline", "office"
        ],
        TagCategory.LIFE.value: [
            "生活", "日常", "家里", "做饭", "购物", "家务", "快递",
            "外卖", "超市", "周末", "假期", "休息", "生活",
            "life", "home", "shopping", "weekend"
        ],
        TagCategory.STUDY.value: [
            "学习", "课程", "考试", "作业", "笔记", "复习", "预习",
            "学校", "老师", "同学", "论文", "研究", "study", "learn",
            "course", "exam", "homework", "school", "university"
        ],
        TagCategory.HEALTH.value: [
            "健康", "医院", "医生", "吃药", "体检", "运动", "健身",
            "减肥", "睡眠", "头痛", "感冒", "预约", "门诊",
            "health", "hospital", "doctor", "exercise", "gym"
        ],
        TagCategory.FINANCE.value: [
            "钱", "工资", "转账", "还款", "账单", "投资", "理财",
            "银行", "信用卡", "花呗", "借呗", "余额", "存款",
            "money", "salary", "bank", "invest", "finance", "payment"
        ],
        TagCategory.TECH.value: [
            "代码", "编程", "bug", "开发", "部署", "服务器", "数据库",
            "api", "git", "python", "java", "javascript", "框架",
            "code", "programming", "develop", "server", "database"
        ],
        TagCategory.ENTERTAINMENT.value: [
            "电影", "游戏", "音乐", "小说", "追剧", "综艺", "动漫",
            "演唱会", "旅游", "美食", "餐厅", "电影票",
            "movie", "game", "music", "entertainment", "fun"
        ],
        TagCategory.TRAVEL.value: [
            "旅游", "旅行", "机票", "酒店", "景点", "签证", "护照",
            "出差", "高铁", "飞机", "行程", "目的地",
            "travel", "trip", "flight", "hotel", "vacation"
        ],
        TagCategory.RELATIONSHIP.value: [
            "朋友", "家人", "父母", "孩子", "老公", "老婆", "男朋友",
            "女朋友", "生日", "纪念日", "结婚", "聚会",
            "friend", "family", "parents", "relationship", "love"
        ]
    }
    
    IMPORTANCE_PATTERNS = {
        TagImportance.HIGH.value: [
            r"紧急", r"重要", r"必须", r"务必", r"千万",
            r"deadline", r"urgent", r"important", r"critical",
            r"不要忘", r"一定要", r"绝对", r"务必"
        ],
        TagImportance.LOW.value: [
            r"随便", r"无所谓", r"不重要", r"可选", r"有空",
            r"optional", r"maybe", r"whenever", r"不急"
        ]
    }
    
    TIME_PATTERNS = [
        r"明天", r"后天", r"下周", r"下个月", r"周末",
        r"今天", r"昨天", r"前天", r"上周", r"上个月",
        r"\d+月\d+日", r"\d+号", r"\d+点", r"\d+:\d+",
        r"tomorrow", r"yesterday", r"next week", r"today",
        r"上午", r"下午", r"晚上", r"早上", r"中午"
    ]
    
    ENTITY_PATTERNS = {
        "person": [
            r"([\\u4e00-\\u9fa5]{2,3})(说|告诉|问|让|请|帮)",
            r"(同事|朋友|老师|医生|老板)([\\u4e00-\\u9fa5]{2,3})?"
        ],
        "location": [
            r"在([\\u4e00-\\u9fa5]{2,10})(工作|生活|学习|出差)",
            r"去([\\u4e00-\\u9fa5]{2,10})(旅游|出差|开会)"
        ],
        "time": [
            r"(\\d{4}年\\d{1,2}月\\d{1,2}日)",
            r"(\\d{1,2}月\\d{1,2}[日号])",
            r"(明天|后天|下周[一二三四五六日]?)"
        ]
    }
    
    SENTIMENT_PATTERNS = {
        TagSentiment.POSITIVE.value: [
            r"开心", r"高兴", r"喜欢", r"棒", r"好", r"成功",
            r"完成", r"解决", r"谢谢", r"感谢", r"满意",
            r"happy", r"great", r"good", r"nice", r"love", r"thanks"
        ],
        TagSentiment.NEGATIVE.value: [
            r"烦", r"讨厌", r"累", r"难过", r"失败", r"问题",
            r"错误", r"麻烦", r"担心", r"焦虑", r"压力",
            r"sad", r"angry", r"tired", r"bad", r"hate", r"worry"
        ]
    }
    
    PROTECTED_PATTERNS = [
        r"不[，。]", r"不是", r"不对", r"更正", r"修改",
        r"取消", r"删除", r"忽略", r"算了"
    ]
    
    def __init__(self, config: TagConfig = None):
        self.config = config or TagConfig()
        self._compiled_patterns = {}
        self._compile_patterns()
    
    def _compile_patterns(self):
        """预编译正则表达式"""
        for category, patterns in self.IMPORTANCE_PATTERNS.items():
            self._compiled_patterns[f"importance_{category}"] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
        
        for sentiment, patterns in self.SENTIMENT_PATTERNS.items():
            self._compiled_patterns[f"sentiment_{sentiment}"] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
        
        self._compiled_patterns["time"] = [
            re.compile(p, re.IGNORECASE) for p in self.TIME_PATTERNS
        ]
        
        self._compiled_patterns["protected"] = [
            re.compile(p, re.IGNORECASE) for p in self.PROTECTED_PATTERNS
        ]
    
    def tag(self, text: str) -> MemoryTag:
        """
        对文本进行规则预标记
        
        Args:
            text: 待标记文本
        
        Returns:
            MemoryTag 对象
        """
        if not text:
            return MemoryTag()
        
        text_lower = text.lower()
        
        category = self._detect_category(text_lower)
        importance = self._detect_importance(text)
        sentiment = self._detect_sentiment(text)
        time_sensitive = self._detect_time_sensitive(text)
        entities = self._extract_entities(text)
        topics = self._extract_topics(text_lower)
        
        confidence = self._calculate_confidence(
            category, importance, sentiment, time_sensitive
        )
        
        return MemoryTag(
            category=category,
            importance=importance,
            sentiment=sentiment,
            entities=entities,
            time_sensitive=time_sensitive,
            topics=topics,
            source=TagSource.RULE.value,
            confidence=confidence
        )
    
    def _detect_category(self, text: str) -> str:
        """检测分类"""
        scores = {}
        
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scores[category] = score
        
        if not scores:
            return TagCategory.OTHER.value
        
        max_score = max(scores.values())
        candidates = [c for c, s in scores.items() if s == max_score]
        
        return candidates[0]
    
    def _detect_importance(self, text: str) -> str:
        """检测重要性"""
        for importance, patterns in self.IMPORTANCE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return importance
        
        return TagImportance.MEDIUM.value
    
    def _detect_sentiment(self, text: str) -> str:
        """检测情感"""
        positive_count = 0
        negative_count = 0
        
        for pattern in self._compiled_patterns.get("sentiment_positive", []):
            if pattern.search(text):
                positive_count += 1
        
        for pattern in self._compiled_patterns.get("sentiment_negative", []):
            if pattern.search(text):
                negative_count += 1
        
        if positive_count > negative_count:
            return TagSentiment.POSITIVE.value
        elif negative_count > positive_count:
            return TagSentiment.NEGATIVE.value
        
        return TagSentiment.NEUTRAL.value
    
    def _detect_time_sensitive(self, text: str) -> bool:
        """检测是否时间敏感"""
        for pattern in self._compiled_patterns.get("time", []):
            if pattern.search(text):
                return True
        return False
    
    def _extract_entities(self, text: str) -> List[str]:
        """提取实体"""
        entities = []
        
        for entity_type, patterns in self.ENTITY_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    if isinstance(match, tuple):
                        entity = match[0] if match[0] else match[-1]
                    else:
                        entity = match
                    
                    if entity and len(entity) >= 2:
                        entities.append(entity)
        
        entities = list(dict.fromkeys(entities))
        
        return entities[:self.config.max_entities]
    
    def _extract_topics(self, text: str) -> List[str]:
        """提取主题关键词"""
        topics = []
        
        all_keywords = []
        for keywords in self.CATEGORY_KEYWORDS.values():
            all_keywords.extend(keywords)
        
        for kw in all_keywords:
            if kw in text and kw not in topics:
                topics.append(kw)
        
        return topics[:self.config.max_topics]
    
    def _calculate_confidence(
        self, 
        category: str, 
        importance: str, 
        sentiment: str,
        time_sensitive: bool
    ) -> float:
        """计算置信度"""
        confidence = 0.5
        
        if category != TagCategory.OTHER.value:
            confidence += 0.2
        
        if importance != TagImportance.MEDIUM.value:
            confidence += 0.1
        
        if sentiment != TagSentiment.NEUTRAL.value:
            confidence += 0.1
        
        if time_sensitive:
            confidence += 0.1
        
        return min(confidence, 1.0)


class LLMTagger:
    """
    LLM 标签分类器
    
    异步处理，不阻塞主流程
    在压缩时复用 LLM 调用
    
    依赖注入：
    - sqlite_store: 通过 set_sqlite_store() 注入，避免循环依赖
    """
    
    LLM_TAG_PROMPT = """请分析以下记忆内容，返回分类标签。

记忆内容：
{memory_text}

请返回 JSON 格式的标签：
{{
  "category": "work|life|study|health|finance|tech|entertainment|travel|relationship|other",
  "importance": "high|medium|low",
  "sentiment": "positive|negative|neutral",
  "entities": ["提取的人名、地点、事件等"],
  "time_sensitive": true/false,
  "topics": ["主题关键词"]
}}

只返回 JSON，不要其他解释。"""
    
    def __init__(self, config: TagConfig = None, sqlite_store=None):
        self.config = config or TagConfig()
        self._sqlite_store = sqlite_store
        self._task_queue = queue.Queue(maxsize=100)
        self._result_cache: Dict[str, MemoryTag] = {}
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._llm_client = None
        self._log = self._get_logger()
    
    def set_sqlite_store(self, sqlite_store):
        """
        设置 SQLite 存储实例（依赖注入）
        
        Args:
            sqlite_store: SQLiteStore 实例
        """
        self._sqlite_store = sqlite_store
    
    def _get_sqlite_store(self):
        """
        获取 SQLite 存储实例
        
        Returns:
            SQLiteStore 实例或 None
        """
        if self._sqlite_store is not None:
            return self._sqlite_store
        
        try:
            from sqlite_store import get_sqlite_store
            self._sqlite_store = get_sqlite_store()
            return self._sqlite_store
        except Exception as e:
            self._log.error("GET_SQLITE_STORE_FAILED", error=str(e))
            return None
    
    def _get_logger(self):
        """获取日志器"""
        try:
            from logger import get_logger
            return get_logger("LLMTagger")
        except Exception:
            class SimpleLog:
                def info(self, event, **kwargs):
                    print(f"[INFO] {event}: {kwargs}")
                def error(self, event, **kwargs):
                    print(f"[ERROR] {event}: {kwargs}")
            return SimpleLog()
    
    def start(self):
        """启动异步处理线程"""
        if self._running:
            return
        
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
        self._log.info("LLM_TAGGER_STARTED")
    
    def stop(self):
        """停止异步处理线程"""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        self._log.info("LLM_TAGGER_STOPPED")
    
    def _worker(self):
        """后台处理线程"""
        while self._running:
            try:
                task = self._task_queue.get(timeout=1.0)
                if task:
                    self._process_task(task)
            except queue.Empty:
                continue
            except Exception as e:
                self._log.error("LLM_TAGGER_WORKER_ERROR", error=str(e))
    
    def _process_task(self, task: Dict):
        """处理单个任务"""
        memory_id = task.get("memory_id")
        memory_text = task.get("memory_text")
        callback = task.get("callback")
        use_atomic_update = task.get("use_atomic_update", True)
        
        if not memory_text:
            return
        
        try:
            tag = self._classify_with_llm(memory_text)
            
            if tag:
                tag.source = TagSource.LLM.value
                tag.updated_at = datetime.now().isoformat()
                
                self._result_cache[memory_id] = tag
                
                if use_atomic_update:
                    try:
                        memory_id_int = int(memory_id)
                        sqlite_store = self._get_sqlite_store()
                        
                        if sqlite_store:
                            sqlite_store.update_metadata_field(
                                record_id=memory_id_int,
                                field_path="semantic_tag",
                                field_value=tag.to_dict()
                            )
                            self._log.info(
                                "LLM_TAG_ATOMIC_UPDATE",
                                memory_id=memory_id,
                                category=tag.category
                            )
                        else:
                            self._log.warning(
                                "LLM_TAG_SQLITE_NOT_AVAILABLE",
                                memory_id=memory_id
                            )
                    except (ValueError, Exception) as e:
                        self._log.warning(
                            "LLM_TAG_ATOMIC_FAILED",
                            memory_id=memory_id,
                            error=str(e)
                        )
                
                if callback:
                    callback(memory_id, tag)
                
                self._log.info(
                    "LLM_TAG_COMPLETED",
                    memory_id=memory_id,
                    category=tag.category
                )
        except Exception as e:
            self._log.error("LLM_TAG_FAILED", memory_id=memory_id, error=str(e))
    
    def _classify_with_llm(self, text: str) -> Optional[MemoryTag]:
        """使用 LLM 进行分类"""
        if not self._llm_client:
            try:
                from cloud_client import CloudClientFactory
                from config import config
                
                if config.cloud.api_key and config.cloud.api_key != "your-api-key-here":
                    self._llm_client = CloudClientFactory.create(
                        provider=config.cloud.provider,
                        api_key=config.cloud.api_key,
                        model=config.cloud.model,
                        base_url=config.cloud.base_url
                    )
            except Exception as e:
                self._log.error("LLM_CLIENT_INIT_FAILED", error=str(e))
                return None
        
        if not self._llm_client:
            return None
        
        try:
            prompt = self.LLM_TAG_PROMPT.format(memory_text=text[:500])
            
            response = self._llm_client.chat([
                {"role": "user", "content": prompt}
            ])
            
            if response and response.get("content"):
                return self._parse_llm_response(response["content"])
        except Exception as e:
            self._log.error("LLM_CALL_FAILED", error=str(e))
        
        return None
    
    def _parse_llm_response(self, response: str) -> Optional[MemoryTag]:
        """
        解析 LLM 响应
        
        使用多层策略提取 JSON：
        1. 优先提取 Markdown JSON 代码块
        2. 使用括号栈匹配算法提取完整 JSON 对象
        3. 支持嵌套 JSON 结构
        """
        if not response:
            return None
        
        json_str = self._extract_json(response)
        
        if not json_str:
            return None
        
        try:
            data = json.loads(json_str)
            return MemoryTag(
                category=data.get("category", "other"),
                importance=data.get("importance", "medium"),
                sentiment=data.get("sentiment", "neutral"),
                entities=data.get("entities", []),
                time_sensitive=data.get("time_sensitive", False),
                topics=data.get("topics", []),
                confidence=0.9
            )
        except json.JSONDecodeError as e:
            self._log.error("JSON_DECODE_FAILED", error=str(e), json_preview=json_str[:100])
            return None
    
    def _extract_json(self, text: str) -> Optional[str]:
        """
        从文本中提取 JSON 字符串
        
        策略：
        1. 提取 Markdown JSON 代码块 ```json ... ```
        2. 提取普通代码块 ``` ... ```
        3. 使用括号栈匹配提取完整 JSON 对象
        """
        import re
        
        json_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        matches = re.findall(json_block_pattern, text, re.IGNORECASE)
        
        for match in matches:
            candidate = match.strip()
            if candidate.startswith('{') and candidate.endswith('}'):
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    continue
        
        json_str = self._extract_json_by_brace_stack(text)
        if json_str:
            return json_str
        
        return None
    
    def _extract_json_by_brace_stack(self, text: str) -> Optional[str]:
        """
        使用括号栈匹配算法提取完整的 JSON 对象
        
        支持嵌套结构，正确处理字符串内的括号
        """
        start_idx = -1
        brace_count = 0
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\' and in_string:
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if in_string:
                continue
            
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx >= 0:
                    candidate = text[start_idx:i+1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        start_idx = -1
                        brace_count = 0
        
        return None
    
    def submit_tag_task(
        self, 
        memory_id: str, 
        memory_text: str,
        callback: Optional[Callable] = None,
        use_atomic_update: bool = True
    ):
        """
        提交异步标记任务
        
        Args:
            memory_id: 记忆ID
            memory_text: 记忆文本
            callback: 完成回调函数 callback(memory_id, tag)
            use_atomic_update: 是否使用原子更新（默认 True）
        """
        if not self.config.llm_tagging_enabled:
            return
        
        if self.config.async_llm_tagging:
            task = {
                "memory_id": memory_id,
                "memory_text": memory_text,
                "callback": callback,
                "use_atomic_update": use_atomic_update
            }
            try:
                self._task_queue.put(task, block=False)
            except queue.Full:
                self._log.warning(
                    "LLM_TAG_QUEUE_FULL",
                    memory_id=memory_id,
                    queue_size=self._task_queue.qsize()
                )
        else:
            tag = self._classify_with_llm(memory_text)
            if tag and callback:
                callback(memory_id, tag)
    
    def get_cached_tag(self, memory_id: str) -> Optional[MemoryTag]:
        """获取缓存的标签"""
        return self._result_cache.get(memory_id)
    
    def classify_sync(self, text: str) -> Optional[MemoryTag]:
        """同步分类（用于压缩时复用）"""
        return self._classify_with_llm(text)


class TagMerger:
    """
    标签合并策略
    
    处理合并记忆时的标签一致性
    """
    
    IMPORTANCE_ORDER = {
        TagImportance.HIGH.value: 3,
        TagImportance.MEDIUM.value: 2,
        TagImportance.LOW.value: 1
    }
    
    def merge(self, tags_list: List[MemoryTag]) -> MemoryTag:
        """
        合并多个标签
        
        策略：
        1. 用户修正的标签有一票否决权（最高优先级）
        2. category: 用户修正优先，否则投票决定
        3. importance: 取最高
        4. sentiment: 用户修正优先，否则投票决定
        5. entities: 合并去重
        6. time_sensitive: 任一为 true 则为 true
        7. topics: 合并去重
        """
        if not tags_list:
            return MemoryTag()
        
        if len(tags_list) == 1:
            return tags_list[0]
        
        user_corrected_tags = [t for t in tags_list if t.user_corrected]
        has_user_correction = len(user_corrected_tags) > 0
        
        if has_user_correction:
            base_tag = user_corrected_tags[0]
            category = base_tag.category
            sentiment = base_tag.sentiment
            confidence = base_tag.confidence
        else:
            category = self._vote_category(tags_list)
            sentiment = self._vote_sentiment(tags_list)
            confidence = self._average_confidence(tags_list)
        
        return MemoryTag(
            category=category,
            importance=self._max_importance(tags_list),
            sentiment=sentiment,
            entities=self._merge_entities(tags_list),
            time_sensitive=self._any_time_sensitive(tags_list),
            topics=self._merge_topics(tags_list),
            source=TagSource.MERGED.value,
            confidence=confidence,
            user_corrected=has_user_correction
        )
    
    def _vote_category(self, tags_list: List[MemoryTag]) -> str:
        """投票决定分类"""
        categories = [t.category for t in tags_list if t.category]
        if not categories:
            return TagCategory.OTHER.value
        
        counter = Counter(categories)
        return counter.most_common(1)[0][0]
    
    def _max_importance(self, tags_list: List[MemoryTag]) -> str:
        """取最高重要性"""
        max_imp = TagImportance.LOW.value
        for tag in tags_list:
            if self.IMPORTANCE_ORDER.get(tag.importance, 0) > self.IMPORTANCE_ORDER.get(max_imp, 0):
                max_imp = tag.importance
        return max_imp
    
    def _vote_sentiment(self, tags_list: List[MemoryTag]) -> str:
        """投票决定情感"""
        sentiments = [t.sentiment for t in tags_list if t.sentiment]
        if not sentiments:
            return TagSentiment.NEUTRAL.value
        
        counter = Counter(sentiments)
        return counter.most_common(1)[0][0]
    
    def _merge_entities(self, tags_list: List[MemoryTag]) -> List[str]:
        """合并实体列表"""
        all_entities = []
        for tag in tags_list:
            all_entities.extend(tag.entities)
        return list(dict.fromkeys(all_entities))[:10]
    
    def _any_time_sensitive(self, tags_list: List[MemoryTag]) -> bool:
        """任一为时间敏感则为 true"""
        return any(t.time_sensitive for t in tags_list)
    
    def _merge_topics(self, tags_list: List[MemoryTag]) -> List[str]:
        """合并主题列表"""
        all_topics = []
        for tag in tags_list:
            all_topics.extend(tag.topics)
        return list(dict.fromkeys(all_topics))[:5]
    
    def _average_confidence(self, tags_list: List[MemoryTag]) -> float:
        """计算平均置信度"""
        confidences = [t.confidence for t in tags_list]
        return sum(confidences) / len(confidences) if confidences else 0.0


class TagCorrectionManager:
    """
    用户修正管理
    
    处理用户手动修正标签
    """
    
    def __init__(self):
        self._correction_log: List[Dict] = []
        self._log = self._get_logger()
    
    def _get_logger(self):
        """获取日志器"""
        try:
            from logger import get_logger
            return get_logger("TagCorrection")
        except Exception:
            class SimpleLog:
                def info(self, event, **kwargs):
                    print(f"[INFO] {event}: {kwargs}")
            return SimpleLog()
    
    def correct_tag(
        self,
        memory_id: str,
        old_tag: MemoryTag,
        new_tag: MemoryTag,
        sqlite_store=None
    ) -> bool:
        """
        用户修正标签
        
        Args:
            memory_id: 记忆ID
            old_tag: 原标签
            new_tag: 新标签
            sqlite_store: 存储实例
        
        Returns:
            是否成功
        """
        new_tag.user_corrected = True
        new_tag.source = TagSource.USER.value
        new_tag.updated_at = datetime.now().isoformat()
        
        if sqlite_store:
            try:
                record = sqlite_store.get(memory_id)
                if record:
                    if not record.metadata:
                        record.metadata = {}
                    
                    record.metadata[MemoryTags.SEMANTIC_TAG] = json.dumps(new_tag.to_dict(), ensure_ascii=False)
                    record.metadata[MemoryTags.TAG_CORRECTED_AT] = datetime.now().isoformat()
                    
                    sqlite_store.add(record)
                    
                    self._log_correction(memory_id, old_tag, new_tag)
                    
                    return True
            except Exception as e:
                self._log.info(
                    "TAG_CORRECTION_FAILED",
                    memory_id=memory_id,
                    error=str(e)
                )
                return False
        
        return True
    
    def _log_correction(self, memory_id: str, old_tag: MemoryTag, new_tag: MemoryTag):
        """记录修正日志"""
        correction = {
            "memory_id": memory_id,
            "old_category": old_tag.category,
            "new_category": new_tag.category,
            "old_importance": old_tag.importance,
            "new_importance": new_tag.importance,
            "timestamp": datetime.now().isoformat()
        }
        self._correction_log.append(correction)
        
        self._log.info(
            "TAG_CORRECTED",
            memory_id=memory_id,
            old_category=old_tag.category,
            new_category=new_tag.category
        )
    
    def get_correction_patterns(self) -> Dict[str, Any]:
        """获取修正模式（用于优化规则）"""
        if not self._correction_log:
            return {}
        
        category_corrections = Counter()
        for log in self._correction_log:
            key = f"{log['old_category']}->{log['new_category']}"
            category_corrections[key] += 1
        
        return {
            "total_corrections": len(self._correction_log),
            "category_patterns": dict(category_corrections.most_common(10))
        }


class TagClassifier:
    """
    标签分类器主入口
    
    单例模式，整合所有标签功能
    
    依赖注入：
    - sqlite_store: 通过 set_sqlite_store() 或构造函数注入，避免循环依赖
    """
    
    _instance: Optional['TagClassifier'] = None
    _lock = threading.Lock()
    
    def __new__(cls, sqlite_store=None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, sqlite_store=None):
        if self._initialized:
            if sqlite_store is not None:
                self._sqlite_store = sqlite_store
            return
        
        self._initialized = True
        self._sqlite_store = sqlite_store
        self.config = TagConfig()
        self.stats = TagStats()
        
        self._rule_tagger = RuleBasedTagger(self.config)
        self._llm_tagger = LLMTagger(self.config)
        self._tag_merger = TagMerger()
        self._correction_manager = TagCorrectionManager()
        
        self._log = self._get_logger()
        
        if self.config.enabled and self.config.llm_tagging_enabled:
            self._llm_tagger.start()
    
    def set_sqlite_store(self, sqlite_store):
        """
        设置 SQLite 存储实例（依赖注入）
        
        Args:
            sqlite_store: SQLiteStore 实例
        """
        self._sqlite_store = sqlite_store
    
    def _get_sqlite_store(self):
        """
        获取 SQLite 存储实例
        
        优先使用注入的实例，否则延迟获取
        
        Returns:
            SQLiteStore 实例或 None
        """
        if self._sqlite_store is not None:
            return self._sqlite_store
        
        try:
            from sqlite_store import get_sqlite_store
            self._sqlite_store = get_sqlite_store()
            return self._sqlite_store
        except Exception as e:
            self._log.error("GET_SQLITE_STORE_FAILED", error=str(e))
            return None
    
    def _get_logger(self):
        """获取日志器"""
        try:
            from logger import get_logger
            return get_logger("TagClassifier")
        except Exception:
            class SimpleLog:
                def info(self, event, **kwargs):
                    print(f"[INFO] {event}: {kwargs}")
                def error(self, event, **kwargs):
                    print(f"[ERROR] {event}: {kwargs}")
            return SimpleLog()
    
    def tag_memory(
        self, 
        text: str, 
        memory_id: str = "",
        enable_llm: bool = False,
        callback: Optional[Callable] = None
    ) -> MemoryTag:
        """
        标记记忆
        
        Args:
            text: 记忆文本
            memory_id: 记忆ID（用于异步 LLM 标记）
            enable_llm: 是否启用 LLM 标记
            callback: LLM 标记完成回调
        
        Returns:
            规则预标记结果（立即可用）
        """
        if not self.config.enabled:
            return MemoryTag()
        
        tag = self._rule_tagger.tag(text)
        
        self.stats.total_tagged += 1
        self.stats.rule_tagged += 1
        
        category = tag.category
        self.stats.category_distribution[category] = \
            self.stats.category_distribution.get(category, 0) + 1
        
        if enable_llm and self.config.llm_tagging_enabled and memory_id:
            self._llm_tagger.submit_tag_task(
                memory_id=memory_id,
                memory_text=text,
                callback=callback
            )
        
        return tag
    
    def tag_for_compression(self, text: str) -> MemoryTag:
        """
        压缩时标记（复用 LLM 调用）
        
        Args:
            text: 记忆文本
        
        Returns:
            LLM 标记结果，失败则返回规则标记
        """
        if not self.config.enabled:
            return MemoryTag()
        
        if self.config.llm_tagging_enabled:
            llm_tag = self._llm_tagger.classify_sync(text)
            if llm_tag:
                self.stats.llm_tagged += 1
                return llm_tag
        
        return self._rule_tagger.tag(text)
    
    def merge_tags(self, tags_list: List[MemoryTag]) -> MemoryTag:
        """
        合并标签
        
        Args:
            tags_list: 待合并的标签列表
        
        Returns:
            合并后的标签
        """
        return self._tag_merger.merge(tags_list)
    
    def correct_tag(
        self,
        memory_id: str,
        new_tag: MemoryTag,
        sqlite_store=None
    ) -> bool:
        """
        用户修正标签
        
        Args:
            memory_id: 记忆ID
            new_tag: 新标签
            sqlite_store: 存储实例
        
        Returns:
            是否成功
        """
        try:
            old_tag = MemoryTag()
            
            if sqlite_store:
                record = sqlite_store.get(memory_id)
                if record and record.metadata:
                    old_tag = MemoryTag.from_dict(
                        record.metadata.get(MemoryTags.SEMANTIC_TAG, {})
                    )
            
            success = self._correction_manager.correct_tag(
                memory_id=memory_id,
                old_tag=old_tag,
                new_tag=new_tag,
                sqlite_store=sqlite_store
            )
            
            if success:
                self.stats.user_corrected += 1
            
            return success
        except Exception as e:
            self._log.error("TAG_CORRECTION_ERROR", error=str(e))
            return False
    
    def get_tag_from_metadata(self, metadata: Dict) -> MemoryTag:
        """
        从元数据中提取标签
        
        Args:
            metadata: 记忆元数据
        
        Returns:
            MemoryTag 对象
        """
        if not metadata:
            return MemoryTag()
        
        tag_data = metadata.get(MemoryTags.SEMANTIC_TAG, {})
        
        if isinstance(tag_data, str):
            try:
                tag_data = json.loads(tag_data)
            except json.JSONDecodeError:
                tag_data = {}
        
        return MemoryTag.from_dict(tag_data)
    
    def update_metadata_with_tag(
        self, 
        metadata: Dict, 
        tag: MemoryTag
    ) -> Dict:
        """
        更新元数据中的标签
        
        Args:
            metadata: 原元数据
            tag: 标签对象
        
        Returns:
            更新后的元数据
        
        注意：ChromaDB 不支持嵌套 dict，因此将 tag 序列化为 JSON 字符串
        """
        if metadata is None:
            metadata = {}
        
        metadata[MemoryTags.SEMANTIC_TAG] = json.dumps(tag.to_dict(), ensure_ascii=False)
        
        return metadata
    
    def update_tag_atomically(
        self, 
        record_id: int, 
        tag: MemoryTag,
        sqlite_store = None
    ) -> bool:
        """
        原子更新数据库中的 semantic_tag 字段
        
        使用 SQLite json_set 进行原子操作，避免并发覆盖
        
        Args:
            record_id: 记录ID
            tag: 标签对象
            sqlite_store: SQLite 存储实例（可选，使用依赖注入）
        
        Returns:
            是否成功
        """
        if sqlite_store is None:
            sqlite_store = self._get_sqlite_store()
        
        if sqlite_store is None:
            self._log.error("SQLITE_STORE_NOT_AVAILABLE")
            return False
        
        try:
            return sqlite_store.update_metadata_field(
                record_id=record_id,
                field_path="semantic_tag",
                field_value=tag.to_dict()
            )
        except Exception as e:
            self._log.error("ATOMIC_TAG_UPDATE_FAILED", error=str(e))
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_tagged": self.stats.total_tagged,
            "rule_tagged": self.stats.rule_tagged,
            "llm_tagged": self.stats.llm_tagged,
            "user_corrected": self.stats.user_corrected,
            "category_distribution": self.stats.category_distribution
        }
    
    def shutdown(self):
        """关闭标签系统"""
        if self._llm_tagger:
            self._llm_tagger.stop()
        
        self._log.info("TAG_CLASSIFIER_SHUTDOWN")


def get_tag_classifier(sqlite_store=None) -> TagClassifier:
    """
    获取标签分类器单例
    
    Args:
        sqlite_store: SQLite 存储实例（可选，用于依赖注入）
    
    Returns:
        TagClassifier 实例
    """
    return TagClassifier(sqlite_store=sqlite_store)
