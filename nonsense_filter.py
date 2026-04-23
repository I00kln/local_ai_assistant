# nonsense_filter.py
# 废话过滤器 - 三层过滤架构（规则→密度→向量）
import os
import re
import json
import threading
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from config import config
from memory_tags import MemoryTagHelper, MemoryConstants


@dataclass
class FilterResult:
    """过滤结果"""
    is_nonsense: bool
    reason: str
    confidence: float  # 0-1，越高越是废话
    storage_type: str = "full"  # "full"=存向量库, "sqlite_only"=仅存SQLite, "discard"=丢弃
    metadata: Dict = None  # 额外元数据标签


class NonsenseFilter:
    """
    废话过滤器 - 混合三层过滤
    
    第一层：规则过滤（快速拦截明显废话）
    第二层：信息密度评分（基于语义向量）
    第三层：向量相似度匹配（与废话库对比）
    
    存储策略：
    - full: 存入向量库+SQLite（有价值记忆）
    - sqlite_only: 仅存SQLite（保持对话完整性，不污染向量空间）
    - discard: 完全丢弃（纯噪音）
    """
    
    # 保护模式 - 这些内容不应被判定为废话
    PROTECTED_PATTERNS = [
        r"不[，,]?\s*是",           # 纠错："不，是100"
        r"不对[，,]?\s*",           # 纠错："不对，应该是..."
        r"不对\s*\d+",              # 纠错："不对200"
        r"更正[：:，,]?\s*",        # 更正："更正：..."
        r"更正一下[是为]?\s*",      # 更正："更正一下是"
        r"纠正[一下]?\s*[是为]?\s*", # 纠正
        r"其实是[是为]?\s*",        # 更正："其实是"
        r"记错[了]?\s*[是为]?\s*",  # 更正："记错了是"
        r"应该是[是为]?\s*",        # 更正："应该是"
        r"搞错[了]?\s*[是为]?\s*",  # 更正："搞错了是"
        r"弄错[了]?\s*[是为]?\s*",  # 更正："弄错了是"
        r"说错[了]?\s*[是为]?\s*",  # 更正："说错了是"
        r"写错[了]?\s*[是为]?\s*",  # 更正："写错了是"
        r"刚才[是为]?\s*",          # 更正："刚才是"
        r"之前[是为]?\s*",          # 更正："之前是"
        r"\d+(?:\.\d+)?(?:元|美元|块|万|千|百|亿|%|％|度|kg|ml|GB|MB|TB|cm|mm|m|km)",  # 有意义的数字（价格、百分比、单位）
        r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}",  # 日期格式
        r"\d{1,2}:\d{2}",          # 时间格式
        r"是\s*\d+",               # 确认数值："是100"
        r"等于\s*\d+",             # 等于数值
        r"总共\s*\d+",             # 总数
        r"共\s*\d+",               # 共计
        r"花了\s*\d+",             # 花费
        r"买了\s*\d+",             # 购买数量
        r"设置[为到]\s*\d+",       # 设置数值
        r"配置[为到]\s*\d+",       # 配置数值
        r"太[棒好美赞酷厉害]了",    # 情感反馈（反映用户偏好）
        r"非常[棒好美赞酷厉害]",    # 情感反馈
        r"[最很特别]+[棒好美赞酷厉害]",  # 情感反馈
        r"[不没]错",               # 确认/纠错
        r"等等[，,]?",             # 补充说明
        r"还有[，,]?",             # 补充说明
        r"注意[：:]?",             # 重要提示
        r"警告[：:]?",             # 警告
        r"重要[：:]?",             # 重要信息
    ]
    
    # 情感关键词 - 反映用户偏好，应保留
    EMOTION_KEYWORDS = {
        "太棒了", "太好了", "太美了", "太赞了", "太酷了", "太厉害了",
        "非常好", "很好", "极好", "优秀", "出色",
        "喜欢", "爱", "推荐", "建议",
        "不满意", "不喜欢", "问题", "错误", "bug",
    }
    
    # 默认废话库（可扩展）
    DEFAULT_NONSENSE_TEMPLATES = [
        # 单字/短回应
        "嗯", "哦", "啊", "哈", "好", "行", "对", "是", "没错", "对的",
        "嗯嗯", "哦哦", "啊啊", "哈哈", "呵呵", "嘿嘿", "好的", "好吧", "行了",
        # 确认类
        "知道了", "明白了", "了解了", "清楚了", "懂了", "收到",
        "好的知道了", "好的明白了", "我知道了", "我明白了",
        # 追问类
        "然后呢", "接着呢", "还有吗", "继续说", "讲下去",
        # 附和类
        "你说得对", "有道理", "确实如此", "没错没错", "赞同",
        # 情绪类（无信息）
        "哈哈哈", "哈哈哈哈", "笑死", "太搞笑了", "绝了",
        # 元对话
        "重复一下问题", "再说一遍", "我没听清", "你刚才说什么",
    ]
    
    # 规则过滤模式
    NONSENSE_PATTERNS = [
        r"^[嗯哦啊哈嘻嘿嗯哼]+[。！]?$",  # 纯语气词
        r"^[好的行对是嗯晓得]+[。！]?$",  # 简短确认
        r"^[哈哈呵呵嘿嘿]+[。！]?$",  # 纯笑声
        r"^[\s\n\r]*$",  # 纯空白
        r"^[0-9\s!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]+$",  # 纯数字和标点
    ]
    
    # 关键词黑名单（快速匹配）
    NONSENSE_KEYWORDS = {
        "嗯", "哦", "啊", "哈", "嘻", "嘿", "哼",
        "嗯嗯", "哦哦", "啊啊", "哈哈", "呵呵", "嘿嘿",
        "好的", "好吧", "行了", "可以", "对", "是的", "没错", "对的",
        "知道了", "明白了", "了解了", "清楚了", "懂了", "收到",
        "好的知道了", "好的明白了", "我知道了", "我明白了",
        "然后呢", "接着呢", "还有吗", "继续说", "讲下去",
        "你说得对", "有道理", "确实如此", "没错没错", "赞同",
        "哈哈哈", "哈哈哈哈", "笑死", "太搞笑了", "绝了",
        "重复一下问题", "再说一遍", "我没听清", "你刚才说什么",
    }
    
    def __init__(self, nonsense_db_path: str = "nonsense_library.json"):
        self.nonsense_db_path = nonsense_db_path
        self.dimension = config.embedding_dimension
        
        self._embedding_service = None
        self._model_lock = threading.Lock()
        self._ready_event = threading.Event()
        self._warmup_started = False
        
        self.nonsense_texts: List[str] = []
        self.nonsense_vectors: Optional[np.ndarray] = None
        self._vectors_computed = False
        self._load_nonsense_library()
        
        self._exact_hashes = self._build_exact_hashes()
        
        self.length_threshold = 10
        self.density_threshold = 0.15
        self.similarity_threshold = 0.85
        
        self.stats = {
            "total_checked": 0,
            "rule_filtered": 0,
            "hash_filtered": 0,
            "density_filtered": 0,
            "vector_filtered": 0,
            "passed": 0
        }
    
    def warmup(self):
        """
        预热模型和向量计算
        
        应在系统启动时显式调用，而非在构造函数中自动启动后台线程
        """
        if self._warmup_started:
            return
        self._warmup_started = True
        self._start_vector_precompute_thread()
    
    def _build_exact_hashes(self) -> set:
        """构建高频废话的MD5哈希集合"""
        import hashlib
        hashes = set()
        for text in self.nonsense_texts:
            text_hash = hashlib.md5(text.strip().encode('utf-8')).hexdigest()
            hashes.add(text_hash)
        for text in self.NONSENSE_KEYWORDS:
            text_hash = hashlib.md5(text.strip().encode('utf-8')).hexdigest()
            hashes.add(text_hash)
        return hashes
    
    def _ensure_model_loaded(self):
        """确保 EmbeddingService 已初始化"""
        if self._embedding_service is not None:
            return
        
        with self._model_lock:
            if self._embedding_service is not None:
                return
            
            from embedding_service import get_embedding_service
            self._embedding_service = get_embedding_service()
            self._ready_event.set()
    
    @property
    def tokenizer(self):
        self._ensure_model_loaded()
        return self._embedding_service._tokenizer
    
    @property
    def session(self):
        self._ensure_model_loaded()
        return self._embedding_service._session
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """获取文本的嵌入向量"""
        self._ensure_model_loaded()
        return self._embedding_service.embed_single(text, use_prefix=True)
    
    def _load_nonsense_library(self):
        """加载废话库（仅加载文本，不计算向量）"""
        if os.path.exists(self.nonsense_db_path):
            try:
                with open(self.nonsense_db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.nonsense_texts = data.get("templates", self.DEFAULT_NONSENSE_TEMPLATES)
                print(f"废话过滤器：加载 {len(self.nonsense_texts)} 条模板（向量计算推迟到后台）")
            except Exception as e:
                print(f"加载废话库失败: {e}，使用默认模板")
                self.nonsense_texts = self.DEFAULT_NONSENSE_TEMPLATES.copy()
        else:
            self.nonsense_texts = self.DEFAULT_NONSENSE_TEMPLATES.copy()
            self._save_nonsense_library()
    
    def _start_vector_precompute_thread(self):
        """启动后台线程预计算向量"""
        def compute_vectors_background():
            """后台计算向量"""
            try:
                self._ready_event.wait(timeout=30.0)
                
                self._ensure_model_loaded()
                
                if not self._embedding_service.is_available:
                    return
                
                vectors = []
                for text in self.nonsense_texts:
                    try:
                        vec = self._get_embedding(text)
                        vectors.append(vec)
                    except Exception:
                        pass
                
                if vectors:
                    self.nonsense_vectors = np.array(vectors)
                    self._vectors_computed = True
                    print(f"废话过滤器：后台预计算向量完成（{len(vectors)} 条）")
            except Exception as e:
                print(f"废话过滤器：后台向量计算失败: {e}")
        
        thread = threading.Thread(target=compute_vectors_background, daemon=True)
        thread.start()
    
    def _save_nonsense_library(self):
        """保存废话库"""
        try:
            with open(self.nonsense_db_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "templates": self.nonsense_texts,
                    "updated_at": datetime.now().isoformat()
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存废话库失败: {e}")
    
    def _precompute_vectors(self):
        """预计算废话库向量（同步版本，用于手动触发）"""
        # 如果后台已经计算完成，直接返回
        if self._vectors_computed and self.nonsense_vectors is not None:
            return
        
        # 确保模型已加载
        self._ensure_model_loaded()
        
        if not self.nonsense_texts or not self._session:
            return
        
        try:
            vectors = []
            for text in self.nonsense_texts:
                vec = self._get_embedding(text)
                vectors.append(vec[0])
            self.nonsense_vectors = np.array(vectors)
            self._vectors_computed = True
            print(f"预计算废话库向量完成: {len(vectors)} 条")
        except Exception as e:
            print(f"预计算向量失败: {e}")
            self.nonsense_vectors = None
    
    def add_nonsense_template(self, text: str) -> bool:
        """添加新的废话模板"""
        text = text.strip()
        if not text or text in self.nonsense_texts:
            return False
        
        self.nonsense_texts.append(text)
        self._save_nonsense_library()
        # 标记向量需要重新计算，但不立即计算（下次使用时后台计算）
        self._vectors_computed = False
        self._start_vector_precompute_thread()
        return True
    
    def remove_nonsense_template(self, text: str) -> bool:
        """移除废话模板"""
        if text in self.nonsense_texts:
            self.nonsense_texts.remove(text)
            self._save_nonsense_library()
            # 标记向量需要重新计算
            self._vectors_computed = False
            self._start_vector_precompute_thread()
            return True
        return False
    
    def _rule_filter(self, text: str) -> Tuple[bool, float]:
        """
        第一层：规则过滤
        返回: (是否是废话, 置信度)
        """
        text_stripped = text.strip()
        
        # 空内容检查
        if not text_stripped:
            return True, 0.99
        
        # 提取用户和助理的对话内容
        lines = text_stripped.split('\n')
        user_line = ""
        assistant_line = ""
        for line in lines:
            if line.startswith("用户:"):
                user_line = line[3:].strip()
            elif line.startswith("助理:"):
                assistant_line = line[3:].strip()
        
        # 检查是否有实质内容
        combined_content = user_line + assistant_line
        if not combined_content:
            return True, 0.99
        
        # 长度检查 - 太短的内容直接判定为废话
        if len(combined_content) <= 2:
            return True, 0.95
        
        # 检查用户输入是否为废话关键词
        if user_line in self.NONSENSE_KEYWORDS:
            return True, 0.9
        
        # 检查助理回复是否为废话关键词
        if assistant_line in self.NONSENSE_KEYWORDS:
            return True, 0.9
        
        # 检查合并后内容是否全为语气词
        if combined_content and all(c in "嗯哦啊哈嘻嘿哼哈哈呵呵嘿嘿" for c in combined_content):
            return True, 0.9
        
        # 正则匹配
        for pattern in self.NONSENSE_PATTERNS:
            try:
                if re.match(pattern, text_stripped, re.UNICODE):
                    return True, 0.85
            except re.error:
                continue
        
        # 纯表情检查
        if self._is_pure_emoji(text_stripped):
            return True, 0.85
        
        # 重复字符检查（如"哈哈哈哈哈哈"）
        if self._is_repetitive(text_stripped):
            return True, 0.8
        
        return False, 0.0
    
    def _is_pure_emoji(self, text: str) -> bool:
        """检查是否纯表情"""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        return bool(emoji_pattern.fullmatch(text))
    
    def _is_repetitive(self, text: str, threshold: float = 0.7) -> bool:
        """检查文本是否过度重复"""
        if len(text) < 5:
            return False
        
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        max_repeat = max(char_counts.values())
        return max_repeat / len(text) > threshold
    
    def _calculate_information_density(self, text: str) -> float:
        """
        第二层：计算信息密度
        基于语义向量的范数变化来估计信息丰富度
        """
        if not self.session:
            return 0.5  # 默认中等密度
        
        try:
            # 获取文本向量
            vec = self._get_embedding(text)[0]
            
            # 计算向量的统计特征
            vec_mean = np.mean(np.abs(vec))
            vec_std = np.std(vec)
            vec_entropy = self._calculate_entropy(vec)
            
            # 综合评分（越高表示信息越丰富）
            density = (vec_mean * 0.3 + vec_std * 0.4 + vec_entropy * 0.3)
            
            density = min(1.0, max(0.0, density * MemoryConstants.DENSITY_NORMALIZATION_MULTIPLIER))
            
            return density
        except Exception as e:
            print(f"计算信息密度失败: {e}")
            return 0.5
    
    def _calculate_entropy(self, vec: np.ndarray) -> float:
        """计算向量熵值"""
        # 将向量转为概率分布
        probs = np.abs(vec) + 1e-10
        probs = probs / np.sum(probs)
        
        # 计算香农熵
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        return entropy / np.log2(len(vec))  # 归一化
    
    def _vector_filter(self, text: str) -> Tuple[bool, float]:
        """
        第三层：向量相似度过滤
        与废话库进行相似度匹配
        """
        if self.nonsense_vectors is None or len(self.nonsense_vectors) == 0:
            return False, 0.0
        
        try:
            text_vec = self._get_embedding(text)[0]
            
            # 计算与废话库的相似度
            similarities = np.dot(self.nonsense_vectors, text_vec)
            max_similarity = float(np.max(similarities))
            
            if max_similarity > self.similarity_threshold:
                return True, max_similarity
            
            return False, max_similarity
        except Exception as e:
            print(f"向量过滤失败: {e}")
            return False, 0.0
    
    def _check_protected(self, text: str) -> bool:
        """
        检查是否为受保护内容
        
        受保护内容包括：
        - 纠错类对话
        - 包含数字的内容
        - 情感反馈
        - 重要提示
        """
        for pattern in self.PROTECTED_PATTERNS:
            try:
                if re.search(pattern, text, re.UNICODE):
                    return True
            except re.error:
                continue
        
        # 检查情感关键词
        for keyword in self.EMOTION_KEYWORDS:
            if keyword in text:
                return True
        
        return False
    
    def filter(self, user_input: str, assistant_response: str) -> FilterResult:
        """
        主过滤方法 - 三层过滤 + 保护机制
        
        流程：
        0. 保护检查（防止误删重要内容）
        1. Hash快速匹配（毫秒级拦截高频废话）
        2. 规则过滤（快速拦截90%明显废话）
        3. 信息密度评分（语义层面）
        4. 向量相似度匹配（精确匹配）
        
        存储策略：
        - full: 存入向量库+SQLite
        - sqlite_only: 仅存SQLite
        - discard: 完全丢弃
        """
        import hashlib
        self.stats["total_checked"] += 1
        
        combined_text = f"用户: {user_input}\n助理: {assistant_response}"
        
        if self._check_protected(combined_text):
            self.stats["passed"] += 1
            protected_metadata = MemoryTagHelper.mark_protected({}, "auto_detected_by_filter")
            return FilterResult(
                is_nonsense=False,
                reason="受保护内容（纠错/情感/重要信息）",
                confidence=0.0,
                storage_type="full",
                metadata=protected_metadata
            )
        
        text_hash = hashlib.md5(combined_text.strip().encode('utf-8')).hexdigest()
        if text_hash in self._exact_hashes:
            self.stats["hash_filtered"] += 1
            return FilterResult(
                is_nonsense=True,
                reason="Hash精确匹配：高频废话",
                confidence=0.99,
                storage_type="discard"
            )
        
        is_nonsense, confidence = self._rule_filter(combined_text)
        if is_nonsense:
            self.stats["rule_filtered"] += 1
            storage_type = "discard" if confidence > 0.9 else "sqlite_only"
            return FilterResult(
                is_nonsense=True,
                reason="规则过滤：明显无意义内容",
                confidence=confidence,
                storage_type=storage_type
            )
        
        # 短文本进入深度检查
        if len(combined_text) < self.length_threshold * 2:
            effective_threshold = self.density_threshold
            
            if len(combined_text) < 15:
                effective_threshold = 0.08
            
            if re.search(r'\d+', combined_text):
                effective_threshold *= 0.6
            
            density = self._calculate_information_density(combined_text)
            if density < effective_threshold:
                self.stats["density_filtered"] += 1
                return FilterResult(
                    is_nonsense=True,
                    reason=f"信息密度过低: {density:.2f}",
                    confidence=1.0 - density,
                    storage_type="sqlite_only"
                )
            
            if self.nonsense_vectors is not None and len(self.nonsense_vectors) > 0:
                is_nonsense, similarity = self._vector_filter(combined_text)
                if is_nonsense:
                    self.stats["vector_filtered"] += 1
                    return FilterResult(
                        is_nonsense=True,
                        reason=f"与废话库相似度: {similarity:.2f}",
                        confidence=similarity,
                        storage_type="sqlite_only"
                    )
        
        self.stats["passed"] += 1
        return FilterResult(
            is_nonsense=False,
            reason="通过所有过滤层",
            confidence=0.0,
            storage_type="full"
        )
    
    def get_stats(self) -> Dict:
        """获取过滤统计信息"""
        total = self.stats["total_checked"]
        if total == 0:
            return self.stats.copy()
        
        stats_with_rate = self.stats.copy()
        stats_with_rate["filter_rate"] = round(
            (total - self.stats["passed"]) / total * 100, 2
        )
        return stats_with_rate
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            "total_checked": 0,
            "rule_filtered": 0,
            "density_filtered": 0,
            "vector_filtered": 0,
            "passed": 0
        }


# 全局过滤器实例
_nonsense_filter: Optional[NonsenseFilter] = None


def get_nonsense_filter() -> NonsenseFilter:
    """获取全局废话过滤器实例"""
    global _nonsense_filter
    if _nonsense_filter is None:
        _nonsense_filter = NonsenseFilter()
    return _nonsense_filter


def should_store_to_memory(user_input: str, assistant_response: str) -> bool:
    """
    便捷函数：判断是否应该存入记忆
    
    使用示例：
    if should_store_to_memory(user_input, assistant_response):
        memory.add(conversation_text)
    """
    filter_result = get_nonsense_filter().filter(user_input, assistant_response)
    return not filter_result.is_nonsense
