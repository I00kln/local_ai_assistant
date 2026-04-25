# context_builder.py
# 上下文构建器 - 使用三层记忆管理器
from typing import List, Dict, Tuple, Set, Optional
from config import config
from memory_tags import MemoryConstants
from logger import get_logger
import re
from datetime import datetime, timedelta

SHORT_QUERY_STOPWORDS = {
    "好", "好的", "嗯", "哦", "是", "对", "行", "可以", "继续",
    "明白", "知道了", "谢谢", "感谢", "不客气", "再见", "拜拜",
    "ok", "yes", "no", "hi", "hello", "hey"
}

PRIORITY_BUCKET_A = "system_fixed"
PRIORITY_BUCKET_B = "recent_l1"
PRIORITY_BUCKET_C = "important_l1"
PRIORITY_BUCKET_D = "retrieved_l2_l3"

class ContextBuilder:
    """上下文构建器：多级记忆检索 + 动态长度控制 + 优先级聚合"""
    
    def __init__(self, memory_manager):
        self.memory = memory_manager
        self.max_context = config.local.max_context
        self.max_output = config.local.max_output_tokens
        self.max_memory_tokens = config.max_memory_tokens
        self.system_reserve = config.system_prompt_reserve
        
        self.l1_min_results = config.l1_min_results
        self.l2_default_threshold = config.similarity_threshold
        self.l2_lower_threshold = config.l2_lower_threshold
        self._log = get_logger()
        
        self._recent_conversation_ids: Set[str] = set()
        self._important_memory_ids: Set[str] = set()
        self._l1_content_hashes: Set[str] = set()
    
    def get_thresholds(self, mode: str = "local") -> Dict[str, float]:
        """
        根据模式获取threshold配置
        
        Args:
            mode: "local" (本地压缩) | "cloud_only" (仅云端) | "hybrid" (混合)
        
        Returns:
            {"l1": float, "l2": float, "l3": float}
        """
        if mode == "cloud_only":
            return {
                "l1": config.cloud_l1_threshold,
                "l2": config.cloud_l2_threshold,
                "l3": config.cloud_l3_threshold
            }
        else:
            return {
                "l1": config.local_l1_threshold,
                "l2": config.local_l2_threshold,
                "l3": config.local_l3_threshold
            }
    
    def get_max_retrieve(self, mode: str = "local") -> int:
        """根据模式获取最大检索数量"""
        if mode == "cloud_only":
            return config.cloud.max_retrieve_results
        return config.max_retrieve_results
    
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
            mode: "local" (本地压缩) | "cloud_only" (仅云端) | "hybrid" (混合)
        
        返回：
        - memory_context: 格式化后的记忆字符串
        - processed_user_input: 处理后的用户输入
        - retrieved_memories: 检索到的记忆列表
        - has_memories: 是否检索到有效记忆
        """
        thresholds = self.get_thresholds(mode)
        max_retrieve = self.get_max_retrieve(mode)
        
        retrieved = self._multi_level_retrieve(
            user_input, 
            conversation_history, 
            thresholds, 
            max_retrieve
        )
        
        memory_context, processed_user_input, _ = self._control_context_length(
            retrieved, user_input, mode, conversation_history
        )
        
        has_memories = bool(memory_context and memory_context.strip())
        
        self._log.info(
            "BUILD_CONTEXT_COMPLETE",
            query=user_input[:50],
            retrieved_count=len(retrieved),
            has_memories=has_memories,
            context_length=len(memory_context) if memory_context else 0,
            mode=mode
        )
        
        return memory_context, processed_user_input, retrieved, has_memories
    
    def _is_valid_query(self, query: str) -> bool:
        """判断查询是否值得检索"""
        query = query.strip().lower()
        
        if len(query) < 2:
            return False
        
        if query in SHORT_QUERY_STOPWORDS:
            return False

        if re.match(r'^[^\u4e00-\u9fa5a-zA-Z0-9]+$', query):
            return False
        
        return True
    
    def _expand_query(
        self, 
        current_input: str, 
        conversation_history: List[Dict] = None,
        max_history_turns: int = 2
    ) -> str:
        """
        Query 扩展：合并当前输入与前几轮对话核心实体
        
        策略：
        1. 提取当前输入的关键词
        2. 从前 N 轮对话中提取实体（名词短语）
        3. 合并去重后作为扩展查询
        
        Args:
            current_input: 当前用户输入
            conversation_history: 对话历史
            max_history_turns: 最多回溯的对话轮数
        
        Returns:
            扩展后的查询字符串
        """
        if not conversation_history or len(conversation_history) == 0:
            return current_input
        
        entities = set()
        
        current_entities = self._extract_entities(current_input)
        entities.update(current_entities)
        
        recent_history = conversation_history[-max_history_turns:] if conversation_history else []
        
        for conv in recent_history:
            user_text = conv.get("user", "")
            assistant_text = conv.get("assistant", "")
            
            user_entities = self._extract_entities(user_text)
            assistant_entities = self._extract_entities(assistant_text)
            
            entities.update(user_entities)
            entities.update(assistant_entities)
        
        entities.difference_update(current_entities)
        
        if entities:
            expanded = f"{current_input} {' '.join(list(entities)[:5])}"
            return expanded
        
        return current_input
    
    def _extract_entities(self, text: str) -> Set[str]:
        """
        从文本中提取实体（名词短语）
        
        策略：
        1. 提取中文名词短语（2-4字）
        2. 提取英文单词（大写开头或全大写）
        3. 提取数字+单位组合
        4. 过滤停用词
        
        Args:
            text: 输入文本
        
        Returns:
            实体集合
        """
        if not text:
            return set()
        
        entities = set()
        
        chinese_noun_pattern = r'[\u4e00-\u9fa5]{2,4}(?:项目|系统|模块|功能|文件|配置|服务|数据|接口|版本)'
        chinese_matches = re.findall(chinese_noun_pattern, text)
        entities.update(chinese_matches)
        
        english_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b|\b[A-Z]{2,}\b'
        english_matches = re.findall(english_pattern, text)
        entities.update(english_matches)
        
        number_unit_pattern = r'\d+(?:\.\d+)?(?:元|美元|块|万|千|百|亿|%|％|度|kg|ml|GB|MB|TB|ms|秒|分钟|小时|天|周|月|年)'
        number_matches = re.findall(number_unit_pattern, text)
        entities.update(number_matches)
        
        stopwords = {"这个", "那个", "什么", "怎么", "如何", "为什么", "是不是", "有没有", "可以", "能够", "需要", "应该"}
        entities.difference_update(stopwords)
        
        return entities
    
    def _multi_level_retrieve(
        self, 
        query: str, 
        conversation_history: List[Dict] = None,
        thresholds: Dict[str, float] = None,
        max_retrieve: int = None,
        enable_query_expansion: bool = True
    ) -> List[Dict]:
        """
        多级记忆检索
        
        重构说明：
        - 核心检索委托给 MemoryManager.search()，激活 RRF 融合和读锁保护
        - 保留 Query 扩展、去重、时间窗口过滤等后处理逻辑
        
        检索策略：
        1. Query 扩展（可选）
        2. 调用 MemoryManager.search() 获取 L1/L2/L3 结果
        3. 去重处理
        4. 时间窗口过滤
        """
        if not self._is_valid_query(query):
            self._log.info("QUERY_INVALID", query=query[:50])
            return []
        
        if thresholds is None:
            thresholds = {"l1": self.l2_default_threshold, "l2": self.l2_default_threshold, "l3": self.l2_lower_threshold}
        
        if max_retrieve is None:
            max_retrieve = config.max_retrieve_results
        
        self._update_l1_tracking(conversation_history)
        
        expanded_query = query
        if enable_query_expansion and conversation_history:
            expanded_query = self._expand_query(query, conversation_history)
        
        threshold = thresholds.get("l2", self.l2_default_threshold)
        
        self._log.info(
            "MULTI_LEVEL_RETRIEVE_START",
            query=query[:50],
            expanded_query=expanded_query[:50] if expanded_query != query else "same",
            threshold=threshold,
            max_retrieve=max_retrieve
        )
        
        search_results = self.memory.search(
            query=expanded_query,
            top_k=max_retrieve * 2,
            include_l3=True,
            threshold=threshold,
            include_l1=True
        )
        
        results = []
        for r in search_results:
            results.append({
                "text": r.text,
                "similarity": r.similarity,
                "source": r.source,
                "weight": r.weight,
                "metadata": r.metadata,
                "rrf_score": r.rrf_score,
                "rank": r.rank
            })
        
        results = self._deduplicate_against_l1(results, conversation_history)
        results = self._time_window_filter(results, conversation_history)
        
        self._log.info(
            "MULTI_LEVEL_RETRIEVE_COMPLETE",
            raw_count=len(search_results),
            final_count=len(results[:max_retrieve])
        )
        
        return results[:max_retrieve]
    
    def _update_l1_tracking(self, conversation_history: List[Dict] = None):
        """
        更新 L1 追踪集合（用于去重）
        
        修复：使用组装后的 combined_text 计算哈希，与检索结果格式一致
        
        Args:
            conversation_history: 对话历史
        """
        self._l1_content_hashes.clear()
        self._recent_conversation_ids.clear()
        
        if not conversation_history:
            return
        
        import hashlib
        
        for conv in conversation_history[-10:]:
            user_text = conv.get("user", "")
            assistant_text = conv.get("assistant", "")
            
            combined_text = f"用户: {user_text}\n助理: {assistant_text}"
            content_hash = hashlib.md5(combined_text.encode('utf-8')).hexdigest()[:16]
            self._l1_content_hashes.add(content_hash)
            
            if user_text:
                user_hash = hashlib.md5(user_text.encode('utf-8')).hexdigest()[:16]
                self._l1_content_hashes.add(user_hash)
            
            if assistant_text:
                assistant_hash = hashlib.md5(assistant_text.encode('utf-8')).hexdigest()[:16]
                self._l1_content_hashes.add(assistant_hash)
            
            conv_id = conv.get("id") or conv.get("conversation_id")
            if conv_id:
                self._recent_conversation_ids.add(str(conv_id))
    
    def _deduplicate_against_l1(
        self, 
        results: List[Dict], 
        conversation_history: List[Dict] = None
    ) -> List[Dict]:
        """
        对检索结果进行 L1 去重
        
        策略：
        1. ID 级硬去重：排除已存在于 L1 的记录
        2. 内容哈希去重：排除内容与 L1 对话相同的记录
        3. 相似度去重：排除与 L1 对话语义相似度 > 0.85 的记录
        
        注意：时间窗口过滤在 _time_window_filter 中单独处理
        
        Args:
            results: 检索结果列表
            conversation_history: L1 对话历史
        
        Returns:
            去重后的结果列表
        """
        if not results:
            return []
        
        import hashlib
        
        deduplicated = []
        
        l1_texts = []
        if conversation_history:
            for conv in conversation_history:
                user_text = conv.get("user", "")
                assistant_text = conv.get("assistant", "")
                if user_text:
                    l1_texts.append(user_text)
                if assistant_text:
                    l1_texts.append(assistant_text)
        
        for item in results:
            item_id = item.get("id") or item.get("record_id")
            if item_id and str(item_id) in self._recent_conversation_ids:
                continue
            
            text = item.get("text", "")
            if text:
                content_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:16]
                if content_hash in self._l1_content_hashes:
                    continue
                
                if l1_texts and self._is_similar_to_l1(text, l1_texts):
                    continue
            
            deduplicated.append(item)
        
        return deduplicated
    
    def _is_similar_to_l1(self, text: str, l1_texts: List[str], threshold: float = 0.85) -> bool:
        """
        检查文本是否与 L1 对话相似
        
        使用 Jaccard 相似度进行快速近似判断
        
        Args:
            text: 待检查文本
            l1_texts: L1 对话文本列表
            threshold: 相似度阈值
        
        Returns:
            是否相似
        """
        text_words = set(text.lower().split())
        if not text_words:
            return False
        
        for l1_text in l1_texts:
            l1_words = set(l1_text.lower().split())
            if not l1_words:
                continue
            
            intersection = len(text_words & l1_words)
            union = len(text_words | l1_words)
            
            if union > 0:
                similarity = intersection / union
                if similarity > threshold:
                    return True
        
        return False
    
    def _time_window_filter(
        self, 
        results: List[Dict], 
        conversation_history: List[Dict] = None,
        window_minutes: int = 10
    ) -> List[Dict]:
        """
        时间窗口过滤：仅排除与 L1 对话时间重叠的检索结果
        
        注意：此方法已重构，不再简单排除所有最近的记忆。
        时间窗口过滤仅用于避免重复显示 L1 中已有对话时间段的内容。
        
        策略：
        1. 获取 L1 对话的时间范围
        2. 仅排除时间戳在 L1 时间范围内的检索结果
        3. 保留所有其他结果（包括最近的记忆）
        
        Args:
            results: 检索结果列表
            conversation_history: L1 对话历史
            window_minutes: 时间窗口（分钟），用于扩展 L1 时间范围
        
        Returns:
            过滤后的结果列表
        """
        if not results:
            return results
        
        if not conversation_history:
            return results
        
        l1_time_ranges = []
        for conv in conversation_history[-10:]:
            timestamp_str = conv.get("timestamp")
            if timestamp_str:
                try:
                    conv_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    time_start = conv_time.replace(tzinfo=None) - timedelta(minutes=2)
                    time_end = conv_time.replace(tzinfo=None) + timedelta(minutes=2)
                    l1_time_ranges.append((time_start, time_end))
                except (ValueError, TypeError):
                    pass
        
        if not l1_time_ranges:
            return results
        
        filtered = []
        for item in results:
            timestamp_str = item.get("timestamp") or item.get("created_time")
            if timestamp_str:
                try:
                    item_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    item_time_naive = item_time.replace(tzinfo=None)
                    
                    in_l1_range = False
                    for time_start, time_end in l1_time_ranges:
                        if time_start <= item_time_naive <= time_end:
                            in_l1_range = True
                            break
                    
                    if in_l1_range:
                        continue
                except (ValueError, TypeError):
                    pass
            
            filtered.append(item)
        
        return filtered
    
    def _calculate_text_overlap(self, text1: str, text2: str) -> float:
        """
        计算两个文本的重叠度（基于字符级 Jaccard 相似度）
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
        
        Returns:
            重叠度 (0-1)
        """
        if not text1 or not text2:
            return 0.0
        
        chars1 = set(text1)
        chars2 = set(text2)
        
        if not chars1 or not chars2:
            return 0.0
        
        intersection = len(chars1 & chars2)
        union = len(chars1 | chars2)
        
        return intersection / union if union > 0 else 0.0
    
    def _estimate_tokens(self, text: str) -> int:
        """估算文本的 token 数量"""
        from token_utils import estimate_tokens
        return estimate_tokens(text)
    
    def _estimate_tokens_cached(self, text: str, metadata: dict = None) -> int:
        """
        估算文本的 token 数量（带缓存）
        
        设计原则：
        - 优先使用 metadata 中缓存的 token_count
        - 未缓存时计算并返回（不自动缓存，由调用方决定）
        
        Args:
            text: 文本内容
            metadata: 记忆元数据（可选）
        
        Returns:
            token 数量
        """
        from memory_tags import MemoryTagHelper
        
        cached = MemoryTagHelper.get_token_count(metadata)
        if cached is not None:
            return cached
        
        return self._estimate_tokens(text)
    
    def _calculate_available_tokens(
        self, 
        memories: List[str], 
        user_input: str
    ) -> Tuple[int, int, int]:
        """
        动态计算可用token分配
        
        分配策略：
        - 基础比例：L1:L2:L3 = 2:3:5（根据记忆来源动态调整）
        - 时间相关查询：增加L1权重
        - 长查询：增加记忆权重
        - 短查询：减少记忆权重
        """
        user_tokens = self._estimate_tokens(user_input)
        memory_tokens_raw = sum(self._estimate_tokens(m) + 10 for m in memories)
        
        query_type = self._detect_query_type(user_input)
        
        memory_weight = {
            "time_related": 0.7,
            "factual": 0.6,
            "casual": 0.3,
            "complex": 0.5,
            "default": 0.4
        }.get(query_type, 0.4)
        
        available_for_memory = int(
            (self.max_context - self.system_reserve - self.max_output) * memory_weight
        )
        
        available_for_memory = max(available_for_memory, 500)
        
        total_needed = self.system_reserve + memory_tokens_raw + user_tokens + self.max_output
        
        if total_needed <= self.max_context:
            return memory_tokens_raw, user_tokens, False
        
        if user_tokens > 200:
            user_limit = min(user_tokens, 300)
            memory_limit = available_for_memory - user_limit
        else:
            user_limit = user_tokens
            memory_limit = available_for_memory
        
        memory_limit = max(memory_limit, 300)
        
        return memory_limit, user_limit, False
    
    def _detect_query_type(self, query: str) -> str:
        """
        检测查询类型，用于动态调整Token分配
        
        Returns:
            "time_related" | "factual" | "casual" | "complex" | "default"
        """

        query_lower = query.lower()
        
        time_patterns = r"昨天|今天|上周|之前|刚才|什么时候|何时|when|yesterday|today|before"
        if re.search(time_patterns, query_lower):
            return "time_related"
        
        factual_patterns = r"是什么|什么是|怎么|如何|为什么|谁|哪|what|how|why|who|which"
        if re.search(factual_patterns, query_lower):
            return "factual"
        
        casual_patterns = r"你好|嗨|哈喽|谢谢|再见|ok|好的|嗯|hello|hi|thanks|bye"
        if re.search(casual_patterns, query_lower):
            return "casual"
        
        if len(query) > 100 or query.count("?") > 1 or query.count("？") > 1:
            return "complex"
        
        return "default"
    
    def _compress_memories(
        self, 
        memories: List[str], 
        max_tokens: int
    ) -> Tuple[str, int]:
        """
        压缩记忆到指定token数
        
        策略：
        1. 优先保留包含专有名词和数值的记忆
        2. 按相关性排序后选择
        3. 截断时保留关键信息
        """
        if not memories or max_tokens <= 0:
            return "", 0
        
        scored_memories = []
        for m in memories:
            score = self._calculate_memory_importance(m)
            scored_memories.append((m, score))
        
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        selected_memories = []
        total_tokens = 0
        
        for m, score in scored_memories:
            m_tokens = self._estimate_tokens(m) + 10
            if total_tokens + m_tokens <= max_tokens:
                selected_memories.append(m)
                total_tokens += m_tokens
            else:
                remaining = max_tokens - total_tokens
                if remaining > 50:
                    truncated = self._smart_truncate(m, remaining)
                    if self._estimate_tokens(truncated) <= remaining:
                        selected_memories.append(truncated)
                        total_tokens += self._estimate_tokens(truncated) + 10
                break
        
        if not selected_memories and memories:
            first_memory = memories[0]
            first_tokens = self._estimate_tokens(first_memory)
            if first_tokens > max_tokens:
                truncated = self._smart_truncate(first_memory, max_tokens)
                selected_memories = [truncated]
            else:
                selected_memories = [first_memory]
        
        context_text = "\n".join([f"• {m}" for m in selected_memories])
        used_tokens = self._estimate_tokens(context_text) if context_text else 0
        
        return context_text, used_tokens
    
    def _calculate_memory_importance(self, text: str) -> float:
        """
        计算记忆的重要性分数
        
        优先级：
        1. 包含数值（价格、日期、数量）
        2. 包含专有名词（人名、地名、产品名）
        3. 包含关键动词（决定、选择、购买）
        """

        score = 0.0
        
        number_patterns = r'\d+(?:\.\d+)?(?:元|美元|块|万|千|百|亿|%|％|度|kg|kg|ml|GB|MB|TB)?'
        numbers = re.findall(number_patterns, text)
        score += len(numbers) * MemoryConstants.NUMBER_WEIGHT_SCORE
        
        proper_noun_patterns = r'[A-Z][a-z]+|[\u4e00-\u9fa5]{2,4}(?:公司|项目|产品|版本|系统|模块)'
        proper_nouns = re.findall(proper_noun_patterns, text)
        score += len(proper_nouns) * 1.5
        
        key_verbs = ['决定', '选择', '购买', '设置', '配置', '修改', '删除', '创建', '安装', '更新']
        for verb in key_verbs:
            if verb in text:
                score += 1.0
        
        date_patterns = r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?|\d{1,2}月\d{1,2}日|昨天|今天|明天|上周|下周'
        dates = re.findall(date_patterns, text)
        score += len(dates) * 1.5
        
        return score
    
    def _smart_truncate(self, text: str, max_tokens: int) -> str:
        """
        智能截断文本，保留关键信息
        
        策略：
        1. 提取所有数值和专有名词
        2. 在截断边界处保留完整句子
        3. 追加关键信息摘要
        """

        max_chars = int(max_tokens / 2) - 20
        if max_chars <= 0:
            return text[:50] + "..."
        
        key_info = []
        
        number_patterns = r'\d+(?:\.\d+)?(?:元|美元|块|万|千|百|亿|%|％|度|kg|kg|ml|GB|MB|TB)?'
        numbers = re.findall(number_patterns, text)
        if numbers:
            key_info.extend(numbers[:3])
        
        date_patterns = r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?|\d{1,2}月\d{1,2}日'
        dates = re.findall(date_patterns, text)
        if dates:
            key_info.extend(dates[:2])
        
        truncated = text[:max_chars]
        
        last_period = max(
            truncated.rfind('。'),
            truncated.rfind('.'),
            truncated.rfind('！'),
            truncated.rfind('？')
        )
        
        if last_period > max_chars * 0.5:
            truncated = truncated[:last_period + 1]
        
        if key_info:
            key_str = " | 关键信息: " + ", ".join(key_info[:5])
            truncated = truncated + key_str
        
        return truncated + "..."
    
    def _truncate_user_input(self, user_input: str, max_tokens: int) -> str:
        """截断用户输入到指定token数"""
        if max_tokens <= 0:
            return ""
        
        input_tokens = self._estimate_tokens(user_input)
        if input_tokens <= max_tokens:
            return user_input
        
        max_chars = int(max_tokens / 2)
        return user_input[:max_chars] + "...(原文已截断)"
    
    def _control_context_length(
        self, 
        memories: List[Dict], 
        current_query: str,
        mode: str = "local",
        conversation_history: List[Dict] = None
    ) -> Tuple[str, str, bool]:
        """
        控制上下文长度（优先级聚合版）
        
        优先级分桶策略：
        - Bucket A (System/Fixed): 系统提示词（由调用方处理）
        - Bucket B (Recent L1): 最近 3-5 轮对话
        - Bucket C (Important L1): 标记为重要的记忆
        - Bucket D (Retrieved L2/L3): 检索到的历史记忆
        
        Token 分配：
        - Bucket B: 30% 预算
        - Bucket C: 20% 预算（上限）
        - Bucket D: 50% 预算
        
        Args:
            memories: 检索到的记忆列表
            current_query: 当前用户输入
            mode: "local" (本地压缩) | "cloud_only" (仅云端) | "hybrid" (混合)
            conversation_history: L1 对话历史
        
        返回：
        - memory_context: 记忆上下文
        - processed_user_input: 处理后的用户输入
        - skip_user_input: 是否跳过用户原文
        """
        if mode == "cloud_only":
            memory_texts = [m.get("text", "") for m in memories if m.get("text")]
            memory_context = "\n".join([f"• {m}" for m in memory_texts])
            return memory_context, current_query, False
        
        max_memory_tokens, max_user_tokens, skip_user = self._calculate_available_tokens(
            [], current_query
        )
        
        if mode == "local":
            hard_limit = self.max_context - self.system_reserve - self.max_output - 200
            user_tokens = self._estimate_tokens(current_query)
            max_memory_tokens = min(max_memory_tokens, hard_limit - user_tokens)
            max_memory_tokens = max(max_memory_tokens, 200)
        
        buckets = self._build_priority_buckets(memories, conversation_history)
        
        memory_context = self._fill_by_priority(buckets, max_memory_tokens)
        
        processed_user_input = self._truncate_user_input(current_query, max_user_tokens)
        
        return memory_context, processed_user_input, skip_user
    
    def _build_priority_buckets(
        self, 
        memories: List[Dict], 
        conversation_history: List[Dict] = None
    ) -> Dict[str, List[str]]:
        """
        构建优先级分桶
        
        Args:
            memories: 检索到的记忆列表
            conversation_history: L1 对话历史
        
        Returns:
            {
                PRIORITY_BUCKET_B: 最近对话文本列表,
                PRIORITY_BUCKET_C: 重要记忆文本列表,
                PRIORITY_BUCKET_D: 检索记忆文本列表
            }
        """
        buckets = {
            PRIORITY_BUCKET_B: [],
            PRIORITY_BUCKET_C: [],
            PRIORITY_BUCKET_D: []
        }
        
        if conversation_history:
            recent_turns = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
            for conv in recent_turns:
                user_text = conv.get("user", "")
                assistant_text = conv.get("assistant", "")
                if user_text:
                    buckets[PRIORITY_BUCKET_B].append(f"用户: {user_text}")
                if assistant_text:
                    buckets[PRIORITY_BUCKET_B].append(f"助理: {assistant_text}")
        
        important_memories = []
        if self.memory and hasattr(self.memory, 'sqlite') and self.memory.sqlite:
            try:
                important_records = self.memory.sqlite.get_important_memories(limit=10)
                for record in important_records:
                    text = record.compressed_text or record.text
                    if text:
                        important_memories.append(text)
            except Exception:
                pass
        
        for m in memories:
            if m.get("is_important") or m.get("metadata", {}).get("tags", {}).get("important"):
                text = m.get("text", "")
                if text and text not in important_memories:
                    important_memories.append(text)
        
        buckets[PRIORITY_BUCKET_C] = important_memories
        
        retrieved_texts = []
        seen_texts = set(buckets[PRIORITY_BUCKET_B] + buckets[PRIORITY_BUCKET_C])
        
        for m in memories:
            text = m.get("text", "")
            if text and text not in seen_texts:
                retrieved_texts.append(text)
                seen_texts.add(text)
        
        buckets[PRIORITY_BUCKET_D] = retrieved_texts
        
        return buckets
    
    def _fill_by_priority(
        self, 
        buckets: Dict[str, List[str]], 
        max_tokens: int
    ) -> str:
        """
        按优先级填充 Token 预算（带动态预算流转）
        
        分配策略：
        - Bucket B (Recent L1): 初始 30% 预算，未用完流转给 D
        - Bucket C (Important): 初始 20% 预算，未用完流转给 D
        - Bucket D (Retrieved): 至少 50% 预算 + B/C 结余
        
        动态流转：
        - 若 B/C 未用完预算，剩余部分自动滚入 D
        - 确保不会因 B/C 内容少而浪费 Token 预算
        
        Args:
            buckets: 优先级分桶
            max_tokens: 最大 Token 数
        
        Returns:
            格式化的记忆上下文
        """
        MIN_BUCKET_D_RATIO = 0.2
        MIN_BUCKET_D_TOKENS = 200
        
        bucket_b_budget = int(max_tokens * 0.3)
        bucket_c_budget = int(max_tokens * 0.2)
        
        bucket_d_budget = max(
            max_tokens - bucket_b_budget - bucket_c_budget,
            int(max_tokens * MIN_BUCKET_D_RATIO),
            MIN_BUCKET_D_TOKENS
        )
        
        if bucket_d_budget > max_tokens - bucket_b_budget - bucket_c_budget:
            shortage = bucket_d_budget - (max_tokens - bucket_b_budget - bucket_c_budget)
            bucket_b_budget = max(int(bucket_b_budget - shortage * 0.6), 0)
            bucket_c_budget = max(int(bucket_c_budget - shortage * 0.4), 0)
        
        result_parts = []
        total_used = 0
        
        bucket_b_texts = buckets.get(PRIORITY_BUCKET_B, [])
        if bucket_b_texts:
            bucket_b_content, b_used = self._compress_memories(bucket_b_texts, bucket_b_budget)
            if bucket_b_content:
                result_parts.append("【最近对话】\n" + bucket_b_content)
                total_used += b_used
            b_rollover = bucket_b_budget - b_used
        else:
            b_rollover = bucket_b_budget
        
        bucket_c_texts = buckets.get(PRIORITY_BUCKET_C, [])
        if bucket_c_texts:
            bucket_c_content, c_used = self._compress_memories(bucket_c_texts, bucket_c_budget)
            if bucket_c_content:
                result_parts.append("【重要记忆】\n" + bucket_c_content)
                total_used += c_used
            c_rollover = bucket_c_budget - c_used
        else:
            c_rollover = bucket_c_budget
        
        bucket_d_budget = bucket_d_budget + b_rollover + c_rollover
        bucket_d_budget = min(bucket_d_budget, max_tokens - total_used)
        
        bucket_d_texts = buckets.get(PRIORITY_BUCKET_D, [])
        if bucket_d_texts:
            bucket_d_content, d_used = self._compress_memories(bucket_d_texts, bucket_d_budget)
            if bucket_d_content:
                result_parts.append("【历史参考】\n" + bucket_d_content)
        
        return "\n\n".join(result_parts)
    
    def truncate_for_local_llm(
        self, 
        system_prompt: str, 
        memory_context: str, 
        user_input: str,
        max_tokens: int = None
    ) -> Tuple[str, str]:
        """
        为本地LLM截断内容以适应token限制
        
        将截断逻辑从UI层下沉到此处，确保：
        1. 按段落/记忆条目颗粒度截断，而非硬切
        2. 保留完整句子边界
        3. 优先保留关键信息
        
        Args:
            system_prompt: 系统提示词
            memory_context: 记忆上下文
            user_input: 用户输入
            max_tokens: 最大token数（默认从config读取）
        
        Returns:
            (截断后的记忆上下文, 截断后的用户输入)
        """
        if max_tokens is None:
            max_tokens = self.max_context
        
        system_tokens = self._estimate_tokens(system_prompt)
        user_tokens = self._estimate_tokens(user_input)
        format_overhead = 50
        
        available_for_memory = max_tokens - system_tokens - user_tokens - self.max_output - format_overhead
        available_for_memory = max(available_for_memory, 200)
        
        memory_tokens = self._estimate_tokens(memory_context)
        
        if memory_tokens <= available_for_memory:
            return memory_context, user_input
        
        memory_items = memory_context.split('\n')
        
        selected_items = []
        current_tokens = 0
        
        for item in memory_items:
            if not item.strip():
                continue
            item_tokens = self._estimate_tokens(item) + 5
            if current_tokens + item_tokens <= available_for_memory:
                selected_items.append(item)
                current_tokens += item_tokens
            else:
                break
        
        if selected_items:
            truncated_memory = '\n'.join(selected_items)
            truncated_memory += "\n...[内容已截断以适应token限制]"
        else:
            truncated_memory = self._smart_truncate(memory_context, available_for_memory)
        
        return truncated_memory, user_input
