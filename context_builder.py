# context_builder.py
# 上下文构建器 - 使用三层记忆管理器
from typing import List, Dict, Tuple
from config import config
from memory_tags import MemoryConstants
from logger import get_logger
import re

SHORT_QUERY_STOPWORDS = {
    "好", "好的", "嗯", "哦", "是", "对", "行", "可以", "继续",
    "明白", "知道了", "谢谢", "感谢", "不客气", "再见", "拜拜",
    "ok", "yes", "no", "hi", "hello", "hey"
}

class ContextBuilder:
    """上下文构建器：多级记忆检索 + 动态长度控制"""
    
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
        
        has_memories = len(retrieved) > 0
        
        memory_context, processed_user_input, _ = self._control_context_length(
            retrieved, user_input, mode
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
    
    def _multi_level_retrieve(
        self, 
        query: str, 
        conversation_history: List[Dict] = None,
        thresholds: Dict[str, float] = None,
        max_retrieve: int = None
    ) -> List[Dict]:
        """
        多级记忆检索
        
        L1: 内存中的最近对话历史
        L2: 向量库（ChromaDB）
        L3: 数据库（SQLite）
        
        检索策略：
        1. 优先搜索L1，如果结果足够则返回
        2. L1不足则搜索L2
        3. L2结果不足则降低置信度重试
        4. L2仍不足则搜索L3（SQLite全文搜索）
        5. 都没有则返回空列表
        """
        if not self._is_valid_query(query):
            return []
        
        if thresholds is None:
            thresholds = {"l1": self.l2_default_threshold, "l2": self.l2_default_threshold, "l3": self.l2_lower_threshold}
        
        if max_retrieve is None:
            max_retrieve = config.max_retrieve_results
        
        results = []
        
        l1_results = self._search_l1(query, conversation_history, thresholds["l1"])
        if l1_results:
            results.extend(l1_results)
        
        l1_count = len(results)
        
        if l1_count < self.l1_min_results:
            l2_results = self._search_l2(query, thresholds["l2"])
            
            if len(l2_results) < self.l1_min_results - l1_count:
                l2_lower = self._search_l2(query, thresholds["l3"])
                l2_results = self._merge_results(l2_results, l2_lower)
            
            results = self._merge_results(results, l2_results)
        
        if len(results) < self.l1_min_results:
            l3_results = self._search_l3(query, thresholds["l3"])
            results = self._merge_results(results, l3_results)
        
        return results[:max_retrieve]
    
    def _search_l1(self, query: str, conversation_history: List[Dict] = None, threshold: float = 0.9) -> List[Dict]:
        """
        L1: 搜索内存中的最近对话历史
        
        简单的关键词匹配，返回最近的对话
        """
        if not conversation_history:
            return []
        
        results = []
        query_lower = query.lower()
        query_keywords = set(query_lower.split())
        
        for conv in reversed(conversation_history[-20:]):
            user_text = conv.get("user", "").lower()
            assistant_text = conv.get("assistant", "").lower()
            
            score = 0
            matched_keywords = 0
            for kw in query_keywords:
                if kw in user_text:
                    score += 2
                    matched_keywords += 1
                if kw in assistant_text:
                    score += 1
                    matched_keywords += 1
            
            if score > 0:
                combined_text = f"用户: {conv.get('user', '')}\n助理: {conv.get('assistant', '')}"
                max_possible_score = len(query_keywords) * 3
                similarity = score / max_possible_score if max_possible_score > 0 else 0
                
                if similarity >= threshold:
                    results.append({
                        "text": combined_text,
                        "similarity": similarity,
                        "source": "L1",
                        "timestamp": conv.get("timestamp", "")
                    })
        
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:5]
    
    def _search_l2(self, query: str, threshold: float) -> List[Dict]:
        """
        L2: 搜索向量库（ChromaDB）
        """
        try:
            results = self.memory.search(query, top_k=10, threshold=threshold, include_l1=False)
            formatted_results = []
            for r in results:
                formatted_results.append({
                    "text": r.text,
                    "similarity": r.similarity,
                    "source": r.source,
                    "weight": r.weight,
                    "metadata": r.metadata
                })
            return formatted_results
        except Exception as e:
            self._log.warning("L2_SEARCH_FALLBACK",
                             error=str(e),
                             query_preview=query[:50] if query else "",
                             fallback="L3_FTS5")
            return []
    
    def _search_l3(self, query: str, threshold: float = 0.8) -> List[Dict]:
        """
        L3: 搜索SQLite数据库（全文搜索）
        
        使用FTS5进行全文搜索，作为向量检索的补充
        """
        try:
            if not self.memory.sqlite:
                return []
            
            results = self.memory.sqlite.search(query, limit=10)
            formatted_results = []
            
            query_keywords = set(query.lower().split())
            
            for record in results:
                text = record.compressed_text or record.text
                text_lower = text.lower()
                
                matched_keywords = sum(1 for kw in query_keywords if kw in text_lower)
                keyword_ratio = matched_keywords / len(query_keywords) if query_keywords else 0
                similarity = keyword_ratio * 0.8
                
                if similarity >= threshold:
                    formatted_results.append({
                        "text": text,
                        "similarity": similarity,
                        "source": "L3",
                        "weight": record.weight,
                        "timestamp": record.created_time,
                        "metadata": record.metadata
                    })
            
            return formatted_results
        except Exception as e:
            self._log.warning("L3_SEARCH_FAILED",
                             error=str(e),
                             query_preview=query[:50] if query else "")
            return []
    
    def _merge_results(self, existing: List[Dict], new_results: List[Dict]) -> List[Dict]:
        """
        合并检索结果，去重 + 多样性控制 + 来源权重排序
        
        策略：
        1. 合并所有结果
        2. 应用来源权重
        3. 多样性过滤（文本重叠度 > 0.8 视为重复）
        4. 按综合得分排序
        """
        seen = set()
        merged = []
        
        all_items = existing + new_results
        
        for item in all_items:
            text = item.get("text", "")
            if not text or text in seen:
                continue
            
            seen.add(text)
            
            similarity = item.get("similarity", 0.0)
            weight = item.get("weight", 1.0)
            source = item.get("source", "L3")
            
            source_weights = {
                "L1": config.source_weight_l1,
                "L2": config.source_weight_l2,
                "L3": config.source_weight_l3,
                "L3_LOW_QUALITY": config.source_weight_l3 * 0.5,
            }
            source_weight = source_weights.get(source, config.source_weight_l3)
            
            item["final_score"] = similarity * weight * source_weight
            merged.append(item)
        
        merged.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        
        filtered = []
        for item in merged:
            is_duplicate = False
            item_text = item.get("text", "")
            for selected in filtered:
                selected_text = selected.get("text", "")
                overlap = self._calculate_text_overlap(item_text, selected_text)
                if overlap >= config.diversity_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
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
        mode: str = "local"
    ) -> Tuple[str, str, bool]:
        """
        控制上下文长度
        
        Args:
            memories: 检索到的记忆列表
            current_query: 当前用户输入
            mode: "local" (本地压缩) | "cloud_only" (仅云端) | "hybrid" (混合)
        
        返回：
        - memory_context: 记忆上下文
        - processed_user_input: 处理后的用户输入
        - skip_user_input: 是否跳过用户原文
        
        Token限制：
        - 本地模式：总token不超过 max_context (默认8192)
        - 云端模式：无硬性限制
        """
        memory_texts = [m.get("text", "") for m in memories if m.get("text")]
        
        if mode == "cloud_only":
            memory_context = "\n".join([f"• {m}" for m in memory_texts])
            return memory_context, current_query, False
        
        max_memory_tokens, max_user_tokens, skip_user = self._calculate_available_tokens(
            memory_texts, current_query
        )
        
        if mode == "local":
            hard_limit = self.max_context - self.system_reserve - self.max_output - 200
            user_tokens = self._estimate_tokens(current_query)
            max_memory_tokens = min(max_memory_tokens, hard_limit - user_tokens)
            max_memory_tokens = max(max_memory_tokens, 200)
        
        memory_context, memory_used = self._compress_memories(
            memory_texts, max_memory_tokens
        )
        
        processed_user_input = self._truncate_user_input(current_query, max_user_tokens)
        
        return memory_context, processed_user_input, skip_user
    
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
