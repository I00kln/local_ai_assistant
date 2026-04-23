"""
提示词管理模块

整合四种核心提示词：
1. 记忆提取 (L1→L2): 从原始对话提取原子记忆
2. 语义压缩 (L2→L3): 聚类压缩相关记忆
3. 智能重排: LLM相关性重排
4. 冲突检测: 检测新旧记忆事实冲突

注意：本模块的提示词用于后台异步处理，与以下场景区分：
- config.system_prompt: 用于 chat_window 实时压缩检索记忆（生成回复前）
- config.local_assistant_prompt: 用于本地LLM直接回答用户问题
"""

from typing import List, Dict, Any


class MemoryPrompts:
    """记忆相关提示词集合"""
    
    EXTRACTION_SYSTEM = """你是一个记忆提取专家。你的任务是将用户与 AI 的对话片段压缩为一条"原子记忆"。

提取准则：
1. 去噪：剔除"你好"、"嗯"、"谢谢"等无信息量内容。
2. 实体归一化：确保主体（User/AI）明确，不要使用"他/她"等代词。
3. 意图捕获：保留用户的偏好、历史事实、以及未完成的任务。
4. 格式：输出为 [核心事实] + (上下文补充)。

示例：
输入：用户说他家在上海，但他下周要去北京出差，问我天气。
输出：用户居住地为上海；下周行程：前往北京出差。"""
    
    EXTRACTION_USER_TEMPLATE = """请提取以下对话的核心记忆：

用户：{user_input}
助理：{assistant_response}

提取结果："""
    
    COMPRESSION_SYSTEM = """你是一个记忆归档器。请将以下多条语义相关的记忆片段合并为一条长效背景知识。

要求：
- 时间敏感：如果记忆之间存在时间冲突，以最近的为准。
- 逻辑聚合：例如"用户喜欢苹果"和"用户不喜欢吃梨"应合并为"用户水果偏好：喜爱苹果，排斥梨"。
- 保留权重：识别出用户反复强调的信息，并在描述中体现（如"极度重视..."）。
- 去重：合并重复信息，保留唯一事实。

输出格式：直接输出合并后的记忆，不要添加解释。"""
    
    COMPRESSION_USER_TEMPLATE = """待合并条目：
{memory_list}

合并结果："""
    
    RERANK_SYSTEM = """你是检索重排专家。给定用户的当前问题和检索回来的 K 条候选记忆，请按相关性评分。

评分公式参考：
Score = α × 语义相似度 + β × 事实一致性

任务：
判断候选记忆是否能直接辅助回答当前问题。如果记忆虽然语义接近但逻辑无关（例如问"苹果手机"回"苹果水果"），请将其标记为 0 分。

输出格式：
返回 JSON 数组，每个元素包含 {{"id": "记忆ID", "score": 分数, "reason": "简短理由"}}"""
    
    RERANK_USER_TEMPLATE = """当前问题：{current_query}

候选记忆：
{candidate_memories}

请评分并排序："""
    
    CONFLICT_SYSTEM = """你是记忆一致性审查员。请对比新记忆与历史记忆。

检测逻辑：
1. 冲突定义：同一主体的同一属性发生了不可并存的变化（如：单身 vs 已婚）。
2. 处理建议：
   - Update：新信息更可靠，建议覆盖。
   - Correct：旧信息是误导，建议删除。
   - Coexist：两者是状态演进，建议均保留并打上时间戳。

输出格式：
返回 JSON：{{"has_conflict": bool, "action": "Update|Correct|Coexist|None", "reason": "说明"}}"""
    
    CONFLICT_USER_TEMPLATE = """新记忆：{new_record}

历史记录：{old_record}

请判断是否存在冲突："""


class PromptManager:
    """
    提示词管理器
    
    职责：
    1. 管理所有提示词模板
    2. 提供提示词渲染接口
    3. 支持提示词版本管理
    """
    
    _instance = None
    _lock = __import__('threading').Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._prompts = MemoryPrompts()
        self._version = "1.0.0"
    
    def get_extraction_prompt(self, user_input: str, assistant_response: str) -> List[Dict[str, str]]:
        """
        获取记忆提取提示词
        
        Args:
            user_input: 用户输入
            assistant_response: 助理响应
        
        Returns:
            消息列表 [{"role": "system", "content": ...}, ...]
        """
        return [
            {"role": "system", "content": self._prompts.EXTRACTION_SYSTEM},
            {"role": "user", "content": self._prompts.EXTRACTION_USER_TEMPLATE.format(
                user_input=user_input,
                assistant_response=assistant_response
            )}
        ]
    
    def get_compression_prompt(self, memory_list: List[str]) -> List[Dict[str, str]]:
        """
        获取语义压缩提示词
        
        Args:
            memory_list: 待合并的记忆列表
        
        Returns:
            消息列表
        """
        formatted_list = "\n".join([f"{i+1}. {mem}" for i, mem in enumerate(memory_list)])
        
        return [
            {"role": "system", "content": self._prompts.COMPRESSION_SYSTEM},
            {"role": "user", "content": self._prompts.COMPRESSION_USER_TEMPLATE.format(
                memory_list=formatted_list
            )}
        ]
    
    def get_rerank_prompt(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        获取智能重排提示词
        
        Args:
            query: 当前查询
            candidates: 候选记忆列表 [{"id": ..., "text": ..., "score": ...}]
        
        Returns:
            消息列表
        """
        formatted_candidates = "\n".join([
            f"[ID:{c['id']}] (相似度:{c.get('score', 0):.2f}) {c['text'][:100]}..."
            for c in candidates
        ])
        
        return [
            {"role": "system", "content": self._prompts.RERANK_SYSTEM},
            {"role": "user", "content": self._prompts.RERANK_USER_TEMPLATE.format(
                current_query=query,
                candidate_memories=formatted_candidates
            )}
        ]
    
    def get_conflict_prompt(self, new_record: str, old_record: str) -> List[Dict[str, str]]:
        """
        获取冲突检测提示词
        
        Args:
            new_record: 新记忆
            old_record: 旧记忆
        
        Returns:
            消息列表
        """
        return [
            {"role": "system", "content": self._prompts.CONFLICT_SYSTEM},
            {"role": "user", "content": self._prompts.CONFLICT_USER_TEMPLATE.format(
                new_record=new_record,
                old_record=old_record
            )}
        ]
    
    @property
    def version(self) -> str:
        return self._version


def get_prompt_manager() -> PromptManager:
    """获取提示词管理器单例"""
    return PromptManager()
