# memory_manager.py
# 三层记忆管理器 - 统一管理L1/L2/L3记忆层
import os
import threading
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from config import config
from vector_store import VectorStore, get_vector_store
from sqlite_store import SQLiteStore, MemoryRecord, get_sqlite_store


@dataclass
class MemorySearchResult:
    """记忆搜索结果"""
    text: str
    source: str  # L1, L2, L3
    similarity: float
    weight: float
    metadata: Dict[str, Any]


class MemoryManager:
    """
    三层记忆管理器
    
    L1: 内存层 - 当前对话历史（由chat_window管理）
    L2: 向量库层 - ChromaDB热数据
    L3: 数据库层 - SQLite冷数据
    
    功能：
    - 统一搜索接口（L1→L2→L3）
    - 自动回填（L3命中→L2）
    - 定期压缩（L2→L3）
    - 权重管理与遗忘
    """
    
    def __init__(self, vector_store: VectorStore = None, sqlite_store: SQLiteStore = None):
        self.vector_store = vector_store or get_vector_store()  # L2 向量库
        self.sqlite = sqlite_store or get_sqlite_store()  # L3 数据库
        
        self.lock = threading.RLock()
        
        # L1 内存层（当前对话历史，由外部管理）
        self.conversation_history: List[Dict] = []
        self.max_l1_size = 25  # L1最大保留对话数
        
        # 压缩器
        self.compressor = None
        self._init_compressor()
        
        # 统计信息
        self.stats = {
            "l1_hits": 0,
            "l2_hits": 0,
            "l3_hits": 0,
            "l3_backfills": 0,
            "compressions": 0,
            "forgotten": 0
        }
    
    def _init_compressor(self):
        """初始化记忆压缩器"""
        if not config.compression_enabled:
            return
        
        # 延迟初始化，避免循环导入
        pass
    
    def _get_compressor(self):
        """获取压缩器实例"""
        if self.compressor is None and config.compression_enabled:
            try:
                from llm_client import LlamaClient
                self.compressor = LlamaClient()
            except Exception as e:
                print(f"初始化压缩器失败: {e}")
        return self.compressor
    
    def add_conversation(self, user_input: str, assistant_response: str, metadata: Dict = None):
        """
        添加对话到L1内存层
        
        Args:
            user_input: 用户输入
            assistant_response: 助理回复
            metadata: 元数据
        
        注意：溢出处理在锁外执行，避免阻塞其他线程
        """
        overflow = None
        
        with self.lock:
            self.conversation_history.append({
                "user": user_input,
                "assistant": assistant_response,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            })
            
            # 保持L1大小限制
            if len(self.conversation_history) > self.max_l1_size:
                overflow = self.conversation_history[:-self.max_l1_size]
                self.conversation_history = self.conversation_history[-self.max_l1_size:]
        
        # 溢出处理在锁外执行（涉及向量存储，可能耗时）
        if overflow:
            self._overflow_to_l2(overflow)
    
    def _overflow_to_l2(self, conversations: List[Dict]):
        """将溢出的L1对话存入L2向量库"""
        if not conversations:
            return
        
        texts = []
        metadata_list = []
        
        for conv in conversations:
            text = f"用户: {conv['user']}\n助理: {conv['assistant']}"
            texts.append(text)
            metadata_list.append({
                "timestamp": conv["timestamp"],
                "type": "conversation",
                "source": conv.get("metadata", {}).get("source", "local")
            })
        
        self.vector_store.add(texts, metadata_list)
        print(f"L1溢出: {len(texts)} 条对话存入L2向量库")
    
    def search(self, query: str, top_k: int = None, include_l3: bool = True, threshold: float = None, include_l1: bool = True) -> List[MemorySearchResult]:
        """
        统一搜索接口 - L1→L2→L3
        
        Args:
            query: 搜索查询
            top_k: 返回结果数量
            include_l3: 是否搜索L3数据库
            threshold: 相似度阈值，低于此值的结果将被过滤
            include_l1: 是否搜索L1内存层（默认True，context_builder单独搜索L1时会设为False）
        
        Returns:
            搜索结果列表（已去重、已应用时间衰减、已过滤阈值）
        """
        if top_k is None:
            top_k = config.max_retrieve_results
        
        if threshold is None:
            threshold = 0.0
        
        results = []
        seen_texts = set()
        
        time_context = self._detect_time_context(query)
        
        if include_l1:
            l1_results = self._search_l1(query, top_k, threshold)
            for r in l1_results:
                if r.text not in seen_texts:
                    seen_texts.add(r.text)
                    results.append(r)
            self.stats["l1_hits"] += len(l1_results)
        
        if len(results) < top_k:
            remaining = top_k - len(results)
            l2_results = self._search_l2(query, remaining)
            for r in l2_results:
                if r.text not in seen_texts:
                    seen_texts.add(r.text)
                    r.similarity = self._apply_time_decay(r, time_context)
                    results.append(r)
        self.stats["l2_hits"] += len(l2_results)
        
        if include_l3 and len(results) < top_k and config.sqlite_enabled:
            remaining = top_k - len(results)
            l3_results = self._search_l3(query, remaining)
            for r in l3_results:
                if r.text not in seen_texts:
                    seen_texts.add(r.text)
                    r.similarity = self._apply_time_decay(r, time_context)
                    results.append(r)
            
            for result in l3_results:
                self._backfill_to_l2(result)
        
        results.sort(key=lambda x: x.similarity * x.weight, reverse=True)
        
        filtered_results = [r for r in results if r.similarity >= threshold]
        
        return filtered_results[:top_k]
    
    def _detect_time_context(self, query: str) -> Dict[str, Any]:
        """
        检测查询中的时间上下文
        
        Returns:
            {
                "has_time_ref": bool,
                "time_type": str,  # "recent", "yesterday", "week", "month"
                "decay_factor": float
            }
        """
        import re
        
        query_lower = query.lower()
        
        time_patterns = {
            "today": (r"今天|今日|today", "recent", 1.5),
            "yesterday": (r"昨天|昨日|yesterday", "yesterday", 2.0),
            "week": (r"上周|这周|本周|last week|this week", "week", 1.3),
            "recent": (r"最近|刚才|刚才|recently|just now", "recent", 1.4),
            "before": (r"之前|以前|before|earlier", "older", 0.8),
        }
        
        for time_type, (pattern, _, decay) in time_patterns.items():
            if re.search(pattern, query_lower):
                return {
                    "has_time_ref": True,
                    "time_type": time_type,
                    "decay_factor": decay
                }
        
        return {
            "has_time_ref": False,
            "time_type": "any",
            "decay_factor": 1.0
        }
    
    def _apply_time_decay(self, result: MemorySearchResult, time_context: Dict) -> float:
        """
        应用时间衰减因子
        
        根据记忆的时间戳和查询的时间上下文调整相似度
        """
        if not time_context["has_time_ref"]:
            return result.similarity
        
        timestamp_str = result.metadata.get("timestamp", "")
        if not timestamp_str:
            return result.similarity
        
        try:
            from datetime import datetime
            
            memory_time = datetime.fromisoformat(timestamp_str)
            now = datetime.now()
            age_hours = (now - memory_time).total_seconds() / 3600
            
            decay = 1.0
            time_type = time_context["time_type"]
            
            if time_type == "recent":
                # 最近：24小时内的记忆加权
                if age_hours < 24:
                    decay = 1.5
                elif age_hours < 48:
                    decay = 1.2
                else:
                    decay = 0.7
            
            elif time_type == "yesterday":
                # 昨天：24-48小时内的记忆大幅加权
                if 24 <= age_hours < 48:
                    decay = 2.0
                elif age_hours < 24:
                    decay = 1.0  # 今天的也给一定权重
                else:
                    decay = 0.5
            
            elif time_type == "week":
                # 本周：7天内的记忆加权
                if age_hours < 168:  # 7天
                    decay = 1.3
                else:
                    decay = 0.6
            
            return result.similarity * decay
            
        except Exception:
            return result.similarity
    
    def _search_l1(self, query: str, top_k: int, threshold: float = 0.0) -> List[MemorySearchResult]:
        """搜索L1内存层"""
        results = []
        query_lower = query.lower()
        query_keywords = set(query_lower.split())
        
        for conv in reversed(self.conversation_history):
            user_text = conv.get("user", "").lower()
            assistant_text = conv.get("assistant", "").lower()
            
            score = 0
            for kw in query_keywords:
                if kw in user_text:
                    score += 2
                if kw in assistant_text:
                    score += 1
            
            if score > 0:
                combined_text = f"用户: {conv['user']}\n助理: {conv['assistant']}"
                max_possible_score = len(query_keywords) * 3
                similarity = score / max_possible_score if max_possible_score > 0 else 0
                
                if similarity >= threshold:
                    results.append(MemorySearchResult(
                        text=combined_text,
                        source="L1",
                        similarity=similarity,
                        weight=1.0,
                        metadata={"timestamp": conv.get("timestamp", "")}
                    ))
        
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:top_k]
    
    def _search_l2(self, query: str, top_k: int) -> List[MemorySearchResult]:
        """搜索L2向量库"""
        results = []
        
        try:
            l2_results = self.vector_store.search(query, n_results=top_k)
            
            for r in l2_results:
                results.append(MemorySearchResult(
                    text=r["text"],
                    source="L2",
                    similarity=r["similarity"],
                    weight=1.0,
                    metadata=r.get("metadata", {})
                ))
        except Exception as e:
            print(f"L2搜索失败: {e}")
        
        return results
    
    def _search_l3(self, query: str, top_k: int) -> List[MemorySearchResult]:
        """搜索L3数据库"""
        results = []
        
        try:
            # 提取关键词
            keywords = query.split()
            l3_results = self.sqlite.search_by_keywords(keywords, limit=top_k)
            
            for record in l3_results:
                # 使用压缩后的文本（如果有）
                text = record.compressed_text or record.text
                
                results.append(MemorySearchResult(
                    text=text,
                    source="L3",
                    similarity=0.7,  # 数据库搜索没有相似度，给个默认值
                    weight=record.weight,
                    metadata={
                        "id": record.id,
                        "access_count": record.access_count,
                        "timestamp": record.created_time
                    }
                ))
        except Exception as e:
            print(f"L3搜索失败: {e}")
        
        return results
    
    def _backfill_to_l2(self, result: MemorySearchResult):
        """
        L3命中时回填到L2向量库
        
        同时增加L3中的权重
        """
        if result.source != "L3":
            return
        
        record_id = result.metadata.get("id")
        if record_id is None:
            return
        
        try:
            # 存入L2向量库
            self.vector_store.add([result.text], [{
                "timestamp": datetime.now().isoformat(),
                "type": "backfill",
                "source": "L3",
                "original_id": record_id
            }])
            
            # 增加L3中的权重
            self.sqlite.update_weight(record_id, boost=True)
            
            self.stats["l3_backfills"] += 1
            print(f"L3回填: 记录 {record_id} 已存入L2并提升权重")
            
        except Exception as e:
            print(f"L3回填失败: {e}")
    
    def compress_memories(self, use_local: bool = True) -> Dict[str, int]:
        """
        压缩记忆：L2→L3
        
        将向量库中的低频访问记忆压缩后存入数据库
        
        Args:
            use_local: 是否使用本地LLM压缩
        
        Returns:
            压缩统计信息
        """
        if not config.compression_enabled:
            return {"compressed": 0, "skipped": 0}
        
        stats = {"compressed": 0, "skipped": 0, "errors": 0}
        
        try:
            # 获取长时间未访问的记忆
            unaccessed = self.sqlite.get_unaccessed_memories(
                days=config.memory_decay_days // 2,
                limit=50
            )
            
            if not unaccessed:
                return stats
            
            compressor = self._get_compressor() if use_local else None
            
            for record in unaccessed:
                try:
                    # 检查是否需要压缩
                    if len(record.text) < config.compression_min_length:
                        # 短文本直接存入L3
                        self.sqlite.add(record)
                        stats["skipped"] += 1
                        continue
                    
                    # 使用LLM压缩
                    if compressor:
                        compressed_text = self._compress_text(compressor, record.text)
                    else:
                        compressed_text = record.text  # 无压缩器则原样存储
                    
                    # 存入L3
                    compressed_record = MemoryRecord(
                        text=record.text,
                        compressed_text=compressed_text,
                        source="compressed",
                        weight=record.weight,
                        metadata={"original_source": record.source}
                    )
                    self.sqlite.add(compressed_record)
                    stats["compressed"] += 1
                    
                except Exception as e:
                    print(f"压缩记忆失败: {e}")
                    stats["errors"] += 1
            
            self.stats["compressions"] += stats["compressed"]
            
        except Exception as e:
            print(f"压缩流程失败: {e}")
        
        return stats
    
    def _compress_text(self, compressor, text: str) -> str:
        """使用LLM压缩文本"""
        prompt = f"""请将以下对话记录压缩为简洁的摘要，保留所有关键信息、专有名词和数值。

原始内容：
{text}

压缩要求：
1. 保留所有专有名词、人名、地名、数值
2. 保留因果关系和关键决策
3. 去除冗余和修饰性内容
4. 压缩后长度约为原文的30%-50%

压缩结果："""

        try:
            messages = [
                {"role": "system", "content": "你是一个记忆压缩专家，擅长保留关键信息的同时大幅压缩文本长度。"},
                {"role": "user", "content": prompt}
            ]
            
            result = compressor.chat(messages, max_tokens=500)
            return result.strip() if result else text
            
        except Exception as e:
            print(f"LLM压缩失败: {e}")
            return text
    
    def decay_and_forget(self) -> Dict[str, int]:
        """
        执行记忆衰减和遗忘
        
        Returns:
            遗忘统计信息
        """
        stats = {"decayed": 0, "forgotten": 0}
        
        try:
            # L3数据库权重衰减
            result = self.sqlite.decay_weights(config.memory_decay_days)
            stats["decayed"] = result.get("decayed", 0)
            stats["forgotten"] = result.get("forgotten", 0)
            
            self.stats["forgotten"] += stats["forgotten"]
            
        except Exception as e:
            print(f"记忆衰减失败: {e}")
        
        return stats
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        l2_count = len(self.vector_store)
        l3_stats = self.sqlite.get_stats() if config.sqlite_enabled else {}
        
        return {
            "l1_count": len(self.conversation_history),
            "l2_count": l2_count,
            "l3_count": l3_stats.get("total_count", 0),
            "l1_hits": self.stats["l1_hits"],
            "l2_hits": self.stats["l2_hits"],
            "l3_hits": self.stats["l3_hits"],
            "l3_backfills": self.stats["l3_backfills"],
            "compressions": self.stats["compressions"],
            "forgotten": self.stats["forgotten"],
            "l3_details": l3_stats
        }
    
    def __len__(self) -> int:
        """返回L1内存层的记忆数量"""
        return len(self.conversation_history)
    
    def clear_l1(self):
        """清空L1内存层"""
        with self.lock:
            self.conversation_history.clear()
    
    def clear_all(self, keep_l3: bool = False):
        """
        清空所有记忆层
        
        Args:
            keep_l3: 是否保留L3数据库（SQLite）
        """
        with self.lock:
            # 清空L1
            self.conversation_history.clear()
            
            # 清空L2向量库
            self.vector_store.clear()
            
            # 清空L3数据库
            if not keep_l3 and self.sqlite:
                with self.sqlite.lock:
                    with self.sqlite._get_connection() as conn:
                        conn.execute("DELETE FROM memories")
                        conn.commit()
            
            # 重置统计
            self.stats = {
                "l1_hits": 0,
                "l2_hits": 0,
                "l3_hits": 0,
                "l3_backfills": 0,
                "compressions": 0,
                "forgotten": 0
            }
    
    def cleanup_resources(self):
        """
        清理资源（用于会话结束）
        
        释放内存、关闭连接、清理缓存
        """
        import gc
        
        with self.lock:
            # 清空L1
            self.conversation_history.clear()
            
            # 清理SQLite连接池
            if self.sqlite:
                self.sqlite._cleanup_connections()
            
            # 触发垃圾回收
            gc.collect()
    
    def save_all(self):
        """保存所有层"""
        # ChromaDB自动持久化，无需额外操作
        # SQLite自动持久化，无需额外操作
        pass


# 全局实例
_memory_manager: Optional[MemoryManager] = None
_instance_lock = threading.Lock()


def get_memory_manager(vector_store: VectorStore = None) -> MemoryManager:
    """获取全局记忆管理器实例（线程安全单例）"""
    global _memory_manager
    if _memory_manager is None:
        with _instance_lock:
            if _memory_manager is None:
                _memory_manager = MemoryManager(vector_store)
    return _memory_manager


def reset_memory_manager():
    """
    重置全局记忆管理器实例
    
    用于清空会话或重新初始化
    """
    global _memory_manager
    with _instance_lock:
        if _memory_manager is not None:
            _memory_manager.cleanup_resources()
            _memory_manager = None
