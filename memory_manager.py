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
from event_bus import get_event_bus, EventType
from memory_transaction import get_transaction_coordinator
from logger import get_logger


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
    
    线程安全：
    - 使用迁移锁协调检索与迁移操作
    - 使用版本号检测并发修改
    - 检索时等待迁移完成
    """
    
    def __init__(self, vector_store: VectorStore = None, sqlite_store: SQLiteStore = None):
        self.vector_store = vector_store or get_vector_store()
        self.sqlite = sqlite_store or get_sqlite_store()
        self._event_bus = get_event_bus()
        self._tx_coordinator = get_transaction_coordinator()
        self._log = get_logger()
        
        self.lock = threading.RLock()
        self._read_lock = threading.Lock()
        self._write_version = 0
        self._last_sync_time = time.time()
        
        self.conversation_history: List[Dict] = []
        self.max_l1_size = 25
        
        self.stats = {
            "l1_hits": 0,
            "l2_hits": 0,
            "l3_hits": 0,
            "l3_backfills": 0,
            "compressions": 0,
            "forgotten": 0,
            "retry_reads": 0
        }
        
        self._event_bus.subscribe(EventType.MEMORY_WRITTEN, self._handle_memory_written)
    
    def _handle_memory_written(self, event):
        """处理记忆写入事件（更新版本号）"""
        with self.lock:
            self._write_version += 1
            self._last_sync_time = time.time()
        self._log.debug("MEMORY_WRITE_NOTIFIED", version=self._write_version)
    
    def add_conversation(self, user_input: str, assistant_response: str, metadata: Dict = None):
        """
        添加对话到L1内存层
        
        Args:
            user_input: 用户输入
            assistant_response: 助理回复
            metadata: 元数据
        
        注意：L1仅作为内存缓存，溢出时直接丢弃
        持久化工作已由 AsyncProcessor 保证
        """
        with self.lock:
            self.conversation_history.append({
                "user": user_input,
                "assistant": assistant_response,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            })
            
            if len(self.conversation_history) > self.max_l1_size:
                overflow_count = len(self.conversation_history) - self.max_l1_size
                self.conversation_history = self.conversation_history[-self.max_l1_size:]
                self._event_bus.publish(EventType.L1_OVERFLOW, {
                    "overflow_count": overflow_count,
                    "current_size": len(self.conversation_history),
                    "max_size": self.max_l1_size
                })
                self._log.debug("L1_OVERFLOW_PUBLISHED", 
                               overflow_count=overflow_count,
                               current_size=len(self.conversation_history))
    
    def search(self, query: str, top_k: int = None, include_l3: bool = True, threshold: float = None, include_l1: bool = True) -> List[MemorySearchResult]:
        """
        统一搜索接口 - L1→L2→L3
        
        线程安全：
        - 等待迁移操作完成后再执行检索
        - 使用版本号检测并发修改
        - 检测到修改时自动重试（最多2次）
        
        Args:
            query: 搜索查询
            top_k: 返回结果数量
            include_l3: 是否搜索L3数据库
            threshold: 相似度阈值，低于此值的结果将被过滤
            include_l1: 是否搜索L1内存层（默认True，context_builder单独搜索L1时会设为False）
        
        Returns:
            搜索结果列表（已去重、已应用时间衰减、已过滤阈值）
        """
        import time
        from metrics import get_metrics_collector
        
        start_time = time.perf_counter()
        
        if self._tx_coordinator.is_migration_active():
            self._tx_coordinator.wait_for_migration(timeout=5.0)
            self._tx_coordinator.release_migration_wait()
        
        if top_k is None:
            top_k = config.max_retrieve_results
        
        if threshold is None:
            threshold = 0.0
        
        max_retries = 3
        for attempt in range(max_retries):
            start_version = self._write_version
            
            results = self._do_search(query, top_k, include_l3, threshold, include_l1)
            
            if self._write_version == start_version:
                break
            
            self.stats["retry_reads"] += 1
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        metrics = get_metrics_collector()
        metrics.record_retrieval_latency(duration_ms)
        metrics.record_retrieval_hits(
            self.stats.get("l1_hits", 0),
            self.stats.get("l2_hits", 0),
            self.stats.get("l3_hits", 0)
        )
        
        return results
    
    def _do_search(self, query: str, top_k: int, include_l3: bool, threshold: float, include_l1: bool) -> List[MemorySearchResult]:
        """
        实际执行搜索（内部方法）
        
        优化：使用超额采样 + 内存重排策略
        1. 扩大召回范围（top_k * 3）
        2. 在内存中结合时间衰减、权重等因素重新打分
        3. 最后截断输出最终的Top-K
        4. 过滤掉 forgotten 记忆
        """
        results = []
        seen_texts = set()
        
        time_context = self._detect_time_context(query)
        
        recall_multiplier = 3
        recall_limit = top_k * recall_multiplier
        
        if include_l1:
            l1_results = self._search_l1(query, recall_limit, threshold)
            for r in l1_results:
                if r.text not in seen_texts:
                    if MemoryTagHelper.is_forgotten(r.metadata):
                        continue
                    seen_texts.add(r.text)
                    r.combined_score = r.similarity * r.weight
                    results.append(r)
            self.stats["l1_hits"] += len(l1_results)
        
        if len(results) < recall_limit:
            remaining = recall_limit - len(results)
            l2_results = self._search_l2(query, remaining)
            for r in l2_results:
                if r.text not in seen_texts:
                    if MemoryTagHelper.is_forgotten(r.metadata):
                        continue
                    seen_texts.add(r.text)
                    time_decay = self._apply_time_decay(r, time_context)
                    r.combined_score = time_decay * r.weight
                    results.append(r)
        self.stats["l2_hits"] += len(l2_results)
        
        if include_l3 and len(results) < recall_limit and config.sqlite_enabled:
            remaining = recall_limit - len(results)
            l3_results = self._search_l3(query, remaining)
            for r in l3_results:
                if r.text not in seen_texts:
                    if MemoryTagHelper.is_forgotten(r.metadata):
                        continue
                    seen_texts.add(r.text)
                    time_decay = self._apply_time_decay(r, time_context)
                    r.combined_score = time_decay * r.weight
                    results.append(r)
            
            for result in l3_results:
                if not MemoryTagHelper.is_forgotten(result.metadata):
                    self._backfill_to_l2(result)
        
        results.sort(key=lambda x: x.combined_score, reverse=True)
        
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
            "today": (r"今天|今日|today", "today", 1.5),
            "yesterday": (r"昨天|昨日|yesterday", "yesterday", 2.0),
            "last_week": (r"上周|last week", "last_week", 1.8),
            "this_week": (r"这周|本周|this week", "this_week", 1.3),
            "recent": (r"最近|刚才|recently|just now", "recent", 1.4),
            "last_month": (r"上个月|上月|last month", "last_month", 1.2),
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
    
    def _calculate_time_decay(self, timestamp: str, time_context: Dict = None) -> float:
        """
        统一的时间衰减计算
        
        基础衰减（无时间上下文）：
        - 最近1小时内: 1.0
        - 1-6小时: 0.8
        - 6-24小时: 0.6
        - 1-7天: 0.4
        - 7天以上: 0.2
        
        结合时间上下文：
        - "今天"：24小时内加权
        - "昨天"：24-48小时内加权
        - "上周"：7-14天内加权
        - "这周"：7天内加权
        """
        if not timestamp:
            return 0.5
        
        try:
            conv_time = datetime.fromisoformat(timestamp)
            now = datetime.now()
            delta = now - conv_time
            hours = delta.total_seconds() / 3600
            
            base_decay = 1.0
            if hours < 1:
                base_decay = 1.0
            elif hours < 6:
                base_decay = 0.8
            elif hours < 24:
                base_decay = 0.6
            elif hours < 168:
                base_decay = 0.4
            else:
                base_decay = 0.2
            
            if not time_context or not time_context.get("has_time_ref"):
                return base_decay
            
            time_type = time_context.get("time_type", "any")
            context_decay = 1.0
            
            if time_type == "today":
                if hours < 24:
                    context_decay = 1.5
                else:
                    context_decay = 0.5
            
            elif time_type == "yesterday":
                if 24 <= hours < 48:
                    context_decay = 2.0
                elif hours < 24:
                    context_decay = 1.0
                else:
                    context_decay = 0.4
            
            elif time_type == "last_week":
                if 168 <= hours < 336:
                    context_decay = 1.8
                elif hours < 168:
                    context_decay = 1.0
                else:
                    context_decay = 0.5
            
            elif time_type == "this_week":
                if hours < 168:
                    context_decay = 1.3
                else:
                    context_decay = 0.5
            
            elif time_type == "recent":
                if hours < 24:
                    context_decay = 1.4
                elif hours < 72:
                    context_decay = 1.1
                else:
                    context_decay = 0.7
            
            elif time_type == "last_month":
                if 720 <= hours < 1440:
                    context_decay = 1.2
                elif hours < 720:
                    context_decay = 0.9
                else:
                    context_decay = 0.6
            
            return base_decay * context_decay
            
        except (ValueError, TypeError):
            return 0.5
    
    def _apply_time_decay(self, result: MemorySearchResult, time_context: Dict) -> float:
        """
        应用时间衰减因子（使用统一的时间衰减函数）
        """
        timestamp_str = result.metadata.get(MemoryTags.TIMESTAMP, "")
        decay = self._calculate_time_decay(timestamp_str, time_context)
        return result.similarity * decay
    
    def _search_l1(self, query: str, top_k: int, threshold: float = 0.0) -> List[MemorySearchResult]:
        """
        搜索L1内存层
        
        评分策略：
        1. 关键词覆盖率（基础分）
        2. 词权重（长词权重更高）
        3. 位置权重（用户输入权重高于助理回复）
        4. 时间衰减（最近的对话权重更高）
        """
        results = []
        query_lower = query.lower()
        query_keywords = set(query_lower.split())
        
        if not query_keywords:
            return results
        
        keyword_weights = {}
        for kw in query_keywords:
            if len(kw) > 4:
                keyword_weights[kw] = 3.0
            elif len(kw) > 2:
                keyword_weights[kw] = 2.0
            else:
                keyword_weights[kw] = 1.0
        
        for conv in reversed(self.conversation_history):
            user_text = conv.get("user", "").lower()
            assistant_text = conv.get("assistant", "").lower()
            
            score = 0.0
            matched_keywords = 0
            
            for kw, kw_weight in keyword_weights.items():
                if kw in user_text:
                    score += kw_weight * 2.0
                    matched_keywords += 1
                if kw in assistant_text:
                    score += kw_weight * 1.0
                    matched_keywords += 1
            
            if score > 0:
                combined_text = f"用户: {conv['user']}\n助理: {conv['assistant']}"
                
                max_possible_score = sum(keyword_weights.values()) * 3
                base_similarity = score / max_possible_score if max_possible_score > 0 else 0
                
                coverage = matched_keywords / (len(query_keywords) * 2)
                
                time_decay = self._calculate_time_decay(conv.get("timestamp", ""))
                
                similarity = (
                    base_similarity * 0.5 +
                    coverage * 0.3 +
                    time_decay * 0.2
                )
                
                if similarity >= threshold:
                    results.append(MemorySearchResult(
                        text=combined_text,
                        source="L1",
                        similarity=similarity,
                        weight=1.0,
                        metadata={MemoryTags.TIMESTAMP: conv.get("timestamp", "")}
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
        """
        搜索L3数据库
        
        优化：直接使用FTS5的BM25分数，移除Python层冗余的字符串匹配
        
        对于 sqlite_only 记录（is_vectorized=-1）：
        - 降低权重（乘以 0.5）
        - 标记来源为 L3_SQLITE_ONLY
        """
        results = []
        
        try:
            l3_results = self.sqlite.search(query, limit=top_k * 2)
            
            for record in l3_results:
                text = record.compressed_text or record.text
                
                is_sqlite_only = (record.is_vectorized == -1)
                
                weight = record.weight
                source = "L3"
                
                if is_sqlite_only:
                    weight *= 0.5
                    source = "L3_LOW_QUALITY"
                
                similarity = min(1.0, record.weight / 2.0)
                
                results.append(MemorySearchResult(
                    text=text,
                    source=source,
                    similarity=similarity,
                    weight=weight,
                    metadata={
                        "id": record.id,
                        "access_count": record.access_count,
                        MemoryTags.TIMESTAMP: record.created_time,
                        MemoryTags.IS_SQLITE_ONLY: is_sqlite_only
                    }
                ))
            
            results.sort(key=lambda x: x.similarity * x.weight, reverse=True)
            
        except Exception as e:
            print(f"L3搜索失败: {e}")
        
        return results[:top_k]
    
    def _calculate_text_similarity(self, query: str, text: str, query_keywords: set = None) -> float:
        """
        计算文本相似度（基于关键词匹配）
        
        综合考虑：
        1. 关键词覆盖率
        2. 关键词位置权重
        3. 文本长度归一化
        4. IDF权重（简化版）
        """
        if not query or not text:
            return 0.0
        
        if query_keywords is None:
            query_keywords = set(query.split())
        
        if not query_keywords:
            return 0.0
        
        matched_keywords = 0
        position_score = 0.0
        idf_score = 0.0
        
        for kw in query_keywords:
            if kw in text:
                matched_keywords += 1
                
                pos = text.find(kw)
                if pos < 100:
                    position_score += 1.0
                elif pos < 300:
                    position_score += 0.7
                else:
                    position_score += 0.5
                
                if len(kw) > 3:
                    idf_score += 1.5
                else:
                    idf_score += 1.0
        
        coverage = matched_keywords / len(query_keywords)
        
        max_position_score = len(query_keywords)
        normalized_position = position_score / max_position_score if max_position_score > 0 else 0
        
        max_idf_score = len(query_keywords) * 1.5
        normalized_idf = idf_score / max_idf_score if max_idf_score > 0 else 0
        
        similarity = (
            coverage * 0.5 +
            normalized_position * 0.3 +
            normalized_idf * 0.2
        )
        
        return min(similarity, 1.0)
    
    def _backfill_to_l2(self, result: MemorySearchResult):
        """
        L3命中时回填到L2向量库
        
        同时增加L3中的权重
        
        注意：sqlite_only 记录不会被回填（保持低质量记忆不污染向量空间）
        """
        if result.source != "L3":
            return
        
        if result.metadata.get(MemoryTags.IS_SQLITE_ONLY, False):
            return
        
        record_id = result.metadata.get("id")
        if record_id is None:
            return
        
        try:
            self.vector_store.add([result.text], [{
                MemoryTags.TIMESTAMP: datetime.now().isoformat(),
                "type": "backfill",
                "source": "L3",
                "original_id": record_id
            }])
            
            self.sqlite.update_weight(record_id, boost=True)
            
            self.stats["l3_backfills"] += 1
            print(f"L3回填: 记录 {record_id} 已存入L2并提升权重")
            
        except Exception as e:
            print(f"L3回填失败: {e}")
    
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
    
    def mark_forgotten(self, record_id: int, reason: str = "user_request") -> bool:
        """
        标记记忆为遗忘状态
        
        遗忘的记忆：
        - 不参与检索
        - 优先被合并/删除
        - 保留原始数据（可恢复）
        - 从 L2 向量库中移除（防止"僵尸记忆"复活）
        
        Args:
            record_id: 记忆ID
            reason: 遗忘原因
        
        Returns:
            是否成功
        """
        try:
            record = self.sqlite.get(record_id)
            if not record:
                return False
            
            record.metadata = MemoryTagHelper.mark_forgotten(record.metadata, reason)
            
            if record.vector_id:
                try:
                    self.vector_store.delete(ids=[record.vector_id])
                    print(f"已从 L2 向量库删除: {record.vector_id}")
                except Exception as e:
                    print(f"L2 向量删除失败（非致命）: {e}")
                
                record.vector_id = ""
                record.is_vectorized = 0
            
            self.sqlite.add(record)
            
            self.stats["forgotten"] += 1
            print(f"记忆 {record_id} 已标记为遗忘")
            
            return True
            
        except Exception as e:
            print(f"标记遗忘失败: {e}")
            return False
    
    def unmark_forgotten(self, record_id: int) -> bool:
        """
        取消遗忘标记（恢复记忆）
        
        恢复后需要重新向量化才能参与 L2 检索
        
        Args:
            record_id: 记忆ID
        
        Returns:
            是否成功
        """
        try:
            record = self.sqlite.get(record_id)
            if not record:
                return False
            
            record.metadata = MemoryTagHelper.unmark_forgotten(record.metadata)
            
            record.metadata[MemoryTags.NEEDS_REVECTORIZATION] = True
            
            self.sqlite.add(record)
            
            print(f"记忆 {record_id} 已恢复，需要重新向量化")
            
            return True
            
        except Exception as e:
            print(f"恢复记忆失败: {e}")
            return False
    
    def get_forgotten_memories(self, limit: int = 100) -> List[Dict]:
        """
        获取已遗忘的记忆列表
        
        用于用户查看和管理已遗忘的记忆
        
        Args:
            limit: 返回数量限制
        
        Returns:
            遗忘记忆列表
        """
        try:
            all_records = self.sqlite.get_recent_memories(limit=limit * 10)
            forgotten = []
            
            for record in all_records:
                if MemoryTagHelper.is_forgotten(record.metadata):
                    forgotten.append({
                        "id": record.id,
                        "text": record.text[:100] + "..." if len(record.text) > 100 else record.text,
                        "forgotten_time": record.metadata.get(MemoryTags.FORGOTTEN_TIME),
                        "reason": record.metadata.get(MemoryTags.FORGOTTEN_REASON),
                        "created_time": record.created_time
                    })
                    
                    if len(forgotten) >= limit:
                        break
            
            return forgotten
            
        except Exception as e:
            print(f"获取遗忘记忆失败: {e}")
            return []
    
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
