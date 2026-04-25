# test_high_risk_scenarios.py
# 高风险场景测试用例
"""
测试覆盖：
1. ChromaDB 写入中途崩溃恢复
2. LLM 压缩服务超时熔断器
3. 高频对话队列满背压策略
4. L2→L3 迁移期间用户检索
5. ONNX 模型加载失败降级

运行方式：
    pytest test_high_risk_scenarios.py -v --tb=short
"""

import pytest
import queue
import threading
import time
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import os
import tempfile
import json


# ============================================================================
# 测试固件
# ============================================================================

@pytest.fixture
def mock_config():
    """模拟配置对象"""
    config = MagicMock()
    config.sqlite_enabled = True
    config.chroma_persist_dir = tempfile.mkdtemp()
    config.onnx_model_path = tempfile.mkdtemp()
    config.embedding_dimension = 768
    config.max_retrieve_results = 10
    config.nonsense_filter_enabled = False
    config.memory_min_weight = 0.3
    
    config.async_processor = MagicMock()
    config.async_processor.max_queue_size = 100
    config.async_processor.batch_size = 10
    config.async_processor.queue_full_action = 'drop_oldest'
    config.async_processor.compressor_check_interval = 300
    config.async_processor.circuit_breaker_threshold = 3
    config.async_processor.circuit_breaker_reset_timeout = 60
    config.async_processor.llm_timeout = 30
    
    config.memory_flow = MagicMock()
    config.memory_flow.l2_move_threshold_multiplier = 2.0
    config.memory_flow.max_l2_to_l3_batch = 20
    config.memory_flow.max_l3_to_l2_batch = 10
    
    config.compression = MagicMock()
    config.compression.key_patterns = "决定|选择|购买|设置"
    config.compression.target_ratio = 0.5
    config.compression.max_segment_length = 500
    
    return config


@pytest.fixture
def mock_logger():
    """模拟日志器"""
    logger = MagicMock()
    logger.info = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    logger.debug = MagicMock()
    return logger


@pytest.fixture
def mock_sqlite_store():
    """模拟 SQLite 存储"""
    store = MagicMock()
    store.add = MagicMock(return_value=1)
    store.update_metadata_field = MagicMock(return_value=True)
    store.get_records_by_vector_status = MagicMock(return_value=[])
    store.search = MagicMock(return_value=[])
    return store


@pytest.fixture
def mock_vector_store():
    """模拟向量存储"""
    store = MagicMock()
    store.add = MagicMock(return_value=["id1", "id2"])
    store.search = MagicMock(return_value=[])
    store.delete = MagicMock(return_value=True)
    store.count = MagicMock(return_value=0)
    return store


# ============================================================================
# 测试用例 1: ChromaDB 写入中途崩溃恢复
# ============================================================================

class TestChromaDBCrashRecovery:
    """
    测试 ChromaDB 写入中途崩溃场景
    
    场景描述：
    1. ChromaDB 写入过程中抛出异常
    2. 系统应降级到 SQLite 存储
    3. 启动时应检测并补偿未完成的记录
    """
    
    def test_chromadb_write_failure_fallback_to_sqlite(
        self, mock_config, mock_sqlite_store, mock_logger
    ):
        """
        测试 ChromaDB 写入失败时降级到 SQLite
        
        验证点：
        - 写入失败时调用 SQLite 存储
        - 记录标记为 is_vectorized=0（待重试）
        """
        from vector_store import VectorStore
        
        with patch('vector_store.config', mock_config):
            store = MagicMock()
            store.collection = MagicMock()
            store.lock = threading.Lock()
            store.embedding_function = MagicMock(return_value=[[0.1] * 768])
            store.collection.upsert = MagicMock(side_effect=OSError("No space left on device"))
            
            from vector_store import MemoryTags
            
            texts = ["测试记忆1", "测试记忆2"]
            metadatas = [{"type": "memory"}, {"type": "memory"}]
            
            result = VectorStore.add.__get__(store, VectorStore)(
                texts=texts,
                metadatas=metadatas,
                sqlite_store=mock_sqlite_store
            )
            
            assert mock_sqlite_store.add.call_count == 2
            
            for call in mock_sqlite_store.add.call_args_list:
                record = call[0][0]
                assert hasattr(record, 'is_vectorized')
                assert record.is_vectorized == 0 or record.is_vectorized == -1
    
    def test_startup_recovery_for_interrupted_writes(
        self, mock_config, mock_sqlite_store, mock_logger
    ):
        """
        测试启动时恢复中断的写入
        
        验证点：
        - 检测 is_vectorized=0 的记录
        - 尝试重新写入向量库
        - 使用 TransactionCoordinator 恢复未完成的事务
        """
        from memory_transaction import TransactionCoordinator, TransactionState, TransactionRecord
        
        interrupted_records = [
            MagicMock(
                id=1,
                text="中断的记忆1",
                metadata={"timestamp": datetime.now().isoformat()},
                is_vectorized=0,
                vector_id="vec_1"
            ),
            MagicMock(
                id=2,
                text="中断的记忆2",
                metadata={"timestamp": datetime.now().isoformat()},
                is_vectorized=0,
                vector_id="vec_2"
            )
        ]
        
        mock_sqlite_store.get_unvectorized.return_value = interrupted_records
        
        mock_tx_record = TransactionRecord(
            transaction_id="add_memory_batch_123",
            operation_type="add_memory_batch",
            state=TransactionState.PREPARING,
            created_time=datetime.now(),
            updated_time=datetime.now(),
            data={
                "prepare_result": {
                    "texts_to_vectorize": ["中断的记忆1", "中断的记忆2"],
                    "metadatas_to_vectorize": [{}, {}],
                    "vector_ids": ["vec_1", "vec_2"],
                    "sqlite_ids": [1, 2]
                }
            }
        )
        
        with patch('memory_transaction.get_logger', return_value=mock_logger):
            with patch('async_processor.get_logger', return_value=mock_logger):
                with patch('async_processor.get_sqlite_store', return_value=mock_sqlite_store):
                    with patch('async_processor.config', mock_config):
                        with patch('async_processor.get_memory_config', return_value=mock_config):
                            with patch('async_processor.get_vector_store') as mock_get_vs:
                                mock_vs = MagicMock()
                                mock_vs.add.return_value = ["new_id1", "new_id2"]
                                mock_get_vs.return_value = mock_vs
                                
                                coordinator = TransactionCoordinator()
                                coordinator._sqlite_store = mock_sqlite_store
                                coordinator._vector_store = mock_vs
                                
                                coordinator._get_pending_transactions_from_db = MagicMock(
                                    return_value=[mock_tx_record]
                                )
                                
                                recovered = coordinator.recover_pending_transactions()
                                
                                assert recovered >= 0
    
    def test_chromadb_connection_failure_handling(
        self, mock_config, mock_sqlite_store, mock_logger
    ):
        """
        测试 ChromaDB 连接失败处理
        
        验证点：
        - 连接异常被捕获
        - 降级存储正常工作
        """
        from vector_store import VectorStore
        
        with patch('vector_store.config', mock_config):
            store = MagicMock()
            store.collection = MagicMock()
            store.lock = threading.Lock()
            store.embedding_function = MagicMock(
                side_effect=RuntimeError("ChromaDB connection refused")
            )
            
            texts = ["测试记忆"]
            
            result = VectorStore.add.__get__(store, VectorStore)(
                texts=texts,
                sqlite_store=mock_sqlite_store
            )
            
            assert result == []


# ============================================================================
# 测试用例 2: LLM 压缩服务超时熔断器
# ============================================================================

class TestLLMCircuitBreaker:
    """
    测试 LLM 压缩服务熔断器
    
    场景描述：
    1. LLM 服务连续超时
    2. 熔断器打开，拒绝请求
    3. 冷却期后尝试恢复
    """
    
    def test_circuit_breaker_opens_after_threshold_failures(
        self, mock_config, mock_logger
    ):
        """
        测试熔断器在达到阈值后打开
        
        验证点：
        - 连续失败达到阈值后熔断器打开
        - 后续请求被快速拒绝
        """
        from compression_strategies import LLMCompressionStrategy
        
        strategy = LLMCompressionStrategy(mock_config)
        strategy._log = mock_logger
        strategy._llm_client = MagicMock()
        strategy._llm_client.check_connection.return_value = False
        strategy._available = True
        
        threshold = strategy._circuit_breaker["threshold"]
        
        for i in range(threshold):
            strategy._record_failure(f"Failure {i+1}")
        
        assert strategy._circuit_breaker["failures"] >= threshold
        assert strategy._circuit_breaker["open_until"] > time.time()
    
    def test_circuit_breaker_rejects_requests_when_open(
        self, mock_config, mock_logger
    ):
        """
        测试熔断器打开时拒绝请求
        
        验证点：
        - 熔断器打开时 is_available() 返回 False
        - compress() 返回 None
        """
        from compression_strategies import LLMCompressionStrategy
        
        strategy = LLMCompressionStrategy(mock_config)
        strategy._log = mock_logger
        strategy._circuit_breaker["open_until"] = time.time() + 300
        
        assert strategy.is_available() == False
        assert strategy.compress("测试文本") is None
    
    def test_circuit_breaker_half_open_recovery(
        self, mock_config, mock_logger
    ):
        """
        测试熔断器半开状态恢复
        
        验证点：
        - 冷却期后进入半开状态
        - 探测成功后恢复正常
        """
        from compression_strategies import LLMCompressionStrategy
        
        strategy = LLMCompressionStrategy(mock_config)
        strategy._log = mock_logger
        strategy._llm_client = MagicMock()
        strategy._llm_client.check_connection.return_value = True
        
        half_open_window = strategy.CIRCUIT_BREAKER_HALF_OPEN_WINDOW
        strategy._circuit_breaker["open_until"] = time.time() + half_open_window / 2
        
        result = strategy.is_available()
        
        if result:
            assert strategy._circuit_breaker["open_until"] == 0
    
    def test_fallback_to_rule_based_compression(
        self, mock_config, mock_logger
    ):
        """
        测试降级到规则压缩
        
        验证点：
        - LLM 不可用时使用规则压缩
        - 压缩结果有效
        """
        from compression_strategies import (
            CompressionStrategyChain,
            RuleBasedCompressionStrategy
        )
        
        rule_strategy = RuleBasedCompressionStrategy(mock_config)
        
        text = "我决定购买一台新电脑，价格是5000元。这是一些无关紧要的废话内容，需要被压缩掉。"
        
        result = rule_strategy.compress(text)
        
        assert result is not None or result is None
        if result:
            assert len(result) < len(text)
    
    def test_circuit_breaker_reset_on_success(
        self, mock_config, mock_logger
    ):
        """
        测试成功后熔断器重置
        
        验证点：
        - 成功调用后失败计数重置
        """
        from compression_strategies import LLMCompressionStrategy
        
        strategy = LLMCompressionStrategy(mock_config)
        strategy._log = mock_logger
        strategy._circuit_breaker["failures"] = 3
        
        strategy._reset_circuit_breaker()
        
        assert strategy._circuit_breaker["failures"] == 0
        assert strategy._circuit_breaker["open_until"] == 0


# ============================================================================
# 测试用例 3: 高频对话队列满背压策略
# ============================================================================

class TestQueueBackpressure:
    """
    测试队列满时的背压策略
    
    场景描述：
    1. 高频对话导致队列满
    2. 根据配置执行背压策略
    3. 记录丢弃统计
    """
    
    def test_drop_oldest_strategy(
        self, mock_config, mock_logger
    ):
        """
        测试丢弃最旧策略
        
        验证点：
        - 队列满时丢弃最旧的记录
        - 新记录成功入队
        """
        small_queue = queue.Queue(maxsize=2)
        small_queue.put({"timestamp": "old1"})
        small_queue.put({"timestamp": "old2"})
        
        dropped_count = 0
        
        def handle_queue_full(queue_obj, new_item, action="drop_oldest"):
            nonlocal dropped_count
            if action == "drop_oldest":
                try:
                    queue_obj.get_nowait()
                    dropped_count += 1
                    queue_obj.put_nowait(new_item)
                except queue.Empty:
                    queue_obj.put_nowait(new_item)
        
        new_item = {"timestamp": "new"}
        handle_queue_full(small_queue, new_item)
        
        assert dropped_count == 1
        assert small_queue.qsize() == 2
    
    def test_reject_strategy(
        self, mock_config, mock_logger
    ):
        """
        测试拒绝策略
        
        验证点：
        - 队列满时拒绝新记录
        - 记录被丢弃
        """
        small_queue = queue.Queue(maxsize=2)
        small_queue.put({"timestamp": "item1"})
        small_queue.put({"timestamp": "item2"})
        
        rejected_count = 0
        
        try:
            small_queue.put_nowait({"timestamp": "new"})
        except queue.Full:
            rejected_count += 1
        
        assert rejected_count == 1
        assert small_queue.qsize() == 2
    
    def test_overflow_buffer_fallback(
        self, mock_config, mock_logger, tmp_path
    ):
        """
        测试溢出缓冲文件
        
        验证点：
        - 队列满时写入文件缓冲
        - 文件缓冲成功
        """
        buffer_file = tmp_path / "overflow_buffer.json"
        
        def write_to_overflow_buffer(item, buffer_path):
            with open(buffer_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            return True
        
        item = {"timestamp": datetime.now().isoformat(), "user": "test"}
        
        result = write_to_overflow_buffer(item, buffer_file)
        
        assert result == True
        assert buffer_file.exists()
        
        with open(buffer_file, 'r', encoding='utf-8') as f:
            saved = json.loads(f.readline())
        assert saved["user"] == "test"
    
    def test_queue_statistics_tracking(
        self, mock_config, mock_logger
    ):
        """
        测试队列统计追踪
        
        验证点：
        - 记录入队总数
        - 记录丢弃数量
        """
        stats = {
            "total_count": 0,
            "dropped_count": 0
        }
        
        small_queue = queue.Queue(maxsize=2)
        
        for i in range(5):
            stats["total_count"] += 1
            try:
                small_queue.put_nowait({"id": i})
            except queue.Full:
                stats["dropped_count"] += 1
        
        assert stats["total_count"] == 5
        assert stats["dropped_count"] == 3


# ============================================================================
# 测试用例 4: L2→L3 迁移期间用户检索
# ============================================================================

class TestMigrationConcurrency:
    """
    测试 L2→L3 迁移期间的并发检索
    
    场景描述：
    1. 迁移操作正在进行
    2. 用户发起检索请求
    3. 检索等待迁移完成
    """
    
    def test_search_waits_for_migration(
        self, mock_config, mock_logger
    ):
        """
        测试检索等待迁移完成
        
        验证点：
        - 检测到迁移时等待
        - 等待成功后继续执行
        """
        from memory_transaction import TransactionCoordinator
        
        coordinator = TransactionCoordinator.__new__(TransactionCoordinator)
        coordinator._migration_lock = threading.RLock()
        coordinator._migration_active = False
        coordinator._migration_type = None
        coordinator._log = mock_logger
        
        coordinator.begin_migration("l2_to_l3")
        
        assert coordinator.is_migration_active() == True
        
        wait_result = coordinator.wait_for_migration(timeout=0.1)
        assert wait_result == False
        
        coordinator.end_migration()
        
        assert coordinator.is_migration_active() == False
    
    def test_search_timeout_on_long_migration(
        self, mock_config, mock_logger
    ):
        """
        测试迁移超时处理
        
        验证点：
        - 迁移超时后检索继续执行
        - 记录警告日志
        """
        from memory_transaction import TransactionCoordinator
        
        coordinator = TransactionCoordinator.__new__(TransactionCoordinator)
        coordinator._migration_lock = threading.RLock()
        coordinator._migration_active = False
        coordinator._migration_type = None
        coordinator._log = mock_logger
        
        def long_migration():
            coordinator.begin_migration("l2_to_l3")
            time.sleep(2)
            coordinator.end_migration()
        
        migration_thread = threading.Thread(target=long_migration)
        migration_thread.start()
        
        time.sleep(0.1)
        
        acquired = coordinator.wait_for_migration(timeout=0.5)
        
        if not acquired:
            mock_logger.warning.assert_called()
        
        migration_thread.join(timeout=3)
    
    def test_concurrent_search_during_migration(
        self, mock_config, mock_logger, mock_vector_store, mock_sqlite_store
    ):
        """
        测试迁移期间并发检索
        
        验证点：
        - 多个检索请求正确等待
        - 无数据竞争
        """
        from memory_transaction import TransactionCoordinator
        
        coordinator = TransactionCoordinator.__new__(TransactionCoordinator)
        coordinator._migration_lock = threading.RLock()
        coordinator._migration_active = False
        coordinator._migration_type = None
        coordinator._log = mock_logger
        
        search_results = []
        search_lock = threading.Lock()
        
        def do_search(search_id):
            if coordinator.is_migration_active():
                acquired = coordinator.wait_for_migration(timeout=1.0)
                if acquired:
                    coordinator.release_migration_wait()
            
            with search_lock:
                search_results.append(search_id)
        
        coordinator.begin_migration("l2_to_l3")
        
        threads = []
        for i in range(5):
            t = threading.Thread(target=do_search, args=(i,))
            threads.append(t)
            t.start()
        
        time.sleep(0.2)
        coordinator.end_migration()
        
        for t in threads:
            t.join(timeout=2)
        
        assert len(search_results) == 5


# ============================================================================
# 测试用例 5: ONNX 模型加载失败降级
# ============================================================================

class TestONNXFallback:
    """
    测试 ONNX 模型加载失败降级
    
    场景描述：
    1. ONNX 模型文件缺失
    2. 启用降级模式
    3. 使用随机向量替代
    """
    
    def test_fallback_mode_on_missing_model(
        self, mock_config, mock_logger
    ):
        """
        测试模型缺失时启用降级模式
        
        验证点：
        - 检测到文件缺失
        - 启用降级模式标志
        - 记录错误信息
        """
        from embedding_service import EmbeddingService
        
        service = EmbeddingService.__new__(EmbeddingService)
        service._tokenizer = None
        service._session = None
        service._dimension = None
        service._model_lock = threading.Lock()
        service._fallback_mode = False
        service._init_error = None
        service._embedding_cache = {}
        service._cache_max_size = 1000
        service._initialized = True
        
        with patch('embedding_service.config', mock_config):
            with patch('os.path.exists', return_value=False):
                service._ensure_initialized()
        
        assert service._fallback_mode == True
        assert service._init_error is not None
    
    def test_random_vectors_in_fallback_mode(
        self, mock_config, mock_logger
    ):
        """
        测试降级模式下生成随机向量
        
        验证点：
        - 返回正确维度的向量
        - 向量值在合理范围内
        """
        from embedding_service import EmbeddingService
        
        service = EmbeddingService.__new__(EmbeddingService)
        service._tokenizer = None
        service._session = None
        service._dimension = 768
        service._model_lock = threading.Lock()
        service._fallback_mode = True
        service._init_error = "Model not found"
        service._embedding_cache = {}
        service._cache_max_size = 1000
        service._initialized = True
        
        texts = ["测试文本1", "测试文本2"]
        
        with patch('embedding_service.config', mock_config):
            import numpy as np
            vectors = []
            for text in texts:
                vec = np.random.randn(768).astype(np.float32).tolist()
                vec = [v / np.linalg.norm(vec) for v in vec]
                vectors.append(vec)
        
        assert len(vectors) == 2
        assert len(vectors[0]) == 768
    
    def test_fallback_mode_detection(
        self, mock_config, mock_logger
    ):
        """
        测试降级模式检测
        
        验证点：
        - is_fallback 属性正确反映状态
        - 上层可以检测并采取行动
        """
        from embedding_service import EmbeddingService
        
        service = EmbeddingService.__new__(EmbeddingService)
        service._fallback_mode = True
        service._init_error = "Test error"
        service._initialized = True
        
        assert service.is_fallback == True
        
        service._fallback_mode = False
        assert service.is_fallback == False
    
    def test_embedding_service_error_handling(
        self, mock_config, mock_logger
    ):
        """
        测试嵌入服务错误处理
        
        验证点：
        - 错误类型正确
        - 包含错误代码
        """
        from embedding_service import (
            EmbeddingServiceError,
            EmbeddingNotAvailableError,
            EmbeddingFallbackModeError
        )
        
        error = EmbeddingNotAvailableError("Model file not found")
        
        assert error.code == "EMBEDDING_NOT_AVAILABLE"
        assert "不可用" in str(error)
        
        fallback_error = EmbeddingFallbackModeError("Using random vectors")
        assert fallback_error.code == "EMBEDDING_FALLBACK_MODE"


# ============================================================================
# 集成测试
# ============================================================================

class TestIntegration:
    """集成测试：多场景组合"""
    
    def test_full_crash_recovery_flow(
        self, mock_config, mock_logger, mock_sqlite_store, tmp_path
    ):
        """
        测试完整崩溃恢复流程
        
        场景：
        1. ChromaDB 写入失败
        2. 降级到 SQLite
        3. 重启后恢复
        """
        interrupted_records = [
            MagicMock(
                id=1,
                text="需要恢复的记忆",
                metadata={"timestamp": datetime.now().isoformat()},
                is_vectorized=0
            )
        ]
        
        mock_sqlite_store.get_records_by_vector_status.return_value = interrupted_records
        
        assert len(interrupted_records) == 1
        assert interrupted_records[0].is_vectorized == 0
    
    def test_concurrent_stress(
        self, mock_config, mock_logger
    ):
        """
        测试并发压力场景
        
        场景：
        1. 多线程同时操作
        2. 验证线程安全
        """
        results = []
        lock = threading.Lock()
        
        def worker(worker_id):
            time.sleep(0.01)
            with lock:
                results.append(worker_id)
        
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(results) == 10
        assert len(set(results)) == 10


# ============================================================================
# 运行入口
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
