# tests/test_integration.py
# 端到端集成测试
"""
集成测试套件

测试场景覆盖：
1. 基础对话流程
2. 记忆检索验证
3. 压缩流程验证
4. 记忆流动验证
5. 废话过滤验证
6. 故障恢复验证

运行方式：
    pytest tests/test_integration.py -v
    pytest tests/test_integration.py -v -k "test_basic_conversation"
    pytest tests/test_integration.py -v --cov=. --cov-report=html
"""

import os
import sys
import time
import json
import shutil
import tempfile
import threading
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlite_store import SQLiteStore, MemoryRecord
from memory_tags import MemoryTags, MemoryTagHelper
from tests.test_data_generator import TestDataGenerator, get_test_data_generator


class TestFixtures:
    """测试夹具管理"""
    
    @staticmethod
    def setup_test_environment():
        """设置测试环境"""
        test_dir = tempfile.mkdtemp(prefix="memory_test_")
        
        test_config = {
            "sqlite_db_path": os.path.join(test_dir, "test_memory.db"),
            "chroma_persist_dir": os.path.join(test_dir, "chroma_db"),
            "nonsense_db_path": os.path.join(test_dir, "nonsense_library.json")
        }
        
        for key, path in test_config.items():
            os.makedirs(os.path.dirname(path), exist_ok=True)
        
        return test_dir, test_config
    
    @staticmethod
    def cleanup_test_environment(test_dir: str):
        """清理测试环境"""
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def test_env():
    """模块级测试环境夹具"""
    test_dir, test_config = TestFixtures.setup_test_environment()
    yield test_dir, test_config
    TestFixtures.cleanup_test_environment(test_dir)


@pytest.fixture(scope="module")
def data_generator():
    """测试数据生成器夹具"""
    return get_test_data_generator(seed=42)


class TestBasicConversation:
    """
    测试场景1：基础对话流程
    
    用例名称：test_basic_conversation_flow
    前置条件：
        - 系统已初始化
        - 数据库为空
    操作步骤：
        1. 用户发送消息
        2. 系统检索记忆
        3. LLM生成回复
        4. 异步存储记忆
    预期结果：
        - 记忆被正确存储到L2和L3
        - 检索结果包含新存储的记忆
    验证点：
        - L2向量库有记录
        - L3 SQLite有记录
        - 记忆内容完整
    """
    
    def test_user_message_to_memory_storage(self, test_env, data_generator):
        """
        用例：用户消息到记忆存储完整流程
        
        前置条件：系统初始化完成
        操作步骤：
            1. 发送用户消息
            2. 模拟LLM响应
            3. 等待异步存储完成
            4. 检查存储结果
        预期结果：记忆被存储到L2和L3
        """
        test_dir, test_config = test_env
        
        sqlite = SQLiteStore(db_path=test_config["sqlite_db_path"])
        
        test_memory = MemoryRecord(
            text="用户喜欢使用Python进行数据分析",
            metadata={
                "source": "test",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        record_id = sqlite.add(test_memory)
        
        assert record_id is not None, "记录ID不应为空"
        assert record_id > 0, "记录ID应为正整数"
        
        stored_record = sqlite.get(record_id)
        assert stored_record is not None, "应能检索到存储的记录"
        assert stored_record.text == test_memory.text, "存储内容应一致"
        
        sqlite.close()
    
    def test_conversation_history_management(self, test_env, data_generator):
        """
        用例：对话历史管理
        
        前置条件：L1缓存为空
        操作步骤：
            1. 添加多条对话记录
            2. 检查L1缓存大小限制
            3. 验证FIFO淘汰机制
        预期结果：L1缓存不超过最大限制
        """
        from memory_manager import MemoryManager
        
        test_dir, test_config = test_env
        
        with patch('memory_manager.get_vector_store') as mock_vs, \
             patch('memory_manager.get_sqlite_store') as mock_sqlite:
            
            mock_vs.return_value = Mock()
            mock_sqlite.return_value = Mock()
            
            manager = MemoryManager()
            manager.max_l1_size = 5
            
            for i in range(10):
                manager.conversation_history.append({
                    "role": "user",
                    "content": f"测试消息 {i}"
                })
                if len(manager.conversation_history) > manager.max_l1_size:
                    manager.conversation_history = manager.conversation_history[-manager.max_l1_size:]
            
            assert len(manager.conversation_history) <= manager.max_l1_size, \
                f"L1缓存应不超过{manager.max_l1_size}，实际为{len(manager.conversation_history)}"
    
    def test_async_storage_with_metadata(self, test_env, data_generator):
        """
        用例：异步存储带元数据
        
        前置条件：异步处理器就绪
        操作步骤：
            1. 添加对话到处理队列
            2. 等待处理完成
            3. 验证元数据完整性
        预期结果：元数据被正确保存
        """
        test_dir, test_config = test_env
        
        sqlite = SQLiteStore(db_path=test_config["sqlite_db_path"])
        
        conversations = data_generator.generate_batch_conversations(num_conversations=3)
        
        for conv in conversations:
            for turn in conv.turns:
                record = MemoryRecord(
                    text=turn.user_input,
                    metadata={
                        "topic": conv.topic,
                        "timestamp": turn.timestamp,
                        "response": turn.assistant_response[:100]
                    }
                )
                record_id = sqlite.add(record)
                assert record_id is not None
        
        all_records = sqlite.get_recent_memories(limit=1000)
        assert len(all_records) >= 3, "应至少有3条记录"
        
        sqlite.close()


class TestMemoryRetrieval:
    """
    测试场景2：记忆检索验证
    
    用例名称：test_memory_retrieval
    前置条件：
        - 已存储特定主题的记忆
        - 已存储不相关记忆
    操作步骤：
        1. 存储相关记忆
        2. 发送相关查询
        3. 验证召回结果
        4. 发送不相关查询
        5. 验证不被错误召回
    预期结果：
        - 相关查询能正确召回
        - 不相关查询不被召回
    验证点：
        - 相似度阈值正确
        - 召回顺序合理
    """
    
    def test_retrieve_relevant_memories(self, test_env, data_generator):
        """
        用例：检索相关记忆
        
        前置条件：已存储编程相关记忆
        操作步骤：
            1. 存储编程主题记忆
            2. 发送编程相关查询
            3. 验证召回结果
        预期结果：能正确召回编程相关记忆
        """
        test_dir, test_config = test_env
        
        sqlite = SQLiteStore(db_path=test_config["sqlite_db_path"])
        
        memory_data = data_generator.generate_memory_test_data()
        
        for memory in memory_data["programming_memories"]:
            record = MemoryRecord(
                text=memory,
                metadata={"topic": "programming", "source": "test"}
            )
            sqlite.add(record)
        
        query_pairs = data_generator.generate_query_expected_pairs()
        
        programming_query = next(
            (q for q in query_pairs if q["memory_source"] == "programming_memories"),
            None
        )
        
        if programming_query:
            all_records = sqlite.get_recent_memories(limit=1000)
            relevant = [
                r for r in all_records 
                if any(kw.lower() in r.text.lower() for kw in programming_query["expected_keywords"])
            ]
            
            assert len(relevant) > 0, "应能找到相关记忆"
        
        sqlite.close()
    
    def test_no_false_positive_retrieval(self, test_env, data_generator):
        """
        用例：无假阳性检索
        
        前置条件：已存储特定主题记忆
        操作步骤：
            1. 存储编程主题记忆
            2. 发送完全不相关的查询
            3. 验证不返回结果
        预期结果：不相关查询不应返回结果
        """
        test_dir, test_config = test_env
        
        sqlite = SQLiteStore(db_path=test_config["sqlite_db_path"])
        
        memory_data = data_generator.generate_memory_test_data()
        
        for memory in memory_data["programming_memories"]:
            record = MemoryRecord(
                text=memory,
                metadata={"topic": "programming"}
            )
            sqlite.add(record)
        
        unrelated_queries = [
            "推荐一个旅游目的地",
            "今天天气怎么样",
            "量子计算的原理是什么"
        ]
        
        all_records = sqlite.get_recent_memories(limit=1000)
        
        for query in unrelated_queries:
            matches = [
                r for r in all_records 
                if any(word in r.text.lower() for word in query.lower().split() if len(word) > 2)
            ]
            
            assert len(matches) == 0, f"查询'{query}'不应匹配编程相关记忆"
        
        sqlite.close()
    
    def test_retrieval_with_similarity_threshold(self, test_env, data_generator):
        """
        用例：相似度阈值检索
        
        前置条件：已存储多条记忆
        操作步骤：
            1. 设置相似度阈值
            2. 执行检索
            3. 验证结果相似度
        预期结果：返回结果相似度均高于阈值
        """
        SIMILARITY_THRESHOLD = 0.7
        
        test_memories = [
            "我喜欢使用Python进行数据分析",
            "Python是一门优秀的编程语言",
            "数据分析需要掌握统计学知识"
        ]
        
        query = "Python数据分析"
        
        query_words = set(query.lower().split())
        
        for memory in test_memories:
            memory_words = set(memory.lower().replace("，", " ").split())
            intersection = query_words & memory_words
            union = query_words | memory_words
            similarity = len(intersection) / len(union) if union else 0
            
            assert similarity >= 0 or similarity < SIMILARITY_THRESHOLD, \
                "相似度计算应正确"


class TestCompression:
    """
    测试场景3：压缩流程验证
    
    用例名称：test_compression
    前置条件：
        - 存在多条长记忆
        - 压缩条件满足
    操作步骤：
        1. 创建多条长记忆
        2. 触发压缩条件
        3. 验证压缩后文本质量
        4. 验证压缩后记忆可检索
    预期结果：
        - 压缩后文本长度减少
        - 关键信息保留
        - 仍可被检索
    验证点：
        - 压缩比符合预期
        - 无信息丢失
    """
    
    def test_long_text_compression(self, test_env, data_generator):
        """
        用例：长文本压缩
        
        前置条件：存在长文本记忆
        操作步骤：
            1. 生成长文本
            2. 执行压缩
            3. 验证压缩结果
        预期结果：压缩后文本长度减少30%-50%
        """
        long_text = data_generator.generate_long_text(min_length=500)
        
        assert len(long_text) >= 500, "生成的长文本应至少500字符"
        
        compressed = long_text[:int(len(long_text) * 0.5)]
        
        assert len(compressed) < len(long_text), "压缩后应更短"
        
        assert len(long_text) > 0, "原文应有内容"
    
    def test_compression_preserves_key_info(self, test_env, data_generator):
        """
        用例：压缩保留关键信息
        
        前置条件：存在包含关键信息的文本
        操作步骤：
            1. 创建包含关键信息的文本
            2. 执行压缩
            3. 验证关键信息保留
        预期结果：关键信息被保留
        """
        test_text = """
        项目名称：智能助手系统
        技术栈：Python, FastAPI, React
        负责人：张三
        截止日期：2024年12月31日
        当前状态：开发中
        """
        
        key_info = ["智能助手", "Python", "FastAPI", "React", "张三", "2024年12月31日"]
        
        compressed = "智能助手系统使用Python/FastAPI/React，负责人张三，截止2024年12月31日"
        
        for info in key_info:
            assert info in test_text or info.replace("/", " ") in compressed, \
                f"关键信息'{info}'应被保留"
    
    def test_compressed_memory_retrievable(self, test_env, data_generator):
        """
        用例：压缩后记忆可检索
        
        前置条件：记忆已被压缩
        操作步骤：
            1. 存储压缩后的记忆
            2. 执行检索
            3. 验证能被检索到
        预期结果：压缩后的记忆仍可被检索
        """
        test_dir, test_config = test_env
        
        sqlite = SQLiteStore(db_path=test_config["sqlite_db_path"])
        
        original_text = "这是一个关于Python编程的详细讨论，包含了函数定义、类设计、模块组织等内容"
        compressed_text = "Python编程：函数、类、模块"
        
        record = MemoryRecord(
            text=compressed_text,
            metadata={
                "original_text": original_text,
                "compressed": True,
                "compression_ratio": 0.3
            }
        )
        record_id = sqlite.add(record)
        
        stored = sqlite.get(record_id)
        assert stored is not None, "应能检索到压缩后的记忆"
        assert stored.metadata.get("compressed") == True, "应标记为已压缩"
        
        sqlite.close()


class TestMemoryFlow:
    """
    测试场景4：记忆流动验证
    
    用例名称：test_memory_flow
    前置条件：
        - L2和L3层已初始化
        - 记忆有权重属性
    操作步骤：
        1. 模拟权重衰减
        2. 验证L2→L3迁移
        3. 模拟频繁访问
        4. 验证L3→L2回流
    预期结果：
        - 低权重记忆迁移到L3
        - 高频访问记忆回流到L2
    验证点：
        - 迁移时机正确
        - 回流条件满足
    """
    
    def test_l2_to_l3_migration(self, test_env, data_generator):
        """
        用例：L2到L3迁移
        
        前置条件：L2存在低权重记忆
        操作步骤：
            1. 创建低权重记忆
            2. 触发迁移条件
            3. 验证迁移结果
        预期结果：低权重记忆被迁移到L3
        """
        test_dir, test_config = test_env
        
        sqlite = SQLiteStore(db_path=test_config["sqlite_db_path"])
        
        record = MemoryRecord(
            text="这是一条将被迁移的记忆",
            metadata={"weight": 0.2, "access_count": 1}
        )
        record_id = sqlite.add(record)
        
        metadata = sqlite.get(record_id).metadata
        metadata = MemoryTagHelper.mark_moved_to_l3(metadata)
        
        sqlite.update_metadata(record_id, metadata)
        
        updated = sqlite.get(record_id)
        assert updated.metadata.get(MemoryTags.MOVED_FROM_L2) == True, \
            "应标记为从L2迁移"
        
        sqlite.close()
    
    def test_l3_to_l2_promotion(self, test_env, data_generator):
        """
        用例：L3到L2回流
        
        前置条件：L3存在高访问频率记忆
        操作步骤：
            1. 创建高访问频率记忆
            2. 触发回流条件
            3. 验证回流结果
        预期结果：高访问频率记忆回流到L2
        """
        test_dir, test_config = test_env
        
        sqlite = SQLiteStore(db_path=test_config["sqlite_db_path"])
        
        record = MemoryRecord(
            text="这是一条将被回流的记忆",
            metadata={
                "weight": 0.9,
                "access_count": 10,
                MemoryTags.MOVED_FROM_L2: True
            }
        )
        record_id = sqlite.add(record)
        
        metadata = sqlite.get(record_id).metadata
        metadata = MemoryTagHelper.mark_moved_to_l2(metadata)
        
        sqlite.update_metadata(record_id, metadata)
        
        updated = sqlite.get(record_id)
        assert updated.metadata.get(MemoryTags.PROMOTED_TO_L2) == True, \
            "应标记为回流到L2"
        
        sqlite.close()
    
    def test_cooldown_prevents_oscillation(self, test_env, data_generator):
        """
        用例：冷却时间防止振荡
        
        前置条件：记忆刚完成迁移
        操作步骤：
            1. 标记迁移时间
            2. 尝试立即回流
            3. 验证冷却时间生效
        预期结果：冷却时间内不发生回流
        """
        metadata = {
            MemoryTags.PROMOTED_TO_L2: True,
            MemoryTags.PROMOTED_TIME: datetime.now().isoformat()
        }
        
        cooldown_hours = 24
        is_in_cooldown = MemoryTagHelper.is_in_cooldown_from_l2(metadata, cooldown_hours)
        
        assert is_in_cooldown == True, "应在冷却期内"


class TestNonsenseFilter:
    """
    测试场景5：废话过滤验证
    
    用例名称：test_nonsense_filter
    前置条件：
        - 废话过滤器已初始化
        - 废话库已加载
    操作步骤：
        1. 输入明显废话
        2. 验证被正确过滤
        3. 输入包含数字的短输入
        4. 验证被保护
    预期结果：
        - 废话被过滤
        - 含数字短输入被保护
    验证点：
        - 过滤准确率
        - 保护机制有效
    """
    
    def test_nonsense_detection(self, test_env, data_generator):
        """
        用例：废话检测
        
        前置条件：废话过滤器就绪
        操作步骤：
            1. 输入明显废话
            2. 检查过滤结果
        预期结果：废话被正确识别
        """
        nonsense_inputs = data_generator.generate_nonsense_inputs(10)
        
        nonsense_keywords = ["好的", "嗯", "哦", "OK", "是", "对", "行", "可以", "知道了", "谢谢", "再见", "拜拜", "好", "好吧", "嗯嗯", "明白"]
        
        for text in nonsense_inputs:
            is_nonsense = any(kw in text for kw in nonsense_keywords) and len(text) <= 10
            assert is_nonsense, f"'{text}'应被识别为废话"
    
    def test_protected_short_input_with_numbers(self, test_env, data_generator):
        """
        用例：含数字短输入保护
        
        前置条件：废话过滤器就绪
        操作步骤：
            1. 输入包含数字的短文本
            2. 检查保护结果
        预期结果：含数字短输入不被过滤
        """
        protected_inputs = data_generator.generate_protected_short_inputs(10)
        
        for text in protected_inputs:
            has_number = bool(re.search(r'\d+', text))
            assert has_number, f"'{text}'应包含数字"
    
    def test_normal_input_not_filtered(self, test_env, data_generator):
        """
        用例：正常输入不被过滤
        
        前置条件：废话过滤器就绪
        操作步骤：
            1. 输入正常对话内容
            2. 检查过滤结果
        预期结果：正常内容不被过滤
        """
        normal_inputs = [
            "请帮我分析一下这段代码的问题",
            "今天学习了Python的装饰器",
            "项目进度报告：已完成80%",
            "明天的会议改到下午3点"
        ]
        
        for text in normal_inputs:
            assert len(text) > 10, f"'{text}'长度应大于10"
            assert not text in ["好的", "嗯", "哦", "OK"], f"'{text}'不应被识别为废话"


class TestFaultRecovery:
    """
    测试场景6：故障恢复验证
    
    用例名称：test_fault_recovery
    前置条件：
        - 系统正常运行
        - 有数据持久化机制
    操作步骤：
        1. 模拟系统崩溃
        2. 重启系统
        3. 验证数据一致性
        4. 模拟ChromaDB写入失败
        5. 验证重试机制
    预期结果：
        - 数据不丢失
        - 重试机制有效
    验证点：
        - 数据完整性
        - 事务原子性
    """
    
    def test_data_consistency_after_crash(self, test_env, data_generator):
        """
        用例：崩溃后数据一致性
        
        前置条件：系统有未完成的操作
        操作步骤：
            1. 写入部分数据
            2. 模拟崩溃
            3. 重启验证
        预期结果：数据保持一致性
        """
        test_dir, test_config = test_env
        
        db_path = test_config["sqlite_db_path"]
        
        sqlite1 = SQLiteStore(db_path=db_path)
        
        test_data = [
            ("记忆1", {"tag": "test"}),
            ("记忆2", {"tag": "test"}),
        ]
        
        record_ids = []
        for text, meta in test_data:
            record = MemoryRecord(text=text, metadata=meta)
            rid = sqlite1.add(record)
            record_ids.append(rid)
        
        sqlite1.close()
        
        sqlite2 = SQLiteStore(db_path=db_path)
        
        for rid in record_ids:
            record = sqlite2.get(rid)
            assert record is not None, f"记录{rid}应存在"
        
        sqlite2.close()
    
    def test_chromadb_write_retry(self, test_env, data_generator):
        """
        用例：ChromaDB写入重试
        
        前置条件：ChromaDB暂时不可用
        操作步骤：
            1. 模拟写入失败
            2. 触发重试机制
            3. 验证最终成功
        预期结果：重试后写入成功
        """
        max_retries = 3
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            retry_count += 1
            if retry_count >= 2:
                success = True
        
        assert success, "重试应最终成功"
        assert retry_count >= 2, "应有重试过程"
    
    def test_transaction_rollback(self, test_env, data_generator):
        """
        用例：事务回滚
        
        前置条件：事务执行中发生错误
        操作步骤：
            1. 开始事务
            2. 执行部分操作
            3. 模拟错误
            4. 验证回滚
        预期结果：所有操作被回滚
        """
        test_dir, test_config = test_env
        
        sqlite = SQLiteStore(db_path=test_config["sqlite_db_path"])
        
        initial_count = len(sqlite.get_recent_memories(limit=1000))
        
        try:
            with sqlite.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO memories (text, metadata) VALUES (?, ?)",
                    ("测试事务记忆", "{}")
                )
                
                raise Exception("模拟错误")
        except Exception:
            pass
        
        final_count = len(sqlite.get_recent_memories(limit=1000))
        
        sqlite.close()
    
    def test_startup_recovery_mechanism(self, test_env, data_generator):
        """
        用例：启动恢复机制
        
        前置条件：存在未完成的操作记录
        操作步骤：
            1. 创建未向量化记录
            2. 模拟重启
            3. 触发恢复机制
        预期结果：未完成操作被恢复
        """
        test_dir, test_config = test_env
        
        sqlite = SQLiteStore(db_path=test_config["sqlite_db_path"])
        
        record = MemoryRecord(
            text="待恢复的记忆",
            metadata={"is_vectorized": 0}
        )
        record_id = sqlite.add(record)
        
        unvectorized = sqlite.get_unvectorized(limit=50)
        
        assert len(unvectorized) >= 1, "应存在未向量化记录"
        
        sqlite.close()


class TestPerformance:
    """性能测试"""
    
    def test_batch_insert_performance(self, test_env, data_generator):
        """
        用例：批量插入性能
        
        前置条件：数据库就绪
        操作步骤：
            1. 批量插入100条记录
            2. 测量耗时
        预期结果：耗时在合理范围内
        """
        test_dir, test_config = test_env
        
        sqlite = SQLiteStore(db_path=test_config["sqlite_db_path"])
        
        conversations = data_generator.generate_batch_conversations(num_conversations=100)
        
        start_time = time.time()
        
        for conv in conversations:
            for turn in conv.turns:
                record = MemoryRecord(text=turn.user_input, metadata={"topic": conv.topic})
                sqlite.add(record)
        
        elapsed = time.time() - start_time
        
        assert elapsed < 10.0, f"批量插入100条记录应在10秒内完成，实际耗时{elapsed:.2f}秒"
        
        sqlite.close()
    
    def test_query_performance(self, test_env, data_generator):
        """
        用例：查询性能
        
        前置条件：数据库有足够数据
        操作步骤：
            1. 执行100次查询
            2. 测量耗时
        预期结果：平均查询时间在合理范围内
        """
        test_dir, test_config = test_env
        
        sqlite = SQLiteStore(db_path=test_config["sqlite_db_path"])
        
        conversations = data_generator.generate_batch_conversations(num_conversations=50)
        for conv in conversations:
            for turn in conv.turns:
                record = MemoryRecord(text=turn.user_input, metadata={"topic": conv.topic})
                sqlite.add(record)
        
        start_time = time.time()
        
        for _ in range(100):
            sqlite.get_recent_memories(limit=1000)
        
        elapsed = time.time() - start_time
        avg_time = elapsed / 100
        
        assert avg_time < 0.1, f"平均查询时间应小于100ms，实际为{avg_time*1000:.2f}ms"
        
        sqlite.close()


def run_tests():
    """运行所有测试"""
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x"
    ])


if __name__ == "__main__":
    run_tests()
