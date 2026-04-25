# conftest.py
# pytest 共享配置和固件
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """测试数据目录"""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture
def mock_all_dependencies():
    """
    模拟所有外部依赖
    
    用于需要完全隔离环境的测试
    """
    mocks = {}
    
    mocks['config'] = MagicMock()
    mocks['config'].sqlite_enabled = True
    mocks['config'].chroma_persist_dir = "/tmp/test_chroma"
    mocks['config'].onnx_model_path = "/tmp/test_onnx"
    mocks['config'].embedding_dimension = 768
    
    return mocks


from unittest.mock import MagicMock
