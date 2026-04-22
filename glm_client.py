# glm_client.py
# GLM (智谱AI) 客户端 - 使用 zai-sdk
from typing import Dict, List, Optional
import re


class GLMClient:
    """
    GLM (智谱AI) API 客户端
    
    使用 zai-sdk 调用智谱AI的GLM系列模型
    支持模型：glm-4, glm-4-flash, glm-4-plus, glm-4-air 等
    
    安装依赖：
        pip install zai-sdk
    
    获取API Key：
        https://open.bigmodel.cn/
    
    API 端点：
        https://open.bigmodel.cn/api/paas/v4
    """
    
    def __init__(self, api_key: str, model: str = "glm-4-flash", base_url: str = None):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or "https://open.bigmodel.cn/api/paas/v4"
        self._client = None
        self._init_client()
    
    def _init_client(self):
        """初始化GLM客户端"""
        if not self.api_key:
            print("[GLM] API 密钥为空，客户端不可用")
            return
        
        try:
            from zhipuai import ZhipuAI
            self._client = ZhipuAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            print(f"[GLM] 客户端初始化成功，模型: {self.model}")
        except ImportError:
            print("[GLM] 未安装智谱AI SDK，请运行: pip install zhipuai 或 pip install zai-sdk")
        except Exception as e:
            print(f"[GLM] 客户端初始化失败: {e}")
    
    def _try_zai_sdk(self):
        """尝试使用 zai-sdk 作为备选（已废弃，逻辑已合并到_init_client）"""
        pass
    
    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 8192) -> str:
        """
        发送对话请求
        
        Args:
            messages: 消息列表，格式 [{"role": "user/assistant/system", "content": "..."}]
            max_tokens: 最大输出token数
        
        Returns:
            模型响应文本
        """
        if not self._client:
            return ""
        
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
            )
            
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                return content.strip() if content else ""
            return ""
        except Exception as e:
            print(f"[GLM] API 调用失败: {e}")
            return ""
    
    def is_available(self) -> bool:
        """检查API是否可用"""
        return self._client is not None and bool(self.api_key)
    
    def get_model_info(self) -> Dict[str, str]:
        """获取当前模型信息"""
        return {
            "provider": "glm",
            "model": self.model,
            "available": self.is_available()
        }


def validate_glm_api_key(api_key: str) -> tuple:
    """
    验证 GLM API 密钥格式
    
    Args:
        api_key: API 密钥
    
    Returns:
        (是否有效, 错误信息)
    """
    if not api_key:
        return False, "GLM API 密钥为空"
    
    if len(api_key) < 20:
        return False, "GLM API 密钥长度不足"
    
    if not re.match(r'^[A-Za-z0-9._-]+$', api_key):
        return False, "GLM API 密钥格式无效"
    
    return True, ""


def create_glm_client(api_key: str, model: str = None, base_url: str = None) -> Optional[GLMClient]:
    """
    创建 GLM 客户端实例
    
    Args:
        api_key: GLM API 密钥
        model: 模型名称，默认 glm-4-flash
        base_url: API 端点，默认 https://open.bigmodel.cn/api/paas/v4
    
    Returns:
        GLMClient 实例，失败返回 None
    """
    is_valid, error_msg = validate_glm_api_key(api_key)
    if not is_valid:
        print(f"[GLM] API 密钥验证失败: {error_msg}")
        return None
    
    return GLMClient(api_key, model or "glm-4-flash", base_url)
