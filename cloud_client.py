# cloud_client.py
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from config import config
import re


def validate_api_key(provider: str, api_key: str) -> Tuple[bool, str]:
    """
    验证 API 密钥格式
    
    Args:
        provider: 提供商 (openai, gemini)
        api_key: API 密钥
    
    Returns:
        (是否有效, 错误信息)
    """
    if not api_key:
        return False, "API 密钥为空"
    
    if provider.lower() == "openai":
        if not api_key.startswith("sk-"):
            return False, "OpenAI API 密钥应以 'sk-' 开头"
        if len(api_key) < 20:
            return False, "OpenAI API 密钥长度不足"
        if not re.match(r'^sk-[A-Za-z0-9_-]+$', api_key):
            return False, "OpenAI API 密钥格式无效"
    
    elif provider.lower() == "gemini":
        if len(api_key) < 20:
            return False, "Gemini API 密钥长度不足"
        if not re.match(r'^[A-Za-z0-9_-]+$', api_key):
            return False, "Gemini API 密钥格式无效"
    
    return True, ""


class CloudAIClient(ABC):
    """云端AI客户端抽象基类"""
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 8192) -> str:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass


class OpenAIClient(CloudAIClient):
    """OpenAI API 客户端 - 使用官方SDK"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", base_url: str = None):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self._client = None
        self._init_client()
    
    def _init_client(self):
        """初始化OpenAI客户端"""
        try:
            from openai import OpenAI
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self._client = OpenAI(**client_kwargs)
        except ImportError:
            print("未安装 openai 库，请运行: pip install openai")
        except Exception as e:
            print(f"OpenAI 客户端初始化失败: {e}")
    
    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 8192) -> str:
        """发送对话请求"""
        if not self._client:
            return ""
        
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except Exception as e:
            print(f"OpenAI API 调用失败: {e}")
            return ""
    
    def is_available(self) -> bool:
        """检查API是否可用"""
        return self._client is not None and bool(self.api_key)


class GeminiClient(CloudAIClient):
    """Google Gemini API 客户端 - 使用新版SDK"""
    
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model = model
        self._client = None
        self._init_client()
    
    def _init_client(self):
        """初始化Gemini客户端"""
        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        except ImportError:
            print("未安装 google-genai 库，请运行: pip install google-genai")
        except Exception as e:
            print(f"Gemini 客户端初始化失败: {e}")
    
    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 8192) -> str:
        """发送对话请求"""
        if not self._client:
            return ""
        
        try:
            from google.genai import types
            
            system_instruction = None
            contents = []
            
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if role == "system":
                    system_instruction = content
                elif role == "user":
                    contents.append(
                        types.Content(
                            role="user",
                            parts=[types.Part(text=content)]
                        )
                    )
                elif role == "assistant":
                    contents.append(
                        types.Content(
                            role="model",
                            parts=[types.Part(text=content)]
                        )
                    )
            
            generate_config = types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=0.7,
                system_instruction=system_instruction
            )
            
            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=generate_config
            )
            
            if response.text:
                return response.text.strip()
            return ""
        except Exception as e:
            print(f"Gemini API 调用失败: {e}")
            return ""
    
    def is_available(self) -> bool:
        """检查API是否可用"""
        return self._client is not None and bool(self.api_key)


class CloudClientFactory:
    """云端客户端工厂"""
    
    @staticmethod
    def create(provider: str, api_key: str, model: str = None, base_url: str = None) -> Optional[CloudAIClient]:
        is_valid, error_msg = validate_api_key(provider, api_key)
        if not is_valid:
            print(f"[安全] API 密钥验证失败: {error_msg}")
            return None
        
        if provider.lower() == "openai":
            return OpenAIClient(api_key, model or "gpt-4o-mini", base_url)
        elif provider.lower() == "gemini":
            return GeminiClient(api_key, model or "gemini-2.5-flash")
        else:
            print(f"不支持的云端AI提供商: {provider}")
            return None


class HybridClient:
    """混合客户端：本地LLM + 云端AI"""
    
    def __init__(self, local_client, cloud_client: Optional[CloudAIClient] = None):
        self.local_client = local_client
        self.cloud_client = cloud_client
        self._sensitive_filter = None
        self._filter_enabled = True
        self._init_sensitive_filter()
        self.cloud_system_prompt = """你是一个智能助手，负责基于用户问题和历史上下文提供高质量的回答。

【任务说明】
1. 用户会提供【历史上下文】和【当前问题】
2. 历史上下文是本地AI对之前对话的压缩摘要
3. 你需要结合历史上下文，回答用户的当前问题

【回答要求】
1. 提供完整、详细、有帮助的回答
2. 如果历史上下文与问题相关，可以参考其中的信息
3. 如果历史上下文与问题无关，直接基于你的知识回答
4. 保持自然、友好的对话风格"""
    
    def _init_sensitive_filter(self):
        """初始化敏感信息过滤器"""
        try:
            from sensitive_filter import get_sensitive_filter
            self._sensitive_filter = get_sensitive_filter()
            self._filter_enabled = config.privacy.sensitive_filter_enabled
            
            excluded_keywords = []
            if config.privacy.excluded_keywords:
                excluded_keywords = [
                    kw.strip() 
                    for kw in config.privacy.excluded_keywords.split(",") 
                    if kw.strip()
                ]
            
            self._sensitive_filter.configure(
                enabled=self._filter_enabled,
                excluded_keywords=excluded_keywords
            )
        except ImportError:
            print("[安全] 敏感信息过滤器未安装")
            self._sensitive_filter = None
    
    def _mask_sensitive(self, text: str) -> Tuple[str, Dict]:
        """
        脱敏处理
        
        Args:
            text: 原始文本
        
        Returns:
            (脱敏后文本, 检测统计)
        """
        if not self._filter_enabled or not self._sensitive_filter:
            return text, {}
        
        return self._sensitive_filter.mask(text)
    
    def process(
        self, 
        user_input: str, 
        memory_context: str, 
        local_response: str,
        metadata: Dict = None
    ) -> str:
        """
        处理流程：
        1. 本地LLM已返回压缩结果
        2. 敏感信息脱敏（如启用）
        3. 将用户输入 + 本地结果发送到云端
        4. 返回云端结果
        
        Args:
            user_input: 用户原始输入
            memory_context: 记忆上下文
            local_response: 本地LLM响应
            metadata: 元数据（包含来源、时间戳等）
        """
        if not self.cloud_client or not self.cloud_client.is_available():
            return local_response
        
        messages = [
            {"role": "system", "content": self.cloud_system_prompt}
        ]
        
        context_parts = []
        total_detected = {}
        
        if metadata:
            meta_info = self._format_metadata(metadata)
            if meta_info:
                context_parts.append(f"【上下文元数据】\n{meta_info}")
        
        if memory_context:
            masked_context, detected = self._mask_sensitive(memory_context)
            context_parts.append(f"【历史上下文】\n{masked_context}")
            total_detected.update(detected)
        
        masked_input, detected = self._mask_sensitive(user_input)
        context_parts.append(f"【当前问题】\n{masked_input}")
        total_detected.update(detected)
        
        masked_response, detected = self._mask_sensitive(local_response)
        context_parts.append(f"【本地参考】\n{masked_response}")
        total_detected.update(detected)
        
        if total_detected and config.privacy.log_sensitive_detection:
            print(f"[安全] 检测到敏感信息: {total_detected}")
        
        user_message = "\n\n".join(context_parts)
        messages.append({"role": "user", "content": user_message})
        
        cloud_response = self.cloud_client.chat(messages)
        
        if cloud_response:
            return cloud_response
        else:
            return local_response
    
    def _format_metadata(self, metadata: Dict) -> str:
        """
        格式化元数据为可读字符串
        
        提取关键标识符：
        - 来源（L1/L2/L3）
        - 时间戳
        - 记忆数量
        """
        if not metadata:
            return ""
        
        parts = []
        
        sources = metadata.get("sources", [])
        if sources:
            unique_sources = list(set(sources))
            parts.append(f"记忆来源: {', '.join(unique_sources)}")
        
        timestamps = metadata.get("timestamps", [])
        if timestamps:
            latest = max(timestamps) if timestamps else None
            if latest:
                parts.append(f"最近记忆时间: {latest}")
        
        memory_count = metadata.get("memory_count", 0)
        if memory_count > 0:
            parts.append(f"检索记忆数: {memory_count}")
        
        context_used = metadata.get("context_used", 0)
        if context_used > 0:
            parts.append(f"上下文长度: {context_used}字符")
        
        return "\n".join(parts) if parts else ""
    
    def direct_chat(self, user_input: str) -> str:
        """直接使用云端AI回答（无记忆时）"""
        if not self.cloud_client or not self.cloud_client.is_available():
            return ""
        
        masked_input, detected = self._mask_sensitive(user_input)
        
        if detected and config.privacy.log_sensitive_detection:
            print(f"[安全] 检测到敏感信息: {detected}")
        
        messages = [
            {"role": "system", "content": "你是一个智能助手，请提供完整、详细、有帮助的回答。"},
            {"role": "user", "content": masked_input}
        ]
        
        return self.cloud_client.chat(messages)
    
    def is_cloud_available(self) -> bool:
        """检查云端是否可用"""
        return self.cloud_client is not None and self.cloud_client.is_available()
