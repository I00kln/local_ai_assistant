# llm_client.py
import requests
import json
from typing import Optional, Dict, Any, List
from config import config

class LlamaClient:
    """llama.cpp API 客户端 - 使用 /v1/chat/completions 接口"""
    
    def __init__(self):
        self.api_url = config.local.api_url
        self.max_context = config.local.max_context
        self.temperature = config.local.temperature
        self.max_output_tokens = config.local.max_output_tokens
        
        if "/completion" in self.api_url:
            self.chat_api_url = self.api_url.replace("/completion", "/v1/chat/completions")
        else:
            self.chat_api_url = self.api_url.rstrip("/") + "/v1/chat/completions"
    
    def chat(self, messages: List[Dict[str, str]], max_tokens: int = None) -> str:
        """发送对话请求 - 使用标准 OpenAI 格式"""
        
        if max_tokens is None:
            max_tokens = self.max_output_tokens
        
        payload = {
            "model": "local",
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(self.chat_api_url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0].get("message", {}).get("content", "")
            else:
                content = result.get("content", "") or result.get("text", "")
            
            if not content:
                print(f"API 响应为空，完整响应: {json.dumps(result, ensure_ascii=False, indent=2)}")
            
            return content.strip()
        except requests.exceptions.RequestException as e:
            print(f"LLM API 调用失败: {e}")
            return self._fallback_chat(messages, max_tokens)
    
    def _fallback_chat(self, messages: List[Dict[str, str]], max_tokens: int) -> str:
        """降级方案：使用 /completion 接口手动构建 prompt"""
        prompt = self._build_prompt(messages)
        
        payload = {
            "prompt": prompt,
            "temperature": self.temperature,
            "n_predict": max_tokens,
            "stop": ["<|im_end|>", "<|im_start|>", "</s>"],
            "echo": False
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            content = result.get("content", "") or result.get("text", "")
            return content.strip()
        except requests.exceptions.RequestException as e:
            return f"抱歉，本地模型调用失败：{str(e)}"
    
    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        """构建 ChatML 格式的 prompt（降级方案使用）"""
        prompt_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>\n")
            elif role == "user":
                prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
            elif role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")
        
        prompt_parts.append("<|im_start|>assistant\n")
        return "".join(prompt_parts)
    
    def check_connection(self) -> bool:
        """检查 llama.cpp 服务是否可用"""
        try:
            response = requests.get(self.api_url.replace("/completion", "/health"), timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            try:
                test_messages = [{"role": "user", "content": "Hi"}]
                test_payload = {"model": "local", "messages": test_messages, "max_tokens": 1}
                response = requests.post(self.chat_api_url, json=test_payload, timeout=5)
                return response.status_code == 200
            except requests.exceptions.RequestException:
                try:
                    test_payload = {"prompt": "Hello", "n_predict": 1}
                    response = requests.post(self.api_url, json=test_payload, timeout=5)
                    return response.status_code == 200
                except requests.exceptions.RequestException:
                    return False
