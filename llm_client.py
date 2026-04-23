# llm_client.py
import requests
import json
import time
import threading
from typing import Dict, List, Optional
from collections import deque
from config import config


class CircuitBreaker:
    """
    熔断器实现
    
    状态：
    - CLOSED: 正常状态，允许请求
    - OPEN: 熔断状态，拒绝请求
    - HALF_OPEN: 半开状态，允许探测请求
    """
    
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"
    
    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 300.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.state = self.CLOSED
        self.last_failure_time = 0.0
        self._lock = threading.Lock()
    
    def can_execute(self) -> bool:
        """检查是否可以执行请求"""
        with self._lock:
            if self.state == self.CLOSED:
                return True
            elif self.state == self.OPEN:
                if time.time() - self.last_failure_time >= self.reset_timeout:
                    self.state = self.HALF_OPEN
                    return True
                return False
            else:
                return True
    
    def record_success(self):
        """记录成功"""
        with self._lock:
            self.failure_count = 0
            self.state = self.CLOSED
    
    def record_failure(self):
        """记录失败"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = self.OPEN
    
    def get_state(self) -> str:
        """获取当前状态"""
        with self._lock:
            return self.state


class AdaptiveTimeout:
    """
    自适应超时管理器
    
    根据历史响应时间动态调整超时值
    """
    
    def __init__(self, initial_timeout: float = 120.0, min_timeout: float = 10.0, max_timeout: float = 300.0):
        self.initial_timeout = initial_timeout
        self.min_timeout = min_timeout
        self.max_timeout = max_timeout
        self.response_times: deque = deque(maxlen=20)
        self._lock = threading.Lock()
    
    def record_response_time(self, duration: float):
        """记录响应时间"""
        with self._lock:
            self.response_times.append(duration)
    
    def get_timeout(self) -> float:
        """
        计算自适应超时值
        
        策略：
        - 基于历史 P95 响应时间的 2 倍
        - 限制在 [min_timeout, max_timeout] 范围内
        """
        with self._lock:
            if not self.response_times:
                return self.initial_timeout
            
            sorted_times = sorted(self.response_times)
            n = len(sorted_times)
            p95_index = int(n * 0.95)
            p95_time = sorted_times[min(p95_index, n - 1)]
            
            adaptive_timeout = p95_time * 2
            return max(self.min_timeout, min(adaptive_timeout, self.max_timeout))
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        with self._lock:
            if not self.response_times:
                return {"count": 0, "avg": 0, "current_timeout": self.initial_timeout}
            
            return {
                "count": len(self.response_times),
                "avg": sum(self.response_times) / len(self.response_times),
                "current_timeout": self.get_timeout()
            }


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
        
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.async_processor.circuit_breaker_threshold,
            reset_timeout=config.async_processor.circuit_breaker_reset_timeout
        )
        self.adaptive_timeout = AdaptiveTimeout(
            initial_timeout=config.async_processor.llm_timeout,
            min_timeout=10.0,
            max_timeout=300.0
        )
        self._slow_response_threshold = 60.0
    
    def chat(self, messages: List[Dict[str, str]], max_tokens: int = None) -> str:
        """发送对话请求 - 使用标准 OpenAI 格式"""
        
        if not self.circuit_breaker.can_execute():
            return self._fallback_chat(messages, max_tokens, skip_circuit=True)
        
        if max_tokens is None:
            max_tokens = self.max_output_tokens
        
        payload = {
            "model": "local",
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens
        }
        
        timeout = self.adaptive_timeout.get_timeout()
        start_time = time.time()
        
        try:
            response = requests.post(self.chat_api_url, json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            
            duration = time.time() - start_time
            self.adaptive_timeout.record_response_time(duration)
            self.circuit_breaker.record_success()
            
            if duration > self._slow_response_threshold:
                print(f"[LLM 警告] 响应时间过长: {duration:.1f}s (超时阈值: {timeout:.1f}s)")
            
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0].get("message", {}).get("content", "")
            else:
                content = result.get("content", "") or result.get("text", "")
            
            if not content:
                print(f"API 响应为空，完整响应: {json.dumps(result, ensure_ascii=False, indent=2)}")
            
            return content.strip()
        except requests.exceptions.Timeout as e:
            duration = time.time() - start_time
            self.circuit_breaker.record_failure()
            print(f"[LLM 超时] 请求超时 ({timeout:.1f}s)，切换降级模式")
            return self._fallback_chat(messages, max_tokens)
        except requests.exceptions.RequestException as e:
            self.circuit_breaker.record_failure()
            print(f"LLM API 调用失败: {e}")
            return self._fallback_chat(messages, max_tokens)
    
    def _fallback_chat(self, messages: List[Dict[str, str]], max_tokens: int, skip_circuit: bool = False) -> str:
        """降级方案：使用 /completion 接口手动构建 prompt"""
        if not skip_circuit and not self.circuit_breaker.can_execute():
            return "抱歉，服务暂时不可用，请稍后重试。"
        
        prompt = self._build_prompt(messages)
        
        payload = {
            "prompt": prompt,
            "temperature": self.temperature,
            "n_predict": max_tokens,
            "stop": ["<|im_end|>", "<|im_start|>", "</s>"],
            "echo": False
        }
        
        timeout = min(self.adaptive_timeout.get_timeout() * 0.5, 60.0)
        start_time = time.time()
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            
            duration = time.time() - start_time
            self.adaptive_timeout.record_response_time(duration)
            self.circuit_breaker.record_success()
            
            content = result.get("content", "") or result.get("text", "")
            return content.strip()
        except requests.exceptions.RequestException as e:
            self.circuit_breaker.record_failure()
            return f"抱歉，本地模型调用失败：{str(e)}"
    
    def get_circuit_breaker_state(self) -> str:
        """获取熔断器状态"""
        return self.circuit_breaker.get_state()
    
    def get_timeout_stats(self) -> dict:
        """获取超时统计"""
        return self.adaptive_timeout.get_stats()
    
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
        base_url = self.api_url.replace("/completion", "")
        
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"[LLM] 连接成功: {base_url}/health")
                return True
        except requests.exceptions.ConnectionError:
            print(f"[LLM] 连接失败: 无法连接到 {base_url}，请确保 llama.cpp 服务已启动")
        except requests.exceptions.Timeout:
            print(f"[LLM] 连接超时: {base_url}/health 响应超过5秒")
        except requests.exceptions.RequestException as e:
            print(f"[LLM] 健康检查失败: {e}")
        
        try:
            test_messages = [{"role": "user", "content": "Hi"}]
            test_payload = {"model": "local", "messages": test_messages, "max_tokens": 1}
            response = requests.post(self.chat_api_url, json=test_payload, timeout=5)
            if response.status_code == 200:
                print(f"[LLM] Chat API 连接成功: {self.chat_api_url}")
                return True
            else:
                print(f"[LLM] Chat API 返回错误: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"[LLM] Chat API 连接失败: {self.chat_api_url}")
        except requests.exceptions.RequestException as e:
            print(f"[LLM] Chat API 测试失败: {e}")
        
        try:
            test_payload = {"prompt": "Hello", "n_predict": 1}
            response = requests.post(self.api_url, json=test_payload, timeout=5)
            if response.status_code == 200:
                print(f"[LLM] Completion API 连接成功: {self.api_url}")
                return True
            else:
                print(f"[LLM] Completion API 返回错误: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"[LLM] Completion API 测试失败: {e}")
        
        print(f"[LLM] 所有连接尝试失败，请检查:")
        print(f"  1. llama.cpp 服务是否已启动")
        print(f"  2. API URL 配置是否正确: {self.api_url}")
        print(f"  3. 端口是否被防火墙阻止")
        return False
