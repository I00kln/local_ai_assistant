# chat_window.py
import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any

from config import config
from models import UIState
from ui_state import UIStateManager


class ChatWindow:
    """本地AI助理对话窗口 - 支持本地+云端混合模式"""
    
    def __init__(self):
        self.memory = None
        self.llm = None
        self.context_builder = None
        self.async_processor = None
        self.hybrid_client = None
        
        self.conversation_history = []
        self._initialized = False
        self._closing = False
        self._after_ids: List[str] = []
        
        self.window = tk.Tk()
        self.window.title("本地AI助理 - ChromaDB + SQLite")
        self.window.geometry("800x600")
        
        self._setup_ui()
        
        self.state_manager = UIStateManager(self.status_label, self.memory_count_label)
        self.state_manager.set_state(UIState.INITIALIZING, "正在加载组件...")
        
        after_id = self.window.after(100, self._async_init)
        self._after_ids.append(after_id)
        
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _setup_ui(self):
        """设置界面"""
        self.status_frame = ttk.Frame(self.window)
        self.status_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.status_label = ttk.Label(self.status_frame, text="状态: 正在加载组件...")
        self.status_label.pack(side=tk.LEFT)
        
        self.memory_count_label = ttk.Label(self.status_frame, text="记忆数: 0")
        self.memory_count_label.pack(side=tk.RIGHT)
        
        self.chat_display = scrolledtext.ScrolledText(
            self.window, 
            wrap=tk.WORD, 
            font=("Microsoft YaHei", 10),
            state=tk.DISABLED
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.chat_display.tag_config("user", foreground="#0066CC", font=("Microsoft YaHei", 10, "bold"))
        self.chat_display.tag_config("assistant", foreground="#009900", font=("Microsoft YaHei", 10))
        self.chat_display.tag_config("cloud", foreground="#990099", font=("Microsoft YaHei", 10))
        self.chat_display.tag_config("system", foreground="#666666", font=("Microsoft YaHei", 9, "italic"))
        self.chat_display.tag_config("memory", foreground="#996600", font=("Microsoft YaHei", 9))
        
        self.input_frame = ttk.Frame(self.window)
        self.input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.input_entry = tk.Text(self.input_frame, height=3, font=("Microsoft YaHei", 10))
        self.input_entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.input_entry.bind("<Control-Return>", lambda e: self._send_message())
        
        self.send_button = ttk.Button(
            self.input_frame, 
            text="发送 (Ctrl+Enter)", 
            command=self._send_message,
            state=tk.DISABLED
        )
        self.send_button.pack(side=tk.RIGHT, padx=(5, 0))
        
        self.control_frame = ttk.Frame(self.window)
        self.control_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.clear_btn = ttk.Button(self.control_frame, text="清空对话", command=self._clear_conversation, state=tk.DISABLED)
        self.clear_btn.pack(side=tk.LEFT, padx=2)
        
        self.save_btn = ttk.Button(self.control_frame, text="立即保存记忆", command=self._force_save, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=2)
        
        self.metrics_btn = ttk.Button(self.control_frame, text="系统监控", command=self._show_metrics, state=tk.DISABLED)
        self.metrics_btn.pack(side=tk.LEFT, padx=2)
        
        self.show_memory_enabled = tk.BooleanVar(value=True)
        self.show_memory_check = ttk.Checkbutton(
            self.control_frame, 
            text="显示检索记忆", 
            variable=self.show_memory_enabled,
            state=tk.DISABLED
        )
        self.show_memory_check.pack(side=tk.LEFT, padx=2)
        
        self.show_local_enabled = tk.BooleanVar(value=True)
        self.show_local_check = ttk.Checkbutton(
            self.control_frame, 
            text="显示本地结果", 
            variable=self.show_local_enabled,
            state=tk.DISABLED
        )
        self.show_local_check.pack(side=tk.LEFT, padx=2)
        
        self.local_var = tk.BooleanVar(value=config.local.enabled)
        self.local_btn = ttk.Checkbutton(
            self.control_frame, 
            text="启用本地压缩", 
            variable=self.local_var,
            state=tk.DISABLED
        )
        self.local_btn.pack(side=tk.LEFT, padx=2)
        
        self.cloud_var = tk.BooleanVar(value=config.cloud.enabled)
        self.cloud_btn = ttk.Checkbutton(
            self.control_frame, 
            text="启用云端AI", 
            variable=self.cloud_var,
            state=tk.DISABLED
        )
        self.cloud_btn.pack(side=tk.LEFT, padx=2)
        
        # 废话过滤器控制
        self.nonsense_var = tk.BooleanVar(value=config.nonsense_filter_enabled)
        self.nonsense_btn = ttk.Checkbutton(
            self.control_frame, 
            text="废话过滤", 
            variable=self.nonsense_var,
            state=tk.DISABLED
        )
        self.nonsense_btn.pack(side=tk.LEFT, padx=2)
        
        self.stats_btn = ttk.Button(
            self.control_frame, 
            text="过滤统计", 
            command=self._show_filter_stats,
            state=tk.DISABLED
        )
        self.stats_btn.pack(side=tk.LEFT, padx=2)
        
        self._append_message("system", "正在初始化系统组件，请稍候...")
    
    def _safe_after(self, delay: int, callback) -> Optional[str]:
        """
        安全的 window.after 调用
        
        Args:
            delay: 延迟毫秒数
            callback: 回调函数
        
        Returns:
            after_id 或 None（如果窗口正在关闭）
        """
        if self._closing:
            return None
        
        try:
            after_id = self.window.after(delay, callback)
            self._after_ids.append(after_id)
            return after_id
        except tk.TclError:
            return None
    
    def _cancel_all_after(self):
        """取消所有待执行的 after 回调"""
        for after_id in self._after_ids:
            try:
                self.window.after_cancel(after_id)
            except tk.TclError:
                pass
        self._after_ids.clear()
    
    def _async_init(self):
        """后台异步初始化组件"""
        def init_components():
            if self._closing:
                return
            
            try:
                self._safe_after(0, lambda: self._append_message("system", "初始化记忆系统..."))
                
                from memory_manager import get_memory_manager
                self.memory = get_memory_manager()
                
                if self._closing:
                    return
                
                l2_count = len(self.memory.vector_store)
                l3_count = self.memory.sqlite.count() if self.memory.sqlite else 0
                self._safe_after(0, lambda msg=f"L2向量库: {l2_count}条, L3数据库: {l3_count}条": 
                    self._append_message("system", f"记忆库加载完成 - {msg}"))
                
                self.state_manager.set_memory_counts(l2_count, l3_count)
                
                from llm_client import LlamaClient
                self.llm = LlamaClient()
                
                from context_builder import ContextBuilder
                self.context_builder = ContextBuilder(self.memory)
                
                from async_processor import AsyncMemoryProcessor
                self.async_processor = AsyncMemoryProcessor()
                self.async_processor.start()
                
                self._init_cloud_client()
                
                self._initialized = True
                
                self._safe_after(0, self._enable_ui)
                self._safe_after(0, self._check_connections)
                
            except Exception as e:
                err_msg = str(e)
                self._safe_after(0, lambda msg=err_msg: self._append_message("system", f"初始化失败: {msg}"))
                self._safe_after(0, lambda: self.state_manager.set_state(UIState.ERROR, "初始化失败"))
        
        self._init_thread = threading.Thread(target=init_components, daemon=True)
        self._init_thread.start()
    
    def _init_cloud_client(self):
        """初始化云端客户端"""
        if not config.cloud.api_key or config.cloud.api_key == "your-api-key-here":
            self._safe_after(0, lambda: self._append_message(
                "system", "未配置云端API密钥，云端AI不可用"
            ))
            return
        
        try:
            from cloud_client import CloudClientFactory, HybridClient
            
            cloud_client = CloudClientFactory.create(
                provider=config.cloud.provider,
                api_key=config.cloud.api_key,
                model=config.cloud.model,
                base_url=config.cloud.base_url
            )
            
            if cloud_client and cloud_client.is_available():
                self.hybrid_client = HybridClient(self.llm, cloud_client)
                self._safe_after(0, lambda: self.cloud_btn.config(state=tk.NORMAL))
                
                provider = config.cloud.provider
                model = config.cloud.model
                
                if config.cloud.enabled:
                    self._safe_after(0, lambda: self.cloud_var.set(True))
                    self._safe_after(0, lambda p=provider, m=model: self._append_message(
                        "system", f"云端AI已启用: {p} ({m})"
                    ))
                else:
                    self._safe_after(0, lambda p=provider, m=model: self._append_message(
                        "system", f"云端AI可用: {p} ({m})，可手动启用"
                    ))
        except Exception as e:
            err_msg = str(e)
            self._safe_after(0, lambda msg=err_msg: self._append_message("system", f"云端AI初始化失败: {msg}"))
    
    def _enable_ui(self):
        """启用 UI 控件"""
        if self._closing:
            return
        
        self.send_button.config(state=tk.NORMAL)
        self.clear_btn.config(state=tk.NORMAL)
        self.show_memory_check.config(state=tk.NORMAL)
        self.show_local_check.config(state=tk.NORMAL)
        self.local_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.NORMAL)
        self.metrics_btn.config(state=tk.NORMAL)
        self.nonsense_btn.config(state=tk.NORMAL)
        self.stats_btn.config(state=tk.NORMAL)
        
        l2_count = len(self.memory.vector_store) if self.memory else 0
        l3_count = self.memory.sqlite.count() if self.memory and self.memory.sqlite else 0
        self.state_manager.set_memory_counts(l2_count, l3_count)
        self.state_manager.set_state(UIState.IDLE)
        self._append_message("system", "系统初始化完成，可以开始对话！")
    
    def _show_filter_stats(self):
        """显示废话过滤器统计信息"""
        try:
            from nonsense_filter import get_nonsense_filter
            stats = get_nonsense_filter().get_stats()
            
            stats_text = f"""
📊 废话过滤统计
━━━━━━━━━━━━━━━━━━━━━
总检查对话数: {stats['total_checked']}
规则过滤: {stats['rule_filtered']} 条
密度过滤: {stats['density_filtered']} 条
向量过滤: {stats['vector_filtered']} 条
通过存储: {stats['passed']} 条
过滤率: {stats.get('filter_rate', 0)}%
━━━━━━━━━━━━━━━━━━━━━
            """.strip()
            
            self._append_message("system", stats_text)
        except Exception as e:
            self._append_message("system", f"获取过滤统计失败: {e}")
    
    def _check_connections(self):
        """检查连接状态（异步）"""
        if self._closing:
            return
        
        def check_in_background():
            local_ok = False
            cloud_ok = False
            
            try:
                local_ok = self.llm and self.llm.check_connection()
            except Exception:
                pass
            
            try:
                cloud_ok = self.hybrid_client and self.hybrid_client.is_cloud_available()
            except Exception:
                pass
            
            status_parts = []
            if local_ok:
                status_parts.append("本地LLM已连接")
            else:
                status_parts.append("本地LLM未连接")
            
            if cloud_ok:
                status_parts.append(f"云端AI({config.cloud.provider})可用")
            
            self._safe_after(0, lambda: self.state_manager.add_status_message(" | ".join(status_parts)))
            
            if not local_ok:
                self._safe_after(0, lambda: self._append_message("system", "无法连接到 llama.cpp，请确保服务已启动"))
        
        check_thread = threading.Thread(target=check_in_background, daemon=True)
        check_thread.start()
    
    def _send_message(self):
        """发送用户消息"""
        if not self._initialized or self._closing:
            return
        
        user_input = self.input_entry.get("1.0", tk.END).strip()
        if not user_input:
            return
        
        self.input_entry.delete("1.0", tk.END)
        
        self._append_message("user", user_input)
        
        self.send_button.config(state=tk.DISABLED)
        
        use_local = self.local_var.get() and self.llm is not None
        use_cloud = self.cloud_var.get() and self.hybrid_client is not None
        show_memory = self.show_memory_enabled.get()
        show_local = self.show_local_enabled.get()
        
        if use_local and use_cloud:
            self.state_manager.set_state(UIState.PROCESSING_HYBRID, "本地+云端处理中")
        elif use_cloud:
            self.state_manager.set_state(UIState.PROCESSING_CLOUD, "云端处理中")
        elif use_local:
            self.state_manager.set_state(UIState.PROCESSING_LOCAL, "本地处理中")
        else:
            self.state_manager.set_state(UIState.ERROR, "无可用AI服务")
            self._safe_after(0, lambda: self.send_button.config(state=tk.NORMAL))
            return
        
        thread = threading.Thread(target=self._process_message, args=(user_input, use_local, use_cloud, show_memory, show_local), daemon=True)
        thread.start()
    
    def _build_retrieval_metadata(self, memories: List[Dict], memory_context: str) -> Dict[str, Any]:
        """
        构建检索元数据
        
        Args:
            memories: 检索到的记忆列表
            memory_context: 记忆上下文字符串
        
        Returns:
            元数据字典，包含来源、时间戳、数量等信息
        """
        if not memories:
            return {}
        
        sources = [m.get("source", "?") for m in memories]
        timestamps = [m.get("timestamp", "") for m in memories if m.get("timestamp")]
        
        return {
            "sources": sources,
            "timestamps": timestamps,
            "memory_count": len(memories),
            "context_used": len(memory_context) if memory_context else 0,
            "avg_similarity": sum(m.get("similarity", 0) for m in memories) / len(memories) if memories else 0
        }
    
    def _process_message(self, user_input: str, use_local: bool, use_cloud: bool, show_memory: bool = True, show_local: bool = True):
        """处理用户消息（后台线程）"""
        if self._closing:
            self._safe_after(0, lambda: self.send_button.config(state=tk.NORMAL))
            return
        
        try:
            mode = "cloud_only" if (use_cloud and not use_local) else "local"
            
            memory_context, processed_input, retrieved_memories, has_memories = self.context_builder.build_context(
                user_input, self.conversation_history, mode
            )
            
            retrieval_metadata = self._build_retrieval_metadata(retrieved_memories, memory_context)
            
            if show_memory and retrieved_memories:
                memory_info = "\n".join([
                    f"  [{m.get('similarity', 0):.2f}][{m.get('source', '?')}] {m.get('text', '')[:50]}..."
                    for m in retrieved_memories[:5]
                ])
                self._safe_after(0, lambda info=memory_info: self._append_message(
                    "memory", f"检索到的记忆:\n{info}"
                ))
            
            local_response = ""
            final_response = ""
            response_source = "assistant"
            cloud_success = False
            
            if use_cloud and not use_local:
                self._safe_after(0, lambda: self.state_manager.set_state(UIState.PROCESSING_CLOUD, "云端直连处理中"))
                
                if has_memories and memory_context:
                    messages = [
                        {"role": "system", "content": "你是一个智能助手。请结合历史上下文回答用户问题。"},
                        {"role": "user", "content": f"【历史上下文】\n{memory_context}\n\n【当前问题】\n{user_input}"}
                    ]
                else:
                    messages = [
                        {"role": "system", "content": "你是一个智能助手，请提供完整、详细、有帮助的回答。"},
                        {"role": "user", "content": user_input}
                    ]
                
                final_response = self.hybrid_client.cloud_client.chat(messages)
                
                if final_response:
                    response_source = "cloud"
                    cloud_success = True
                else:
                    final_response = "云端响应为空，请检查网络连接或API配置。"
                    self._safe_after(0, lambda: self._append_message("system", "云端响应为空"))
            
            elif use_cloud and use_local:
                if has_memories:
                    self._safe_after(0, lambda: self.state_manager.set_state(UIState.PROCESSING_LOCAL, "本地压缩中"))
                    
                    messages = [
                        {"role": "system", "content": config.system_prompt}
                    ]
                    
                    if memory_context:
                        user_message = f"【相关内容】\n{memory_context}\n\n【原文】\n{processed_input}"
                    else:
                        user_message = processed_input
                    
                    messages.append({"role": "user", "content": user_message})
                    local_response = self.llm.chat(messages)
                    
                    if local_response and show_local:
                        self._safe_after(0, lambda resp=local_response: self._append_message("assistant", resp))
                    
                    if local_response:
                        self._safe_after(0, lambda: self.state_manager.set_state(UIState.PROCESSING_CLOUD, "等待云端响应"))
                        
                        cloud_response = self.hybrid_client.process(
                            user_input=user_input,
                            memory_context=memory_context,
                            local_response=local_response,
                            metadata=retrieval_metadata
                        )
                        
                        if cloud_response:
                            final_response = cloud_response
                            response_source = "cloud"
                            cloud_success = True
                        else:
                            final_response = local_response
                    else:
                        final_response = local_response
                else:
                    self._safe_after(0, lambda: self.state_manager.set_state(UIState.PROCESSING_CLOUD, "直连云端"))
                    final_response = self.hybrid_client.direct_chat(user_input)
                    
                    if final_response:
                        response_source = "cloud"
                        cloud_success = True
                    else:
                        self._safe_after(0, lambda: self._append_message("system", "云端响应为空"))
            
            elif use_local:
                self._safe_after(0, lambda: self.state_manager.set_state(UIState.PROCESSING_LOCAL, "本地处理中"))
                
                messages = [
                    {"role": "system", "content": config.system_prompt}
                ]
                
                if has_memories:
                    if memory_context:
                        user_message = f"【相关内容】\n{memory_context}\n\n【原文】\n{processed_input}"
                    else:
                        user_message = processed_input
                else:
                    user_message = user_input
                
                messages.append({"role": "user", "content": user_message})
                local_response = self.llm.chat(messages)
                final_response = local_response
            
            else:
                final_response = "未启用任何AI服务，请在设置中启用本地AI或云端AI。"
                self._safe_after(0, lambda resp=final_response: self._append_message("system", resp))
            
            if not final_response:
                final_response = "抱歉，处理过程中出现问题，请重试。"
            
            if response_source == "cloud":
                self._safe_after(0, lambda resp=final_response: self._append_message(response_source, resp))
            elif use_local and not use_cloud:
                self._safe_after(0, lambda resp=final_response: self._append_message(response_source, resp))
            
            self.conversation_history.append({
                "user": user_input,
                "assistant": final_response,
                "source": "cloud" if response_source == "cloud" else "local",
                "timestamp": datetime.now().isoformat()
            })
            
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            should_store = False
            if use_cloud and cloud_success:
                should_store = True
            elif use_local and not use_cloud and final_response:
                should_store = True
            
            if should_store:
                self.async_processor.add_conversation(user_input, final_response, {
                    "context_used": len(memory_context) if memory_context else 0,
                    "memories_retrieved": len(retrieved_memories),
                    "source": "cloud" if cloud_success else "local"
                })
            
            self._safe_after(0, lambda: self.state_manager.set_state(UIState.IDLE))
            
        except Exception as e:
            error_msg = str(e)
            self._safe_after(0, lambda msg=error_msg: self._append_message("system", f"错误: 处理消息时出错: {msg}"))
            self._safe_after(0, lambda: self.state_manager.set_state(UIState.ERROR, "处理出错"))
        
        finally:
            if not self._closing:
                self._safe_after(0, lambda: self.send_button.config(state=tk.NORMAL))
                l2_count = len(self.memory.vector_store) if self.memory and self.memory.vector_store else 0
                l3_count = len(self.memory.sqlite) if self.memory and self.memory.sqlite else 0
                self._safe_after(0, lambda l2=l2_count, l3=l3_count: self.state_manager.set_memory_counts(l2, l3))
    
    def _append_message(self, role: str, content: str):
        """向对话区域添加消息"""
        if self._closing:
            return
        
        self.chat_display.config(state=tk.NORMAL)
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if role == "user":
            self.chat_display.insert(tk.END, f"\n[{timestamp}] 你:\n", "user")
            self.chat_display.insert(tk.END, f"{content}\n", "user")
        elif role == "assistant":
            self.chat_display.insert(tk.END, f"\n[{timestamp}] 助理(本地):\n", "assistant")
            self.chat_display.insert(tk.END, f"{content}\n", "assistant")
        elif role == "cloud":
            self.chat_display.insert(tk.END, f"\n[{timestamp}] 助理(云端):\n", "cloud")
            self.chat_display.insert(tk.END, f"{content}\n", "cloud")
        elif role == "memory":
            self.chat_display.insert(tk.END, f"\n{content}\n", "memory")
        else:
            self.chat_display.insert(tk.END, f"[{timestamp}] {content}\n", "system")
        
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def _clear_conversation(self):
        """清空当前对话历史"""
        self.conversation_history.clear()
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self.send_button.config(state=tk.NORMAL)
        self.state_manager.set_state(UIState.IDLE)
        self._append_message("system", "对话历史已清空")
    
    def _force_save(self):
        """强制保存记忆"""
        if self.async_processor:
            self.async_processor._flush_buffer()
        if self.memory:
            self.memory.save()
        self._append_message("system", "记忆已手动保存")
    
    def _show_metrics(self):
        """显示系统监控指标"""
        try:
            from health_check import get_health_checker
            
            checker = get_health_checker()
            summary = checker.get_metrics_summary()
            
            self._append_message("system", summary)
            
        except Exception as e:
            self._append_message("system", f"获取监控指标失败: {str(e)}")
    
    def _on_close(self):
        """
        窗口关闭时的清理
        
        清理顺序：
        1. 设置关闭标志，阻止新请求
        2. 取消所有待执行的 UI 回调
        3. 等待初始化线程完成（最多5秒）
        4. 停止异步处理器
        5. 保存记忆状态
        6. 清理 MemoryManager 资源（SQLite 连接池）
        7. 清理 VectorStore 资源（ONNX 模型会话）
        8. 清理 SQLite 资源
        9. 销毁窗口
        """
        self._closing = True
        self._cancel_all_after()
        
        if hasattr(self, '_init_thread') and self._init_thread.is_alive():
            self._init_thread.join(timeout=5)
            if self._init_thread.is_alive():
                print("[警告] 初始化线程未能在5秒内完成")
        
        if self.async_processor:
            self.async_processor.stop()
        
        if self.memory:
            self.memory.save()
            self.memory.cleanup_resources()
        
        try:
            from vector_store import get_vector_store
            vector_store = get_vector_store()
            if vector_store:
                vector_store.close()
        except Exception as e:
            print(f"清理 VectorStore 失败: {e}")
        
        try:
            from sqlite_store import get_sqlite_store
            sqlite_store = get_sqlite_store()
            if sqlite_store:
                sqlite_store.close()
        except Exception as e:
            print(f"清理 SQLite 失败: {e}")
        
        try:
            self.window.destroy()
        except tk.TclError:
            pass
    
    def run(self):
        """启动窗口"""
        self.window.mainloop()


if __name__ == "__main__":
    app = ChatWindow()
    app.run()
