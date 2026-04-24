# chat_window.py
import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any

from config import config
from models import UIState
from ui_state import UIStateManager
from logger import TraceContext, get_trace_id, get_logger


class ChatWindow:
    """本地AI助理对话窗口 - 支持本地+云端混合模式 + 会话持久化"""
    
    def __init__(self):
        self._log = get_logger()
        self.memory = None
        self.llm = None
        self.context_builder = None
        self.async_processor = None
        self.hybrid_client = None
        
        self.conversation_history = []
        self._initialized = False
        self._closing = False
        self._after_ids: List[str] = []
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_restored = False
        
        self.window = tk.Tk()
        self.window.title("本地AI助理 - ChromaDB + SQLite")
        self.window.geometry("800x600")
        
        self._setup_ui()
        
        self.state_manager = UIStateManager(
            self.status_label, 
            self.memory_count_label, 
            self.window
        )
        self.state_manager.set_state(UIState.INITIALIZING, "正在初始化...")
        
        self.window.update()
        
        after_id = self.window.after(50, self._start_background_init)
        self._after_ids.append(after_id)
        
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _start_background_init(self):
        """启动后台初始化（延迟启动）"""
        if self._closing:
            return
        
        self.state_manager.set_state(UIState.INITIALIZING, "正在连接数据库...")
        self._init_thread = threading.Thread(target=self._async_init, daemon=True)
        self._init_thread.start()
    
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
        """后台异步初始化组件（分阶段）"""
        if self._closing:
            return
        
        try:
            self._safe_after(0, lambda: self.state_manager.set_state(
                UIState.INITIALIZING, "正在连接数据库..."
            ))
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
            
            self._safe_after(0, lambda: self.state_manager.set_state(
                UIState.INITIALIZING, "正在加载AI模型..."
            ))
            
            from llm_client import LlamaClient
            self.llm = LlamaClient()
            
            from context_builder import ContextBuilder
            self.context_builder = ContextBuilder(self.memory)
            
            self._safe_after(0, lambda: self.state_manager.set_state(
                UIState.INITIALIZING, "正在启动后台处理器..."
            ))
            
            from async_processor import AsyncMemoryProcessor
            self.async_processor = AsyncMemoryProcessor()
            self.async_processor.start()
            
            self._check_llm_health()
            
            self._init_cloud_client()
            
            self._initialized = True
            
            self._safe_after(0, self._enable_ui)
            
            self._try_restore_session()
            
        except Exception as e:
            err_msg = str(e)
            self._safe_after(0, lambda msg=err_msg: self._append_message("system", f"初始化失败: {msg}"))
            self._safe_after(0, lambda: self.state_manager.set_state(UIState.ERROR, "初始化失败"))
    
    def _check_llm_health(self):
        """检查本地 LLM 服务健康状态"""
        self._safe_after(0, lambda: self.state_manager.set_state(
            UIState.INITIALIZING, "正在检查 LLM 服务..."
        ))
        
        try:
            llm_available = self.llm.check_connection()
            if llm_available:
                self._safe_after(0, lambda: self._append_message("system", "本地 LLM 服务连接成功"))
            else:
                self._safe_after(0, lambda: self._append_message(
                    "system", 
                    "⚠️ 本地 LLM 服务连接失败，请检查 llama.cpp 是否已启动"
                ))
                self._safe_after(0, lambda: self.local_btn.config(state=tk.DISABLED))
        except Exception as e:
            self._safe_after(0, lambda msg=str(e): self._append_message(
                "system", f"⚠️ LLM 健康检查失败: {msg}"
            ))
    
    def _init_cloud_client(self):
        """初始化云端客户端"""
        provider = config.cloud.provider
        api_key = config.cloud.api_key
        model = config.cloud.model
        base_url = config.cloud.base_url
        
        if not api_key or api_key == "your-api-key-here":
            self._safe_after(0, lambda p=provider: self._append_message(
                "system", f"云端AI({p})未配置API密钥，请在 .env 中设置 {p.upper()}_API_KEY"
            ))
            self._safe_after(0, lambda: self.cloud_btn.config(state=tk.NORMAL))
            return
        
        try:
            from cloud_client import CloudClientFactory, HybridClient
            
            cloud_client = CloudClientFactory.create(
                provider=provider,
                api_key=api_key,
                model=model,
                base_url=base_url
            )
            
            if cloud_client and cloud_client.is_available():
                self.hybrid_client = HybridClient(self.llm, cloud_client)
                self._safe_after(0, lambda: self.cloud_btn.config(state=tk.NORMAL))
                
                if config.cloud.enabled:
                    self._safe_after(0, lambda: self.cloud_var.set(True))
                    self._safe_after(0, lambda p=provider, m=model: self._append_message(
                        "system", f"云端AI已启用: {p} ({m})"
                    ))
                else:
                    self._safe_after(0, lambda p=provider, m=model: self._append_message(
                        "system", f"云端AI可用: {p} ({m})，可手动启用"
                    ))
            else:
                self._safe_after(0, lambda p=provider: self._append_message(
                    "system", f"云端AI({p})客户端初始化失败"
                ))
                self._safe_after(0, lambda: self.cloud_btn.config(state=tk.NORMAL))
        except Exception as e:
            err_msg = str(e)
            self._safe_after(0, lambda msg=err_msg: self._append_message("system", f"云端AI初始化失败: {msg}"))
            self._safe_after(0, lambda: self.cloud_btn.config(state=tk.NORMAL))
    
    def _try_restore_session(self):
        """
        尝试恢复上次会话
        
        策略：
        1. 从 SQLite 加载最后一次活跃会话
        2. 恢复对话历史到内存
        3. 恢复 UI 状态
        """
        if not self.memory or not self.memory.sqlite:
            return
        
        try:
            session_data = self.memory.sqlite.load_latest_session()
            
            if not session_data:
                return
            
            self.session_id = session_data.get("session_id", self.session_id)
            
            messages = self.memory.sqlite.get_session_messages(self.session_id, limit=50)
            
            if not messages:
                return
            
            restored_count = 0
            for msg in messages[-20:]:
                user_text = msg.get("user", "")
                assistant_text = msg.get("assistant", "")
                
                if user_text:
                    self.conversation_history.append({
                        "user": user_text,
                        "assistant": assistant_text,
                        "timestamp": msg.get("timestamp", "")
                    })
                    
                    self._append_message("user", user_text)
                    if assistant_text:
                        source = msg.get("source", "local")
                        tag = "cloud" if source == "cloud" else "assistant"
                        self._append_message(tag, assistant_text)
                    
                    restored_count += 1
            
            self._session_restored = True
            
            ui_snapshot = session_data.get("ui_state_snapshot", {})
            if ui_snapshot:
                self.local_var.set(ui_snapshot.get("local_enabled", config.local.enabled))
                self.cloud_var.set(ui_snapshot.get("cloud_enabled", config.cloud.enabled))
                self.nonsense_var.set(ui_snapshot.get("nonsense_enabled", config.nonsense_filter_enabled))
            
            self._safe_after(0, lambda c=restored_count: self._append_message(
                "system", f"已恢复上次会话 ({c} 条对话)"
            ))
            
            self._log.info("SESSION_RESTORED", 
                          session_id=self.session_id,
                          message_count=restored_count)
            
        except Exception as e:
            self._log.warning("SESSION_RESTORE_FAILED", error=str(e))
    
    def _save_session(self):
        """
        保存当前会话状态
        
        策略：
        1. 保存 UI 状态快照
        2. 保存上下文快照
        3. 更新会话消息计数
        """
        if not self.memory or not self.memory.sqlite:
            return
        
        try:
            ui_snapshot = {
                "local_enabled": self.local_var.get(),
                "cloud_enabled": self.cloud_var.get(),
                "nonsense_enabled": self.nonsense_var.get(),
                "show_memory_enabled": self.show_memory_enabled.get(),
                "show_local_enabled": self.show_local_enabled.get()
            }
            
            context_snapshot = {
                "conversation_count": len(self.conversation_history)
            }
            
            self.memory.sqlite.save_session(
                self.session_id,
                ui_state_snapshot=ui_snapshot,
                context_snapshot=context_snapshot
            )
            
        except Exception as e:
            self._log.warning("SESSION_SAVE_FAILED", error=str(e))
    
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
    
    def _estimate_tokens(self, text: str) -> int:
        """估算文本的token数量"""
        from token_utils import estimate_tokens
        return estimate_tokens(text)
    
    def _build_compression_messages(
        self, 
        memory_context: str
    ) -> List[Dict[str, str]]:
        """
        构建记忆压缩消息（仅压缩记忆，不包含用户原文）
        
        Args:
            memory_context: 记忆上下文
        
        Returns:
            消息列表
        """
        system_prompt = config.system_prompt
        
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        max_tokens = config.local.max_context
        system_tokens = self._estimate_tokens(system_prompt)
        format_overhead = 50
        
        available_for_memory = max_tokens - system_tokens - config.local.max_output_tokens - format_overhead
        available_for_memory = max(available_for_memory, 200)
        
        memory_tokens = self._estimate_tokens(memory_context)
        
        if memory_tokens <= available_for_memory:
            truncated_memory = memory_context
        else:
            ratio = available_for_memory / memory_tokens
            target_chars = int(len(memory_context) * ratio * 0.9)
            
            truncated_memory = memory_context[:target_chars]
            
            last_period = max(
                truncated_memory.rfind('。'),
                truncated_memory.rfind('.'),
                truncated_memory.rfind('\n')
            )
            
            if last_period > target_chars * 0.7:
                truncated_memory = truncated_memory[:last_period + 1]
            
            truncated_memory += "\n...[内容已截断]"
        
        user_message = f"【相关内容】\n{truncated_memory}"
        messages.append({"role": "user", "content": user_message})
        return messages
    
    def _build_local_llm_messages(
        self, 
        memory_context: str, 
        user_input: str,
        apply_token_limit: bool = True
    ) -> List[Dict[str, str]]:
        """
        构建本地LLM消息（仅本地模式，包含用户输入）
        
        Args:
            memory_context: 记忆上下文
            user_input: 用户输入
            apply_token_limit: 是否应用token限制
        
        Returns:
            消息列表
        """
        system_prompt = config.local_assistant_prompt
        
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        if apply_token_limit:
            truncated_memory, truncated_input = self.context_builder.truncate_for_local_llm(
                system_prompt, memory_context, user_input
            )
        else:
            truncated_memory, truncated_input = memory_context, user_input
        
        if truncated_memory:
            user_message = f"【相关内容】\n{truncated_memory}\n\n【原文】\n{truncated_input}"
        else:
            user_message = truncated_input
        
        messages.append({"role": "user", "content": user_message})
        return messages
    
    def _process_message(self, user_input: str, use_local: bool, use_cloud: bool, show_memory: bool = True, show_local: bool = True):
        """处理用户消息（后台线程）"""
        if self._closing:
            self._safe_after(0, lambda: self.send_button.config(state=tk.NORMAL))
            return
        
        with TraceContext("user_message"):
            try:
                mode = "cloud_only" if (use_cloud and not use_local) else "local"
                
                memory_context, processed_input, retrieved_memories, has_memories = self.context_builder.build_context(
                    user_input, self.conversation_history, mode
                )
                
                retrieval_metadata = self._build_retrieval_metadata(retrieved_memories, memory_context)
                
                if show_memory and retrieved_memories:
                    try:
                        from sensitive_filter import get_sensitive_filter
                        sensitive_filter = get_sensitive_filter()
                        
                        memory_lines = []
                        for m in retrieved_memories[:5]:
                            text = m.get('text', '')
                            masked_text, _ = sensitive_filter.mask(text)
                            memory_lines.append(
                                f"  [{m.get('similarity', 0):.2f}][{m.get('source', '?')}] {masked_text[:50]}..."
                            )
                        memory_info = "\n".join(memory_lines)
                    except Exception:
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
                        
                        messages = self._build_compression_messages(memory_context)
                        compressed_memory = self.llm.chat(messages)
                        
                        if compressed_memory and show_local:
                            self._safe_after(0, lambda resp=compressed_memory: self._append_message("assistant", f"[压缩记忆]\n{resp[:200]}..."))
                        
                        if compressed_memory:
                            self._safe_after(0, lambda: self.state_manager.set_state(UIState.PROCESSING_CLOUD, "等待云端响应"))
                            
                            cloud_response = self.hybrid_client.process(
                                user_input=user_input,
                                compressed_memory=compressed_memory,
                                metadata=retrieval_metadata
                            )
                            
                            if cloud_response:
                                final_response = cloud_response
                                response_source = "cloud"
                                cloud_success = True
                            else:
                                final_response = "云端响应为空"
                        else:
                            self._safe_after(0, lambda: self.state_manager.set_state(UIState.PROCESSING_CLOUD, "直连云端"))
                            final_response = self.hybrid_client.direct_chat(user_input)
                            
                            if final_response:
                                response_source = "cloud"
                                cloud_success = True
                            else:
                                self._safe_after(0, lambda: self._append_message("system", "云端响应为空"))
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
                    
                    messages = self._build_local_llm_messages(
                        memory_context if has_memories else "", 
                        processed_input if has_memories else user_input,
                        apply_token_limit=True
                    )
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
                
                if should_store:
                    try:
                        from sensitive_filter import get_sensitive_filter
                        sensitive_filter = get_sensitive_filter()
                        
                        masked_user_input, user_detected = sensitive_filter.mask(user_input)
                        masked_response, resp_detected = sensitive_filter.mask(final_response)
                        
                        if user_detected or resp_detected:
                            self._log_sensitive_detection(user_detected, resp_detected)
                    except Exception:
                        masked_user_input = user_input
                        masked_response = final_response
                    
                    self.async_processor.add_conversation(masked_user_input, masked_response, {
                        "context_used": len(memory_context) if memory_context else 0,
                        "memories_retrieved": len(retrieved_memories),
                        "source": "cloud" if cloud_success else "local",
                        "sensitive_masked": True,
                        "trace_id": get_trace_id()
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
    
    def _log_sensitive_detection(self, user_detected: Dict, resp_detected: Dict):
        """记录敏感信息检测日志"""
        if not config.privacy.log_sensitive_detection:
            return
        
        total = {}
        for k, v in user_detected.items():
            total[k] = total.get(k, 0) + v
        for k, v in resp_detected.items():
            total[k] = total.get(k, 0) + v
        
        if total:
            self._safe_after(0, lambda t=total: self._append_message(
                "system", f"[安全] 已脱敏敏感信息: {t}"
            ))
    
    def _clear_conversation(self):
        """清空当前对话历史并结束当前会话"""
        if self.memory and self.memory.sqlite:
            try:
                self.memory.sqlite.end_session(self.session_id)
            except Exception as e:
                self._log.warning("END_SESSION_FAILED", error=str(e))
        
        self.conversation_history.clear()
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self.send_button.config(state=tk.NORMAL)
        self.state_manager.set_state(UIState.IDLE)
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_restored = False
        
        self._append_message("system", "对话历史已清空，已开始新会话")
    
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
        
        使用 LifecycleManager 集中管理资源释放：
        1. 设置关闭标志，阻止新请求
        2. 取消所有待执行的 UI 回调
        3. 保存当前会话状态
        4. 调用 LifecycleManager.shutdown() 关闭所有服务
        5. 销毁窗口
        """
        self._closing = True
        self._cancel_all_after()
        
        if hasattr(self, '_init_thread') and self._init_thread.is_alive():
            self._init_thread.join(timeout=2)
        
        self._save_session()
        
        try:
            from lifecycle_manager import get_lifecycle_manager
            lifecycle = get_lifecycle_manager()
            lifecycle.shutdown(timeout=8)
        except Exception as e:
            self._log.error("LIFECYCLE_SHUTDOWN_FAILED", error=str(e))
        
        if self.memory:
            try:
                self.memory.save()
            except Exception as e:
                self._log.error("MEMORY_SAVE_FAILED", error=str(e))
        
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
