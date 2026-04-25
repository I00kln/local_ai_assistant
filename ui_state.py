# ui_state.py
# UI状态管理器 - 统一管理界面状态
import tkinter as tk
from typing import Callable, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
from contextlib import contextmanager
import time
from models import UIState
from logger import get_logger


@dataclass
class StatusMessage:
    """状态消息"""
    text: str
    level: str = "info"  # info, warning, error, success
    timestamp: str = ""
    persistent: bool = False
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%H:%M:%S")


BACKGROUND_STATES = {
    UIState.BACKGROUND_COMPRESSING,
    UIState.BACKGROUND_DEDUPING,
    UIState.PROCESSING_LOCAL,
    UIState.PROCESSING_CLOUD,
    UIState.PROCESSING_HYBRID,
    UIState.INITIALIZING
}


class ThrottleManager:
    """
    节流管理器
    
    设计原则：
    - SRP: 仅负责 UI 更新节流
    - 防止事件风暴导致界面卡死
    
    特性：
    - 合并短时间内的多次更新请求
    - 可配置节流间隔
    - 支持强制刷新
    """
    
    def __init__(self, interval_ms: int = 50):
        """
        初始化节流管理器
        
        Args:
            interval_ms: 节流间隔（毫秒）
        """
        self._interval_ms = interval_ms
        self._last_update_time = 0.0
        self._pending_update = None
        self._pending_args = None
        self._scheduled = False
    
    def should_update(self) -> bool:
        """
        检查是否应该执行更新
        
        Returns:
            是否应该立即执行更新
        """
        current_time = time.time() * 1000
        elapsed = current_time - self._last_update_time
        
        if elapsed >= self._interval_ms:
            self._last_update_time = current_time
            return True
        
        return False
    
    def schedule_update(self, update_func: Callable, *args, **kwargs):
        """
        调度更新（带节流）
        
        Args:
            update_func: 更新函数
            *args, **kwargs: 更新函数参数
        """
        if self.should_update():
            update_func(*args, **kwargs)
            self._pending_update = None
            self._pending_args = None
        else:
            self._pending_update = update_func
            self._pending_args = (args, kwargs)
    
    def flush(self):
        """强制执行待处理的更新"""
        if self._pending_update:
            args, kwargs = self._pending_args
            self._pending_update(*args, **kwargs)
            self._pending_update = None
            self._pending_args = None
            self._last_update_time = time.time() * 1000


class UIStateManager:
    """
    UI状态管理器
    
    功能：
    - 统一管理界面状态
    - 状态变更通知
    - 后台任务状态显示
    - 状态历史记录
    - 异常自动恢复（上下文管理器）
    - 防抖/节流机制防止事件风暴
    - 超时强制回位机制
    
    线程安全：
    - 所有 UI 更新通过 window.after() 调度到主线程
    - 支持从任意线程安全调用
    - 节流间隔默认 50ms，防止事件风暴
    """
    
    THROTTLE_INTERVAL_MS = 100
    PROCESSING_TIMEOUT_SECONDS = 30.0
    MAX_AFTER_CALLS_PER_SECOND = 10
    
    def __init__(self, status_label: tk.Label = None, memory_label: tk.Label = None, window: tk.Tk = None):
        self._log = get_logger()
        self.status_label = status_label
        self.memory_label = memory_label
        self.window = window
        
        self._current_state = UIState.IDLE
        self._state_stack: List[UIState] = []
        self._background_tasks: Dict[str, str] = {}
        self._status_history: List[StatusMessage] = []
        self._callbacks: Dict[str, List[Callable]] = {}
        
        self._memory_count = 0
        self._l2_count = 0
        self._l3_count = 0
        
        self._throttle = ThrottleManager(self.THROTTLE_INTERVAL_MS)
        self._processing_start_time: Optional[float] = None
        self._timeout_check_id: Optional[str] = None
        self._after_call_times: List[float] = []
    
    def set_window(self, window: tk.Tk):
        """设置窗口引用（用于线程安全 UI 更新）"""
        self.window = window
    
    def _can_schedule_after(self) -> bool:
        """
        检查是否可以调度新的 after 调用
        
        Returns:
            是否允许调度
        """
        current_time = time.time()
        one_second_ago = current_time - 1.0
        
        self._after_call_times = [t for t in self._after_call_times if t > one_second_ago]
        
        return len(self._after_call_times) < self.MAX_AFTER_CALLS_PER_SECOND
    
    def _safe_ui_update(self, update_func: Callable):
        """
        安全的 UI 更新（带节流）
        
        确保在主线程中执行 UI 操作
        使用节流机制防止事件风暴
        限制 window.after 调用频率为每秒 10 次
        
        Args:
            update_func: UI 更新函数
        """
        if self.window:
            try:
                if not self._can_schedule_after():
                    self._log.warning(
                        "UI_AFTER_RATE_LIMITED",
                        message="window.after 调用频率超限，丢弃更新请求"
                    )
                    return
                
                self._after_call_times.append(time.time())
                
                def throttled_update():
                    self._throttle.schedule_update(update_func)
                
                self.window.after(0, throttled_update)
            except tk.TclError:
                pass
        else:
            update_func()
    
    @property
    def current_state(self) -> UIState:
        """获取当前状态"""
        return self._current_state
    
    def _is_processing_state(self, state: UIState) -> bool:
        """检查是否为处理中状态"""
        return state in (
            UIState.PROCESSING_LOCAL,
            UIState.PROCESSING_CLOUD,
            UIState.PROCESSING_HYBRID
        )
    
    def _start_timeout_check(self):
        """启动超时检测"""
        self._cancel_timeout_check()
        
        if self.window and self._is_processing_state(self._current_state):
            self._processing_start_time = time.time()
            self._timeout_check_id = self.window.after(
                int(self.PROCESSING_TIMEOUT_SECONDS * 1000),
                self._on_processing_timeout
            )
    
    def _cancel_timeout_check(self):
        """取消超时检测"""
        if self._timeout_check_id and self.window:
            try:
                self.window.after_cancel(self._timeout_check_id)
            except tk.TclError:
                pass
        self._timeout_check_id = None
        self._processing_start_time = None
    
    def _on_processing_timeout(self):
        """
        处理中状态超时回调
        
        设计原则：
        - PROCESSING 状态超过 30 秒自动重置为 IDLE
        - 记录错误日志便于排查
        - 清空状态栈防止卡死
        """
        self._timeout_check_id = None
        
        if self._is_processing_state(self._current_state):
            duration = time.time() - self._processing_start_time if self._processing_start_time else 0
            self._log.error(
                "UI_STATE_TIMEOUT",
                state=self._current_state.value,
                duration=round(duration, 1),
                message="处理中状态超时，强制重置为 IDLE"
            )
            
            self._state_stack.clear()
            self._current_state = UIState.IDLE
            self._processing_start_time = None
            
            status_text = self._get_status_text(UIState.IDLE, "操作超时，已重置")
            self._update_status_label(status_text)
            
            self.add_status_message("操作超时，已自动重置状态", level="warning")
    
    def set_state(self, state: UIState, message: str = None):
        """
        设置UI状态
        
        Args:
            state: 新状态
            message: 可选的状态消息
        """
        old_state = self._current_state
        self._current_state = state
        
        if self._is_processing_state(state):
            self._start_timeout_check()
        else:
            self._cancel_timeout_check()
        
        status_text = self._get_status_text(state, message)
        self._update_status_label(status_text)
        
        self._notify_callbacks("state_change", {
            "old_state": old_state,
            "new_state": state,
            "message": message
        })
    
    def push_state(self, state: UIState, message: str = None):
        """
        压入状态（用于临时状态）
        
        Args:
            state: 临时状态
            message: 状态消息
        
        安全机制：
        - 最大栈深度限制为 10，防止无限 push
        - 超过限制时自动丢弃最旧的状态
        """
        MAX_STACK_DEPTH = 10
        
        if len(self._state_stack) >= MAX_STACK_DEPTH:
            self._state_stack.pop(0)
        
        self._state_stack.append(self._current_state)
        self.set_state(state, message)
    
    def pop_state(self):
        """
        弹出状态，恢复之前的状态
        
        安全机制：
        - 如果栈为空，自动恢复到 IDLE 状态
        """
        if self._state_stack:
            previous_state = self._state_stack.pop()
            self._current_state = previous_state
            status_text = self._get_status_text(previous_state)
            self._update_status_label(status_text)
        else:
            self.set_state(UIState.IDLE)
    
    def reset_state_stack(self):
        """
        重置状态栈
        
        清空所有压入的状态，恢复到 IDLE
        """
        self._state_stack.clear()
        self.set_state(UIState.IDLE)
    
    @contextmanager
    def temp_state(self, state: UIState, message: str = None, on_error: str = None):
        """
        临时状态上下文管理器
        
        自动在异常时恢复状态
        
        Args:
            state: 临时状态
            message: 状态消息
            on_error: 错误时的额外消息
        
        Usage:
            with ui_state.temp_state(UIState.PROCESSING_LOCAL):
                ... # 异常时自动恢复状态
        """
        self.push_state(state, message)
        try:
            yield
        except Exception as e:
            error_msg = on_error or f"操作失败: {str(e)}"
            self.add_status_message(error_msg, level="error")
            raise
        finally:
            self.pop_state()
    
    @contextmanager
    def background_task_context(self, task_id: str, task_name: str, state: UIState = None):
        """
        后台任务上下文管理器
        
        自动管理任务生命周期和状态
        
        Args:
            task_id: 任务ID
            task_name: 任务名称
            state: 可选的状态（默认不改变状态）
        
        Usage:
            with ui_state.background_task_context("compress", "压缩中"):
                ... # 异常时自动清理任务
        """
        self.start_background_task(task_id, task_name)
        if state:
            self.push_state(state, task_name)
        try:
            yield
        except Exception as e:
            self.add_status_message(f"{task_name}失败: {str(e)}", level="error")
            raise
        finally:
            self.end_background_task(task_id)
            if state:
                self.pop_state()
    
    def start_background_task(self, task_id: str, task_name: str):
        """
        启动后台任务
        
        Args:
            task_id: 任务ID
            task_name: 任务名称
        """
        self._background_tasks[task_id] = task_name
        self._update_background_status()
    
    def end_background_task(self, task_id: str):
        """
        结束后台任务
        
        Args:
            task_id: 任务ID
        """
        if task_id in self._background_tasks:
            del self._background_tasks[task_id]
            self._update_background_status()
    
    def clear_all_background_tasks(self):
        """清除所有后台任务并恢复状态"""
        self._background_tasks.clear()
        while self._state_stack:
            self.pop_state()
        if self._current_state in BACKGROUND_STATES:
            self.set_state(UIState.IDLE)
    
    def set_memory_counts(self, l2_count: int, l3_count: int):
        """
        设置记忆数量
        
        Args:
            l2_count: L2向量库数量
            l3_count: L3数据库数量
        """
        self._l2_count = l2_count
        self._l3_count = l3_count
        self._update_memory_label()
    
    def add_callback(self, event: str, callback: Callable):
        """
        添加事件回调
        
        Args:
            event: 事件名称
            callback: 回调函数
        """
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)
    
    def add_status_message(self, text: str, level: str = "info", persistent: bool = False):
        """
        添加状态消息
        
        Args:
            text: 消息文本
            level: 消息级别
            persistent: 是否持久显示
        """
        message = StatusMessage(text=text, level=level, persistent=persistent)
        self._status_history.append(message)
        
        if len(self._status_history) > 100:
            self._status_history = self._status_history[-100:]
    
    def get_status_history(self, limit: int = 20) -> List[StatusMessage]:
        """获取状态历史"""
        return self._status_history[-limit:]
    
    def _get_status_text(self, state: UIState, message: str = None) -> str:
        """获取状态文本"""
        state_messages = {
            UIState.IDLE: "就绪",
            UIState.INITIALIZING: "初始化中...",
            UIState.PROCESSING_LOCAL: "本地处理中...",
            UIState.PROCESSING_CLOUD: "云端处理中...",
            UIState.PROCESSING_HYBRID: "本地+云端处理中...",
            UIState.BACKGROUND_COMPRESSING: "后台压缩记忆...",
            UIState.BACKGROUND_DEDUPING: "后台去重中...",
            UIState.ERROR: "错误"
        }
        
        base_text = state_messages.get(state, "未知状态")
        
        if message:
            return f"状态: {base_text} | {message}"
        
        return f"状态: {base_text}"
    
    def _update_status_label(self, text: str):
        """更新状态标签（线程安全）"""
        def do_update():
            if self.status_label:
                try:
                    self.status_label.config(text=text)
                except tk.TclError:
                    pass
        
        self._safe_ui_update(do_update)
    
    def _update_memory_label(self):
        """更新记忆数量标签（线程安全）"""
        def do_update():
            if self.memory_label:
                try:
                    total = self._l2_count + self._l3_count
                    self.memory_label.config(text=f"记忆数: L2={self._l2_count}, L3={self._l3_count}")
                except tk.TclError:
                    pass
        
        self._safe_ui_update(do_update)
    
    def _update_background_status(self):
        """更新后台任务状态"""
        if not self._background_tasks:
            if self._current_state in BACKGROUND_STATES:
                self.pop_state()
        else:
            task_names = list(self._background_tasks.values())
            status = f"后台: {', '.join(task_names)}"
            self._update_status_label(f"状态: {status}")
    
    def _notify_callbacks(self, event: str, data: Dict[str, Any]):
        """通知回调"""
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                try:
                    callback(data)
                except Exception as e:
                    self._log.warning("CALLBACK_EXECUTION_FAILED",
                                     event=event,
                                     error=str(e))
    
    def get_full_status(self) -> Dict[str, Any]:
        """获取完整状态信息"""
        return {
            "current_state": self._current_state.value,
            "state_stack": [s.value for s in self._state_stack],
            "background_tasks": self._background_tasks,
            "memory_counts": {
                "l2": self._l2_count,
                "l3": self._l3_count
            }
        }
