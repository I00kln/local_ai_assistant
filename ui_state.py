# ui_state.py
# UI状态管理器 - 统一管理界面状态
import tkinter as tk
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from models import UIState


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


class UIStateManager:
    """
    UI状态管理器
    
    功能：
    - 统一管理界面状态
    - 状态变更通知
    - 后台任务状态显示
    - 状态历史记录
    """
    
    def __init__(self, status_label: tk.Label = None, memory_label: tk.Label = None):
        self.status_label = status_label
        self.memory_label = memory_label
        
        self._current_state = UIState.IDLE
        self._state_stack: List[UIState] = []
        self._background_tasks: Dict[str, str] = {}
        self._status_history: List[StatusMessage] = []
        self._callbacks: Dict[str, List[Callable]] = {}
        
        self._memory_count = 0
        self._l2_count = 0
        self._l3_count = 0
    
    @property
    def current_state(self) -> UIState:
        """获取当前状态"""
        return self._current_state
    
    def set_state(self, state: UIState, message: str = None):
        """
        设置UI状态
        
        Args:
            state: 新状态
            message: 可选的状态消息
        """
        old_state = self._current_state
        self._current_state = state
        
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
        """
        self._state_stack.append(self._current_state)
        self.set_state(state, message)
    
    def pop_state(self):
        """弹出状态，恢复之前的状态"""
        if self._state_stack:
            previous_state = self._state_stack.pop()
            self._current_state = previous_state
            status_text = self._get_status_text(previous_state)
            self._update_status_label(status_text)
    
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
        """更新状态标签"""
        if self.status_label:
            try:
                self.status_label.config(text=text)
            except tk.TclError:
                pass
    
    def _update_memory_label(self):
        """更新记忆数量标签"""
        if self.memory_label:
            try:
                total = self._l2_count + self._l3_count
                self.memory_label.config(text=f"记忆数: L2={self._l2_count}, L3={self._l3_count}")
            except tk.TclError:
                pass
    
    def _update_background_status(self):
        """更新后台任务状态"""
        if not self._background_tasks:
            if self._current_state in [UIState.BACKGROUND_COMPRESSING, UIState.BACKGROUND_DEDUPING]:
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
                    print(f"回调执行失败: {e}")
    
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
