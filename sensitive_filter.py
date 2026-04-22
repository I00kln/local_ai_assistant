"""
敏感信息过滤器

功能：
- 检测并脱敏 PII（个人身份信息）
- 支持多种敏感信息类型
- 可配置开关和规则
- 记录脱敏统计

使用方式：
- 在发送云端前调用 mask() 方法
- 可通过配置启用/禁用
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime


@dataclass
class FilterRule:
    """过滤规则"""
    name: str
    pattern: str
    replacement: str
    description: str
    enabled: bool = True


@dataclass
class FilterStats:
    """过滤统计"""
    total_processed: int = 0
    total_masked: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)
    last_process_time: Optional[str] = None


class SensitiveFilter:
    """
    敏感信息过滤器
    
    支持的敏感信息类型：
    - 手机号
    - 身份证号
    - 邮箱
    - 银行卡号
    - 密码
    - API 密钥
    - 地址（关键词）
    - 姓名（关键词）
    """
    
    DEFAULT_RULES = [
        FilterRule(
            name="phone",
            pattern=r'(?<!\d)(1[3-9]\d{9})(?!\d)',
            replacement=r'1**\*\*\*\*\*\*\2',
            description="手机号"
        ),
        FilterRule(
            name="id_card",
            pattern=r'(?<!\d)(\d{6})(\d{8})(\d{3}[\dXx])(?!\d)',
            replacement=r'\1********\3',
            description="身份证号"
        ),
        FilterRule(
            name="email",
            pattern=r'[\w.+-]+@[\w.-]+\.\w+',
            replacement=r'***@***.***',
            description="邮箱地址"
        ),
        FilterRule(
            name="bank_card",
            pattern=r'(?<!\d)(\d{4})\d{8,11}(\d{4})(?!\d)',
            replacement=r'\1 **** **** \2',
            description="银行卡号"
        ),
        FilterRule(
            name="password",
            pattern=r'(密码|password|pwd|pass|口令)[是为：:＝\s]+[\w!@#$%^&*()\-+=\[\]{}|;:,.<>?/~`]{4,}',
            replacement=r'\1****',
            description="密码"
        ),
        FilterRule(
            name="api_key",
            pattern=r'(api[_-]?key|密钥|secret[_-]?key|token)[是为：:＝\s]+[\w\-]{10,}',
            replacement=r'\1****',
            description="API密钥"
        ),
        FilterRule(
            name="credit_card",
            pattern=r'(?<!\d)(\d{4})\s?\d{4}\s?\d{4}\s?(\d{4})(?!\d)',
            replacement=r'\1 **** **** \2',
            description="信用卡号"
        ),
        FilterRule(
            name="ipv4",
            pattern=r'(?<!\d)(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})(?!\d)',
            replacement=r'\1.***.***.\4',
            description="IP地址"
        ),
    ]
    
    _instance: Optional['SensitiveFilter'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._enabled = True
        self._rules: Dict[str, FilterRule] = {}
        self._stats = FilterStats()
        self._excluded_keywords: List[str] = []
        self._custom_rules: List[FilterRule] = []
        
        for rule in self.DEFAULT_RULES:
            self._rules[rule.name] = rule
    
    def configure(
        self,
        enabled: bool = True,
        excluded_keywords: List[str] = None,
        custom_rules: List[Dict] = None
    ):
        """
        配置过滤器
        
        Args:
            enabled: 是否启用
            excluded_keywords: 排除关键词（包含这些词的内容不过滤）
            custom_rules: 自定义规则列表
        """
        self._enabled = enabled
        
        if excluded_keywords:
            self._excluded_keywords = excluded_keywords
        
        if custom_rules:
            for rule_dict in custom_rules:
                rule = FilterRule(
                    name=rule_dict.get("name", "custom"),
                    pattern=rule_dict.get("pattern", ""),
                    replacement=rule_dict.get("replacement", "****"),
                    description=rule_dict.get("description", "自定义规则"),
                    enabled=rule_dict.get("enabled", True)
                )
                self._rules[rule.name] = rule
    
    def enable_rule(self, rule_name: str, enabled: bool = True):
        """启用/禁用特定规则"""
        if rule_name in self._rules:
            self._rules[rule_name].enabled = enabled
    
    def mask(self, text: str) -> Tuple[str, Dict[str, int]]:
        """
        脱敏处理
        
        Args:
            text: 原始文本
        
        Returns:
            (脱敏后文本, 检测到的敏感信息统计)
        """
        if not self._enabled or not text:
            return text, {}
        
        for keyword in self._excluded_keywords:
            if keyword in text:
                return text, {}
        
        masked = text
        detected: Dict[str, int] = {}
        
        for rule in self._rules.values():
            if not rule.enabled:
                continue
            
            try:
                matches = re.findall(rule.pattern, text, re.IGNORECASE)
                if matches:
                    count = len(matches) if isinstance(matches[0], str) else len(matches)
                    detected[rule.name] = count
                    masked = re.sub(rule.pattern, rule.replacement, masked, flags=re.IGNORECASE)
            except re.error:
                continue
        
        self._stats.total_processed += 1
        if detected:
            self._stats.total_masked += 1
            for rule_name, count in detected.items():
                self._stats.by_type[rule_name] = self._stats.by_type.get(rule_name, 0) + count
        self._stats.last_process_time = datetime.now().isoformat()
        
        return masked, detected
    
    def mask_dict(self, data: Dict) -> Tuple[Dict, Dict[str, int]]:
        """
        脱敏字典数据
        
        Args:
            data: 原始字典
        
        Returns:
            (脱敏后字典, 检测到的敏感信息统计)
        """
        if not self._enabled:
            return data, {}
        
        result = {}
        total_detected: Dict[str, int] = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                masked_value, detected = self.mask(value)
                result[key] = masked_value
                for rule_name, count in detected.items():
                    total_detected[rule_name] = total_detected.get(rule_name, 0) + count
            elif isinstance(value, dict):
                masked_dict, detected = self.mask_dict(value)
                result[key] = masked_dict
                for rule_name, count in detected.items():
                    total_detected[rule_name] = total_detected.get(rule_name, 0) + count
            else:
                result[key] = value
        
        return result, total_detected
    
    def detect(self, text: str) -> Dict[str, List[str]]:
        """
        检测敏感信息（不脱敏）
        
        Args:
            text: 原始文本
        
        Returns:
            {规则名: [匹配到的敏感信息列表]}
        """
        if not text:
            return {}
        
        detected: Dict[str, List[str]] = {}
        
        for rule in self._rules.values():
            if not rule.enabled:
                continue
            
            try:
                matches = re.findall(rule.pattern, text, re.IGNORECASE)
                if matches:
                    if isinstance(matches[0], tuple):
                        detected[rule.name] = [''.join(m) for m in matches]
                    else:
                        detected[rule.name] = matches
            except re.error:
                continue
        
        return detected
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "enabled": self._enabled,
            "total_processed": self._stats.total_processed,
            "total_masked": self._stats.total_masked,
            "mask_rate": (
                self._stats.total_masked / self._stats.total_processed
                if self._stats.total_processed > 0 else 0
            ),
            "by_type": dict(self._stats.by_type),
            "last_process_time": self._stats.last_process_time,
            "rules_count": len(self._rules),
            "enabled_rules": sum(1 for r in self._rules.values() if r.enabled)
        }
    
    def reset_stats(self):
        """重置统计"""
        self._stats = FilterStats()
    
    def get_rules(self) -> List[Dict]:
        """获取所有规则"""
        return [
            {
                "name": rule.name,
                "description": rule.description,
                "enabled": rule.enabled
            }
            for rule in self._rules.values()
        ]


_filter: Optional[SensitiveFilter] = None


def get_sensitive_filter() -> SensitiveFilter:
    """获取全局敏感信息过滤器实例"""
    global _filter
    if _filter is None:
        _filter = SensitiveFilter()
    return _filter
