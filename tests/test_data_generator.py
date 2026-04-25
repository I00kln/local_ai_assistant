# tests/test_data_generator.py
# 测试数据生成器 - 批量生成模拟对话数据
"""
测试数据生成器

功能：
1. 生成模拟对话数据
2. 生成特定主题的记忆
3. 生成废话/非废话测试数据
4. 生成长文本用于压缩测试
"""

import random
import string
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ConversationTurn:
    """单轮对话"""
    user_input: str
    assistant_response: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestConversation:
    """测试对话"""
    turns: List[ConversationTurn] = field(default_factory=list)
    topic: str = ""
    tags: List[str] = field(default_factory=list)
    
    def add_turn(self, user_input: str, assistant_response: str, metadata: Dict = None):
        self.turns.append(ConversationTurn(
            user_input=user_input,
            assistant_response=assistant_response,
            metadata=metadata or {}
        ))
        return self


class TestDataGenerator:
    """测试数据生成器"""
    
    TOPICS = {
        "programming": {
            "keywords": ["Python", "函数", "变量", "循环", "类", "API", "数据库", "调试"],
            "templates": [
                "如何使用{keyword}实现{action}？",
                "{keyword}和{keyword2}有什么区别？",
                "我在使用{keyword}时遇到了{problem}，怎么解决？",
                "请帮我写一个{keyword}的示例代码。",
                "{keyword}的最佳实践是什么？"
            ],
            "actions": ["排序", "搜索", "过滤", "缓存", "异步处理"],
            "problems": ["报错", "性能问题", "内存泄漏", "死锁"]
        },
        "project": {
            "keywords": ["项目", "需求", "迭代", "上线", "测试", "部署", "文档"],
            "templates": [
                "项目{keyword}的进度如何？",
                "我们需要{action}{keyword}。",
                "{keyword}的截止日期是什么时候？",
                "关于{keyword}，我有以下问题：{detail}",
                "请帮我整理{keyword}的文档。"
            ],
            "actions": ["更新", "审核", "提交", "评审", "优化"],
            "details": ["技术方案", "接口设计", "数据库结构", "测试用例"]
        },
        "daily": {
            "keywords": ["会议", "日程", "提醒", "任务", "计划"],
            "templates": [
                "明天{time}有{keyword}吗？",
                "帮我{action}{keyword}。",
                "{keyword}安排在{time}。",
                "取消{time}的{keyword}。",
                "查看本周的{keyword}列表。"
            ],
            "times": ["上午", "下午", "晚上", "9点", "14点"],
            "actions": ["安排", "取消", "修改", "确认"]
        },
        "knowledge": {
            "keywords": ["机器学习", "深度学习", "神经网络", "NLP", "CV"],
            "templates": [
                "什么是{keyword}？",
                "{keyword}的应用场景有哪些？",
                "如何入门{keyword}？",
                "{keyword}和{keyword2}的关系是什么？",
                "请解释{keyword}的原理。"
            ]
        }
    }
    
    NONSENSE_TEMPLATES = [
        "好的",
        "嗯",
        "哦",
        "知道了",
        "行",
        "可以",
        "好的好的",
        "嗯嗯",
        "OK",
        "好",
        "是",
        "对",
        "明白",
        "谢谢",
        "再见",
        "拜拜",
        "嗯好的",
        "好的知道了",
        "行吧",
        "好吧"
    ]
    
    PROTECTED_SHORT_INPUTS = [
        "我的密码是123456",
        "验证码是888888",
        "卡号后四位是6789",
        "订单号是202401010001",
        "金额是1000元",
        "手机号是13800138000",
        "身份证后四位是1234",
        "房间号是888",
        "航班号是CA1234",
        "座位号是12A"
    ]
    
    LONG_TEXT_TEMPLATES = [
        """在软件开发过程中，我们经常遇到各种技术挑战。{intro}
        
首先，让我们讨论一下{topic1}。{detail1}。这是一个非常重要的概念，因为它直接影响到系统的{aspect1}。
在实际项目中，我们发现{example1}。

其次，{topic2}也是一个关键点。{detail2}。很多开发者在这方面容易犯错误，比如{mistake1}。
为了避免这些问题，我们建议{solution1}。

第三，关于{topic3}，我们需要特别注意。{detail3}。
这涉及到{aspect2}和{aspect3}的平衡。

最后，总结一下：{summary}。希望这些内容对你有所帮助。""",
        
        """今天我们来深入探讨{topic1}这个话题。{intro}
        
背景介绍：{background}。这个领域近年来发展迅速，出现了许多新技术和新方法。

核心概念：{concept1}是指{definition1}。它包含以下几个关键要素：
1. {element1}：{desc1}
2. {element2}：{desc2}
3. {element3}：{desc3}

实践应用：在实际项目中，我们通常会{practice1}。例如{example1}。
这样做的好处是{benefit1}，但也需要注意{caution1}。

常见问题：很多初学者会遇到{problem1}。解决方案是{solution1}。
另一个常见问题是{problem2}，可以通过{solution2}来解决。

总结：{summary}。掌握这些知识，将帮助你在{field}领域取得更好的成果。"""
    ]
    
    def __init__(self, seed: int = 42):
        """初始化生成器"""
        random.seed(seed)
    
    def generate_conversation(self, topic: str = None, num_turns: int = 3) -> TestConversation:
        """生成一个测试对话"""
        topic = topic or random.choice(list(self.TOPICS.keys()))
        topic_data = self.TOPICS[topic]
        
        conv = TestConversation(topic=topic, tags=[topic])
        
        for _ in range(num_turns):
            template = random.choice(topic_data["templates"])
            keywords = topic_data["keywords"]
            
            filled = template.format(
                keyword=random.choice(keywords),
                keyword2=random.choice(keywords) if "{keyword2}" in template else "",
                action=random.choice(topic_data.get("actions", ["处理"])),
                problem=random.choice(topic_data.get("problems", ["问题"])),
                detail=random.choice(topic_data.get("details", ["相关内容"])),
                time=random.choice(topic_data.get("times", ["下午"])),
                intro="这是一个关于技术的问题。",
                topic1=random.choice(keywords),
                topic2=random.choice(keywords),
                topic3=random.choice(keywords),
                detail1="这是第一个要点的详细说明",
                detail2="这是第二个要点的详细说明",
                detail3="这是第三个要点的详细说明",
                aspect1="性能",
                aspect2="可维护性",
                aspect3="可扩展性",
                example1="在某项目中应用了这个概念",
                mistake1="过度设计",
                solution1="遵循最佳实践",
                summary="以上是本次讨论的主要内容",
                background="这个领域有着悠久的历史",
                concept1="核心概念",
                definition1="一个重要的技术术语",
                element1="要素一",
                element2="要素二",
                element3="要素三",
                desc1="第一个要素的描述",
                desc2="第二个要素的描述",
                desc3="第三个要素的描述",
                practice1="采用迭代开发的方式",
                benefit1="提高开发效率",
                caution1="不要过度优化",
                problem1="性能瓶颈",
                problem2="代码耦合",
                solution2="重构代码",
                field="软件开发"
            )
            
            response = f"关于您的问题，我来详细解答：{filled[:50]}..."
            
            conv.add_turn(
                user_input=filled,
                assistant_response=response,
                metadata={"topic": topic, "generated": True}
            )
        
        return conv
    
    def generate_batch_conversations(
        self, 
        num_conversations: int = 10,
        topics: List[str] = None,
        turns_per_conversation: tuple = (2, 5)
    ) -> List[TestConversation]:
        """批量生成测试对话"""
        topics = topics or list(self.TOPICS.keys())
        conversations = []
        
        for _ in range(num_conversations):
            topic = random.choice(topics)
            num_turns = random.randint(*turns_per_conversation)
            conv = self.generate_conversation(topic=topic, num_turns=num_turns)
            conversations.append(conv)
        
        return conversations
    
    def generate_nonsense_inputs(self, count: int = 10) -> List[str]:
        """生成废话输入"""
        return random.sample(self.NONSENSE_TEMPLATES, min(count, len(self.NONSENSE_TEMPLATES)))
    
    def generate_protected_short_inputs(self, count: int = 10) -> List[str]:
        """生成受保护的短输入（包含数字）"""
        return random.sample(self.PROTECTED_SHORT_INPUTS, min(count, len(self.PROTECTED_SHORT_INPUTS)))
    
    def generate_long_text(self, min_length: int = 500, topic: str = "programming") -> str:
        """生成用于压缩测试的长文本"""
        template = random.choice(self.LONG_TEXT_TEMPLATES)
        topic_data = self.TOPICS.get(topic, self.TOPICS["programming"])
        keywords = topic_data["keywords"]
        
        text = template.format(
            intro="这是一个技术讨论的详细记录。",
            topic1=random.choice(keywords),
            topic2=random.choice(keywords),
            topic3=random.choice(keywords),
            detail1="这是关于第一个主题的详细说明，包含了大量的技术细节和实践经验",
            detail2="第二个主题同样重要，涉及到系统设计的核心原则",
            detail3="第三个主题关注的是实际应用中的问题和解决方案",
            aspect1="性能和可靠性",
            aspect2="可维护性",
            aspect3="可扩展性",
            example1="在某大型项目中，我们应用了这些原则，取得了显著的效果",
            mistake1="过度设计和过早优化",
            solution1="遵循YAGNI原则，按需开发",
            background="这个领域在过去十年中经历了巨大的变革",
            concept1="核心概念",
            definition1="一个关键的技术术语，指的是系统中最重要的组成部分",
            element1="模块化设计",
            element2="接口隔离",
            element3="依赖注入",
            desc1="将系统分解为独立的模块，每个模块负责特定的功能",
            desc2="定义清晰的接口，降低模块间的耦合度",
            desc3="通过依赖注入实现松耦合，提高系统的灵活性",
            practice1="采用测试驱动开发（TDD）的方法",
            benefit1="代码质量显著提高，bug数量大幅减少",
            caution1="不要为了测试而测试，要关注业务价值",
            problem1="代码复杂度过高",
            problem2="测试覆盖率不足",
            solution2="重构代码，增加单元测试",
            summary="通过以上讨论，我们了解了软件开发中的关键概念和最佳实践",
            field="软件工程"
        )
        
        while len(text) < min_length:
            text += f"\n\n补充说明：{random.choice(keywords)}是{random.choice(['重要的', '关键的', '核心的'])}概念。"
        
        return text[:min_length + 200]
    
    def generate_memory_test_data(self) -> Dict[str, Any]:
        """生成记忆测试数据"""
        return {
            "programming_memories": [
                "我喜欢使用Python进行数据分析",
                "最近在学习机器学习和深度学习",
                "项目使用FastAPI作为后端框架",
                "数据库选用了PostgreSQL",
                "前端使用React框架开发"
            ],
            "project_memories": [
                "项目名称是智能助手",
                "当前处于第二迭代阶段",
                "下周三有项目评审会议",
                "技术负责人是张三",
                "预计下个月上线"
            ],
            "daily_memories": [
                "每天早上9点站会",
                "周三下午有技术分享",
                "本周五提交周报",
                "下周一有客户演示",
                "每天记录工作日志"
            ],
            "unrelated_memories": [
                "今天天气不错",
                "午餐吃了面条",
                "下班后去健身房",
                "周末计划看电影",
                "最近在读一本小说"
            ]
        }
    
    def generate_query_expected_pairs(self) -> List[Dict[str, Any]]:
        """生成查询-预期结果对"""
        memory_data = self.generate_memory_test_data()
        
        return [
            {
                "query": "项目用的是什么技术栈？",
                "expected_keywords": ["Python", "FastAPI", "PostgreSQL", "React"],
                "should_find": True,
                "memory_source": "programming_memories"
            },
            {
                "query": "项目进度怎么样？",
                "expected_keywords": ["第二迭代", "下个月上线", "评审会议"],
                "should_find": True,
                "memory_source": "project_memories"
            },
            {
                "query": "这周有什么安排？",
                "expected_keywords": ["站会", "技术分享", "周报", "演示"],
                "should_find": True,
                "memory_source": "daily_memories"
            },
            {
                "query": "量子计算的原理是什么？",
                "expected_keywords": [],
                "should_find": False,
                "memory_source": None
            },
            {
                "query": "推荐一个旅游目的地",
                "expected_keywords": [],
                "should_find": False,
                "memory_source": None
            }
        ]


def get_test_data_generator(seed: int = 42) -> TestDataGenerator:
    """获取测试数据生成器单例"""
    return TestDataGenerator(seed=seed)
