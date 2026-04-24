# 🧠 MemoryMind

> 基于三层记忆架构的本地AI助理，支持长期记忆、智能压缩和云端协同

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#许可证)

---

## 📖 项目简介

MemoryMind 是一款**隐私优先**的本地AI助理系统，通过创新的三层记忆架构实现长期记忆能力。不同于传统聊天机器人"聊完即忘"的局限，MemoryMind 能够记住你的偏好、习惯和重要信息，并在后续对话中智能检索和引用。

**解决什么问题？**
- 🔒 **隐私焦虑**：敏感数据无需上传云端，本地推理 + 本地存储
- 🧠 **记忆缺失**：AI 记不住之前的对话，每次都是"新朋友"
- 💰 **成本控制**：本地 LLM 免费运行，云端 AI 按需调用
- 🎯 **信息噪音**：智能过滤废话，只保留有价值记忆

**适用场景**
| 场景 | 说明 |
|------|------|
| 📚 个人知识管理 | 记录学习笔记、项目经验，智能检索回顾 |
| 🤖 智能客服助理 | 记住用户偏好和历史问题，提供个性化服务 |
| 👨‍🎓 学习伴侣 | 追踪学习进度，推荐复习内容 |
| 💼 工作助手 | 记住会议决议、任务安排，提醒重要事项 |

**与同类项目的核心差异**

| 特性 | MemoryMind | 典型聊天机器人 | RAG 系统 |
|------|------------|----------------|----------|
| 本地优先 | ✅ 完全本地运行 | ❌ 依赖云端 | ⚠️ 部分本地 |
| 长期记忆 | ✅ 三层架构 | ❌ 无记忆 | ✅ 向量检索 |
| 智能压缩 | ✅ LLM 驱动 | ❌ 无 | ⚠️ 简单截断 |
| 废话过滤 | ✅ 三层过滤 | ❌ 无 | ❌ 无 |
| 云端协同 | ✅ 按需切换 | ❌ 强制云端 | ❌ 单一模式 |
| 记忆合并 | ✅ 相似记忆合并 | ❌ 无 | ❌ 无 |
| 健康监控 | ✅ 组件健康检查 | ❌ 无 | ⚠️ 基础监控 |

---

## ✨ 核心特性

| 特性 | 状态 | 说明 |
|------|------|------|
| 🏠 **本地 LLM 推理** | ✅ 已实现 | 支持 llama.cpp API，完全离线运行 |
| 🧠 **三层记忆架构** | ✅ 已实现 | L1 内存 → L2 向量库 → L3 数据库，自动流转 |
| 🔍 **向量语义检索** | ✅ 已实现 | ONNX + BGE 模型，本地嵌入计算 |
| 🗜️ **智能记忆压缩** | ✅ 已实现 | LLM 驱动压缩，保留关键信息 |
| 🧹 **废话过滤器** | ✅ 已实现 | 规则 + 密度 + 向量三层过滤 |
| ☁️ **云端 AI 协同** | ✅ 已实现 | 支持 OpenAI / Gemini / GLM 按需切换 |
| 🔐 **敏感信息过滤** | ✅ 已实现 | 自动识别并保护密码、密钥等敏感数据 |
| 📊 **系统监控面板** | ✅ 已实现 | 实时查看记忆统计、性能指标 |
| 🔄 **记忆权重衰减** | ✅ 已实现 | 自动遗忘过期记忆，保持系统精简 |
| 🔀 **记忆合并** | ✅ 已实现 | 检测并合并相似记忆，减少冗余 |
| 🔒 **事务一致性** | ✅ 已实现 | 两阶段提交确保跨存储原子性 |
| 🩺 **健康检查** | ✅ 已实现 | 组件状态监控与降级状态追踪 |
| 📈 **性能指标** | ✅ 已实现 | 延迟分布、命中率、队列状态监控 |
| 🏷️ **记忆标签系统** | ✅ 已实现 | 压缩/流转/保护/遗忘标签统一管理 |
| 🎯 **后台任务调度** | ✅ 已实现 | 统一协调压缩/迁移/合并/遗忘任务 |
| 🔄 **生命周期管理** | ✅ 已实现 | 服务启动/关闭顺序控制，优雅退出 |
| 🔌 **线程池管理** | ✅ 已实现 | 按任务类型隔离，拒绝策略保护 |
| 🔗 **多 AI 路由** | 📝 规划中 | 根据任务类型自动选择最佳 AI |
| 🔒 **本地加密存储** | 📝 规划中 | AES 加密敏感记忆 |

---

## 🏗️ 架构概览

### 分层架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         用户界面层 (UI)                               │
│                    chat_window.py + ui_state.py                      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         应用服务层                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ LLM Client   │  │ Cloud Client │  │ Context      │               │
│  │ (llama.cpp)  │  │ (OpenAI/     │  │ Builder      │               │
│  │              │  │ Gemini/GLM)  │  │              │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         记忆管理层                                    │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │              Memory Manager (三层记忆)                       │     │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐                │     │
│  │  │   L1    │───▶│   L2    │───▶│   L3    │                │     │
│  │  │  内存层  │    │ 向量库层 │    │ 数据库层 │                │     │
│  │  │ (当前)  │    │(ChromaDB)│   │ (SQLite) │                │     │
│  │  └─────────┘    └─────────┘    └─────────┘                │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ Nonsense     │  │ Sensitive    │  │ Memory       │               │
│  │ Filter       │  │ Filter       │  │ Transaction  │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ Memory       │  │ Memory       │  │ Background   │               │
│  │ Merger       │  │ Tags         │  │ Scheduler    │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         基础设施层                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ Vector Store │  │ SQLite Store │  │ Embedding    │               │
│  │ (ChromaDB)   │  │ (SQLite)     │  │ Service      │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ Event Bus    │  │ Logger       │  │ Metrics      │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ Health Check │  │ Lifecycle    │  │ Thread Pool  │               │
│  │              │  │ Manager      │  │ Manager      │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
```

### 数据流说明

```
用户输入
    │
    ▼
┌─────────────────┐
│  废话过滤器      │ ──▶ 丢弃纯噪音
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  敏感信息过滤    │ ──▶ 保护密码/密钥
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  记忆检索        │ ◀── L1 → L2 → L3 三层检索
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  上下文构建      │ ──▶ 组装相关记忆 + 用户问题
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  LLM 推理        │ ──▶ 本地 or 云端
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  异步处理        │ ──▶ 去重 → 压缩 → 存储
│  (调度器协调)    │ ──▶ 合并 → 遗忘 → 归档
└─────────────────┘
    │
    ▼
  返回响应
```

---

## 🚀 快速开始

### 环境要求

| 项目 | 要求 |
|------|------|
| Python | 3.9+ |
| 内存 | 建议 8GB+（运行本地 LLM） |
| 存储 | 2GB+（模型 + 数据库） |
| 操作系统 | Windows / macOS / Linux |

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/your-repo/memorymind.git
cd memorymind

# 2. 创建虚拟环境（推荐）
python -m venv venv

# Windows 激活
venv\Scripts\activate

# Linux/macOS 激活
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 下载嵌入模型（首次运行）
# 将 ONNX 格式的 BGE 模型放入 models/bge_onnx_model/ 目录
# 或使用 HuggingFace 下载：
# huggingface-cli download BAAI/bge-small-zh-v1.5 --local-dir models/bge_onnx_model
```

### 配置本地 LLM

```bash
# 下载 llama.cpp 并启动服务
# 方式一：使用预编译版本
llama-server -m your-model.gguf --port 8080

# 方式二：使用 Docker
docker run -p 8080:8080 -v ./models:/models localai/localai:latest
```

### 创建配置文件

```bash
# 创建 .env 文件
cat > .env << EOF
# 本地 LLM 配置
LOCAL_ENABLED=true
LOCAL_API_URL=http://localhost:8080/completion
LOCAL_MAX_CONTEXT=8192

# 云端 AI 配置（可选）
CLOUD_ENABLED=false
CLOUD_PROVIDER=gemini
GEMINI_API_KEY=your-api-key-here

# 记忆配置
MEMORY_DECAY_DAYS=30
COMPRESSION_MIN_LENGTH=100
EOF
```

### 启动应用

```bash
python chat_window.py
```

### 首次对话示例

```
👤 你: 你好，我是张三，我是一名软件工程师

🤖 助理: 你好张三！很高兴认识你。作为软件工程师，你平时主要使用哪些技术栈呢？

👤 你: 我主要用 Python 和 TypeScript，最近在学习 Rust

🤖 助理: 很棒的技术组合！Python 和 TypeScript 覆盖了后端和前端，Rust 则是系统级编程的好选择...

--- 后续对话 ---

👤 你: 还记得我是做什么的吗？

🤖 助理: 当然！你是张三，一名软件工程师，主要使用 Python 和 TypeScript，最近在学习 Rust。
```

---

## ⚙️ 配置说明

### 核心配置项

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `LOCAL_ENABLED` | `true` | 是否启用本地 LLM |
| `LOCAL_API_URL` | `http://localhost:8080/completion` | llama.cpp API 地址 |
| `LOCAL_MAX_CONTEXT` | `8192` | 本地 LLM 上下文长度 |
| `LOCAL_TEMPERATURE` | `0.7` | 生成温度 |
| `CLOUD_ENABLED` | `false` | 是否启用云端 AI |
| `CLOUD_PROVIDER` | `gemini` | 云端提供商：openai/gemini/glm |
| `GEMINI_API_KEY` | - | Gemini API 密钥 |
| `OPENAI_API_KEY` | - | OpenAI API 密钥 |
| `GLM_API_KEY` | - | 智谱 GLM API 密钥 |
| `MEMORY_DECAY_DAYS` | `30` | 记忆遗忘周期（天） |
| `MEMORY_MIN_WEIGHT` | `0.3` | 最小记忆权重阈值 |
| `COMPRESSION_MIN_LENGTH` | `100` | 触发压缩的最小文本长度 |
| `COMPRESSION_TARGET_RATIO` | `0.6` | 压缩目标比例 |
| `NONSENSE_FILTER_ENABLED` | `true` | 是否启用废话过滤 |
| `SENSITIVE_FILTER_ENABLED` | `true` | 是否启用敏感信息过滤 |
| `MAX_RETRIEVE_RESULTS` | `5` | 检索返回的最大记忆数 |
| `SIMILARITY_THRESHOLD` | `0.90` | 相似度阈值 |

### 配置文件位置

```
memorymind/
├── .env                    # 环境变量配置（需创建）
├── config.py               # 默认配置定义
├── memory.db               # SQLite 数据库（自动创建）
├── chroma_db/              # ChromaDB 向量库（自动创建）
└── nonsense_library.json   # 废话库配置
```

---

## 📚 使用指南

### 基本对话操作

| 操作 | 说明 |
|------|------|
| `Ctrl + Enter` | 发送消息 |
| `清空对话` | 清空当前会话（不影响记忆） |
| `立即保存记忆` | 手动触发记忆持久化 |
| `系统监控` | 查看记忆统计和性能指标 |

### 记忆管理功能

```
┌─────────────────────────────────────────────────────────────────────┐
│  记忆生命周期                                                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   新对话 ──▶ L1 内存层（25条）──▶ L2 向量库 ──▶ L3 数据库            │
│                │                  │              │                  │
│                │                  │              │                  │
│                ▼                  ▼              ▼                  │
│            当前会话           热数据检索      冷数据归档              │
│                              智能压缩        权重衰减                │
│                              去重存储        自动遗忘                │
│                              记忆合并        事务保护                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 云端 AI 切换

在界面底部勾选「启用云端 AI」即可切换：

| 模式 | 说明 |
|------|------|
| 仅本地 | 完全离线，隐私最佳 |
| 仅云端 | 需要网络，能力更强 |
| 混合模式 | 本地压缩 + 云端回答 |

### 废话过滤策略

```
输入文本
    │
    ▼
┌─────────────────┐
│ 第一层：规则过滤  │ ──▶ 快速拦截明显废话（"好的"、"嗯嗯"）
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 第二层：密度评分  │ ──▶ 计算信息密度，低密度标记
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 第三层：向量匹配  │ ──▶ 与废话库对比相似度
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 存储决策         │
│ • full: 存向量库 │
│ • sqlite_only    │
│ • discard: 丢弃  │
└─────────────────┘
```

---

## 📁 项目结构

```
memorymind/
├── chat_window.py          # 主界面入口
├── config.py               # 全局配置管理
├── models.py               # 数据模型定义
│
├── llm_client.py           # 本地 LLM 客户端 (llama.cpp)
├── cloud_client.py         # 云端 AI 客户端 (OpenAI/Gemini/GLM)
├── glm_client.py           # 智谱 GLM 专用客户端
│
├── memory_manager.py       # 三层记忆管理器
├── vector_store.py         # L2 向量存储 (ChromaDB)
├── sqlite_store.py         # L3 数据库存储 (SQLite)
│
├── context_builder.py      # 上下文构建器
├── async_processor.py      # 异步处理器（压缩/去重/存储）
│
├── nonsense_filter.py      # 废话过滤器
├── sensitive_filter.py     # 敏感信息过滤器
│
├── embedding_service.py    # ONNX 嵌入服务
├── memory_archiver.py      # 记忆归档器
├── memory_merger.py        # 记忆合并器
├── memory_tags.py          # 记忆标签系统
├── memory_transaction.py   # 记忆事务协调器
│
├── background_scheduler.py # 后台任务调度器
├── compression_strategies.py # 压缩策略模块
├── thread_pool_manager.py  # 线程池管理器
├── lifecycle_manager.py    # 生命周期管理器
│
├── event_bus.py            # 事件总线
├── logger.py               # 结构化日志
├── metrics.py              # 性能指标收集
├── health_check.py         # 健康检查
├── system_health_checklist.py # 系统健康检查清单
├── token_utils.py          # Token 计算工具
├── prompts.py              # 提示词模板
├── ui_state.py             # UI 状态管理
├── tag_classifier.py       # 标签分类器
│
├── requirements.txt        # Python 依赖
├── nonsense_library.json   # 废话库配置
└── AGENTS.md               # 开发规范文档
```

### 核心文件职责

| 文件 | 职责 |
|------|------|
| [memory_manager.py](memory_manager.py) | 三层记忆统一管理，检索/存储/迁移协调 |
| [vector_store.py](vector_store.py) | ChromaDB 向量存储，语义检索 |
| [sqlite_store.py](sqlite_store.py) | SQLite 持久化存储，冷数据归档 |
| [async_processor.py](async_processor.py) | 后台异步处理：压缩/去重/遗忘 |
| [background_scheduler.py](background_scheduler.py) | 统一协调所有记忆状态变更操作 |
| [memory_transaction.py](memory_transaction.py) | 两阶段提交，跨存储原子性保证 |
| [memory_merger.py](memory_merger.py) | 相似记忆检测与合并 |
| [memory_tags.py](memory_tags.py) | 记忆标签统一管理 |
| [nonsense_filter.py](nonsense_filter.py) | 三层废话过滤，存储策略决策 |
| [context_builder.py](context_builder.py) | 组装检索记忆 + 用户问题 |
| [lifecycle_manager.py](lifecycle_manager.py) | 服务启动/关闭顺序控制 |
| [thread_pool_manager.py](thread_pool_manager.py) | 线程池隔离与拒绝策略 |
| [health_check.py](health_check.py) | 组件健康状态监控 |
| [metrics.py](metrics.py) | 性能指标收集与统计 |
| [config.py](config.py) | 全局配置，支持环境变量覆盖 |

---

## 🗓️ 开发进度

### 当前版本：v0.3.0

### ✅ 已完成功能

- [x] 三层记忆架构 (L1/L2/L3)
- [x] 本地 LLM 推理 (llama.cpp)
- [x] 云端 AI 支持 (OpenAI/Gemini/GLM)
- [x] 向量语义检索 (ChromaDB + ONNX)
- [x] 智能记忆压缩
- [x] 废话三层过滤
- [x] 敏感信息保护
- [x] 记忆权重衰减
- [x] 系统监控面板
- [x] 事件驱动架构
- [x] 结构化日志
- [x] 熔断器保护
- [x] 记忆合并（相似记忆检测与合并）
- [x] 事务协调器（两阶段提交）
- [x] 后台任务调度器
- [x] 生命周期管理器
- [x] 线程池管理器
- [x] 健康检查器
- [x] 性能指标收集
- [x] 记忆标签系统

### 📋 下一阶段规划

- [ ] 多 AI 路由（根据任务自动选择）
- [ ] 本地加密存储
- [ ] Web UI 界面
- [ ] API 接口开放
- [ ] 插件系统
- [ ] 移动端适配

---

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出建议！

```bash
# 1. Fork 项目
# 2. 创建特性分支
git checkout -b feature/your-feature

# 3. 提交更改
git commit -m "Add: your feature description"

# 4. 推送分支
git push origin feature/your-feature

# 5. 创建 Pull Request
```

**代码规范**：请参考 [AGENTS.md](AGENTS.md) 中的工程规范。

---

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

---

<p align="center">
  Made with ❤️ by MemoryMind Team
</p>
