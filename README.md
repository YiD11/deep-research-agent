# Deep Research Agent 🕵️‍♂️

**不仅仅是问答，而是深度的研究与思考。**

Deep Research Agent 是一个基于 **LangGraph** 和 **LangChain** 构建的下一代智能研究助手。它超越了传统的 RAG（检索增强生成）系统，通过模拟人类研究员的思维过程——**拆解问题、并行探索、综合汇总**——来解决复杂的查询任务。

## 🌟 核心亮点

*   **🧠 动态思维链**: 能够自动分析模糊的用户指令，将其重写并拆解为多个清晰的研究子问题。
*   **⚡ 并行研究引擎**: 针对拆解出的子问题，并行启动多个 Agent 进行独立探索，大幅提升研究效率。
*   **🔍 深度 RAG 检索**: 结合语义理解与文档切片技术，深入挖掘私有知识库细节，提供有理有据的精准回答。
*   **📝 结构化综合**: 将来自不同视角的检索结果智能聚合成逻辑严密、内容详实的最终答案，拒绝简单的片段拼接。
*   **🧩 MCP 生态兼容**: 原生支持 **Model Context Protocol (MCP)**，可轻松扩展工具库，连接无限可能。

## 🛠️ 技术栈

本项目站在巨人的肩膀上：
*   **Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph) (Stateful Agents)
*   **Framework**: [LangChain](https://github.com/langchain-ai/langchain)
*   **Vector DB**: Qdrant / Milvus
*   **Serving**: FastAPI & Uvicorn
*   **UI**: Gradio

## 🚀 快速开始

### 1. 环境准备

本项目使用 `uv` 进行依赖管理（也可以使用 pip）。

```bash
# 使用 uv (推荐)
uv sync

# 或者使用 pip
pip install .
```

### 2. 配置

复制 `.env.example` 为 `.env` 并填入你的 API Key 和配置信息。

```bash
cp .env.example .env
```

### 3. 启动基础设施

本项目依赖 Qdrant，请确保已安装 Docker 和 Docker Compose。

```bash
docker-compose up -d
```

### 4. 启动应用

```bash
python main.py
```
