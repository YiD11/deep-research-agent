# Deep Research Agent

一个基于 LangGraph/LangChain 的研究助手，参考 [RAGentA 论文](https://github.com/tobiasschreieder/LiveRAG) 的 Claim Analysis 机制实现。

## 核心功能

1. **问题分解** - 将复杂问题拆解为可独立检索的子问题
2. **向量检索** - 基于 Qdrant 的语义搜索
3. **声明提取** - 从答案中提取带引用的原子声明
4. **覆盖度分析** - 判断答案是否完整覆盖问题各组成部分
5. **智能补全** - 对未回答的方面补充检索

## 架构

### 整体流程

```
用户输入 → [summarize] → [analyze_rewrite] → [process_question并行子图] → [aggregate] → 最终答案
                              ↓
                    rag_agent → extract_answer → extract_claims → analyze_claims → process_follow_ups
```

### 子图详解 (process_question)

每个子问题进入以下处理流程：

1. **rag_agent** - 使用向量检索+MCP工具生成带 `[X]` 引用格式的答案
2. **extract_answer** - 从agent消息中提取最终答案
3. **extract_claims** - 用正则/LLM从答案中提取原子声明及其引用
4. **analyze_claims** - LLM分析：
   - 问题结构 (SINGLE/MULTIPLE)
   - 各声明回答了哪些问题组件
   - 每个组件的覆盖状态 (FULLY/PARTIALLY/NOT_ANSWERED)
   - 生成未回答组件的后续问题
5. **process_follow_ups** - 对每个后续问题：
   - 检索新文档
   - 生成答案并标注引用
   - 整合进原答案

### 状态定义

**GraphState** (主图):
- `rewrittenQuestions`: 分解后的子问题列表
- `agent_answers`: 各子问题的答案
- `all_claims`: 所有声明
- `claim_analysis`: 声明分析结果

**QuestionAnswerState** (子图):
- `question`: 当前子问题
- `answer_with_citations`: 带引用的答案
- `claims`: 提取的声明列表
- `claim_analysis`: 分析结果

## 快速开始

```bash
# 安装依赖
uv sync

# 启动Qdrant
docker-compose up -d

# 运行
uv run python main.py      # FastAPI (8000)
uv run python app.py       # Gradio UI (8080)
```

## 目录结构

```
├── core/rag.py      # RagSystem入口
├── pkg/graph.py     # LangGraph定义 + RAGentA逻辑
├── const/prompt.py  # 提示词模板
├── model/qa.py      # 数据模型
├── db/              # 存储层 (Qdrant + 文件)
└── ui/              # Gradio界面
```

## 主要文件

| 文件 | 职责 |
|------|------|
| `pkg/graph.py` | claim提取、分析、补全节点 |
| `const/prompt.py` | CLAIM_ANALYSIS_PROMPT 等 |
| `model/qa.py` | Claim, ClaimAnalysis 模型 |

## 环境变量

见 `.env.example`：
- LLM配置 (`LLM_MODEL_NAME`, `LLM_BASE_URL`, `LLM_API_KEY`)
- Qdrant配置 (`QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_COLLECTION_NAME`)
- 知识库路径 (`MARKDOWN_FILE_PATH`)

## 致谢

受 [RAGentA: Multi-Agent RAG for Attributed Question Answering](https://github.com/tobiasschreieder/LiveRAG) 启发实现了 Claim Analysis 机制。
