# Ghost Agent · 智能 OnCall Agent

企业级 AI 运维自动化助手。整合三大核心 Agent 能力，提供问题自动应答与故障智能排查一体化服务：

- **知识库 Agent（RAG）**：文档加载 → 分片 → 嵌入 → 向量检索 → 增强生成
- **对话 Agent（ReAct）**：思考-行动交替循环，自动调用工具与知识库
- **运维 Agent（Plan-Execute-Replan）**：规划 → 执行 → 重规划，自动排查并上报

## 技术栈（Python）

FastAPI · LangChain · LangGraph · pymilvus (Milvus) · Doubao-embedding-text-240715 · MCP · Hypothesis

## 目录结构

```
src/ghost_agent/
├── clients/     # Doubao / Milvus 客户端封装
├── models/      # Pydantic 数据模型与错误模型
├── core/        # Loader/Transformer/Indexer/Retriever/Chat_Model/Prompt/Tool/MCP
├── vector_db/   # Milvus vector_store
├── memory/      # Memory_Module（短期/长期记忆）
├── agents/      # 知识库 / 对话 / 运维 Agent
└── api/         # FastAPI 路由与启动装配
tests/           # 单元测试 + 属性测试（Hypothesis）
```

## 环境搭建（conda）

```bash
conda create -n ghost_agent python=3.11 -y
conda activate ghost_agent
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

复制 `.env.example` 为 `.env` 并填入火山引擎 Doubao 的 API Key。

## 运行测试

```bash
pytest
```
