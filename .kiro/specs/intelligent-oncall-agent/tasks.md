# Implementation Plan: 智能 OnCall Agent（Ghost Agent）— Python 技术栈

## Overview

本实现计划基于已批准的 requirements.md 与 design.md，针对 **Python 技术栈** 落地智能 OnCall Agent。技术选型遵循设计文档的技术栈映射表：

- **API 接口层**：FastAPI（`StreamingResponse` 实现 SSE）
- **Agent 编排**：LangChain + LangGraph（`StateGraph` 实现 ReAct 与 Plan-Execute-Replan；`create_react_agent` 实现 ReAct 循环）
- **向量库客户端**：`pymilvus`（Milvus Vector_Database）
- **Embedding**：Doubao（volcengine）SDK（Doubao-embedding-text-240715）
- **工具调用**：LangChain Tools / `bind_tools`（Function Call）
- **MCP 客户端**：`mcp`（python sdk）
- **属性测试**：Hypothesis（每条属性至少运行 100 次迭代）

实现方法论：测试驱动、增量推进、逐步装配，每个任务都建立在前序任务之上，最终在启动装配阶段把所有组件接线为可运行服务，不留孤立代码。所有任务描述使用中文，代码标识符、库名与结构标记保持英文。

属性测试标签统一格式：`Feature: intelligent-oncall-agent, Property {number}: {property_text}`，并使用 Hypothesis 的 `@settings(max_examples=100)`（或更高）保证至少 100 次迭代；外部依赖（Embedding_Model、Chat_Model、Milvus、MCP、send_msg）以 mock/stub 替身隔离。

## Tasks

- [x] 1. 项目脚手架与配置基础
  - [x] 1.1 初始化 Python 项目结构与依赖
    - 创建 `src/oncall_agent/` 包目录（`clients/`、`models/`、`core/`、`vector_db/`、`memory/`、`agents/`、`api/`）与 `tests/` 目录
    - 编写 `pyproject.toml`/`requirements.txt`，使用 pinned 版本固定依赖：`fastapi`、`uvicorn`、`langchain`、`langgraph`、`pymilvus`、`volcengine-python-sdk`（Doubao）、`mcp`、`pydantic`、`pytest`、`hypothesis`
    - 配置 `pytest.ini`/`pyproject` 的 pytest 与 Hypothesis profile（默认 `max_examples>=100`），创建 `tests/conftest.py` 公共 fixtures 占位
    - _Requirements: 23.1_

  - [x] 1.2 实现配置管理模块与技术栈启动校验
    - 创建 `src/oncall_agent/config.py`，集中管理所有可配置参数：`/chat` 处理超时(60s)、SSE 空闲超时(30s)、嵌入最大重试次数(0–5,默认3)、ReAct 最大迭代次数(1–50,默认10)、工具调用超时(1–300s,默认30s)、模型调用超时(1–120s,默认60s)、最大重规划次数(1–50,默认10)、历史消息 Top-K(1–50)、检索 Top-K(1–100,默认5)、最小相似度阈值、短期记忆保留条数上限(≥1)、单分片最小/最大长度、Embedding 维度与最大输入长度、受支持文档类型集合、单文件大小上限、最大步骤数上限
    - 实现 `tech_stack` 配置项与 `validate_tech_stack()`：仅允许 `{python, go, java}`，不在集合内时抛出"不支持该技术栈"错误；并校验单次部署仅启用一种技术栈
    - 实现参数范围校验（越界时拒绝并报错）
    - _Requirements: 23.1, 23.6, 23.7_

  - [ ]* 1.3 编写 config 单元测试
    - 测试参数范围越界拒绝、技术栈 guard（合法/非法值）、默认值加载
    - _Requirements: 23.6, 23.7_

- [x] 2. 数据模型层
  - [x] 2.1 实现统一错误模型与枚举
    - 创建 `src/oncall_agent/models/errors.py`：`ErrorResponse`(error_code, message, details?) 与领域异常类（如 `EmptyMessageError`、`FileTooLargeError`、`DimensionMismatchError`、`TemplateNotFoundError`、`ToolValidationError`、`ToolNotFoundError` 等），统一错误码常量
    - _Requirements: 1.3, 1.6, 16.3, 16.6, 20.5, 21.4_

  - [x] 2.2 实现核心数据模型与校验
    - 创建 `src/oncall_agent/models/` 下各 Pydantic 模型：`Chunk`(chunk_id, source_file_id, seq, start_offset, end_offset, text, parent_chunk_id?；约束 start_offset<=end_offset)、`VectorRecord`(id, vector, text, source_id, vector_type∈{DOC_CHUNK,MESSAGE}, metadata)、`Session`/`Message`(role∈{USER,ASSISTANT}, created_at)/`ShortTermMemory`/`LongTermSummary`、`IngestTask`(status∈{PENDING,RUNNING,COMPLETED,FAILED}, chunk_count?, failure_reason?)、`TroubleshootingTask`(trigger_type, target, alarm, status, replan_count, report_status?)、`AnalysisSummary`(root_cause/suggestions/executed_actions 支持 NO_CONTENT)、`Plan`/`Step`(order, tool_name, goal, status)/`StepResult`/`StepFailure`/`ReplanVerdict`(COMPLETED/CONTINUE/REPLAN)、`ToolDefinition`(name, description, params, source∈{BUILTIN,MCP})/`ParamDef`(name, type, required)
    - 实现各模型字段类型与必填校验
    - _Requirements: 6.3, 18.6, 21.3, 3.2, 4.2, 11.2, 13.1, 14.2, 16.1_

  - [ ]* 2.3 编写数据模型单元测试
    - 测试各模型字段校验、枚举取值、序列化/反序列化往返
    - _Requirements: 6.3, 16.1, 21.3_

- [x] 3. 基础设施客户端与向量库层
  - [x] 3.1 实现 Doubao Embedding 客户端与 Milvus 客户端封装
    - 创建 `src/oncall_agent/clients/doubao_client.py`：封装 Doubao（volcengine）SDK 调用 Doubao-embedding-text-240715，提供 `embed(texts) -> vectors`，暴露输出维度 `dim` 与最大输入长度 `max_input_len`；统一异常封装供上层重试
    - 创建 `src/oncall_agent/clients/milvus_client.py`：封装 `pymilvus` 连接（含连接超时配置）、collection 句柄获取
    - _Requirements: 21.1, 21.2, 23.4_

  - [x] 3.2 实现 Vector_Database 层（vector_store）
    - 创建 `src/oncall_agent/vector_db/vector_store.py`：基于 `pymilvus` 创建/管理 collection（统一存储 DOC_CHUNK 与 MESSAGE 向量）
    - 实现 `write(record)`：写入前校验 `len(vector)==dim`，不一致则拒绝并返回维度不一致错误；写入时同时持久化原始文本、来源标识(source_id)、向量类型(vector_type) 与 metadata
    - 实现 `search(query_vector, top_k, min_score, source_scope?)`、`delete_by_source(source_file_id) -> deleted_count`、连接超时返回向量库不可用错误且保留未写入数据不丢失
    - _Requirements: 21.1, 21.3, 21.4, 21.5, 19.1_

  - [ ]* 3.3 编写属性测试：向量写入记录必含元数据与维度校验
    - **Feature: intelligent-oncall-agent, Property 26: 对任意向量写入请求，当向量维度等于 Embedding_Model 输出维度时写入成功且持久化记录同时包含原始文本、来源标识与向量类型；当向量维度不等于该维度时拒绝写入并返回维度不一致错误。**
    - 使用 Hypothesis 生成维度等于/不等于 dim 的向量与随机元数据；以内存替身 Milvus 隔离；`max_examples>=100`
    - _Requirements: 21.3, 21.4_

  - [ ]* 3.4 编写 vector_store 单元测试
    - 测试连接超时返回不可用错误且保留未写入数据、delete_by_source 计数、search 阈值过滤
    - _Requirements: 21.5_

- [x] 4. 核心组件：Loader 与 Transformer（分片）
  - [x] 4.1 实现 Loader（文档加载）
    - 创建 `src/oncall_agent/core/loader.py`：将受支持类型文件解析为统一纯文本 `ParseResult`(text, sections[title,level,paragraphs], meta)，保留标题层级顺序与段落归属
    - 解析失败/解析超时/内容为空时终止、不移交 Transformer，返回对应错误并将 `IngestTask` 标记为 FAILED；成功时附带至少含来源文件标识、文件名、文件格式的元数据
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

  - [ ]* 4.2 编写 Loader 单元测试
    - 测试结构化标题层级保留、解析失败/超时/空内容三种失败路径标记任务为 FAILED
    - _Requirements: 5.2, 5.3, 5.5, 5.6_

  - [x] 4.3 实现 Transformer（分片策略）
    - 创建 `src/oncall_agent/core/transformer.py`：实现 BY_HEADING、BY_PARAGRAPH、BY_SEMANTIC 三种策略，未指定时取配置默认策略
    - 将非空内容切分为 ≥1 个 Chunk（除最后一个外长度介于 [minLen,maxLen]）；为每个 Chunk 附来源文件标识、顺序序号(0..n-1)、起止位置；单 Chunk 超过 Embedding 最大输入长度时二次切分为不超过该长度的子 Chunk（设置 parent_chunk_id）
    - 内容为空不生成 Chunk、不移交；策略不可应用时终止并返回错误；完成后移交 Indexer
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

  - [ ]* 4.4 编写属性测试：分片长度边界
    - **Feature: intelligent-oncall-agent, Property 1: 对任意非空已解析文档内容与任一受支持的分片策略，切分产生的 Chunk 数量至少为 1，且除最后一个 Chunk 外，每个 Chunk 的文本长度均介于配置的单分片最小长度与最大长度之间（含端点）。**
    - Hypothesis 生成随机文本（含纯空白、Unicode、超长）+ 随机策略 + 长度参数；`max_examples>=100`
    - _Requirements: 6.1_

  - [ ]* 4.5 编写属性测试：超长 Chunk 二次切分上界
    - **Feature: intelligent-oncall-agent, Property 2: 对任意文本长度超过 Embedding_Model 最大输入长度的内容，分片完成后所有最终 Chunk 的文本长度均不超过该最大输入长度。**
    - Hypothesis 生成超过 max_input_len 的内容；断言所有最终 Chunk 长度 <= max_input_len；`max_examples>=100`
    - _Requirements: 6.4_

  - [ ]* 4.6 编写属性测试：Chunk 元数据完整且序号位置单调
    - **Feature: intelligent-oncall-agent, Property 3: 对任意由分片产生的 Chunk 集合，每个 Chunk 均带有来源文件标识、顺序序号与起止位置；序号在集合内连续递增（0..n-1），且每个 Chunk 的 start_offset <= end_offset，起止位置随序号单调不减。**
    - `max_examples>=100`
    - _Requirements: 6.3_

- [x] 5. 核心组件：Indexer（嵌入与索引）
  - [x] 5.1 实现 Indexer
    - 创建 `src/oncall_agent/core/indexer.py`：`index(chunks) -> {success_count, failure_count, failures}`，为每个 Chunk 调 Doubao Embedding 生成向量并写入 vector_store；嵌入失败按配置最大重试次数(0–5)重试；达上限仍失败或写入失败则记录失败、计入 failure_count 并继续其余 Chunk；空集合不调 Embedding 返回 (0,0)
    - 实现 `index_message(session_id, user_msg, answer)`：为消息生成向量并连同 session_id 写入（vector_type=MESSAGE）；失败仅记录不中断对话流程
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 19.1, 19.2_

  - [ ]* 5.2 编写属性测试：索引计数守恒
    - **Feature: intelligent-oncall-agent, Property 4: 对任意 Chunk 集合（含空集合）及任意成功/失败分布，Indexer 返回的成功写入数量与失败数量之和恒等于接收到的 Chunk 总数。**
    - Hypothesis 生成随机 Chunk 集合 + 随机成功/失败 mock（含空集合）；`max_examples>=100`
    - _Requirements: 7.5, 7.6, 22.4_

  - [ ]* 5.3 编写属性测试：嵌入重试次数上界
    - **Feature: intelligent-oncall-agent, Property 5: 对任意嵌入失败序列，Indexer 对单个 Chunk 的累计重试次数不超过配置的最大重试次数（取值范围 0–5，默认 3）。**
    - 以计数 mock 统计调用次数；`max_examples>=100`
    - _Requirements: 7.3_

  - [ ]* 5.4 编写 Indexer 单元测试
    - 测试消息向量写入失败仅记录不中断（19.2）、单 Chunk 写入失败隔离
    - _Requirements: 19.2, 7.4_

- [x] 6. 核心组件：Retriever（检索召回）
  - [x] 6.1 实现 Retriever
    - 创建 `src/oncall_agent/core/retriever.py`：`retrieve(query, opts{topK,minScore,hybrid?,rerank?,sessionScope?})`，非空查询转查询向量后从 vector_store 召回相似度 >= minScore 且最高的 Chunk（数量 <= topK）；Top-K 边界存在分数并列时全部返回（可超 Top-K）
    - 启用 hybrid 时合并向量+关键词结果、按 chunk_id 去重、按融合分数降序、返回 <= topK；启用 rerank 时按相关性降序重排返回
    - 无满足阈值结果返回空集；查询为空/trim 后为空返回错误；查询向量生成失败终止并返回错误；sessionScope 限定当前 Session 范围召回历史消息（数量由历史消息 Top-K 决定），无历史消息返回空集
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 19.3, 19.4, 19.5_

  - [ ]* 6.2 编写属性测试：检索 Top-K 与阈值不变量
    - **Feature: intelligent-oncall-agent, Property 6: 对任意向量库与非空查询，Retriever 返回结果的相似度分数均不低于配置的最小相似度阈值，结果按相似度降序排列，且在不存在边界并列的情况下返回数量不超过 Top-K（取值范围 1–100，默认 5）。**
    - Hypothesis 生成随机向量库 + 查询 + 分数；`max_examples>=100`
    - _Requirements: 8.2_

  - [ ]* 6.3 编写属性测试：混合检索去重、降序与数量上界
    - **Feature: intelligent-oncall-agent, Property 7: 对任意向量检索结果集合与关键词检索结果集合，启用混合检索后合并结果不含重复的 chunk_id，按融合分数降序排列，且返回数量不超过 Top-K。**
    - `max_examples>=100`
    - _Requirements: 8.4_

  - [ ]* 6.4 编写属性测试：重排是降序排列且为输入的排列
    - **Feature: intelligent-oncall-agent, Property 8: 对任意待重排的 Chunk 集合，重排结果按相关性分数降序排列，且重排结果恰为输入集合的一个排列（不增删元素）。**
    - 断言重排后为输入的 multiset permutation 且降序；`max_examples>=100`
    - _Requirements: 8.5_

  - [ ]* 6.5 编写属性测试：消息向量召回的 Session 范围与数量上界
    - **Feature: intelligent-oncall-agent, Property 9: 对任意跨多个 Session 的消息向量库与当前 Session 标识，在该 Session 范围内的召回结果中每条消息都归属于当前 Session，且返回数量不超过配置的历史消息 Top-K（取值范围 1–50）。**
    - Hypothesis 生成多 Session 交错消息向量；`max_examples>=100`
    - _Requirements: 19.4, 10.1_

- [ ] 7. 核心组件：Chat_Model 与 Prompt_Module
  - [ ] 7.1 实现 Chat_Model 封装
    - 创建 `src/oncall_agent/core/chat_model.py`：基于 LangChain ChatModel 封装 `generate(prompt, opts) -> Completion` 与 `stream(prompt, opts) -> AsyncIterator[Delta]`，支持工具调用（Function Call）输出结构（`bind_tools`）；超时与错误向上层透传
    - _Requirements: 9.2, 10.2, 2.2_

  - [ ] 7.2 实现 Prompt_Module（提示词工程）
    - 创建 `src/oncall_agent/core/prompt_module.py`：以唯一名称管理模板，`build(templateName, vars) -> Prompt`；提示词包含角色定义、任务目标、输出结构/格式约束说明；模板标注需分步推理时加入分步思考指令；支持不改 Agent 调用代码更新/替换同名模板；引用模板不存在时停止构造返回缺失模板名错误
    - 实现 RAG 增强提示词构造：将用户查询与召回集合全部 Chunk 构造为单条提示词并标注每个 Chunk 来源文件标识
    - _Requirements: 9.1, 20.1, 20.2, 20.3, 20.4, 20.5_

  - [ ]* 7.3 编写属性测试：增强提示词包含查询与全部来源标识
    - **Feature: intelligent-oncall-agent, Property 10: 对任意非空召回 Chunk 集合与用户查询，构造出的增强提示词包含该用户查询，且召回集合中每个 Chunk 的来源文件标识均出现在提示词中。**
    - Hypothesis 生成随机召回集合 + 查询；`max_examples>=100`
    - _Requirements: 9.1_

  - [ ]* 7.4 编写属性测试：Prompt 必含字段
    - **Feature: intelligent-oncall-agent, Property 12: 对任意提示词构造请求，生成的提示词均包含角色定义、任务目标与输出结构/格式约束说明；当模板被标注为需要分步推理时，提示词额外包含分步思考指令。**
    - `max_examples>=100`
    - _Requirements: 20.1, 20.2, 20.3_

  - [ ]* 7.5 编写 Prompt_Module 单元测试
    - 测试同名模板热替换（20.4）、引用缺失模板返回错误（20.5）
    - _Requirements: 20.4, 20.5_

- [ ] 8. 工具集与 MCP 客户端
  - [ ] 8.1 实现 Tool_Registry 与内置工具
    - 创建 `src/oncall_agent/core/tool_registry.py`：维护每个工具唯一名称、参数定义(名称/类型/必填)、功能描述；`register(toolDef)`、`invoke(name, params)`，调用时按参数定义校验类型与必填，不符合则拒绝并返回具体不符合项，工具名不存在则返回不存在错误
    - 注册内置工具 query_internal_docs、query_cls_log、query_prometheus_alarm、send_msg（以 LangChain Tools 形式，可被 `bind_tools` 使用）
    - _Requirements: 16.1, 16.2, 16.3, 16.4, 16.5, 16.6_

  - [ ]* 8.2 编写属性测试：工具参数校验
    - **Feature: intelligent-oncall-agent, Property 21: 对任意工具定义与任意调用参数，Tool_Registry 当且仅当所有必填参数齐备且各参数类型与定义匹配时放行执行；否则拒绝调用、不执行该工具并返回指明具体不符合项的校验错误。**
    - Hypothesis 生成随机工具定义 + 参数（含缺失必填、类型不符、未知工具）；`max_examples>=100`
    - _Requirements: 16.2, 16.3, 16.5_

  - [ ] 8.3 实现 MCP_Client
    - 创建 `src/oncall_agent/core/mcp_client.py`：基于 `mcp`（python sdk）`connect(server)` 获取工具清单并将每个工具注册到 Tool_Registry；`invoke(name, params)` 经 MCP 协议转发并回传响应
    - 命名冲突拒绝注册、记录冲突、保留已注册同名工具不变；连接失败不注册任何该服务端工具且不影响内置工具；MCP 工具错误响应返回执行失败错误；调用超时返回超时错误且不影响其余工具
    - _Requirements: 17.1, 17.2, 17.3, 17.4, 17.5, 17.6_

  - [ ]* 8.4 编写属性测试：MCP 命名冲突保留既有工具
    - **Feature: intelligent-oncall-agent, Property 22: 对任意已注册工具集合与一个名称冲突的待注册 MCP 工具，注册被拒绝，已注册的同名工具定义保持不变，且工具集内工具总数不变。**
    - `max_examples>=100`
    - _Requirements: 17.3_

  - [ ]* 8.5 编写 MCP 集成测试
    - 以 stub MCP 服务端测试工具注册与调用、错误响应、调用超时（17.1, 17.2, 17.5, 17.6）
    - _Requirements: 17.1, 17.2, 17.5, 17.6_

- [ ] 9. 横切能力：Memory_Module（多轮对话记忆）
  - [ ] 9.1 实现 Memory_Module
    - 创建 `src/oncall_agent/memory/memory_module.py`：`append(sessionId, userMsg, answer)` 按时间顺序追加写入对应 Session 的 Short_Term_Memory；超过保留条数上限(≥1)时将溢出较早消息总结后写入 Long_Term_Memory 并从短期记忆移除使其数量 <= 上限；总结/写入长期记忆失败时保留消息在短期记忆并记录失败
    - `load(sessionId) -> {shortTerm, longTerm}` 按时间顺序返回；保证每个 Session 记忆仅来源于且仅作用于该 Session（隔离）
    - _Requirements: 18.1, 18.2, 18.3, 18.4, 18.5, 18.6_

  - [ ]* 9.2 编写属性测试：记忆按 Session 隔离
    - **Feature: intelligent-oncall-agent, Property 23: 对任意跨多个 Session 交错写入的对话消息，加载某一 Session 的记忆时返回的所有内容均归属于该 Session，不同 Session 之间记忆内容互不泄漏。**
    - Hypothesis 生成多 Session 多轮交错消息；`max_examples>=100`
    - _Requirements: 18.6_

  - [ ]* 9.3 编写属性测试：短期记忆容量上界与溢出归档
    - **Feature: intelligent-oncall-agent, Property 24: 对任意任意轮数的对话消息序列，短期记忆中的消息数量在任意时刻均不超过配置的保留条数上限（≥1），溢出的较早消息被总结并纳入长期记忆覆盖范围。**
    - `max_examples>=100`
    - _Requirements: 18.2, 18.3_

  - [ ]* 9.4 编写属性测试：消息追加保持时间顺序
    - **Feature: intelligent-oncall-agent, Property 25: 对任意多轮对话序列，写入后短期记忆中的消息按时间先后顺序排列，且每轮的用户消息与应答均按发生顺序被追加。**
    - `max_examples>=100`
    - _Requirements: 18.1_

- [ ] 10. Checkpoint — 核心组件层完成
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 11. Knowledge_Base_Agent（RAG 入库 + 检索生成 + 同步）
  - [ ] 11.1 实现 KBA 入库管线
    - 创建 `src/oncall_agent/agents/knowledge_base_agent.py`：实现 `ingest(file, meta) -> IngestResult`，编排 Loader → Transformer → Indexer 全流程；按阶段更新 `IngestTask` 状态，成功置 COMPLETED 并提供 chunk_count，失败置 FAILED 并提供 failure_reason；为每个 Chunk 附来源文件标识
    - _Requirements: 5.1, 6.5, 7.1, 3.6, 3.7, 22.1_

  - [ ] 11.2 实现 KBA 检索增强生成（answer）
    - 在 `knowledge_base_agent.py` 增加 `answer(query) -> {answer, cited_sources}`：编排 Retriever → Prompt_Module → Chat_Model；获非空召回集合后要求 Chat_Model 仅依据 Chunk 生成答案并附引用来源文件标识列表；召回为空时明确告知未检索到且不臆造；生成错误/超时终止并返回错误，不返回部分/臆造答案
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_

  - [ ]* 11.3 编写属性测试：答案引用来源是召回来源的子集
    - **Feature: intelligent-oncall-agent, Property 11: 对任意召回 Chunk 集合与据此生成的答案，答案附带的引用来源文件标识列表是召回集合来源标识集合的子集（不包含召回集合以外的来源）。**
    - 以 stub Chat_Model 隔离；`max_examples>=100`
    - _Requirements: 9.4_

  - [ ] 11.4 实现 KBA 知识库同步与移除（sync / remove）
    - 在 `knowledge_base_agent.py` 增加 `sync(file) -> {status, success_count, failure_count}`：执行加载→分片→嵌入→写入完整流程；同源文件再次同步时以来源文件标识为依据用新 Chunk 替换旧 Chunk 且不残留旧 Chunk；任一阶段失败终止、记录失败阶段与原因并保持已有 Chunk 不变
    - 增加 `remove(sourceFileId) -> {deleted_count}`：删除该来源全部 Chunk 并返回数量；移除不存在来源不删除任何 Chunk 并返回不存在提示
    - _Requirements: 22.2, 22.3, 22.4, 22.5, 22.6_

  - [ ]* 11.5 编写属性测试：知识库同源替换不残留旧 Chunk
    - **Feature: intelligent-oncall-agent, Property 27: 对任意来源文件，对其连续同步两版内容后，Vector_Database 中归属该来源文件的 Chunk 集合恰等于新一版生成的 Chunk 集合，不残留任何旧版 Chunk。**
    - 以内存替身 vector_store 隔离；`max_examples>=100`
    - _Requirements: 22.2_

  - [ ]* 11.6 编写属性测试：来源删除计数往返
    - **Feature: intelligent-oncall-agent, Property 28: 对任意已写入 N 个 Chunk 的来源文件，移除该来源后返回的删除数量等于 N，且该来源在 Vector_Database 中剩余 Chunk 数为 0。**
    - `max_examples>=100`
    - _Requirements: 22.3_

  - [ ]* 11.7 编写属性测试：同步失败保持既有 Chunk 不变
    - **Feature: intelligent-oncall-agent, Property 29: 对任意在加载、分片、嵌入或写入任一阶段失败的同步流程，失败后该来源文件在 Vector_Database 中已有的 Chunk 集合与失败前完全一致（失败不破坏既有状态）。**
    - Hypothesis 随机选取失败阶段注入；`max_examples>=100`
    - _Requirements: 22.5_

- [ ] 12. Conversation_Agent（ReAct，LangGraph）
  - [ ] 12.1 实现 ReAct 循环并装配
    - 创建 `src/oncall_agent/agents/conversation_agent.py`：使用 LangGraph `create_react_agent`/`StateGraph` 实现 `handle(sessionId, message)`；构造提示词时召回当前 Session 相关历史消息（历史消息 Top-K，1–50）；输出含工具调用请求时经 Tool_Registry 调用并将响应作为新观察结果送回模型继续循环；输出不再含工具调用时退出循环返回最终内容
    - 迭代次数达上限(默认10,范围1–50)终止并返回已生成内容 + 达上限提示；工具错误/超时(默认30s)将失败原因作为观察结果送回模型；模型错误/超时(默认60s)终止循环、保留已生成内容、返回应答失败错误
    - 应答完成后接线 Memory_Module 写入与 Indexer 消息向量入库
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 18.1, 19.1_

  - [ ]* 12.2 编写属性测试：ReAct 迭代次数上界
    - **Feature: intelligent-oncall-agent, Property 13: 对任意 Chat_Model 行为（包括始终请求工具调用的情形），Conversation_Agent 的 ReAct 循环迭代次数不超过配置的最大迭代次数上限（默认 10，范围 1–50），且达上限时返回的应答携带"已达最大迭代次数"的提示信息。**
    - 以始终请求工具的 mock 模型 + 迭代计数；`max_examples>=100`
    - _Requirements: 10.5_

  - [ ]* 12.3 编写 Conversation_Agent 单元测试
    - 测试工具错误/超时作为观察结果送回（10.6）、模型错误/超时终止并保留已生成内容（10.7）
    - _Requirements: 10.6, 10.7_

- [ ] 13. Ops_Agent（Plan-Execute-Replan，多 Agent，LangGraph）
  - [ ] 13.1 实现 Planner_Agent
    - 创建 `src/oncall_agent/agents/planner.py`：`plan(alarm) -> Plan`，启动后调 query_internal_docs 查询处理步骤；生成有序步骤计划(步骤数 1..maxSteps)，每步标注待调用工具(取自 Tool_Registry)与目标；query_internal_docs 无返回/失败/超时则生成通用计划并标注 grounded=false；完成后移交 Executor
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6_

  - [ ]* 13.2 编写属性测试：规划步骤数边界与步骤合法性
    - **Feature: intelligent-oncall-agent, Property 14: 对任意由 Planner_Agent 生成的执行计划，步骤数量介于 1 与配置的最大步骤数上限之间，步骤序号连续，且每个步骤标注的工具名均属于 Tool_Registry 已注册工具名集合并带有非空目标。**
    - `max_examples>=100`
    - _Requirements: 11.2, 11.3_

  - [ ] 13.3 实现 Executor_Agent
    - 创建 `src/oncall_agent/agents/executor.py`：`execute(step) -> StepResult | StepFailure`，从第一步按 order 升序每次执行一步；调用步骤标注工具(来自 Tool_Registry，含 query_cls_log、query_prometheus_alarm、send_msg 与 MCP 工具)；成功记录含步骤标识与响应的结果并移交 Replanner；失败记录含步骤标识与原因的失败信息、暂停后续步骤并移交 Replanner；工具调用超时判定为该步骤失败
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

  - [ ]* 13.4 编写属性测试：执行顺序与失败短路
    - **Feature: intelligent-oncall-agent, Property 15: 对任意执行计划及任一失败步骤位置，Executor_Agent 按步骤序号升序逐步执行；一旦某步骤工具调用失败，该步骤之后的步骤不再被执行，且失败信息包含所属步骤标识。**
    - Hypothesis 生成随机计划 + 失败步索引；`max_examples>=100`
    - _Requirements: 12.1, 12.4_

  - [ ] 13.5 实现 Replanner_Agent
    - 创建 `src/oncall_agent/agents/replanner.py`：`evaluate(plan, stepResult) -> {verdict, newPlan?, replanCount}`，输出三态评估之一(COMPLETED/CONTINUE/REPLAN)；COMPLETED 终止并基于已执行结果生成总结；CONTINUE 指示执行下一步；REPLAN 生成新计划、replan_count+1、从新计划第一步执行；重规划次数达上限(默认10,范围1–50)终止并生成含"因达上限未完成"说明的总结；模型错误终止、保留已执行结果并生成含失败原因说明的总结
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6_

  - [ ]* 13.6 编写属性测试：重规划评估取值封闭
    - **Feature: intelligent-oncall-agent, Property 16: 对任意当前计划与执行结果组合，Replanner_Agent 输出的评估结果恒为 {任务已完成, 任务未完成且剩余计划仍适用, 任务未完成且剩余计划不再适用} 三个取值之一。**
    - `max_examples>=100`
    - _Requirements: 13.1_

  - [ ]* 13.7 编写属性测试：重规划次数上界
    - **Feature: intelligent-oncall-agent, Property 17: 对任意持续触发重规划的排查流程，重规划次数不超过配置的最大重规划次数上限（默认 10，范围 1–50），且达上限时终止流程并生成含"因达到最大重规划次数而未完成"说明的分析结果总结。**
    - 以持续 REPLAN mock + 计数；`max_examples>=100`
    - _Requirements: 13.5_

  - [ ] 13.8 实现 Ops_Agent 编排、触发管理与结果上报
    - 创建 `src/oncall_agent/agents/ops_agent.py`：使用 LangGraph 自定义多节点 `StateGraph` 编排 Planner → Executor → Replanner 协作循环；实现 `start(trigger)` 受理任务返回 task_id 与初始状态
    - 触发：支持 manual/scheduled/webhook(签名校验通过)启动；缺告警信息或排查目标拒绝并返回缺失字段错误；同一排查目标已有进行中流程时不启动新流程并返回指向已有任务标识的提示
    - 上报 `report(summary)`：生成总结(含根因分析、处理建议、已执行操作记录三部分，无内容部分以明确无内容说明标注)后在上报时限内调 send_msg 发送到目标群组；发送失败按间隔重试至最大次数；达上限仍失败记录失败并保留总结；成功记录上报状态与时间并标记已上报；目标群组未配置/不存在跳过发送、记录错误并保留总结
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 15.1, 15.2, 15.3, 15.4, 15.5_

  - [ ]* 13.9 编写属性测试：分析结果总结三部分齐全
    - **Feature: intelligent-oncall-agent, Property 18: 对任意分析结果总结（其中任意部分的可填充内容可能为空），总结均包含根因分析、处理建议与已执行操作记录三个部分，且对无可填充内容的部分以明确的无内容说明标注。**
    - Hypothesis 生成各部分内容随机为空/非空；`max_examples>=100`
    - _Requirements: 14.2_

  - [ ]* 13.10 编写属性测试：上报重试次数上界
    - **Feature: intelligent-oncall-agent, Property 19: 对任意 send_msg 持续失败的情形，Ops_Agent 的累计重试次数不超过配置的最大重试次数。**
    - 以持续失败 send_msg mock + 计数；`max_examples>=100`
    - _Requirements: 14.3_

  - [ ]* 13.11 编写属性测试：同一排查目标的进行中流程互斥
    - **Feature: intelligent-oncall-agent, Property 20: 对任意针对同一排查目标的并发或重复触发请求，系统至多存在一个进行中的排查流程，后续触发不启动新流程并返回指向已有进行中任务标识的提示。**
    - Hypothesis 生成同 target 并发/重复触发序列；`max_examples>=100`
    - _Requirements: 15.5_

- [ ] 14. Checkpoint — Agent 层完成
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 15. API 接口层（FastAPI）
  - [ ] 15.1 实现 /chat 接口
    - 创建 `src/oncall_agent/api/routes_chat.py`（APIRouter）：校验顺序 message 存在且 trim 后非空(否则 400 且不路由) → 长度 <= 8000(否则 400 且不路由)；缺 session_id 生成新标识并回传；校验通过路由至 Conversation_Agent.handle 并以单次响应返回完整应答与 session_id；处理超时 60s 返回 504 且保留会话上下文不变；生成错误返回错误响应且保留会话上下文不变
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_

  - [ ]* 15.2 编写属性测试：消息字段校验拒绝空白输入
    - **Feature: intelligent-oncall-agent, Property 30: 对任意缺失用户消息字段或用户消息去除首尾空白后长度为 0 的 /chat 或 /chat_stream 请求，API_Gateway 返回 HTTP 400 且不将请求路由至 Conversation_Agent（/chat_stream 场景下不建立 SSE 连接）。**
    - 使用 FastAPI TestClient 覆盖 /chat 与 /chat_stream 两路由；Hypothesis 生成缺失/纯空白消息；以 spy 断言未路由至 Conversation_Agent；`max_examples>=100`
    - _Requirements: 1.3, 2.6_

  - [ ]* 15.3 编写属性测试：消息长度上界校验
    - **Feature: intelligent-oncall-agent, Property 31: 对任意 /chat 请求，当用户消息长度超过 8000 个字符时返回 HTTP 400 且不路由至 Conversation_Agent；当长度不超过 8000 且去空白后非空时通过校验。**
    - Hypothesis 生成边界附近(8000±)长度消息；`max_examples>=100`
    - _Requirements: 1.6_

  - [ ] 15.4 实现 /chat_stream SSE 接口
    - 创建 `src/oncall_agent/api/routes_stream.py`（APIRouter）：入参校验同 /chat（缺失/空返回 400 且不建立 SSE）；用 FastAPI `StreamingResponse` 建立 SSE 单向连接；首事件 `session` 回传 session_id；按生成先后顺序推送 `delta` 事件；完成发送 `done` 并关闭；客户端断开停止生成并释放资源；Chat_Model 错误发送 `error` 事件并关闭(不撤回已推送)；空闲超时(默认30s)发送空闲超时 `error` 事件并关闭
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8_

  - [ ]* 15.5 编写属性测试：增量流式顺序保持
    - **Feature: intelligent-oncall-agent, Property 34: 对任意 Chat_Model 产出的内容增量序列，/chat_stream 推送给客户端的 delta 事件按生成先后顺序排列，且所有 delta 文本按序拼接后等于原始生成内容的拼接结果。**
    - 以可控增量序列 stub 模型；断言 delta 顺序与拼接相等；`max_examples>=100`
    - _Requirements: 2.2_

  - [ ] 15.6 实现 /upload_file 接口
    - 创建 `src/oncall_agent/api/routes_upload.py`（APIRouter）：multipart 接收文件；校验顺序 文件存在且非空(否则 400) → 文档类型受支持(否则 415 返回受支持类型列表) → 大小 <= 上限(否则 413 返回上限数值)；通过则移交 KBA.ingest 返回 task_id 与初始状态 PENDING；提供 `GET /upload_file/{task_id}` 返回状态、chunk_count?、failure_reason?
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

  - [ ]* 15.7 编写属性测试：文档类型与大小校验
    - **Feature: intelligent-oncall-agent, Property 32: 对任意上传文件，当文档类型不属于受支持类型集合时返回 HTTP 415、当文件大小超过单文件上限时返回 HTTP 413，且两种情形均不进行入库处理；仅当类型受支持且大小不超上限且非空时移交入库。**
    - Hypothesis 生成随机类型/大小组合；以 spy 断言未/已移交入库；`max_examples>=100`
    - _Requirements: 3.4, 3.5_

  - [ ] 15.8 实现 /ai_ops 接口
    - 创建 `src/oncall_agent/api/routes_ops.py`（APIRouter）：接收触发请求路由至 Ops_Agent.start，5s 内返回 task_id 与 "ACCEPTED" 状态；缺告警信息或排查目标返回 400 且不创建任务；trigger_type=webhook 时先校验来源签名，失败返回 401；Ops_Agent 不可用或并发达上限返回 503
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [ ]* 15.9 编写属性测试：排查触发缺失字段校验
    - **Feature: intelligent-oncall-agent, Property 33: 对任意缺少告警信息或缺少排查目标的排查触发请求（经由 /ai_ops 或 Ops_Agent 触发），系统拒绝该次触发、不创建排查任务并返回描述缺失字段的错误信息。**
    - Hypothesis 生成缺 alarm / 缺 target / 两者皆缺的请求；`max_examples>=100`
    - _Requirements: 4.3, 15.4_

  - [ ]* 15.10 编写 API 路由单元测试
    - 测试请求路由(1.1, 4.1)、会话标识生成(1.4, 2.7)、任务状态转换(3.6, 3.7, 14.5)、webhook 签名通过路径(4.4)、503 并发上限(4.6)
    - _Requirements: 1.1, 1.4, 2.7, 4.1, 4.4, 4.6_

- [ ] 16. 集成与启动装配
  - [ ] 16.1 实现应用启动装配与启动校验
    - 创建 `src/oncall_agent/api/app.py` 与 `main.py`：构造 FastAPI app、include 全部 APIRouter；启动时执行配置与技术栈 guard(仅 python 启用、非 {python,go,java} 拒绝启动)；初始化 Doubao Embedding 与 Milvus 客户端(唯一 Vector_Database 与唯一 Embedding_Model)；注册四个内置工具；连接 MCP 服务端注册 MCP 工具(连接失败不影响内置工具)；将 Conversation_Agent、Knowledge_Base_Agent、Ops_Agent 与各核心组件接线为依赖
    - _Requirements: 16.4, 21.1, 21.2, 23.4, 23.6, 23.7_

  - [ ]* 16.2 编写端到端集成测试
    - 以替身 Embedding/Chat_Model/Milvus/MCP 跑通：上传→入库→检索→生成答案管线(22.1)；/ai_ops 触发→规划→执行→重规划→上报流程；Milvus 连接超时与数据保留(21.5)
    - _Requirements: 22.1, 21.5, 17.1, 17.2_

  - [ ]* 16.3 编写冒烟测试
    - 验证四个内置工具已注册(16.4)、Milvus 为唯一向量库(21.1)、Doubao 为唯一嵌入模型(21.2, 23.4)、单次部署仅启用一种技术栈(23.6)、不支持技术栈拒绝启动(23.7)
    - _Requirements: 16.4, 21.1, 21.2, 23.4, 23.6, 23.7_

- [ ] 17. Final Checkpoint — 全量验证
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- 标记 `*` 的子任务为可选（单元测试、属性测试、集成/冒烟测试），可为加速 MVP 跳过；顶层任务不带 `*`。
- 每个任务标注其实现的具体需求条款，保证可追溯性。
- Checkpoint 任务用于阶段性增量验证。
- 属性测试覆盖设计文档全部 34 条 Correctness Properties（P1–P34），每条独立为一个属性测试，使用 Hypothesis 且至少运行 100 次迭代，标签格式为 `Feature: intelligent-oncall-agent, Property {number}: {property_text}`。
- 单元测试聚焦具体示例、状态转换与错误条件（如路由、会话标识生成、超时/错误传播、任务状态转换、降级行为、模板热替换）。
- 外部依赖（Embedding_Model、Chat_Model、Milvus、MCP 服务端、send_msg）在测试中以 mock/stub 隔离，使属性测试聚焦本系统逻辑并可低成本运行 100+ 次。
- 属性与任务映射：P1→4.4, P2→4.5, P3→4.6, P4→5.2, P5→5.3, P6→6.2, P7→6.3, P8→6.4, P9→6.5, P10→7.3, P11→11.3, P12→7.4, P13→12.2, P14→13.2, P15→13.4, P16→13.6, P17→13.7, P18→13.9, P19→13.10, P20→13.11, P21→8.2, P22→8.4, P23→9.2, P24→9.3, P25→9.4, P26→3.3, P27→11.5, P28→11.6, P29→11.7, P30→15.2, P31→15.3, P32→15.7, P33→15.9, P34→15.5。

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "2.1"] },
    { "id": 2, "tasks": ["1.3", "2.2"] },
    { "id": 3, "tasks": ["2.3", "3.1", "4.1", "7.1", "7.2", "8.1"] },
    { "id": 4, "tasks": ["3.2", "4.2", "4.3", "7.3", "7.4", "7.5", "8.2", "8.3", "9.1", "13.1", "13.3", "13.5"] },
    { "id": 5, "tasks": ["3.3", "3.4", "4.4", "4.5", "4.6", "5.1", "6.1", "8.4", "8.5", "9.2", "9.3", "9.4", "13.2", "13.4", "13.6", "13.7", "13.8"] },
    { "id": 6, "tasks": ["5.2", "5.3", "5.4", "6.2", "6.3", "6.4", "6.5", "11.1", "12.1", "13.9", "13.10", "13.11", "15.8"] },
    { "id": 7, "tasks": ["11.2", "12.2", "12.3", "15.1", "15.4", "15.6", "15.9"] },
    { "id": 8, "tasks": ["11.3", "11.4", "15.2", "15.3", "15.5", "15.7", "15.10", "16.1"] },
    { "id": 9, "tasks": ["11.5", "11.6", "11.7", "16.2", "16.3"] }
  ]
}
```
