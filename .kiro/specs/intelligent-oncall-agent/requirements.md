# Requirements Document

## Introduction

本项目构建一个企业级智能 OnCall Agent（智能值守代理，又称 "Ghost Agent"），目标是解决传统 OnCall 值班中人工值守、人工排查效率低下的痛点。系统通过整合三大核心 Agent 能力，实现"问题自动应答"与"故障智能排查"的一体化运维服务，从而降低团队 OnCall 人力成本、提高团队效率，并解决实际的企业级运维问题。

系统采用分层架构：

- **API 接口层**：对外提供 `/chat`、`/chat_stream`、`/upload_file`、`/ai_ops` 四个接口。
- **Agent 层**：包含知识库 Agent（基于 RAG）、对话 Agent（基于 ReAct 模式）、运维 Agent（基于 Plan-Execute-Replan 模式），以及可扩展的其他 Agent。
- **核心组件层**：Loader（加载器）、Indexer（索引器）、Retriever（检索器）、Transformer（转换器）、Chat Model（对话模型）、Prompt（提示词）、Tool（工具）、MCP（模型上下文协议）。
- **知识库层**：将业务接入手册、告警处理手册、历史工单记录等文档同步到向量数据库。

关键技术包括：RAG（检索增强生成）、ReAct（推理-行动交替）、Plan-Execute-Replan（计划-执行-重规划）、Multi-Agent（多智能体）、MCP（模型上下文协议）、向量数据库（Milvus）、Prompt 工程、SSE 流式输出、多轮对话记忆（短期记忆与长期记忆）、Function Call/工具调用。系统提供三种语言技术栈实现选项（Python：FastAPI + LangChain + LangGraph；Go：Goframe + Eino；Java：SpringBoot + Spring AI Alibaba），统一使用 Milvus 向量数据库与 Doubao-embedding-text-240715 嵌入模型。

本文档采用 EARS 模式描述所有需求，并遵循 INCOSE 质量规则。

## Glossary

- **System（系统）**：智能 OnCall Agent 系统整体。
- **API_Gateway（API 接口层）**：对外暴露 HTTP 接口（`/chat`、`/chat_stream`、`/upload_file`、`/ai_ops`）的服务层。
- **Knowledge_Base_Agent（知识库 Agent）**：基于 RAG 模式，先检索相关内容再生成答案的智能体。
- **Conversation_Agent（对话 Agent）**：基于 ReAct 模式，理解问题、调用工具与知识库并精准回答的智能交互智能体。
- **Ops_Agent（运维 Agent）**：基于 Plan-Execute-Replan 模式，自动接收告警、排查问题、分析根因并执行标准化操作的智能体。
- **Planner_Agent（规划智能体）**：运维 Agent 中负责查询处理步骤并制定执行计划的子智能体。
- **Executor_Agent（执行智能体）**：运维 Agent 中负责按计划步骤调用工具集执行的子智能体。
- **Replanner_Agent（重规划智能体）**：运维 Agent 中负责评估执行结果并决定继续、修订或结束计划的子智能体。
- **Loader（加载器）**：负责加载原始文档内容的核心组件。
- **Indexer（索引器）**：负责对文档分片生成嵌入向量并建立索引的核心组件。
- **Retriever（检索器）**：负责根据查询从向量数据库召回相关内容的核心组件。
- **Transformer（转换器）**：负责文档分片与格式转换的核心组件。
- **Chat_Model（对话模型）**：底层大语言模型（LLM）。
- **Prompt_Module（提示词模块）**：负责构造提示词的核心组件。
- **Tool_Registry（工具集）**：注册并管理可被 Agent 调用的工具集合，含内置工具与 MCP 工具。
- **MCP_Client（MCP 客户端）**：基于模型上下文协议（Model Context Protocol）接入外部工具的客户端。
- **Vector_Database（向量数据库）**：使用 Milvus 存储文档分片嵌入向量与消息嵌入向量。
- **Embedding_Model（嵌入模型）**：Doubao-embedding-text-240715，将文本转换为向量。
- **Memory_Module（记忆模块）**：管理多轮对话记忆的组件，含短期记忆与长期记忆。
- **Short_Term_Memory（短期记忆）**：保存近期历史消息的记忆。
- **Long_Term_Memory（长期记忆）**：对关键信息进行总结后保存的记忆。
- **Chunk（分片）**：文档按策略切分后的文本片段。
- **RAG（检索增强生成）**：先检索相关内容、再增强大模型生成答案的技术。
- **ReAct（推理-行动模式）**：思考与行动交替进行的智能体执行模式。
- **Plan_Execute_Replan（计划-执行-重规划模式）**：先制定计划、再执行、再根据结果重规划的智能体执行模式。
- **SSE（服务器发送事件，Server-Sent Events）**：服务器向客户端单向推送的流式输出协议。
- **query_internal_docs（内部文档查询工具）**：查询知识库内部处理步骤的工具。
- **query_cls_log（日志查询工具）**：查询 CLS 日志的工具。
- **query_prometheus_alarm（告警查询工具）**：查询 Prometheus 告警的工具。
- **send_msg（消息发送工具）**：向群组发送消息的工具。
- **Rerank（重排）**：对召回结果按相关性重新排序的过程。
- **Hybrid_Retrieval（混合检索）**：结合向量检索与关键词检索的检索方式。
- **Session（会话）**：一组带有唯一会话标识的多轮交互上下文。

## Requirements

### Requirement 1: 聊天问答接口（/chat）

**User Story:** 作为运维人员，我希望通过 `/chat` 接口以一次性返回方式提交问题，以便获得完整的问答结果。

#### Acceptance Criteria

1. WHEN API_Gateway 通过 `/chat` 接口接收到包含会话标识与用户消息的请求，THE API_Gateway SHALL 将请求路由至 Conversation_Agent 进行处理。
2. WHEN Conversation_Agent 完成应答生成，THE API_Gateway SHALL 以单次 HTTP 响应返回完整应答内容与对应的会话标识。
3. IF `/chat` 请求缺少用户消息字段，或用户消息在去除首尾空白后长度为 0，THEN THE API_Gateway SHALL 拒绝该请求、返回 HTTP 400 状态码与指明用户消息字段缺失或为空的错误信息，并且不将请求路由至 Conversation_Agent。
4. IF `/chat` 请求未携带会话标识，THEN THE API_Gateway SHALL 生成新的会话标识并在响应中返回该会话标识。
5. IF Conversation_Agent 自请求被路由起处理时间达到 60 秒仍未返回应答，THEN THE API_Gateway SHALL 终止该请求处理、返回 HTTP 504 状态码与超时错误信息，并保留该会话标识对应的会话上下文不变。
6. IF `/chat` 请求的用户消息长度超过 8000 个字符，THEN THE API_Gateway SHALL 拒绝该请求、返回 HTTP 400 状态码与指明最大允许长度（8000 个字符）的错误信息，并且不将请求路由至 Conversation_Agent。
7. IF Conversation_Agent 在处理过程中返回生成错误，THEN THE API_Gateway SHALL 终止该请求处理、返回指明应答生成失败的错误响应，并保留该会话标识对应的会话上下文不变。

### Requirement 2: 流式聊天问答接口（/chat_stream）

**User Story:** 作为运维人员，我希望通过 `/chat_stream` 接口以流式方式获取应答，以便在长结果场景下实时查看打字机效果的输出。

#### Acceptance Criteria

1. WHEN API_Gateway 通过 `/chat_stream` 接口接收到包含用户消息的聊天请求，THE API_Gateway SHALL 使用 SSE 协议建立单向流式连接。
2. WHILE Conversation_Agent 正在生成应答，THE API_Gateway SHALL 将生成的内容增量按生成先后顺序作为独立的 SSE 数据事件依次推送给客户端。
3. WHEN 应答内容全部生成完成，THE API_Gateway SHALL 通过 SSE 发送表示流结束的事件并关闭该 SSE 连接。
4. IF SSE 连接在生成过程中被客户端断开，THEN THE API_Gateway SHALL 停止该请求的内容生成并释放关联资源。
5. IF 内容生成过程中 Chat_Model 返回错误，THEN THE API_Gateway SHALL 停止内容推送、通过 SSE 发送表示生成失败的错误事件并关闭连接，且不撤回已推送的内容。
6. IF `/chat_stream` 请求缺少用户消息字段，或用户消息在去除首尾空白后长度为 0，THEN THE API_Gateway SHALL 拒绝该请求、返回 HTTP 400 状态码与指明用户消息字段缺失或为空的错误信息，并且不建立 SSE 连接。
7. IF `/chat_stream` 请求未携带会话标识，THEN THE API_Gateway SHALL 生成新的会话标识并在首个 SSE 事件中返回该会话标识。
8. IF Conversation_Agent 在配置的空闲超时时间（默认 30 秒）内未产生新的内容增量，THEN THE API_Gateway SHALL 通过 SSE 发送表示空闲超时的错误事件并关闭连接。

### Requirement 3: 文件上传接口（/upload_file）

**User Story:** 作为知识库管理员，我希望通过 `/upload_file` 接口上传文档，以便将运维知识纳入知识库供后续检索。

#### Acceptance Criteria

1. WHEN API_Gateway 通过 `/upload_file` 接口接收到文档文件且该文件通过文档类型与文件大小校验，THE API_Gateway SHALL 将文件移交 Knowledge_Base_Agent 进行入库处理。
2. WHEN API_Gateway 接受通过文档类型与文件大小校验的上传文件，THE API_Gateway SHALL 返回唯一入库任务标识与初始入库任务状态。
3. IF `/upload_file` 请求未包含文件或文件内容为空（大小为 0 字节），THEN THE API_Gateway SHALL 拒绝该请求、不进行入库处理，并返回 HTTP 400 状态码与描述文件缺失或为空的错误信息。
4. IF 上传文件的文档类型不属于系统配置的受支持文档类型集合，THEN THE API_Gateway SHALL 拒绝该文件、不进行入库处理，并返回 HTTP 415 状态码与受支持文档类型列表。
5. IF 上传文件大小超过系统配置的单文件大小上限，THEN THE API_Gateway SHALL 拒绝该文件、不进行入库处理，并返回 HTTP 413 状态码与单文件大小上限数值。
6. WHEN 文件成功完成入库处理，THE API_Gateway SHALL 将对应入库任务标识的状态更新为已完成，并在该任务状态查询结果中提供该文件生成的分片数量。
7. IF 文件入库处理失败，THEN THE API_Gateway SHALL 将对应入库任务标识的状态更新为失败，并在该任务状态查询结果中提供失败原因。

### Requirement 4: 运维处理接口（/ai_ops）

**User Story:** 作为运维人员，我希望通过 `/ai_ops` 接口触发故障智能排查，以便系统自动完成排查并反馈处理建议。

#### Acceptance Criteria

1. WHEN API_Gateway 通过 `/ai_ops` 接口接收到运维触发请求，THE API_Gateway SHALL 将请求路由至 Ops_Agent 启动排查流程。
2. WHEN Ops_Agent 受理排查任务，THE API_Gateway SHALL 在 5 秒内返回该排查任务的唯一任务标识与表示"已受理待排查"的初始任务状态。
3. IF `/ai_ops` 请求缺少告警信息或排查目标，THEN THE API_Gateway SHALL 拒绝请求、不创建排查任务并返回 HTTP 400 状态码与描述缺失字段的错误信息。
4. WHERE 触发方式为 webhook，THE API_Gateway SHALL 在校验请求来源签名通过后启动排查流程。
5. IF webhook 请求来源签名校验失败，THEN THE API_Gateway SHALL 拒绝该请求、不启动排查流程并返回 HTTP 401 状态码与描述签名校验失败的错误信息。
6. IF Ops_Agent 处于不可用状态或并发排查任务数已达到配置的并发上限而无法受理新任务，THEN THE API_Gateway SHALL 拒绝请求、不创建新排查任务并返回 HTTP 503 状态码与描述服务暂不可用的错误信息。

### Requirement 5: 知识库 Agent 文档加载（Loader）

**User Story:** 作为知识库管理员，我希望系统能加载多种格式的运维文档，以便统一进行后续处理。

#### Acceptance Criteria

1. WHEN Knowledge_Base_Agent 接收到格式属于系统支持的文档类型集合的待入库文件，THE Loader SHALL 将文件解析为统一的纯文本内容表示。
2. WHERE 文件包含结构化标题与段落，THE Loader SHALL 在解析结果中保留各标题的层级顺序及其所属段落的归属关系。
3. IF Loader 无法将文件解析为纯文本内容，THEN THE Loader SHALL 终止该文件解析、不将其移交 Transformer，并返回描述解析失败原因的错误信息，同时将该文件的入库任务标记为失败。
4. WHEN Loader 完成文件解析，THE Loader SHALL 将解析结果连同至少包含来源文件标识、文件名与文件格式的文件元数据移交 Transformer 处理。
5. IF Loader 解析单个文件的耗时达到配置的解析超时时间上限仍未完成，THEN THE Loader SHALL 终止该文件解析并返回解析超时的错误信息，同时将该文件的入库任务标记为失败。
6. IF 文件解析完成但提取到的纯文本内容为空，THEN THE Loader SHALL 将该文件视为解析失败、不将其移交 Transformer，并返回内容为空的错误信息，同时将该文件的入库任务标记为失败。

### Requirement 6: 知识库 Agent 文档分片（Transformer 与分片策略）

**User Story:** 作为知识库管理员，我希望系统按合理策略对文档进行分片，以便提升检索精度并适配嵌入模型输入限制。

#### Acceptance Criteria

1. WHEN Transformer 接收到非空的已解析文档内容，THE Transformer SHALL 依据当前生效的分片策略将内容切分为一个或多个 Chunk，其中除最后一个 Chunk 外，每个 Chunk 的文本长度均介于配置的单分片最小长度与单分片最大长度之间。
2. THE Transformer SHALL 支持按标题、按段落与按语义三种分片策略，并在请求未显式指定分片策略时采用配置的默认分片策略作为当前生效的分片策略。
3. THE Transformer SHALL 为每个 Chunk 附加来源文件标识、该 Chunk 在源文档中的顺序序号，以及该 Chunk 在源文档中的起止位置信息。
4. IF 单个 Chunk 的文本长度超过 Embedding_Model 的最大输入长度，THEN THE Transformer SHALL 将该 Chunk 进一步切分为多个文本长度均不超过该最大输入长度的子 Chunk。
5. WHEN Transformer 完成分片，THE Transformer SHALL 将 Chunk 集合移交 Indexer 处理。
6. IF Transformer 接收到的已解析文档内容为空，THEN THE Transformer SHALL 不生成任何 Chunk，并返回内容为空的提示信息且不移交 Indexer 处理。
7. IF 当前生效的分片策略无法应用于已解析文档内容，THEN THE Transformer SHALL 终止该文档的分片处理并返回描述分片失败原因的错误信息。

### Requirement 7: 知识库 Agent 索引与嵌入（Indexer）

**User Story:** 作为知识库管理员，我希望系统对文档分片生成嵌入向量并建立索引，以便支持向量检索。

#### Acceptance Criteria

1. WHEN Indexer 接收到非空的 Chunk 集合，THE Indexer SHALL 调用 Embedding_Model（Doubao-embedding-text-240715）为集合中每个 Chunk 生成嵌入向量。
2. WHEN 某个 Chunk 成功生成嵌入向量，THE Indexer SHALL 将该 Chunk 的文本、嵌入向量与元数据写入 Vector_Database。
3. IF Embedding_Model 对某 Chunk 调用失败，THEN THE Indexer SHALL 对该 Chunk 重试，且累计重试次数不超过配置的最大重试次数（取值范围 0 至 5 次，默认 3 次）。
4. IF 某 Chunk 在达到最大重试次数后仍嵌入失败，或该 Chunk 写入 Vector_Database 失败，THEN THE Indexer SHALL 记录该 Chunk 的失败信息、将其计入失败数量并继续处理其余 Chunk。
5. WHEN Indexer 完成全部 Chunk 处理，THE Indexer SHALL 返回成功写入数量与失败数量，且成功写入数量与失败数量之和等于接收到的 Chunk 总数。
6. IF Indexer 接收到的 Chunk 集合为空，THEN THE Indexer SHALL 不调用 Embedding_Model，并返回成功写入数量为 0、失败数量为 0。

### Requirement 8: 知识库 Agent 检索召回（Retriever 与混合检索、重排）

**User Story:** 作为运维人员，我希望系统针对我的问题召回最相关的知识片段，以便生成准确的答案。

#### Acceptance Criteria

1. WHEN Retriever 接收到非空的用户查询，THE Retriever SHALL 调用 Embedding_Model 将查询文本转换为查询向量。
2. WHEN Retriever 获得查询向量，THE Retriever SHALL 从 Vector_Database 召回相似度不低于配置的最小相似度阈值且相似度最高的 Chunk，召回数量不超过配置的 Top-K 参数（取值范围 1 至 100，默认 5）。
3. WHERE 召回结果在 Top-K 边界处存在相似度分数相等的 Chunk，THE Retriever SHALL 返回全部相似度相等的 Chunk，此时返回数量可超过 Top-K。
4. WHERE 启用混合检索（Hybrid_Retrieval），THE Retriever SHALL 合并向量检索结果与关键词检索结果，对合并结果去重后按融合分数降序排序，并返回不超过 Top-K 个 Chunk。
5. WHERE 启用重排（Rerank），THE Retriever SHALL 对召回的 Chunk 集合按相关性分数降序重新排序后返回。
6. IF Vector_Database 中无相似度不低于最小相似度阈值的 Chunk，THEN THE Retriever SHALL 返回空召回结果集。
7. IF Retriever 接收到的用户查询为空或去除首尾空白后长度为 0，THEN THE Retriever SHALL 拒绝该查询并返回查询为空的错误信息。
8. IF Embedding_Model 生成查询向量失败，THEN THE Retriever SHALL 终止本次检索并返回查询向量生成失败的错误信息。

### Requirement 9: 知识库 Agent 增强生成（RAG）

**User Story:** 作为运维人员，我希望系统基于召回的知识片段生成答案，以便获得有依据且准确的回答。

#### Acceptance Criteria

1. WHEN Knowledge_Base_Agent 获得非空的召回 Chunk 集合，THE Prompt_Module SHALL 将用户查询与召回集合中的全部 Chunk 共同构造为单条增强提示词，并在提示词中标注每个 Chunk 的来源文件标识。
2. WHEN 增强提示词构造完成，THE Knowledge_Base_Agent SHALL 将增强提示词发送给 Chat_Model，并要求其仅依据提示词中提供的 Chunk 内容生成最终答案。
3. IF 召回的 Chunk 集合为空，THEN THE Knowledge_Base_Agent SHALL 在答案中明确告知用户未在知识库中检索到相关内容，且不臆造知识库以外的内容。
4. WHEN Chat_Model 返回生成答案，THE Knowledge_Base_Agent SHALL 在答案中附带所引用 Chunk 的来源文件标识列表。
5. IF Chat_Model 在生成答案过程中返回错误，THEN THE Knowledge_Base_Agent SHALL 终止本次生成、返回指明答案生成失败的错误信息，且不返回部分或臆造的答案。
6. IF Chat_Model 在配置的生成超时时间内未返回答案，THEN THE Knowledge_Base_Agent SHALL 终止本次生成并返回生成超时的错误信息。

### Requirement 10: 对话 Agent ReAct 循环

**User Story:** 作为运维人员，我希望对话 Agent 能在思考与行动之间交替，自动调用工具与知识库来回答我的问题，以便处理高频重复的咨询场景。

#### Acceptance Criteria

1. WHEN Conversation_Agent 接收到用户消息，THE Prompt_Module SHALL 构造提示词，其中包含从 Vector_Database 召回的相关历史消息，且召回的历史消息数量由配置的历史消息 Top-K 参数确定（取值范围 1 到 50 条）。
2. WHEN 提示词构造完成，THE Conversation_Agent SHALL 进入 ReAct 循环并将提示词发送给 Chat_Model。
3. WHILE Chat_Model 的输出包含工具调用请求，THE Conversation_Agent SHALL 从 Tool_Registry 调用对应工具，并将工具响应作为新的观察结果送回 Chat_Model 继续循环。
4. WHEN Chat_Model 的输出不再包含工具调用请求，THE Conversation_Agent SHALL 退出 ReAct 循环并将最终内容作为应答返回用户。
5. IF ReAct 循环的迭代次数达到配置的最大迭代次数上限（默认 10 次，可配置范围 1 到 50 次），THEN THE Conversation_Agent SHALL 终止循环并返回已生成的内容以及表明已达到最大迭代次数上限的提示信息。
6. IF 被调用工具返回错误，或被调用工具在配置的工具调用超时时间（默认 30 秒，可配置范围 1 到 300 秒）内未返回响应，THEN THE Conversation_Agent SHALL 终止该次工具调用并将描述失败原因的错误信息作为观察结果送回 Chat_Model，由 Chat_Model 决定后续行动。
7. IF 在 ReAct 循环中 Chat_Model 返回错误，或在配置的模型调用超时时间（默认 60 秒，可配置范围 1 到 120 秒）内未返回输出，THEN THE Conversation_Agent SHALL 终止 ReAct 循环、保留本轮已生成的内容，并向用户返回表明应答生成失败的错误信息。

### Requirement 11: 运维 Agent 规划阶段（Planner）

**User Story:** 作为运维人员，我希望运维 Agent 像资深工程师一样先查询处理手册再制定排查计划，以便排查过程有章可循。

#### Acceptance Criteria

1. WHEN Ops_Agent 启动排查流程，THE Planner_Agent SHALL 调用 query_internal_docs 工具查询与告警相关的处理步骤。
2. WHEN Planner_Agent 获得相关处理步骤，THE Planner_Agent SHALL 生成由有序步骤组成的执行计划，且步骤数量介于 1 至配置的最大步骤数上限之间。
3. THE Planner_Agent SHALL 在执行计划中为每个步骤标注待调用的工具（取自 Tool_Registry 中已注册的工具）与该步骤的目标。
4. IF query_internal_docs 未返回相关处理步骤，THEN THE Planner_Agent SHALL 基于告警信息生成通用排查计划并标注该计划为无手册依据。
5. IF query_internal_docs 工具调用失败或在配置的查询超时时间内未返回，THEN THE Planner_Agent SHALL 基于告警信息生成通用排查计划并标注该计划为无手册依据。
6. WHEN Planner_Agent 完成执行计划生成，THE Planner_Agent SHALL 将该执行计划移交 Executor_Agent。

### Requirement 12: 运维 Agent 执行阶段（Executor）

**User Story:** 作为运维人员，我希望运维 Agent 按计划逐步执行排查动作并调用相应工具，以便自动完成标准化排查操作。

#### Acceptance Criteria

1. WHEN Executor_Agent 接收到执行计划，THE Executor_Agent SHALL 从执行计划的第一个步骤开始，按步骤的标注顺序每次执行一个步骤。
2. WHILE 执行当前步骤，THE Executor_Agent SHALL 调用该步骤标注的工具，且所调用工具来自 Tool_Registry（包含 query_cls_log、query_prometheus_alarm、send_msg 与 MCP 工具）。
3. WHEN 当前步骤的工具调用成功返回，THE Executor_Agent SHALL 记录包含所属步骤标识与工具响应内容的执行结果，并将该执行结果移交 Replanner_Agent。
4. IF 当前步骤的工具调用失败，THEN THE Executor_Agent SHALL 记录包含所属步骤标识与失败原因的失败信息、暂停执行后续步骤，并将失败结果移交 Replanner_Agent。
5. IF 当前步骤的工具调用等待响应的时间达到配置的工具调用超时时间仍未返回，THEN THE Executor_Agent SHALL 终止该工具调用并将其判定为该步骤的工具调用失败。

### Requirement 13: 运维 Agent 重规划阶段（Replanner）

**User Story:** 作为运维人员，我希望运维 Agent 在每步执行后评估进展并决定是否继续、修订计划或结束，以便排查过程能够动态适应实际情况。

#### Acceptance Criteria

1. WHEN Replanner_Agent 接收到当前计划与 Executor_Agent 移交的工具执行结果，THE Replanner_Agent SHALL 输出一个排查完成情况评估结果，该结果取以下三种取值之一：「任务已完成」、「任务未完成且剩余计划仍适用」、「任务未完成且剩余计划不再适用」。
2. IF 评估结果为「任务已完成」，THEN THE Replanner_Agent SHALL 终止后续步骤执行、结束排查流程，并基于已执行步骤的工具结果生成分析结果总结。
3. IF 评估结果为「任务未完成且剩余计划仍适用」，THEN THE Replanner_Agent SHALL 指示 Executor_Agent 继续执行剩余计划中的下一个步骤。
4. IF 评估结果为「任务未完成且剩余计划不再适用」，THEN THE Replanner_Agent SHALL 生成修订后的新计划（New Plan）、将重规划次数加 1，并交由 Executor_Agent 从新计划的第一个步骤开始执行。
5. IF 重规划次数达到配置的最大重规划次数上限（默认 10 次，可配置范围为 1 至 50 次），THEN THE Replanner_Agent SHALL 终止后续步骤执行、结束排查流程，并生成包含「因达到最大重规划次数而未完成」说明的分析结果总结。
6. IF Replanner_Agent 在评估或生成新计划过程中 Chat_Model 返回错误，THEN THE Replanner_Agent SHALL 终止排查流程、保留已执行步骤的工具结果，并生成包含评估失败原因说明的分析结果总结。

### Requirement 14: 运维 Agent 结果上报

**User Story:** 作为运维人员，我希望运维 Agent 将根因分析与处理建议自动发送到值班群，以便团队及时获知排查结论。

#### Acceptance Criteria

1. WHEN Ops_Agent 生成分析结果总结，THE Ops_Agent SHALL 在生成总结后的配置的上报时限内调用 send_msg 工具将总结发送到配置的目标群组。
2. WHEN Ops_Agent 构造分析结果总结，THE Ops_Agent SHALL 使总结包含根因分析、处理建议与已执行操作记录三个部分，且对无可填充内容的部分以明确的无内容说明标注。
3. IF send_msg 工具调用失败，THEN THE Ops_Agent SHALL 按配置的重试间隔重试发送，且累计重试次数不超过配置的最大重试次数。
4. IF send_msg 在达到配置的最大重试次数后仍失败，THEN THE Ops_Agent SHALL 记录包含失败原因的发送失败信息，并在配置的保留期限内保留分析结果总结供后续查询。
5. WHEN send_msg 工具返回发送成功，THE Ops_Agent SHALL 记录上报成功状态与发送时间，并将该排查任务标记为已上报。
6. IF 未配置目标群组或配置的目标群组不存在，THEN THE Ops_Agent SHALL 跳过发送、记录目标群组不可用的错误信息，并保留分析结果总结供后续查询。

### Requirement 15: 运维 Agent 触发方式

**User Story:** 作为运维人员，我希望运维 Agent 支持多种触发方式，以便覆盖手动排查、周期巡检与告警自动响应场景。

#### Acceptance Criteria

1. WHERE 触发方式为手动触发，WHEN Ops_Agent 接收到手动触发请求，THE Ops_Agent SHALL 启动排查流程。
2. WHERE 触发方式为定时触发，WHEN 配置的预定触发时间到达，THE Ops_Agent SHALL 启动排查流程。
3. WHERE 触发方式为 webhook 触发，WHEN Ops_Agent 接收到来源签名校验通过的 webhook 事件，THE Ops_Agent SHALL 启动排查流程。
4. IF 触发请求缺少告警信息或排查目标，THEN THE Ops_Agent SHALL 拒绝该次触发、不启动排查流程并返回描述缺失字段的错误信息。
5. WHILE 同一排查目标已存在进行中的排查流程，IF Ops_Agent 接收到针对该排查目标的新触发请求，THEN THE Ops_Agent SHALL 不启动新的排查流程并返回指向已有进行中任务标识的提示信息。

### Requirement 16: 工具集与 Function Call（Tool_Registry）

**User Story:** 作为系统集成者，我希望系统通过统一工具集管理可调用工具，以便各 Agent 以 Function Call 方式调用工具。

#### Acceptance Criteria

1. THE Tool_Registry SHALL 为每个已注册工具维护在工具集内唯一的名称、参数定义（包含每个参数的名称、类型与是否必填）与功能描述。
2. WHEN Agent 请求调用某工具，THE Tool_Registry SHALL 按该工具参数定义中的参数类型与必填要求校验调用参数。
3. IF 工具调用参数不符合参数定义，THEN THE Tool_Registry SHALL 拒绝调用、不执行该工具并返回指明具体不符合项的参数校验错误信息。
4. THE Tool_Registry SHALL 提供 query_internal_docs、query_cls_log、query_prometheus_alarm 与 send_msg 内置工具。
5. WHEN Agent 请求调用的工具存在且其调用参数通过校验，THE Tool_Registry SHALL 执行该工具并向调用方返回工具响应。
6. IF Agent 请求调用的工具名称在 Tool_Registry 中不存在，THEN THE Tool_Registry SHALL 拒绝调用并返回指明该工具不存在的错误信息。

### Requirement 17: MCP 工具集成（MCP_Client）

**User Story:** 作为系统集成者，我希望系统通过 MCP 协议接入外部工具，以便在不修改核心代码的情况下扩展 Agent 能力。

#### Acceptance Criteria

1. WHEN MCP_Client 成功连接到 MCP 服务端，THE MCP_Client SHALL 获取该服务端提供的工具清单，并将清单中每个工具的名称、参数定义与功能描述注册到 Tool_Registry。
2. WHEN Agent 调用某 MCP 工具，THE MCP_Client SHALL 通过 MCP 协议向对应服务端转发调用，并在收到服务端响应后将工具响应返回给调用方。
3. IF 某 MCP 工具的名称与 Tool_Registry 中已注册工具的名称冲突，THEN THE MCP_Client SHALL 拒绝注册该工具，记录命名冲突信息，并保留已注册的同名工具不变。
4. IF MCP 服务端连接失败，THEN THE MCP_Client SHALL 记录连接失败信息，不将该服务端的任何工具注册到 Tool_Registry，且保持 Tool_Registry 中内置工具的可用性不受影响。
5. IF 某 MCP 工具在服务端返回错误响应，THEN THE MCP_Client SHALL 终止该次调用并向调用方返回指示该 MCP 工具执行失败的错误信息。
6. IF MCP 工具调用时间达到配置的超时时间仍未收到服务端响应，THEN THE MCP_Client SHALL 终止该次调用，向调用方返回指示该 MCP 工具调用超时的错误信息，且不影响 Tool_Registry 中其余工具的可用性。

### Requirement 18: 多轮对话记忆（Memory_Module）

**User Story:** 作为运维人员，我希望系统在多轮对话中记住上下文，以便我无需重复说明背景信息。

#### Acceptance Criteria

1. WHEN Conversation_Agent 完成一轮应答，THE Memory_Module SHALL 将本轮用户消息与应答按时间先后顺序追加写入对应 Session 的 Short_Term_Memory。
2. WHEN Short_Term_Memory 中的消息数量超过配置的保留条数上限（不小于 1 的正整数），THE Memory_Module SHALL 将超出上限的较早消息总结后写入 Long_Term_Memory。
3. WHEN Memory_Module 将较早消息总结写入 Long_Term_Memory，THE Memory_Module SHALL 从 Short_Term_Memory 中移除这些已总结的较早消息，并使 Short_Term_Memory 的消息数量不超过配置的保留条数上限。
4. IF Memory_Module 在总结较早消息或写入 Long_Term_Memory 过程中失败，THEN THE Memory_Module SHALL 将相关消息保留在 Short_Term_Memory 中并记录该失败信息。
5. WHEN Prompt_Module 为某 Session 构造提示词，THE Memory_Module SHALL 按时间先后顺序提供该 Session 的 Short_Term_Memory 与 Long_Term_Memory 内容。
6. THE Memory_Module SHALL 使每个 Session 的记忆内容仅来源于该 Session 且仅作用于该 Session，不同 Session 之间的记忆内容相互隔离。

### Requirement 19: 消息向量召回

**User Story:** 作为运维人员，我希望系统能从历史消息中召回与当前问题相关的内容，以便在长对话中保持上下文相关性。

#### Acceptance Criteria

1. WHEN Conversation_Agent 完成一轮应答，THE Indexer SHALL 调用 Embedding_Model（Doubao-embedding-text-240715）为本轮用户消息与对应应答生成嵌入向量，并连同其所属 Session 标识写入 Vector_Database。
2. IF Indexer 为本轮消息生成嵌入向量或写入 Vector_Database 失败，THEN THE Indexer SHALL 记录该失败信息并继续处理当前会话的后续请求，且不中断对话流程。
3. WHEN Prompt_Module 为某 Session 构造对话提示词，THE Retriever SHALL 调用 Embedding_Model 将当前用户消息转换为查询向量。
4. WHEN Retriever 获得当前用户消息的查询向量，THE Retriever SHALL 在限定为当前 Session 的历史消息范围内从 Vector_Database 召回相似度最高的历史消息，召回数量由配置的 Top-K 参数确定。
5. IF 当前 Session 在 Vector_Database 中无可召回的历史消息，THEN THE Retriever SHALL 返回空召回结果集。

### Requirement 20: Prompt 工程（Prompt_Module）

**User Story:** 作为系统维护者，我希望提示词遵循一致的工程规范，以便提升模型输出的稳定性与准确性。

#### Acceptance Criteria

1. WHEN Prompt_Module 构造提示词，THE Prompt_Module SHALL 在提示词中包含角色定义与任务目标。
2. WHERE 提示词模板被标注为需要分步推理，WHEN Prompt_Module 构造提示词，THE Prompt_Module SHALL 在提示词中加入分步思考指令。
3. WHEN Prompt_Module 构造提示词，THE Prompt_Module SHALL 在提示词中包含对输出内容结构与格式的约束说明。
4. THE Prompt_Module SHALL 以唯一名称标识管理每个提示词模板，并支持在不修改 Agent 调用代码的情况下更新或替换同名模板的内容。
5. IF Prompt_Module 在构造提示词时引用的具名模板不存在，THEN THE Prompt_Module SHALL 停止该次提示词构造并返回描述缺失模板名称的错误信息。

### Requirement 21: 向量数据库与嵌入模型（Vector_Database 与 Embedding_Model）

**User Story:** 作为系统维护者，我希望系统统一使用 Milvus 与 Doubao 嵌入模型，以便文档与消息向量化处理保持一致。

#### Acceptance Criteria

1. THE System SHALL 使用 Milvus 作为唯一的 Vector_Database，统一存储文档分片嵌入向量与消息嵌入向量。
2. THE System SHALL 使用 Doubao-embedding-text-240715 作为唯一的 Embedding_Model，对文档分片文本与消息文本进行向量化。
3. WHEN System 向 Vector_Database 写入向量，THE System SHALL 同时存储该向量对应的原始文本、来源标识与向量类型（文档分片向量或消息向量）。
4. IF 待写入向量的维度与 Embedding_Model 输出向量的维度不一致，THEN THE System SHALL 拒绝该次写入并返回向量维度不一致的错误信息。
5. IF System 在配置的连接超时时间内无法连接 Vector_Database，THEN THE System SHALL 返回向量库不可用的错误信息，并保留尚未写入的数据不丢失。

### Requirement 22: 知识库内容同步

**User Story:** 作为知识库管理员，我希望将业务接入手册、告警处理手册与历史工单记录同步至向量数据库，以便这些知识可被检索使用。

#### Acceptance Criteria

1. WHEN 知识库管理员提交知识库文档同步请求，THE Knowledge_Base_Agent SHALL 对文档执行加载、分片、嵌入与写入 Vector_Database 的完整流程，并为写入的每个 Chunk 附加来源文件标识。
2. WHEN 同一来源文件被再次同步，THE Knowledge_Base_Agent SHALL 以来源文件标识为依据，用新生成的 Chunk 替换 Vector_Database 中该来源文件已有的全部 Chunk，且不残留该来源文件的旧 Chunk。
3. WHEN 知识库管理员请求移除某来源文件，THE Knowledge_Base_Agent SHALL 从 Vector_Database 删除该来源文件的全部 Chunk 并返回被删除的 Chunk 数量。
4. WHEN 文档同步流程完成，THE Knowledge_Base_Agent SHALL 返回同步状态、成功写入的 Chunk 数量与失败的 Chunk 数量。
5. IF 文档同步流程在加载、分片、嵌入或写入任一阶段失败，THEN THE Knowledge_Base_Agent SHALL 终止本次同步、记录失败阶段与失败原因，并保持该来源文件在 Vector_Database 中已有的 Chunk 不变。
6. IF 知识库管理员请求移除的来源文件在 Vector_Database 中不存在，THEN THE Knowledge_Base_Agent SHALL 不删除任何 Chunk 并返回该来源文件不存在的提示信息。

### Requirement 23: 技术栈实现选项

**User Story:** 作为开发团队，我希望系统提供多语言技术栈实现选项，以便团队按自身技术栈选择落地方案。

#### Acceptance Criteria

1. WHERE 选择 Python 技术栈，THE System SHALL 基于 FastAPI、LangChain 与 LangGraph 实现 `/chat`、`/chat_stream`、`/upload_file`、`/ai_ops` 四个接口及知识库 Agent、对话 Agent、运维 Agent 三大 Agent 能力。
2. WHERE 选择 Go 技术栈，THE System SHALL 基于 Goframe 与 Eino 实现 `/chat`、`/chat_stream`、`/upload_file`、`/ai_ops` 四个接口及知识库 Agent、对话 Agent、运维 Agent 三大 Agent 能力。
3. WHERE 选择 Java 技术栈，THE System SHALL 基于 SpringBoot 与 Spring AI Alibaba 实现 `/chat`、`/chat_stream`、`/upload_file`、`/ai_ops` 四个接口及知识库 Agent、对话 Agent、运维 Agent 三大 Agent 能力。
4. THE System SHALL 在任一技术栈实现中统一使用 Milvus 作为 Vector_Database 与 Doubao-embedding-text-240715 作为 Embedding_Model。
5. THE System SHALL 使 Python、Go 与 Java 三种技术栈实现对相同的合法请求返回功能等价的可观测应答。
6. THE System SHALL 在单次部署中仅启用 Python、Go 与 Java 三种技术栈实现中的一种。
7. IF 部署配置指定的技术栈不属于 Python、Go 与 Java 三者之一，THEN THE System SHALL 拒绝以该技术栈启动并返回指明不支持该技术栈的错误信息。
