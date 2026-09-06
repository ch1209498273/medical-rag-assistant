# 系统架构

> **设计原则：** 每一层都能回答三个问题——输入是什么、失败怎么停、结果凭什么被相信。这个原则把“模型能力”拆成可测试的知识、检索、生成和发布边界。

## 定位与边界

医疗知识问答助手 v1.0 是一个本机优先的、证据约束的 RAG 作品集。当前公开演示只索引四份原创虚构血透机构培训资料。产品运行时是一个单一编排工作流：解析、检索、重排、生成、核验和持久化由一个 FastAPI 应用协调；开发阶段的多 Agent 分工不等于请求期间的运行时能力。

## 组件关系

```text
┌────────────────────────────────────────────────────────────────┐
│ React/Vite frontend                                            │
│  ChatPage · DocumentsPage · SSE parser · runtime-mode badge    │
└──────────────────────────────┬─────────────────────────────────┘
                               │ same-origin HTTP/SSE
┌──────────────────────────────▼─────────────────────────────────┐
│ FastAPI API boundary                                           │
│  health · documents · chat · evaluations                       │
│  fixed reason codes, citation validation, safe error shaping    │
└──────────────┬──────────────────────────────┬─────────────────┘
               │                              │
┌──────────────▼─────────────┐    ┌───────────▼──────────────────┐
│ ChatOrchestrator            │    │ IngestionService             │
│ sessions · rewrite · RAG    │    │ scan · extract · chunk       │
│ terminal persistence        │    │ version · publish            │
└──────────────┬─────────────┘    └───────────┬──────────────────┘
               │                              │
┌──────────────▼─────────────┐    ┌───────────▼──────────────────┐
│ RetrievalService             │    │ Provider Protocols           │
│ recall ≤20 · dedupe ·        │    │ Demo or Cloud implementation  │
│ rerank ≤6 · gate 0.35        │    └───────────┬──────────────────┘
└──────────────┬─────────────┘                │
               │                              │
┌──────────────▼─────────────┐    ┌───────────▼──────────────────┐
│ AnswerService                │    │ SQLite + Qdrant Local         │
│ buffer · JSON · citations    │    │ metadata/session + vectors    │
│ optional claim verifier      │    └──────────────────────────────┘
└─────────────────────────────┘
```

实现位置：`backend/app/api` 暴露安全 API 边界；`backend/app/ingestion` 负责 PDF/DOCX 解析与 Chunk；`backend/app/rag` 负责召回、重排、答案和可选核验；`backend/app/providers` 定义 Demo/Cloud 适配器；`backend/app/storage` 封装 SQLite 与 Qdrant Local；`frontend/src` 负责页面与浏览器侧协议解析。

## 建库数据流

1. 资料目录扫描只接受一层 PDF/DOCX，并拒绝符号链接、Windows reparse point、越界路径和超出资源预算的输入。
2. `IngestionService` 以文件名和 SHA-256 建立版本记录；重复内容可跳过，强制重建会产生新版本。只有完成所有写入的版本才会被激活。
3. PDF/DOCX 提取器生成带 `file_name`、标题层级和页码或段落范围的 `SourceRef`。Chunk 保留这个来源位置和内容哈希，便于引用和去重。
4. Embedding 向量写入 Qdrant Local，文档版本、索引状态、失败原因和会话元数据写入 SQLite。当前 SQLite schema version 为 3；MySQL 不在 v1.0。
5. 图片型 PDF 在 Demo Provider 中不会伪造 OCR 文字；Cloud 模式只有在 OCR Provider 返回明确文本时才继续索引。

## Task15：版本化知识工程与两级切分

生产 Demo 的资料流保持简单可复现：PyMuPDF/python-docx 提取 → 保留 `SourceRef` → 确定性 Chunk → Embedding。针对最新的 8 册标准手册，Task15 另外建立了隔离 staging，不直接覆盖生产库：

- **主候选：** Docling `HybridChunker`，先保留册/章/节标题、表格和来源定位，再按 token 长度合并或拆分；首轮得到 1,114 个 Docling chunks。
- **兜底：** 现有确定性抽取/切分器（Legacy），得到 1,147 个 chunks，用来发现 Docling 不可用、表格序列化异常或覆盖不足时的差异；它不是悄悄替换主通道的黑箱。
- **质量门：** 只有非空、无异常重复、章节/表格覆盖和来源定位通过，切片才进入 staging active。定位修复版 `RUN-20260904-004` 保持 1,114 个 chunk identity 不变，并实现 1,114/1,114 可定位（146 段落、968 表格）。第 9 册不混入当前版本，后续以新的增量版本加入。

源项目已将旧知识索引先移入本地可恢复归档，再把最新标准手册迁移到独立正式运行库并完成本地验收。公开仓库不包含真实手册、数据库或向量文件；公开 Demo 仍使用虚构资料，不能把私有运行库包装成公网服务。

真实手册、切片正文和索引文件仍属于本地私有资产；公开仓库只说明算法、数量、版本和验证结果，不导出原文。

## 问答数据流

```text
用户问题
  -> clean_question（长度、空值、控制字符）
  -> Embedding
  -> active version 过滤的向量召回（最多 20）
  -> content hash / 高重叠文本 / 标题上限去重
  -> Reranker（最多 6）
  -> 最高分 >= 0.35？否则 INSUFFICIENT_EVIDENCE
  -> 带随机边界的 evidence 提示
  -> DemoLanguageModel 或 DeepSeek
  -> 完整缓冲后解析双字段 JSON
  -> 文本、来源编号、页码/段落位置和引用集合校验
  -> answer_delta + final SSE
  -> ChatOrchestrator 持久化消息、引用、状态和反馈
```

生成器不会把中途片段直接展示给浏览器。`AnswerService` 先完整接收、限制字节数/字符数/事件数，确认 `refused`/`answer` 合同和引用集合后才发 `answer_delta`。缺证据、非法 JSON、额外字段、未知来源编号、敏感文本或不可定位来源都会整条拒答，不删除坏引用后继续显示。

## 检索逻辑的两条轨迹

当前代码同时保留两条有意隔离的轨迹：

| 轨迹 | 流程 | 用途与结论 |
|---|---|---|
| v1 默认 | Vector recall ≤20 → 生产去重 → Reranker ≤6 → `0.35` 门 | 公开 Demo 和当前基线，强调稳定、可回滚和成本可控 |
| staging/评测 | Vector + lexical → balanced RRF（`k=60`）→ 同一去重/业务过滤 → Reranker | Task14D/E/F、Task18-R1/D1 的受控实验；检索指标提升，但端到端回答与时延回退，未晋级 |

Task18 的最终 R1 在同一 `final-ai-validated-20260904-005` 和知识快照上，将 Recall@20 从 `0.607` 提升到 `0.679`，但回答可用率由 `72.5%` 降至 `68.75%`、引用有效率由 `77.5%` 降至 `76.25%`，平均回答耗时约由 `2.00s` 升至 `2.28s`。因此系统不会因为“召回更高”就自动切换 Hybrid；生产仍走 Vector，后续若继续只做单变量实验。

## 两种运行模式

| 模式 | 知识 Provider | 语言模型 | 网络/密钥边界 |
| --- | --- | --- | --- |
| Demo | `DemoKnowledgeProvider`：确定性 1024 维向量、词法重排、显式空 OCR | `DemoLanguageModel`：只匹配 `demo/questions.json` 登记场景 | 不读 Key；使用拒绝外呼的 smoke transport 验证无网络 |
| Cloud | SiliconFlow `BAAI/bge-m3`、`BAAI/bge-reranker-v2-m3`、`PaddlePaddle/PaddleOCR-VL-1.5` | DeepSeek `deepseek-v4-flash` | 运行者显式切换并提供自己的 Key；前端永远不接触 Key |

两种模式共用 FastAPI、SQLite、Qdrant Local、SSE、会话、引用和页面流程。Cloud 缺少必需配置时，`require_cloud_keys()` 在 Provider 装配前失败；不会捕获错误后静默切到 Demo。Demo 不创建 Cloud client，也不会把问题或证据发往外部服务。

## 招聘方可快速验证的能力

| 关注点 | 可以从仓库验证什么 | 不应误解为 |
| --- | --- | --- |
| 可复现性 | 无 Key、无网络 Demo 与 smoke 脚本 | 公网托管服务 |
| 证据约束 | active 版本过滤、引用定位、整条答案核验 | 医学事实自动正确 |
| 失败边界 | 固定 reason code、低相关拒答、Cloud 配置 fail closed | 对任意问题都能回答 |
| 工程演进 | Provider Protocol、版本化资料、评测与回滚记录 | 已完成多人 ACL 或生产部署 |

## API 与持久化边界

- `GET /api/health` 只返回 `runtime_mode` 与 Provider 状态（`configured`、`missing`、`not_required` 等），不返回凭据。
- `POST /api/documents/scan` 返回文件名、版本状态和固定失败原因；不会返回数据库路径或原文。
- `POST /api/chat/stream` 通过 SSE 返回有限的 `status`、`answer_delta`、`final`、`error` 事件。公开 API 边界会剥离内部 `retrieval_diagnostics`，错误只保留固定 reason code。
- `/api/chat/sessions` 和会话恢复接口把消息、引用和反馈保存在本地 SQLite；浏览器只保存当前会话 ID，不保存 Provider Key。
- 应用生命周期按 `RagRuntime → DocumentRuntime` 管理资源，正常关闭顺序是语言模型、Ingestion/Provider、Qdrant Local、SQLite；关闭路径幂等，装配异常也会清理已取得资源。

## 失败与安全边界

检索 Provider 故障映射为 `RETRIEVAL_UNAVAILABLE`；生成故障丢弃尚未验证的缓冲区并返回 `GENERATION_UNAVAILABLE`；结构或引用检查失败返回 `ANSWER_NOT_VERIFIABLE`；模型明确拒答返回 `MODEL_REFUSED`。这些代码用于 UI 状态和测试，不把上游响应、系统提示词、Key 或本机路径回显给用户。

结构引用存在性和位置校验只回答“答案是否带有可定位的公开来源”，不证明答案与原文的语义一致性，更不证明医疗正确性。更完整的评测边界见 [评测说明](evaluation.md)，模式选择与存储取舍见 [工程决策](engineering-decisions.md)。

## v2.0 演进接口

Provider Protocol、独立存储层和版本化文档为后续 MySQL、Qdrant Server、多知识域和运行时路由保留接口。当前新增的 2A `AgentWorkflow` 只是默认关闭的开发验收合同，用来比较基线、核验和候选路径；它不改变产品请求路径，也不代表面向用户的运行时多 Agent 已经交付。运行时多 Agent、患者端、登录和 ACL 必须在独立的需求、威胁模型和人工验收后再进入路线图。
