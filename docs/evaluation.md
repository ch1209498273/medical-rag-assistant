# 评测与回滚

本项目把“流程能否运行”“工程边界是否安全”和“答案是否有用”分开记录。自动测试和 Demo smoke 是可重复的工程证据；历史 30 题评测与当前 80 题黄金评测集分别按各自协议记录，只发布脱敏聚合；任何一项都不能被解读为医疗正确率、监管认证或生产就绪证明。

## 评测集版本口径（先看）

为避免把不同阶段的数字混在一起，当前仓库同时保留两套有明确边界的评测资产：

- **历史 30 题集（Task 9/12/14）**：`corrected V1/V2`，用于早期 C1/C2、2A 和检索诊断；文档中的 `9/30`、`14/30`、`20/30` 等数字都属于这套历史协议，不能代表当前黄金集规模。
- **当前 80 题黄金评测集（Task 16/18）**：`2026-standard-manual-v1-golden-v1`，实际版本为 `final-ai-validated-20260904-005`，包含 64 道 `answerable` 与 16 道 `no_answer`，来自 8 册 `2026-standard-manual-v1`；Task18 使用该 80 题集做当前流程评测。
- 当前 80 题集已完成 AI+本地结构校验、无答案全库核验和专业审核冻结，状态为 `frozen`（`review_status=accepted`）。它是本项目当前固定的工程黄金评测集；“黄金”只表示评测基准已冻结，不等于临床认证或医疗准确率证明。

后文保留历史 30 题的完整复盘数字，是为了让面试官看到版本演进和回滚依据；阅读当前质量结果时，应优先看 Task 18 的 80 题章节。

## 当前黄金集与评测结果的绑定关系（2026-09-05）

这次最容易混淆的是“题集版本”和“知识库运行版本”是两条独立的版本轴：

| 资产 | 固定身份 | 作用 | 当前解释 |
|---|---|---|---|
| 黄金题集 | `2026-standard-manual-v1-golden-v1` / `final-ai-validated-20260904-005` | 80 题（64 `answerable` + 16 `no_answer`）的业务评测输入 | 已完成专业审核并冻结；不再与历史 30 题混算 |
| 知识库 staging | `RUN-20260904-004` | 8 册手册的 1,114 条定位修复后切片及对应索引 | 作为 Task18 当前流程的固定知识快照 |
| Task18-G | 上述题集 + staging；`baseline_v1 + vector + c1` | 当前 vector 基线闭环 | 80/80 完成，回答可用 53/80；无答案安全拒答 16/16 |
| Task18-R1 | 上述题集 + staging；`hybrid + balanced RRF + c1` | 混合检索对照实验 | 检索排名指标改善，但回答可用率、引用有效率和时延出现回退，结论 `observe`，不切换生产 |
| Task18-D1/F/H | 上述题集 + staging 的诊断/抽检子集 | 区分检索上下文、生成稳定性和人工可接受性 | 只作诊断和抽检，不覆盖 80 题冻结身份，也不替代全量评测 |

因此，“80 题修改后测试结果有没有变”要按运行记录判断：只要题集、知识快照、Prompt、模型或检索逻辑中任一项变化，就必须产生新的 `RUN-*`；不能拿旧 30 题或旧题集结果冒充当前 80 题结果。当前 Task18 的 80 题结果已经使用 `005 + RUN-20260904-004`，历史 `corrected V1/V2` 的 30 题结果仅保留为回滚和方案演进证据。

`final-ai-validated-20260904-005` 是在前一版 80 题基础上完成修订并由用户专业审核冻结的版本；题目正文、参考要点、证据和审校记录仍只在 `data/private/evaluations/task16/`，公开文档只记录身份、分层、指标和决策，避免把内部资料带入 GitHub。

## Task15/Task18 的切分与检索演进

这条演进链解释了为什么代码里同时能看到“向量默认路径”和“混合检索实验”：

1. **Task15 知识刷新：** 8 册 `2026-standard-manual-v1` 在隔离 staging 中以 Docling `HybridChunker` 为主候选，现有确定性抽取/切分器为 Legacy fallback。首轮得到 Docling 1,114 片、Legacy 1,147 片；通过覆盖质量门后，定位修复版仍保持 1,114 个 chunk identity 不变，并达到 1,114/1,114 可定位（146 段落、968 表格）。第 9 册预留为后续增量版本。
2. **生产默认：** 为保证 v1 可复现，线上 Demo 仍使用 `baseline_v1 + vector + c1`：Embedding → 最多 20 条召回 → 去重 → 最多 6 条 Reranker → `0.35` 相关性门 → 答案/引用合同。Task15 的 Docling staging 不会因为建库完成而自动替换生产 SQLite/Qdrant。
3. **混合检索实验：** staging 另开 `vector + lexical + balanced RRF(k=60)`，再沿用业务过滤、去重和 Reranker。Task14D/E/F 与 Task18-D1/R1 分别拆过向量召回、词法召回、RRF 位移、去重噪声和回答侧波动；最终 R1 的 Recall@20 从 `0.607` 提升到 `0.679`，但回答可用率由 `72.5%` 降至 `68.75%`、引用有效率由 `77.5%` 降至 `76.25%`，平均回答耗时约由 `2.00s` 升至 `2.28s`，所以只保留为可解释的候选实验。

这不是“混合检索无效”，而是一个有边界的工程结论：它改善了检索命中，却没有在当前答案合同和生成配置下转化成端到端收益；在没有新的单变量评测前，生产继续保持 vector 基线。

## 可公开复现的 Demo 检查

Demo 资料是四份原创虚构 PDF/DOCX，共 8 个可索引文件。`scripts/demo_smoke.py --no-network` 在进程内创建应用，使用拒绝所有外呼的 transport，然后执行扫描、一个支持问题、一个资料外拒答和一次会话追问。Task 5 的安全聚合结果为：

```text
indexed_documents=8 answered_questions=2 refused_questions=1 network_calls=0
```

它验证的是索引、检索、引用、拒答、追问、会话和零外呼的连通性。Demo Language Model 只精确匹配 `demo/questions.json` 的登记场景，未知问题拒答，因此不能代表开放域模型能力。

本 Task 的文档契约命令为：

```powershell
backend\.venv\Scripts\python.exe -m pytest -q backend/tests/release/test_public_docs.py
```

文档契约、后端/前端全量测试和独立公开仓库历史扫描都属于发布门，应在每次准备发布时重新执行；不在本页固化旧运行计数，也不把历史结果冒充为本轮证据。

## 历史私有 30 题聚合结论（Task 9/12/14）

真实资料和逐题结果留在本地私有目录，公开文本只记录下面的比例。指标中的“正式回答可用性”是当前评测协议下通过答案/引用边界的题数，不是医学准确率；“无答案拒答率”只针对资料中没有依据的题目。

| 配置 | 正式回答可用性 | 无答案拒答率 | 工程判断 |
| --- | ---: | ---: | --- |
| 回滚后的基线/C1 | 9/30（30.0%） | 100% | 作为当前正式提示词基线 |
| C2 证据检查清单实验 | 14/30（46.7%） | 60% | 可回答题增加，但安全门失败 |

C2 的 14/30 比基线多 5 个可用回答，但无答案拒答率从 100% 降到 60%，低于预设的 90% 安全门。C2 因此没有晋级为正式配置；项目负责人确认回滚，当前正式提示词恢复到 C1 的“两字段 JSON + 证据不足即拒答”行为。C2 的私有复评报告保留用于复盘，公开文档不复制题目、参考答案、资料正文或逐题记录。

## 2A 受控对照实验

为验证“开发阶段的工作流拆分”是否值得进入后续版本，2A 定义了同协议的三条对照臂：A 是当前 `baseline_v1`，B 是基线加一次独立核验，C 是候选工作流。三条对照使用同一冻结评测集、同一资料快照和同一聚合指标；A/B/C 的单题预算分别为 4、3、5 个物理调用，预算由代码计算而不是由提示词约定。

本轮已完成本地授权合同、调用预算 transport、零出网预检、私有结果目录约束、失败回退约束和公开候选导出；随后在项目负责人明确授权后完成一次真实 A/C 对照。正式聚合记录 172 次物理调用，0 重试、0 预算违规，原始 Provider 响应未保存。A 形成 21/30 个正式回答（70.0%），C 形成 9/30 个正式回答（30.0%）；C 有 16 题停在 Router 结构校验，因此这不是“多 Agent 质量下降”的充分证据，而是候选适配器仍需诊断的工程信号。

2A 的当前结论是“真实执行链路完成，Router 失败回退已本地修复，业务效果待同协议复评与人工复核”。默认 v1 路径、C1 正式提示词和既有回滚门保持不变；私有逐题结果不公开，也不把本轮聚合数字称为医学准确率、临床安全或生产就绪证明。

## 指标字典（分子、分母、置信区间与限制）

公开文档只记录可复核的聚合指标。每个比例都必须同时说明分子、分母和限制；样本量过小或标签不完整时，不把比例包装成医学准确率、临床安全或业务成效。

| 指标 | 分子 | 分母 | 置信区间 | 限制与解读 |
| --- | --- | --- | --- | --- |
| 正式回答可用性 | 通过答案结构、引用集合和安全门的正式回答题数 | 本轮可评测题总数 | 二项比例可用 Wilson 95% CI；历史公开表只保留计数 | 限制：20/30 只是一次受控私有观测，不是医学准确率；会受检索、Provider 和题目可答性影响 |
| 无答案拒答率 | 标注为无答案且系统正确拒答的题数 | 标注为无答案的题数 | 同上 | 限制：5/5 只说明这 5 道题在该快照下通过拒答门，不等于零风险或所有越界问题都安全 |
| 引用可见率 | 正式答案中带有可定位运行时引用的题数 | 形成正式答案的题数 | 同上 | 限制：引用存在只证明可定位，不证明答案与原文语义一致，更不证明医学正确 |
| 检索 Recall@6 | Top-6 中命中的已标注相关证据数 | 评测集标注的相关证据总数 | 目前只在私有报告中计算 | 限制：依赖完整相关证据标注；17/35 只能提示检索缺口，不能单独证明 Chunker 是唯一根因 |
| 检索 Precision@6 | Top-6 中命中的已标注相关证据数 | Top-6 返回证据数 | 目前只在私有报告中计算 | 限制：受重复片段、标注粒度和候选上限影响；17/104 不是回答正确率 |
| 语义裁判有效率 | 返回合法裁判结构的题数 | 实际发起裁判的题数 | 同上 | 限制：11/20 是“裁判可用性”；裁判不可用时不能把剩余题当作负例或正例 |

**协议边界：** C1/C2 与 9F-B 使用不同评测协议，不能横向比较。C1/C2 的历史表用于提示词安全门回滚，9F-B 的 20/30、5/5 和检索计数用于一次性私有工程诊断。没有真实用户任务日志，因此本项目不报告用户采纳、临床改善或 ROI 等业务成效。

## 指标口径：逐字覆盖与语义评测

历史 `reference_points_covered` 是**逐字覆盖**指标：参考要点必须以规范化后的子串形式出现在正式答案中。它很适合稳定的自动回归，却会把“意思相同、表达不同”的答案判成未覆盖，因此不能单独当作医学正确率或语义正确率。

项目已在私有边界内实现语义评测基础设施：它在一次受控评测运行的内存中比较问题、已核验答案与已选证据，并将安全聚合状态写入独立 sidecar。该 sidecar 不保存正文、来源编号、原始裁判输出或路径，也不进入用户问答、SSE 或 Stage B verifier。

**语义基线已受控执行一次，但尚不稳定。** 9F-B 完整重跑冻结 C1 后，30 题中有 20 题形成正式答案、无答案题为 5/5 正确拒答；20 次语义裁判仅有 11 次有效结构，其中 10 次判为证据支持，另有 9 次不可用。因为历史报告和本轮聚合都不保存答案正文，这些数值不能作为稳定语义分数、医学准确率或检索改善证明。任何新重放仍需单独的数据出站、预算与落盘授权。

## 评测协议的限制

- 公开 Demo 的 7 个场景（事实、流程、条件、禁止事项、跨资料概括、资料外拒答、上下文追问）是人工可读的虚构样例，不是冻结私有评测集的替代品。
- 自动的来源编号、页码/段落和结构化 JSON 检查只能证明结构可定位；可见引用不等于答案正确，语义一致性、专业正确性和临床风险仍需要独立人工抽查。
- C1/C2 使用的资料、模型、检索参数和评测边界是特定实验快照；不能外推到不同机构、不同资料或不同模型。
- Demo 没有调用真实模型；Cloud smoke 若未来执行，最多使用 5 个公开虚构问题、零自动重试，并另行记录授权、调用次数和聚合结果。
- Task 9 的候选题生成链曾因严格 schema/Provider 稳定性失败而停止；没有把无效候选集包装成 30 题成绩，也没有把原始 Provider 响应写入公开文档。

## 失败后的处理

1. 先保留失败的安全 reason code 和聚合计数，不保存上游原文。
2. 定位是资料快照、检索门、提示词、Provider、结构解析还是引用校验的问题；一次只改变一个参数族。
3. 以同一评测集和同一资料快照复评，预先定义可用性、拒答和耗时安全门。
4. 安全门失败时恢复上一个已验证配置，并留下可审计的决策记录。

这条流程正是 C2 的处理方式：回答数量改善不足以抵消拒答边界退化，于是回滚，而不是把实验数字当作正式能力。工程取舍详情见 [工程决策](engineering-decisions.md)；三分钟演示顺序见 [Demo 指南](demo-guide.md)。

## Task 5 本机零网络验收

为避免把一次云端观测误写成产品成效，本轮补做完全离线的四类场景验收：`all_staff` 的虚构制度流程回答、范围未指定时的安全拒答、业务状态未知时的安全拒答，以及资料外问题的“正式拒答 + 通用参考分栏”。资料由 Demo 运行时生成，Provider 为确定性 fake，不读取真实资料，不访问网络。

验收入口为 `scripts/demo_smoke.py --no-network`，关键聚合输出为：

```text
indexed_documents=8 answered_questions=2 refused_questions=1 network_calls=0
```

该输出只能证明本机索引、资格过滤、引用、拒答、追问、历史和通用参考分栏能够连通；它不代表医护人员真实采纳，也不代表开放域模型或临床场景性能。后续任何真实资料重放都必须重新取得出站授权并建立独立评测快照。

## Task 16：2026 标准手册黄金评测集（80 题已专业审核冻结）

Task 16 为当前 8 册资料建立可追溯、可人工复核、可量化的黄金评测集，固定集合身份为 `2026-standard-manual-v1-golden-v1`。目标规模为 80 题：64 道 `answerable`、16 道 `no_answer`；每册 8 道可回答题；最终拆分为 `dev=60`（48/12）和 `holdout=20`（16/4）。候选问题按事实/定义、流程步骤、条件/例外、表格结构化、多证据综合以及四类无答案原因分层，并设置 20/40/20 的难度配额。

本轮已完成本地协议、资料快照适配、候选生成 packet、候选校验、双人审校导入/合并、配额与泄漏检查、不可覆盖冻结出口、指标聚合和现有评测适配器。Task 15 的 `RUN-20260903-003` 作为固定来源：8 册、Docling 主候选 1,114 个切片，来源快照哈希为 `27f02d6ec421dd87b829fcecf966db067fecce3ceee2bde11bceff391d120fe0`；本地预检生成 32 个批次（每批 5 个候选槽位），网络调用为 0。

当前状态是 `frozen`（`review_status=accepted`）：80 题候选集已经由候选生成、结构校验、配额/泄漏检查、无答案全库核验、AI+本地校验和专业审核冻结流程形成，当前版本为 `final-ai-validated-20260904-005`，包含 64 道 `answerable` 与 16 道 `no_answer`。候选生成阶段的失败与修复记录保留在后文，正式 RAG 评测是另一道授权门。真实问题、参考要点、证据、审校记录和详细结果只写入 `data/private/evaluations/task16/`，不会进入 GitHub。Task 16 不修改生产 SQLite/Qdrant、`baseline_v1 + vector + c1`、Prompt 或检索参数，也不代表医学准确率、临床安全或发布就绪。

Task 16 的核心指标包括 Evidence Recall@20/6、Precision@6、MRR@20、nDCG@6、原子要点覆盖、引用 precision/recall、回答可用性、grounded success、无答案拒答率和 unsafe answer rate。比例使用 Wilson 95% 置信区间，排名/连续分数使用确定性 bootstrap 95% 区间；`unsafe_answer_rate=0` 是 holdout 硬门，其余为项目工程质量目标。当前 80 题已完成专业审核冻结，可作为后续正式 RAG 评测的固定输入；冻结本身不等于临床认证。

### Task 16A：候选结构化输出诊断（A3 已执行）

Task 16 首批候选生成实际只调用 1 次，因 `CANDIDATE_SCHEMA_INVALID` 停止；原始响应未保存，因此不能把 few-shot 缺失直接当作根因。Task 16A 已确认 Spec，并形成 A0→A1→A2→A3→A4 实施计划：先用零网络 fake 响应矩阵定位失败阶段，再用完全合成的五对象 few-shot 变体，最后在独立授权后做最多 1 次 DeepSeek v4 Flash 单批次探针。

Task 16A 代码已按确认计划实现；A3 探针已按独立授权执行 1 次，未写正式 `candidates.jsonl`，也未冻结黄金集。完整候选生成仍需新的独立授权。

候选生成首批已在明确出站授权后实际执行 1 次，但返回内容未通过当前严格候选 JSON 合同，分类为 `CANDIDATE_SCHEMA_INVALID`；因授权约束为 0 重试，运行器立即停止，候选数为 0，未生成候选文件，也未保存原始响应。该事实不能单独归因于模型、资料或 Prompt，任何修复、重试或解析兼容性调整都必须另立任务并重新授权。生产配置、评测集和旧资料不受影响。

#### Task 16A 本地实现状态（2026-09-03）

用户已确认 Task 16A 计划并选择 `Inline Execution`。A0/A1/A2 的本地失败阶段诊断、fake 响应矩阵和完全合成 few-shot 已完成；真实 Task 16 packet 的离线运行保持 `network_calls=0`、`provider_constructed=false`、`api_key_read=false`。A3 单批次 runner 和 `scripts/task16a_probe.py` 已实现并通过本地专项测试/CLI help；随后按独立授权尝试 1 次真实请求，但在 Provider 传输层返回 `PROVIDER_TRANSPORT_ERROR`，0 重试。

Task 16A 的 A4 只写入私有白名单报告：`few_shot_probe_valid` 需要 5 条候选、严格解析通过、证据绑定计数为 0、重复计数为 0 且探针成功；本次因传输层无可解析响应而返回 `diagnosis_insufficient`。探针结果不写正式 `candidates.jsonl`、不冻结黄金集，也不改变生产配置；完整候选生成仍需新的独立授权。

#### Task 16A A3 云端探针结果（2026-09-03）

用户已单独授权一次 A3 真实探针，固定一个 `2026-standard-manual-v1` 批次，DeepSeek v4 Flash，最多 1 次、0 重试、`temperature=0`、`max_tokens=4096`，仅保存本地私有聚合结果。

- 实际请求尝试 `1` 次，重试 `0`；结果为 `probe_invalid`，失败阶段 `provider_envelope`，原因 `PROVIDER_TRANSPORT_ERROR`，候选数 `0`。
- A4 决策为 `diagnosis_insufficient`，因此本次不能判断 few-shot 是否改善结构化输出，也不能将失败归因于 Parser、字段合同、资料或模型。
- 结果只写入 `data/private/evaluations/task16a/cloud-20260903-001/` 的 `probe-summary.json` 和 `task16a-report.json`；未保存原始响应、内部问答内容、答案、证据正文或 Key。
- 未生成正式候选、未冻结黄金集、未修改生产链路；授权已用尽，不重试。Task 16 完整候选生成仍是独立任务和授权门。

### Task 16B：传输与请求契约本地诊断（B0/B1）

Task 16B 用来解释一次候选生成探针的传输层失败，并核对资料最小必要出站边界。系统的数据流保持为：`Word/DOCX → Docling 切片 → 本地 SQLite/Qdrant → 检索/重排 → 必要证据 → DeepSeek`。也就是说，**切片后按需发送**：单个批次只带本批次需要的片段；**不把整份 Word 或全部 1,114 个切片**放进一次请求。

B0 检查本地 packet、批次身份、模型端点、消息数量和大小分桶；B1 使用完全合成的 fake transport 覆盖超时、传输、HTTP、空响应和非法响应封装。两阶段都不构造真实 Provider、不读取 API key、不保存请求/响应正文，输出仅为本地私有的计数、分桶、有限 reason code 和决策。

本轮 B0/B1 的验收状态为 `offline_complete`：`network_calls=0`、`provider_constructed=false`、`api_key_read=false`，请求契约记录了“批次切片数”和“本地来源快照总切片数”的区别。现有直连 transport 与代理感知 transport 的差异只作为 wiring 观察，不作为 A3 失败的因果结论。B2 连通性握手、新的真实探针和 Task 16 完整候选生成均需单独授权，不会自动启动；生产仍保持 `baseline_v1 + vector + c1`。

#### Task 16B-B2：最小连通性握手（真实握手已完成）

B2 是一个独立的传输诊断，不是 RAG 评测，也不是模型质量测试。它只向配置的 DeepSeek base URL 的 `/models` 发送无请求体 GET；一次运行累计最多 5 次独立请求，每次 0 自动重试，默认单次超时 10 秒，关闭重定向，使用代理感知 transport。它不发送 Word、Task16 packet、1,114 个切片、问题、答案或证据，也不调用 `/chat/completions`。

CLI 默认执行零网络 `preflight_ready`，只有私有授权文件、`--allow-network` 和本地 Key 均满足时才允许真实请求。真实运行只记录 HTTP 状态类别/信号计数、reason code、决策和安全布尔字段；不保存 Key、Authorization Header、响应正文、异常正文或完整 URL。2xx/3xx 只能说明 HTTP 路径有响应，401/403 表示路径可达但认证需检查，404/405 表示路径或方法可能不匹配；任何这些结果都不能证明 completion、候选 JSON 或医疗回答可用。

当前 B2 本地专项测试为 `24 passed`；真实运行 `run-cli-a3ceed7625cd4558b920f28f383b014f` 实际调用 `3` 次、0 重试，三次均为 `success_path`，决策为 `PROVIDER_REACHABLE_RESPONSE_UNKNOWN`。摘要只保存在 `data/private/evaluations/task16b/cli-out-a3ceed7625cd4558b920f28f383b014f/handshake-summary.json`，未保存原始响应、Key 或内部资料。该结果只证明 `/models` HTTP 路径在当前配置下可达，不证明 `/chat/completions`、候选 JSON、模型回答质量或医疗准确性；不会自动触发 Task16 候选生成或 Task10 发布。

### Task 16C：DeepSeek completion 合成兼容性诊断（真实探针已完成）

Task16C 专门补齐 B2 未覆盖的 `/chat/completions` 边界，但不把它当作 RAG 质量评测。固定五个合成探针：P1/P2 普通 completion，P3/P4 `response_format=json_object`，P5 使用 `example-volume`/`example-chunk-001` 的候选风格合成请求。每次请求 `stream=false`、`temperature=0`；P1–P4 的 `max_tokens=64`，P5 为 `4096`，单次运行最多 5 次、0 自动重试。

本地 runner 只将响应 JSON 在内存中压缩为 HTTP 类别、外壳/内容布尔值、JSON 可解析性、候选阶段、大小桶和有限 reason code；私有 writer 只生成 `completion-summary.json`。CLI 默认是零网络预检，只有新的授权文件、`--allow-network` 和后置读取的 Key 同时满足才可联网。

本地验收为 Task16C `21 passed`，Task16/16A/16B 回归 `33 passed`，Ruff、公开文档检查和 `git diff --check` 通过；CLI 预检为 `preflight_ready`、网络调用 0、Provider 未构造、Key 未读取。随后按独立授权执行 `run-task16c-20260903-001`：实际 5 次 DeepSeek v4 Flash `/chat/completions` 请求、0 重试，P1/P2/P3/P4 均 `PROBE_OK`，P5 返回可解析 JSON 但候选结构合同不通过，决策为 `CANDIDATE_PATH_CONTRACT_ISSUE`。

该结果证明 completion 传输、响应外壳、文本内容和 JSON mode 可工作；它不能证明真实资料召回、候选题生成、医疗准确性或生产发布。原始响应、headers、完整 URL、Key 和合成正文未保存，聚合结果仅写入 `data/private/evaluations/task16c/task16c-cloud-20260903-001/completion-summary.json`。下一步应另立 Task16D 候选输出合同修复设计，先做 1 个真实资料批次的 5 道小试，再决定是否扩展。

### Task 16D：候选输出合同修复与五题真实小试（本地阶段记录）

Task16D 保持现有严格候选解析合同，不通过静默补字段或放宽证据绑定来“修复”失败。它复用完整合成五对象 few-shot，但通过独立 Prompt 适配器隔离于 Task16 默认生成器；runner 只接受 Task16 `V01-B01` 一个确定性批次，目标 5 个候选，最多 1 次 DeepSeek v4 Flash `/chat/completions`，0 重试，`temperature=0`、`max_tokens=4096`。

本地证据为 Task16D 专项 `28 passed`、Task16/16A/16B/16C 回归 `57 passed`，Ruff、`git diff --check` 和私有目录忽略检查通过。CLI `--help` 及真实 packet 零网络预检为 `preflight_ready`：`network_calls=0`、`provider_constructed=false`、`api_key_read=false`。这些是本地协议/fake 验证，不等于真实模型可生成候选。

本段只记录真实小试前的本地阶段：当时没有读取真实 Key、没有发送内部切片、没有保存原始响应，也没有生成正式候选或冻结黄金集。实际小试结果见下一节；成功或失败都不自动扩展到完整候选池，不改变生产 `baseline_v1 + vector + c1`。

### Task 16D：真实资料五题小试（已执行，JSON 合同失败）

按用户新的独立授权，使用 `2026-standard-manual-v1` 的 `V01-B01` 必要切片向 DeepSeek v4 Flash 发起唯一一次 `/chat/completions` 请求；`temperature=0`、`max_tokens=4096`、0 自动重试。实际聚合结果为 `pilot_failed`：`network_calls=1`、`retry_count=0`、`failure_stage=json_decoding`、`reason_code=JSON_INVALID`、`candidate_count=0`。

这证明本次真实候选输出没有通过 JSON 解码，不证明内部资料、检索质量或医疗准确性。由于原始响应按授权未保存，不能进一步断言失败是文本包裹、截断或其他响应形态。私有目录只有 `pilot-summary.json`，没有候选文件；没有重试、没有放宽严格 parser、没有生成或冻结黄金集，也没有改变生产 `baseline_v1 + vector + c1`。后续如要处理，必须另立 JSON 失败诊断/最小修复设计并重新授权。

### Task 16E：生产正式回答预算与 thinking（本地实现）

Task16E 将正式回答与其他结构化步骤的输出预算分开：正式 `AnswerService` 默认使用 `32768`，共享 DeepSeek 结构化默认保持 `2048`，Router 的调用点仍为 `256`。正式回答请求显式发送 `thinking={"type":"disabled"}`；UI 与历史只接收最终答案、引用和安全状态，不接收或保存 `reasoning_content`。

本地实现包含 Provider 单次预算/思考模式校验、AnswerService 可选参数、Demo 兼容接口、Settings 环境配置及云端运行时接线。新增回归确认含 reasoning-only SSE 时适配器只输出 `delta.content`，并确认默认旧 Provider 调用形状不受影响。Task16E 选定测试集合为 `180 passed`，Ruff 与 `git diff --check` 通过；本轮没有 Key 读取、内部资料出站、云端评测或 GitHub 写入。该配置尚未经过新的真实质量复评，不代表医学准确性或发布批准。
### Task 16D-Fix 修复后真实复测

本次复测只验证 Task16D 候选生成修复后的真实请求路径，不是黄金集评测。固定 `V01-B01` 必要切片，DeepSeek v4 Flash，`temperature=0`、`max_tokens=32768`，最多 1 次、0 重试。实际完成 1 次请求后在 Provider envelope 阶段超时，聚合为 `pilot_failed` / `PROVIDER_TIMEOUT`，候选数为 0。

安全边界：`provider_constructed=true`、`api_key_read=true`、`request_body_sent=true`、`internal_material_sent=true`、`raw_provider_responses_saved=false`；私有目录仅保存 `pilot-summary.json`，没有原始响应、headers、Key、异常原文或候选文件。该运行不能证明 JSON 合同、预算修复、资料质量或模型质量；授权已用尽，未重试、未切换 Provider、未生成/冻结黄金集，后续需另立超时根因诊断。
### Task 16D-R：在线超时诊断首轮结果

Task16D-R 不是黄金集评测，而是同一 `V01-B01` 候选批次的传输/稳定性诊断。固定 DeepSeek v4 Flash、`temperature=0`、`max_tokens=32768`、单次超时 120 秒、最多 50 次顺序独立尝试、每次自动重试 0；首个严格解析通过的五候选结果即停止。输出仅允许本地私有聚合，不保存原始响应或候选正文。

首轮 `run-task16dr-20260903-001` 实际使用 50 次尝试，全部在 `provider_envelope` 阶段归类为 `PROVIDER_TRANSPORT_ERROR`，状态 `retry_exhausted`，候选数 0；没有一次进入模型响应、JSON 解码或证据校验。默认沙箱 TCP 检查失败，而受控环境外 TCP 检查成功，因此当前优先怀疑网络出口环境。该运行不证明 DeepSeek、Prompt、资料或严格 Parser 不可用。

聚合摘要位于 `data/private/evaluations/task16dr/cloud-20260903-001/run-summary.json`；没有保存 Key、headers、异常原文、思考内容、问题/答案/证据正文或候选正文。原授权 50 次已耗尽，继续运行必须新的独立授权；不得自动进入 Task16 全量候选、黄金集冻结或生产评测。

### Task 16D-R：第二轮真实在线联调结果

固定同一 `V01-B01`、DeepSeek v4 Flash、`timeout=120` 秒、`max_tokens=32768`，在已确认可达的受控网络环境中执行。实际 1 次请求即成功，严格解析得到 5 条候选，状态 `pilot_pass`，0 重试；没有保存原始响应或候选正文，仅保存私有聚合摘要 `data/private/evaluations/task16dr/cloud-20260903-002/run-summary.json`。

解释边界：第二轮只验证五题候选小试的在线链路和输出合同，不是完整候选池质量评测、黄金评测集冻结、医学准确性验证或生产发布依据；生产 `baseline_v1 + vector + c1` 保持不变。

### Task 16：第三轮完整候选池生成结果

运行 `run-task16-full-proven-20260903-006` 固定 `2026-standard-manual-v1`，使用 DeepSeek v4 Flash、few-shot Prompt、`thinking=disabled`、`temperature=0`、`max_tokens=32768`、超时 120 秒。32 个批次全部成功，实际调用 32 次、重试 0；严格结构解析和确定性批次前缀 ID 归一化均通过。

| 指标 | 结果 |
|---|---:|
| 候选总数 | 160 |
| 可回答（answerable） | 126 |
| 无答案（no_answer） | 34 |
| 唯一 ID | 160 |
| 字段合同错误 | 0 |
| 原始响应保存 | 否 |

证据摘要仅保存在 `data/private/evaluations/task16/task16-full-proven-20260903-006/run-summary.json`，候选内容保存在同目录 `candidates.jsonl`。这些是候选池生成证据，不是质量得分，也不是 80 题黄金评测集；在人工审核和 64+16 选题冻结前，不得用于正式模型质量结论或发布批准。

### Task 16：候选池类型分组

已对候选池做第一层机械分组，规则为只读取 `case_type` 的精确值，不重新调用模型：

| 分组 | 数量 | 本地文件 |
|---|---:|---|
| `answerable` | 126 | `classification-v1/answerable.jsonl` |
| `no_answer` | 34 | `classification-v1/no_answer.jsonl` |

未知类型为 0，重复 ID 为 0。该结果仅用于组织后续审核，不等于证据、答案或安全性已经人工通过，也不改变黄金集 64+16 的目标配额。

### Task 17A：生产反馈闭环本地实现（2026-09-04）

Task17A 已按确认的计划完成本地实现，目标是把上线后的问题/反馈变成可脱敏、可追溯、可人工复核的审核工作项；用户反馈仍是弱监督线索，不能单独构成医学金标。

| 验收项 | 结果 |
|---|---|
| Task17A 专项测试 | `33 passed` |
| Task16 回归 | `164 passed` |
| 脱敏与失败关闭 | 已覆盖手机号、证件/病案号、姓名/地址、日期、未知长标识；无法确定时不进入可审核正文 |
| 数据边界 | 源 SQLite 只读；派生库强制位于 `data/private/feedback/`；事件追加不覆盖历史 |
| 筛选与晋级 | 确定性问题指纹、P0–P3 分级、去重、证据/负例/版本/专业审核门已实现 |
| 留存 | 默认 90 天；清理只删除反馈投影并写入计数摘要 |
| 出站与秘密 | 网络调用 0；不读取 Key、不保存 Provider 原始响应/headers/reasoning |

本轮没有读取真实生产数据库、没有发送真实反馈到任何云端、没有自动生成或冻结 `golden-v2`，也没有修改生产 `baseline_v1 + vector + c1`。下一阶段只有在用户另行授权后，才可对小批量真实反馈执行“只读投影 → 脱敏审计 → 人工复核”；验收报告必须把反馈采集/脱敏/审核指标与正式 RAG 质量指标分开。

### Task 17B：管理员反馈审核工作台离线验收（2026-09-04）

| 验收项 | 结果 |
|---|---|
| 后端反馈审核/反馈投影专项 | `41 passed` |
| 前端完整测试套件 | `49 passed` |
| 后端完整回归（排除 2 个已知基线失败） | `1513 passed, 10 skipped, 2 deselected` |
| 前端生产构建 | 通过（TypeScript + Vite） |
| compileall / Ruff / `git diff --check` | 全部通过 |
| 网络/Provider 调用 | `0` |
| 真实生产反馈读取、golden-v2 冻结 | 否 |

工作台已接入现有“资料管理”并提供审核队列、脱敏详情、证据版本核验、Trace 摘要、Rubric 和追加式历史。API 默认关闭，只有显式本地开关开启后才读取 Task17A 私有派生库；任何脱敏、证据或专业审核门失败都不会进入 `golden_v2_candidate`。本轮测试使用合成 fixture，不代表真实反馈质量、医学准确性、UAT 或发布批准。

完整回归中保留了既有基线的两个失败断言（Task6 空知识库 `embed_calls` 期望差异、Task 14F 预检业务元数据缺失断言）；排除这两项后无新增失败。根目录直接收集还会混入 `release/` 历史副本，不能作为有效回归命令。

### Task 18-E：D1 结果解释与修复决策（2026-09-05）

Task18-E 是 D1 云端成对诊断后的本地解释阶段，不重新调用模型，也不把诊断结果误当成医学准确率。解释器读取 D1 私有聚合摘要，按重复槽位计算回答率、失败/合同失败率和不稳定率，再将信号归为上下文变化、生成方差、回答合同三类。

| 指标 | Vector | Hybrid |
|---|---:|---:|
| 重复槽位 | 20 | 20 |
| 回答槽位 | 11（55.0%） | 15（75.0%） |
| 合同失败率 | 1/20（5.0%） | 1/20（5.0%） |
| 不稳定率 | 1/10（10.0%） | 1/10（10.0%） |

语义裁判汇总为通过 13、不可用 13、未运行 14，可用率 32.5%；该可用率只表示本轮裁判结果覆盖，不代表答案医学正确率。三类信号均为 `observe`，因此决策为 `observe_no_production_change`：生产继续 `baseline_v1 + vector + c1`，不自动切换 Hybrid 或修改 Prompt、RRF、阈值。建议后续做固定证据上下文的回答合同微实验，再单变量诊断 RRF/去重/Top-K。

报告文件：`data/private/evaluations/task18/e-interpretation-20260905-001/interpretation.json` 与 `interpretation.md`。专项测试 `5 passed`；递归隐私禁止字段检查为 0；本阶段云端调用 0。

### Task 18-F：固定证据回答合同微实验预检（2026-09-05）

本实验将一次检索结果冻结为证据包，再重复回答 3 次，用来判断 D1 中的失败更偏向检索上下文还是回答合同/生成稳定性。它不重新评价证据医学正确性，也不自动改变生产。

| 项目 | 固定合同 |
|---|---|
| 题集 / staging | `final-ai-validated-20260904-005` / `RUN-20260904-004` |
| 样本 | 6 题，按合同失败→不稳定→普通案例选择 |
| 检索 | Vector；每题 Embedding 1 次、Reranker 1 次，证据冻结 |
| 回答 | DeepSeek v4 Flash，c1，thinking disabled，32768 tokens，0 重试；每题 3 次 |
| 预算 | Embedding 6 + Reranker 6 + 回答 18 = 30 次；不调用语义裁判 |

本地预检产物为 `data/private/evaluations/task18/f-preflight-20260905-001/preflight.json`，状态 `preflight_required`、`estimated_calls=30`、`network_calls=0`。真实运行需独立授权；结果只允许本地私有聚合，不保存问题、答案、资料正文、Key、headers、reasoning 或原始 Provider 响应。

### Task 18-F：固定证据回答合同微实验云端结果（2026-09-05）

用户授权后，固定同一题集与 staging，按“检索一次、证据冻结、回答三次”执行 6 题微评测。实际调用 27 次（Embedding 6、Reranker 6、DeepSeek 回答 15），0 重试、0 预算违规；不调用语义裁判、MiniMax、TaoToken 或 Sol。1 个 retrieval_refused 案例未进入回答阶段，因此回答路由少于理论 18 次。

聚合：18 个回答槽位中 5 个 answered，1 个 `answer_text_unsafe`，1 个案例出现重复间状态波动；报告决策 `observe_answer_contract_or_generation`。这说明即使证据完全不变，回答合同或生成稳定性仍可能导致失败，支持后续把 Prompt/解析/输出稳定性作为独立变量诊断；不能据此评定医学准确率或切换生产。

报告仅保存于私有路径 `data/private/evaluations/task18/f-fixed-evidence-20260905-001/task18-f-20260905T114745Z-e4268575.json`，递归禁止字段检查为 0。问题、答案、来源 ID、资料正文、Key、headers、reasoning 和原始响应均未保存；生产配置、知识库、黄金集和 GitHub 未改变。

### Task 18-G：80 题 vector 当前基线闭环（2026-09-05）

为得到可复核的当前基线，本轮固定 `final-ai-validated-20260904-005`（80 题）与 `RUN-20260904-004`（1,114 条 staging 切片），不切换 Hybrid、不改 Prompt/RRF/阈值。旧运行器会落盘题目和来源明细，因此改用内存执行器，只输出完成数、路由计数和聚合指标。

| 指标 | 结果 |
|---|---:|
| 完成题数 | 80/80 |
| 云端调用 | 277（Embedding 80、Reranker 80、回答 64、语义裁判 53） |
| 回答可用率 | 53/80 = 66.25% |
| 无答案安全拒答 | 16/16 = 100% |
| 运行时引用可见率 | 53/53 = 100% |
| Recall@20 / Recall@6 | 0.607 / 0.595 |
| Precision@6 / MRR@20 / nDCG@6 | 0.136 / 0.580 / 0.673 |
| 语义裁判 | 通过 40、不可用 13、未运行 27 |

这轮的安全与检索指标和既有 vector 基线一致，但回答可用率从另一轮的 58/80 波动到 53/80；结合 Task18-F 在固定证据下仍出现状态波动，说明当前优先级是回答合同/生成稳定性诊断。该结论不等于医学准确率，也不批准切换生产。聚合文件为 `data/private/evaluations/task18/baseline-vector-closure-safe-20260905-001/run-summary.json`，递归禁止字段为 0，未保存题目、答案、证据正文、来源 ID、Key、headers、reasoning 或原始响应。

### Task 18-H：15 题人工抽检与指标口径校准（2026-09-05）

Task18-G 的 aggregate 只能告诉我们总量和安全边界，不能替代逐题人工审阅。因此本轮创建本地私有抽检包，固定选择三类各 5 题：

| 样本组 | 数量 | 目的 |
|---|---:|---|
| answerable + answered | 5 | 检查主要结论、要点覆盖和证据支持 |
| answerable + 当前未回答 | 5 | 区分检索不足、回答合同失败或安全误拒 |
| no_answer | 5 | 检查应拒答与拒答理由 |

人工表位于 `data/private/evaluations/task18/manual-spotcheck-20260905-001/spotcheck-review.md`。每题需要在本地页面查看实际回答，并分别判断结论、证据、引用、拒答和越权风险。由于安全闭环未保存回答原文，抽检包不伪造回答；审核人填写的内容也不自动写入生产库。

指标口径修正如下：本次 aggregate 可确认回答总量 `53/80=66.25%`、无答案拒答 `16/16=100%`，但不能由 `53/80` 推导 `53/64` 的 answerable response rate；旧逐题诊断中的 `58/64=90.6%` 仅为样本选择线索。Grounded accuracy、citation 和 semantic judge 结果均是工程代理指标，不等于医疗正确率。人工审核达到主要结论一致性 0.85、引用定位准确 0.90、无答案拒答 0.90 后，才决定进入回答、检索或安全专项修复。
用户随后确认 15 题抽检均可接受，记录为 `preliminary_pass`（15/15）。这只说明当前抽样未发现明显问题，不等同于 80 题全量医疗正确率、临床 UAT 或发布批准；因此本轮不触发 Prompt、检索或安全门修改。

### 知识库版本生命周期（公开说明）

源项目已将旧生产索引与旧 staging 运行包移入本地可恢复归档，并保留最新的 `2026-standard-manual-v1` staging 作为唯一最新知识资产。公开仓库不包含真实手册、数据库、向量文件或归档内容；历史评测只作为不可篡改的工程追溯证据。由于 staging 与正式运行库 schema 不同，正式运行接入需要单独的迁移、回归与验收，不把 staging 归档动作包装成上线。
