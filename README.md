# 医疗知识问答助手

> **证据约束的医疗知识 RAG 参考实现**：让医护培训资料的问答结果可定位、可拒答、可复现。

> Medical Knowledge Q&A Assistant — a local-first, evidence-grounded RAG portfolio project.

**公开发布原则：** 本仓库提供可复验的公开作品集；远程仓库、CI、Git tag 和 GitHub Release 状态均须逐项核验。详见[发布证据与远程发布门](docs/release-evidence.md)。

> **边界提醒：** 这是一个面向本机演示和工程学习的作品集项目。公开 Demo 使用完全虚构的资料，不代表真实医疗制度、医保规则或诊疗建议；它不是公网服务，也不替代专业人员判断。

## 30秒了解项目

一个可扩展的医疗知识问答助手。当前 v1.0 聚焦血透机构医护人员的制度与专业知识检索，采用可追溯引用、证据不足拒答和本地知识库存储；患者端和运行时多 Agent 属于 v2.0 路线。

它把 PDF/DOCX 资料变成可定位的 Chunk，经过 Embedding、向量召回、Reranker 和相关性门后，再生成带引用的结构化答案。系统同时提供一个无需 Key、无需网络的确定性 Demo，以及由运行者自备 Key 的显式 Cloud 模式。

30 秒了解项目：它展示的不是“让模型随便回答”，而是如何把资料版本、检索、证据引用、安全拒答、评测回滚和公开数据边界放在同一条可验证工程链路里。

技术栈：Python 3.12、FastAPI、Pydantic Settings、SQLite、Qdrant Local、PyMuPDF、python-docx、React 19、TypeScript、Vite、Vitest、DeepSeek 和 SiliconFlow。

## 界面预览

下面的素材均来自本机 Demo 的虚构资料和固定演示流程；原图尺寸、哈希和审核状态见[媒体清单](docs/assets/manifest.json)。它们用于作品集演示，不代表生产就绪、真实医疗制度或运行时多 Agent。

![有引用的正式答案](docs/assets/chat-answer.png)

![证据不足拒答](docs/assets/chat-refusal.png)

![历史会话恢复](docs/assets/chat-history.png)

![资料管理](docs/assets/documents.png)

![无 Key 演示模式](docs/assets/runtime-mode.png)

![问题流转短 GIF](docs/assets/question-flow.gif)

运行 Demo 后可以看到：

- 左侧“制度问答”和“资料管理”导航，以及持续显示的“无 Key 演示模式”标识；
- 问答页的检索、重排、生成、引用核验状态；
- 正式答案、可展开引用、证据不足拒答、历史会话恢复和反馈按钮；
- 资料管理页返回安全的文件名、版本状态和失败原因，不返回本机路径或原文数据库。

## 快速体验

### Windows 一键 Demo

在项目根目录执行。首次运行需要 Python 3.12、Node.js/npm 和一个本地 Python 虚拟环境：

```powershell
py -3.12 -m venv backend\.venv
backend\.venv\Scripts\python.exe -m pip install -e ".\backend[dev]"
Set-Location frontend
npm ci
Set-Location ..
powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1
```

然后打开 `http://127.0.0.1:5173`。推荐按这个顺序体验：

1. 问“新员工独立值班前需要完成多少学时岗前培训？”并展开引用；
2. 在同一个会话中问“那没通过怎么办？”，观察追问改写和 7 日补训答案；
3. 问“血液透析费用报销比例是多少？”，确认系统拒答且不展示伪造引用；
4. 打开“资料管理”，查看四份虚构资料的索引状态。

停止本机进程：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stop_local.ps1
```

无网络 Demo 冒烟（在不启动浏览器的情况下验证完整后端链路）：

```powershell
backend\.venv\Scripts\python.exe scripts\demo_smoke.py --no-network
```

### macOS/Linux 手动启动 Demo

先在项目根目录一次性安装依赖：

```bash
python3.12 -m venv backend/.venv
backend/.venv/bin/python -m pip install -e './backend[dev]'
(cd frontend && npm ci)
```

然后打开两个终端，均从项目根目录开始。终端 1 保持运行后，再在终端 2 启动前端：

终端 1 — Demo backend：

```bash
cd backend
APP_RUNTIME_MODE=demo \
SOURCE_DOCUMENTS_DIR=../demo/documents \
DEMO_QUESTIONS_PATH=../demo/questions.json \
PRIVATE_DATA_DIR=../data/runtime/demo/private \
QDRANT_PATH=../data/runtime/demo/qdrant \
SQLITE_PATH=../data/runtime/demo/app.sqlite3 \
HOST=127.0.0.1 PORT=8000 \
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

终端 2 — Demo frontend：

```bash
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

两个进程都只绑定本机地址；分别在两个终端使用 `Ctrl+C` 停止。无浏览器冒烟仍可在项目根目录运行 `backend/.venv/bin/python scripts/demo_smoke.py --no-network`。

### 自备 Key 的 Cloud 模式

Cloud 是显式选择，不是 Demo 的隐式回退：

1. 复制 `.env.example` 为本机 `.env`；
2. 将 `APP_RUNTIME_MODE` 改为 `cloud`，只填入运行者自己的 DeepSeek 和 SiliconFlow Key；
3. 使用 `scripts\start_cloud.ps1`（Windows）或从 `backend` 目录手动启动；
4. 看到 `deepseek=configured` 和 `siliconflow=configured` 后，才会启动正式 Cloud Provider。

macOS/Linux 手动启动 Cloud 时，先在项目根目录完成第 1、2 步，再打开两个终端。首次运行可用下面的命令复制配置模板，再用编辑器填写自己的值；`.env` 由后端从项目根目录读取，启动命令本身不包含 Key：

```bash
cp .env.example .env
```

终端 1 — Cloud backend：

```bash
cd backend
APP_RUNTIME_MODE=cloud \
SOURCE_DOCUMENTS_DIR=../demo/documents \
DEMO_QUESTIONS_PATH=../demo/questions.json \
PRIVATE_DATA_DIR=../data/runtime/cloud/private \
QDRANT_PATH=../data/runtime/cloud/qdrant \
SQLITE_PATH=../data/runtime/cloud/app.sqlite3 \
HOST=127.0.0.1 PORT=8000 \
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

终端 2 — Cloud frontend：

```bash
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

Cloud 也只绑定本机地址；分别在两个终端使用 `Ctrl+C` 停止。若任一 Provider 显示 `missing`，后端会在启动前 fail closed，不会自动回到 Demo，也不会由文档命令发起调用。

前端没有 Key 输入框，也不会从 health、SSE、localStorage、构建产物或日志取得 Key。缺少必需配置时，Cloud 在启动前以不含敏感值的错误失败，绝不自动切到 Demo、其他账户或收费模型。Cloud 请求只使用公开的四份虚构资料，费用和配额由运行者自己承担。

## 已实现功能

- PDF/DOCX 的安全解析、资源上限、内容哈希和增量版本索引；
- SQLite 文档版本、会话、消息、引用和反馈持久化；Qdrant Local 保存向量；
- active version 过滤、最多 20 条召回、去重、最多 6 条 Reranker 结果和 `0.35` 相关性门；
- `retrieving → reranking → generating → validating` 阶段状态与 SSE 问答；
- 结构化 `refused`/`answer` 答案合同，答案先完整缓冲，引用集合通过后才显示；
- 缺证据、模型拒答、Provider 不可用、答案不可核验等固定 reason code；
- 连续会话、追问改写、历史恢复、正式答案与未核验通用参考分栏；
- 确定性 Demo Provider：本地 1024 维向量、词法重排、登记场景匹配和安全拒答；
- 显式 Demo/Cloud 健康状态，以及不泄露凭据的 Windows 启动、停止和 Demo smoke 脚本；
- 白名单式公开导出边界，独立公开副本不携带真实资料、数据库、日志、Key 或内部执行记录。

## 系统架构

```text
PDF/DOCX
   │ 解析、定位、Chunk、哈希
   ├──────────────► SQLite：文档版本、会话、消息、反馈
   └──────────────► Embedding ─► Qdrant Local：向量与 active version

问题 ─► 清理 ─► 向量召回(≤20) ─► 去重 ─► Reranker(≤6)
     ─► 相关性门(0.35) ─► Demo 模型或 DeepSeek
     ─► 结构化答案/引用核验 ─► SSE ─► React 页面与 SQLite 会话
```

模块职责、事件边界和存储关系见 [系统架构说明](docs/architecture.md)。公开导出、演示和文档职责见 [Demo 指南](docs/demo-guide.md) 与 [项目案例](docs/project-case-study.md)。

## 一个问题的数据旅程

以公开 Demo 的培训问题为例：

1. `IngestionService` 只读取配置目录中的一层 PDF/DOCX，先做路径、大小和文件类型检查；
2. 提取器产生带文件名、标题层级和页码/段落位置的 `SourceRef`，Chunker 保留这些位置；
3. Embedding 写入 Qdrant Local，SQLite 记录文件哈希和版本状态；只有 active version 参与召回；
4. 用户问题先经过长度和控制字符清理，再做最多 20 条召回、去重、最多 6 条重排，并要求最高分达到 `0.35`；
5. 证据放在明确边界内送入 Demo Language Model 或 Cloud DeepSeek。Demo 只精确匹配已登记场景，不执行问题或证据里的指令；
6. 答案在内存中完整缓冲，解析双字段 JSON，检查文本安全、来源编号和可定位引用。任一检查失败都整条拒答；
7. 通过后才发出 `answer_delta` 和 `final` SSE 事件，并把会话、状态、引用和反馈写入 SQLite。

## 评测与回滚

公开 Demo 的可执行冒烟结果由脚本输出安全聚合值：四份 PDF/DOCX 共 8 个文件、支持问题 2 个、拒答问题 1 个、网络调用 0 次。该数字描述离线流程连通性，不是医学正确率。

私有冻结评测只在本地授权环境执行，公开页面只保留聚合判断，不包含题目、答案、资料正文或逐题结果：

| 配置 | 正式回答可用性 | 无答案拒答 | 结论 |
| --- | ---: | ---: | --- |
| 回滚后的基线/C1 | 9/30（30.0%） | 100% | 作为当前正式提示词基线 |
| C2 实验 | 14/30（46.7%） | 60% | 回答数增加，但低于 90% 安全门，按确认回滚 |

C2 说明提示词实验的取舍：它减少了可回答问题的过度拒答，却让资料中没有答案的问题更容易被回答，安全边界退化。因此当前正式提示词恢复为 C1；C2 报告仅用于工程复盘，不构成医疗有效性、监管认证或生产可用性证明。更完整的方法和限制见 [评测说明](docs/evaluation.md)。

## 工程质量

**工程亮点：** 用确定性 Demo 把“可复现”与“可选云端路径”拆开；用整条答案缓冲和引用集合校验把“有文本”与“有证据”拆开；用冻结评测和回滚记录把“回答更多”与“安全边界更好”拆开。

- 后端、前端、Demo Provider、SSE、会话和安全边界均有自动测试；每个任务遵循 RED → GREEN → 回归；
- 文档合同测试会检查本 README 的精确章节顺序、所有公开 Markdown 相对链接、Demo/Cloud 边界、凭据/路径、Provider 兼容性以及“生产级/运行时多 Agent”越界叙述；
- `scripts/security_scan.py` 以私有开发工作树的 fail-closed 规则处理二进制资料；独立公开副本的安全发布门是 `scripts/validate_public_release.py`，它能核验允许的虚构 Demo 资产与历史边界，不能把源仓库扫描结果冒充公开发布通过；
- 文档合同可通过 `backend\\.venv\\Scripts\\python.exe -m pytest -q backend/tests/release/test_public_docs.py` 复核；后端/前端全量计数、Demo smoke 和历史扫描均在独立公开副本的最终发布门重新生成，避免把旧运行计数包装成当前证据。

## 多 Agent 开发

开发过程中按任务隔离规划、实现、测试、安全复核和文档交接，由项目负责人在需求、数据授权、指标判断和回滚处做决定。产品运行时当前是一个单一编排的 RAG 工作流，不在请求期间调用开发 Agent。角色、交接格式和审查证据见 [多 Agent 开发说明](docs/multi-agent-development.md)。

## 安全与隐私

公开边界的核心规则：

- 资料全部是虚构演示资料；不导出真实企业资料、私有评测题、逐题答案、数据库、向量库、日志或本机路径；
- Demo 不读 Key、不访问外网；Cloud 只读运行者本机 `.env`，前端不接触 Provider Key；
- health 只返回 `configured`/`missing`/`not_required` 等状态，错误和 SSE 只返回固定 reason code；
- 服务默认绑定本机，不提供托管公网服务，不收集用户 Key，也不承诺审计、登录、ACL 或多用户隔离；
- 发布前使用白名单生成独立副本，再进行工作树和完整 Git 历史检查。

详细报告方式和不在范围内的承诺见 [SECURITY.md](SECURITY.md)。

## 技术决策

v1 选择 SQLite + Qdrant Local，换取本机可复现、低运维和不依赖外部服务；Provider 通过最小 Protocol 隔离 Demo 与 Cloud；结构化答案和整条引用门优先于“尽量给出回答”。这些决策的代价、替代方案和回滚记录见 [工程决策](docs/engineering-decisions.md)。

## 当前限制

- v1 的四份资料和问题均为原创虚构演示，不可用于真实制度、医保、诊疗、处方或患者决策；
- 无登录、角色授权、ACL、患者端、审计后台或公网部署；本机 local storage 不是多用户安全边界；
- Demo 的 OCR 不伪造图片识别结果；Cloud OCR/模型调用需要运行者自己的 Key、网络和费用；
- 引用门证明来源结构和定位，不等于模型答案在语义上或医学上正确；公开 Demo 不提供临床决策支持；
- MySQL、Qdrant Server、运行时多 Agent、多知识域和完整人工评测属于 v2.0；
- 远程仓库、CI、Git tag 和 GitHub Release 的状态必须逐项核验；本地专业化整理完成前，不把任何远程状态或徽章写成既成事实。

## v2.0 路线

v2.0 只记录候选方向，不暗示已实现：登录与 ACL、患者科普边界、多知识域与可见范围、MySQL/Qdrant Server/审计、多用户部署、运行时路由与专业审查、人工评测和成本观测。分栏路线图见 [ROADMAP.md](ROADMAP.md)。

## License

公开副本计划采用 [MIT License](LICENSE)，仅覆盖独立公开仓库中实际发布的脱敏代码、虚构资料和公开文档；真实资料、私有评测、私人手册和未导出的开发资产不属于该许可范围。
