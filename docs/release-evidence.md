# 发布证据与远程发布门

本文只记录公开副本可以在本地复核的事实，以及在触碰 GitHub 前必须确认的事项。它不保存真实资料、私有评测内容、Provider 原始响应、Key、本机路径或运行数据库。

## 本轮作品集展示重构

本轮公开副本按“业务问题 → 证据型 RAG → 质量评测 → Agent 治理 → 安全发布”重排：README 负责 30 秒定位，作品集导览负责能力地图，项目案例负责取舍叙事，架构/评测/多 Agent 文档负责技术追问。公开页面新增的 80 题与正式运行库数字只保留脱敏聚合，不携带真实题目、答案、资料或 Provider 原始响应。

公开页面当前明确区分三种证据：

| 证据层 | 可公开内容 | 不可推断 |
| --- | --- | --- |
| Demo | 虚构资料、无网络 smoke、页面截图 | 开放域模型能力或医学正确率 |
| 私有真实评测 | 冻结 80 题聚合指标、检索代理、拒答比例 | 临床 UAT、专家认证或生产上线 |
| 代码/CI | Provider Protocol、RAG 流程、Agent 合同、测试与发布门 | 多用户 ACL、公网服务或监管合规 |

## 本地可复核内容

- 公开副本由白名单导出脚本生成，只包含脱敏代码、原创虚构 Demo 资料、公开测试和文档。
- 2A 私有评测执行器与其测试不进入公开清单；公开副本只保留受控实验的边界说明，不携带冻结题集、逐题结果或 Provider 原始响应。
- 无网络 Demo smoke 使用拒绝外呼的 transport；成功时应输出 `network_calls=0`。
- `validate_public_release.py` 是独立公开副本的安全发布门：它检查白名单布局、敏感文本/路径、公开文档链接、审核资产和可达 Git 历史。
- 通用 `security_scan.py` 继续以私有开发工作树的 fail-closed 规则处理二进制资料，因此会拦截公开 Demo 的 PDF/DOCX 和截图；它不作为公开副本的通过结论，也不能据此放宽私有资料保护。
- 后端/前端测试、构建、公开发布校验和安全扫描是本地发布门；每次准备发布都应重新执行，不能沿用旧截图或旧计数。
- `CHANGELOG.md` 记录公开副本的面向读者变更；它不是虚构 Git tag 或 GitHub Release 的替代品。
- 反馈审核截图只展示合成案例；它用于证明“脱敏 → 证据核验 → 人工审核 → golden-v2 候选”的产品闭环，不证明真实反馈已经入库或黄金集已经冻结。

## 远程发布记录

本轮在本地发布门通过后核对了远程 `main` 基线，使用普通 fast-forward 将公开副本推送到 GitHub，并用 `git ls-remote origin refs/heads/main` 核验远程提交与本地发布提交一致。未使用 force-push。

本轮未创建或推送 Git tag，未创建 GitHub Release，也未修改仓库 topics、About 或其他远程设置；只同步通过本地发布门的代码、测试和文档提交。GitHub Actions 的状态仍以仓库页面实时结果为准，本地发布记录不代替 CI 结果。

## 最新远程 CI 核验（2026-09-05）

公开副本提交 `95a3931915f52539b37ac176e5f846d550569bc5` 推送后，GitHub Actions run `33974447579` 的 `verify` job 已完成并成功。随后文档关联补充提交 `531b4ab14d5b9a1e3124280481b6ee50118ada82` 也已推送，GitHub Actions run `33974863619` 的 `verify` job 已完成并成功。两次 run 的 Ruff、后端 pytest、`npm ci`、前端测试、前端构建、零网络 Demo smoke 和 `validate_public_release.py --history` 均为 `success`；这些证据只证明公开工程流水线和边界检查通过，不代表医疗准确率、临床 UAT 或上线批准。

## 推荐复核顺序

1. 从独立公开副本运行 `scripts/demo_smoke.py --no-network`，确认零网络。
2. 运行公开文档合同、后端测试、前端测试与前端构建。
3. 对 fresh whitelist export 运行 `validate_public_release.py` 与 `security_scan.py`，再比较导出目录和公开副本差异。
4. 检查 `git status`、`git diff --check` 与本地 tag 列表；确认没有真实资料、密钥、数据库、日志、路径、私有评测或原始 Provider 输出。
5. 将验证结果、发布提交和远程核验结果记录在交付说明中；推送后重新核验远程提交和 CI，不使用 force-push。

## 诚实范围

这些门证明的是工程可复现性与公开数据边界，不等于医学语义正确率、临床验证、渗透测试、生产合规或公网可用性。Cloud 模式始终是运行者自备 Key 的本机选项，费用、资料出站授权和运行风险由运行者自行确认。

## 2026-09-13 二期公开展示同步准备

本轮公开副本新增 `docs/phase2-delivery.md`，并在 README、CHANGELOG、ROADMAP 和作品集导览中补充二期范围、合成截图入口、问题处理思路、验收边界与三期承接关系。截图沿用公开 `docs/assets/` 下的合成素材；没有复制 `data/private/`、真实制度、逐题评测、运行库、凭据或患者信息。

发布前本地门固定为：`demo_smoke.py --no-network`、公开发布校验（含可达历史）、后端/前端必要回归、构建、Ruff 和 `git diff --check`。推送完成后再追加远程 `main` 提交与 CI 核验结果；在远程核验前不将本地提交表述为 GitHub 已更新。

## 2026-09-13 二期公开展示已同步

GitHub `main` 已更新到提交 `bea9145052ca24b0f49c217b8ffff64ce77879a1`，其父提交为 `45457eec9420d59746907964abc8189f27e3d616`；本机 `git ls-remote origin refs/heads/main` 返回同一 SHA。该次更新使用普通非强制引用移动，没有覆盖远端历史，也没有创建 tag 或 GitHub Release。

本地发布门结果：后端公开回归 `164 passed, 5 skipped`，前端 `33 passed`，前端构建通过，Ruff 通过，`git diff --check` 通过，`demo_smoke.py --no-network` 输出 `network_calls=0`，`validate_public_release.py --history` 通过。远程 GitHub Actions `ci` run `34753665238` 的 `verify` job（`103714300691`）已完成并成功；这些结果证明公开工程和数据边界可复核，不代表医学准确率、临床 UAT 或生产批准。

## 2026-09-13 README 二期状态显著化

为避免首页顶部仍像旧版，本次在 README 的公开状态区直接增加“二期交付状态”摘要和[二期交付说明](phase2-delivery.md)入口，保留原有 v1 基线描述和公开/私有边界。远端 `main` 提交为 `da3668ed7f6e27284d9b0a8f3ae93fd89a6cb094`；GitHub Actions `ci` run `34756264613` 的 `verify` job `103721031023` 已成功。

## 2026-09-13 二期界面预览重构

替换公开 `docs/assets/` 中的一期界面截图，新增二期工作台的合成展示：医护问答与证据、资料生命周期、Prompt 管理、账号权限、反馈审核、效果看板和问题流转。同步更新 README、二期交付说明、媒体清单和资产合同测试；预览不使用私有真实截图，公开无 Key Demo 仍保留 v1 运行基线。

