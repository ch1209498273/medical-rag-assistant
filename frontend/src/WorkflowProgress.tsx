import type { WorkflowSummary } from "./api/types";

type Props = { summary: WorkflowSummary | null | undefined };

const ROUTE_LABELS: Record<NonNullable<WorkflowSummary["route"]>, string> = {
  direct: "直接回答路径",
  verify: "需核验路径",
  clarify: "需要澄清路径",
  out_of_scope: "超出范围路径",
};

const OUTCOME_LABELS: Record<WorkflowSummary["outcome"], string> = {
  answered: "已形成回答",
  refused: "依据不足，未形成正式回答",
  error: "处理未完成",
  cancelled: "已取消",
};

const VERIFIER_LABELS: Record<WorkflowSummary["verifier_status"], string> = {
  not_run: "未执行核验",
  passed: "核验通过",
  failed: "核验未通过",
  unavailable: "核验不可用",
};

const REASON_LABELS: Record<string, string> = {
  INSUFFICIENT_EVIDENCE: "资料不足",
  ANSWER_NOT_VERIFIABLE: "回答无法核验",
  MODEL_REFUSED: "模型拒答",
  EVIDENCE_SCOPE_UNCLEAR: "适用人员范围不清",
  DOCUMENT_BUSINESS_STATUS_UNKNOWN: "资料业务状态未确认",
  RETRIEVAL_UNAVAILABLE: "检索暂不可用",
  GENERATION_UNAVAILABLE: "回答生成暂不可用",
  ROUTER_UNAVAILABLE: "处理路径选择暂不可用",
  WORKFLOW_BUDGET_EXCEEDED: "本次处理达到调用上限",
  WORKFLOW_TIMEOUT: "本次处理超时",
  PROVIDER_UNAVAILABLE: "模型服务暂不可用",
  PROVIDER_TIMEOUT: "模型服务超时",
  PROVIDER_TRANSPORT_ERROR: "模型服务通信失败",
  PROVIDER_SCHEMA_INVALID: "模型返回格式异常",
};

export default function WorkflowProgress({ summary }: Props) {
  return (
    <details className="workflow-progress">
      <summary>本次处理过程</summary>
      {summary ? <SummaryFacts summary={summary} /> : <p>历史执行信息未记录</p>}
    </details>
  );
}

function SummaryFacts({ summary }: { summary: WorkflowSummary }) {
  const route = summary.route ? ROUTE_LABELS[summary.route] : "尚未选择路径";
  const reason = summary.reason_code
    ? REASON_LABELS[summary.reason_code] ?? "处理未完成"
    : "无额外原因";
  return (
    <dl className="workflow-progress-facts">
      <div>
        <dt>处理路径</dt>
        <dd>{route}</dd>
      </div>
      <div>
        <dt>处理结果</dt>
        <dd>{OUTCOME_LABELS[summary.outcome]}</dd>
      </div>
      <div>
        <dt>回答核验</dt>
        <dd>{VERIFIER_LABELS[summary.verifier_status]}</dd>
      </div>
      <div>
        <dt>模型调用</dt>
        <dd>{summary.http_calls} 次</dd>
      </div>
      {summary.elapsed_ms !== null && (
        <div>
          <dt>处理耗时</dt>
          <dd>{formatElapsed(summary.elapsed_ms)}</dd>
        </div>
      )}
      {summary.reason_code && (
        <div>
          <dt>补充说明</dt>
          <dd>{reason}</dd>
        </div>
      )}
    </dl>
  );
}

function formatElapsed(value: number): string {
  if (value < 1000) return `${value} 毫秒`;
  return `${(value / 1000).toFixed(1)} 秒`;
}
