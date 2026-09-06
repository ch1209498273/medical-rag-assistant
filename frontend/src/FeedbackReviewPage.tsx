import { useEffect, useMemo, useState } from "react";

import {
  ApiError,
  getFeedbackCase,
  getFeedbackReviewSummary,
  listFeedbackCases,
  promoteFeedbackCase,
  saveFeedbackReview,
} from "./api/client";
import type {
  FeedbackCaseDetailResponse,
  FeedbackCaseFilters,
  FeedbackCaseSummary,
  FeedbackReviewInput,
  FeedbackReviewStatus,
  FeedbackTriagePriority,
  FeedbackPromotionInput,
} from "./api/types";

const EMPTY_INPUT: FeedbackReviewInput = {
  reviewer_role: "product",
  decision: "approve",
  evidence_ok: true,
  points_ok: true,
  safety_ok: true,
  note_code: null,
  review_version: "task17b-v1",
};

const DEFAULT_PROMOTION_INPUT: FeedbackPromotionInput = {
  target_set_id: "2026-standard-manual-v1-golden-v2",
  target_version: "candidate-001",
  target_split: "dev",
  manifest_id: "manifest-synthetic-001",
};

export default function FeedbackReviewPage() {
  const [summary, setSummary] = useState<Record<string, number>>({});
  const [cases, setCases] = useState<FeedbackCaseSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<FeedbackCaseDetailResponse | null>(null);
  const [filters, setFilters] = useState<FeedbackCaseFilters>({ review_status: "unreviewed", limit: 50 });
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [promoting, setPromoting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [reviewInput, setReviewInput] = useState<FeedbackReviewInput>(EMPTY_INPUT);
  const [promotionInput, setPromotionInput] = useState<FeedbackPromotionInput>(DEFAULT_PROMOTION_INPUT);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    void Promise.all([getFeedbackReviewSummary(), listFeedbackCases(filters)])
      .then(([nextSummary, nextCases]) => {
        if (cancelled) return;
        setSummary(nextSummary);
        setCases(nextCases);
        setSelectedId((current) => current && nextCases.some((item) => item.case_id === current)
          ? current
          : nextCases[0]?.case_id ?? null);
      })
      .catch(() => {
        if (!cancelled) setError("反馈审核队列暂时无法加载，请检查本机管理员服务。");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, [filters]);

  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      return;
    }
    let cancelled = false;
    setDetailLoading(true);
    setError(null);
    void getFeedbackCase(selectedId)
      .then((nextDetail) => {
        if (!cancelled) {
          setDetail(nextDetail);
          setReviewInput({ ...EMPTY_INPUT });
        }
      })
      .catch(() => {
        if (!cancelled) setError("案例详情暂时无法加载。");
      })
      .finally(() => {
        if (!cancelled) setDetailLoading(false);
      });
    return () => { cancelled = true; };
  }, [selectedId]);

  const pendingCases = useMemo(
    () => cases.filter((item) => item.review_status === "unreviewed" || item.review_status === "in_review"),
    [cases],
  );

  const updateFilter = <K extends keyof FeedbackCaseFilters>(key: K, value: FeedbackCaseFilters[K]) => {
    setFilters((current) => ({ ...current, [key]: value }));
  };

  const submitReview = async () => {
    if (!selectedId || saving) return;
    setSaving(true);
    setError(null);
    setNotice(null);
    try {
      const result = await saveFeedbackReview(selectedId, reviewInput);
      setNotice("审核已保存，已追加到本地审核历史。");
      setDetail((current) => current ? {
        ...current,
        reviews: [...current.reviews, result.review],
        promotion: result.promotion,
      } : current);
      const next = pendingCases.find((item) => item.case_id !== selectedId);
      if (next) setSelectedId(next.case_id);
    } catch (errorValue) {
      setError(errorValue instanceof ApiError && errorValue.code === "FEEDBACK_REVIEW_SAVE_FAILED"
        ? "审核保存失败，未改变当前案例。"
        : "审核服务暂时不可用。");
    } finally {
      setSaving(false);
    }
  };

  const submitPromotion = async () => {
    if (!selectedId || !detail || promoting || detail.promotion.status !== "golden_v2_candidate") return;
    setPromoting(true);
    setError(null);
    setNotice(null);
    try {
      const result = await promoteFeedbackCase(selectedId, promotionInput);
      setNotice("已加入 golden-v2 候选集");
      setDetail((current) => current ? {
        ...current,
        case: { ...current.case, promotion_status: "golden_v2_candidate" },
        promotion: result.promotion,
        promotions: [...current.promotions, result.record],
      } : current);
    } catch (errorValue) {
      setError(errorValue instanceof ApiError && errorValue.code === "FEEDBACK_PROMOTION_FAILED"
        ? "加入候选集失败，当前案例未改变。"
        : "候选集服务暂时不可用。");
    } finally {
      setPromoting(false);
    }
  };

  return (
    <section className="page feedback-review-page" aria-labelledby="feedback-review-title">
      <header className="page-header">
        <div>
          <p className="eyebrow">反馈审核</p>
          <h2 id="feedback-review-title">反馈审核工作台</h2>
          <p className="page-description">只审核已脱敏的本地反馈，不自动导入真实数据或冻结黄金集。</p>
        </div>
        <div className="feedback-summary-cards" aria-label="审核聚合摘要">
          <span>待审核：{summary.review_unreviewed ?? 0}</span>
          <span>高风险 P0/P1：{(summary.priority_P0 ?? 0) + (summary.priority_P1 ?? 0)}</span>
          <span>需复议：{summary.review_adjudication_required ?? 0}</span>
          <span>golden-v2 候选：{summary.promotion_golden_v2_candidate ?? 0}</span>
        </div>
      </header>

      <div className="data-flywheel" aria-label="数据飞轮流程">
        <strong>数据飞轮</strong>
        <span className="flywheel-step done">① 用户反馈</span>
        <span className="flywheel-arrow">→</span>
        <span className="flywheel-step done">② 脱敏投影</span>
        <span className="flywheel-arrow">→</span>
        <span className="flywheel-step done">③ 分级去重</span>
        <span className="flywheel-arrow">→</span>
        <span className="flywheel-step done">④ 人工审核</span>
        <span className="flywheel-arrow">→</span>
        <span className="flywheel-step active">⑤ golden-v2 候选</span>
        <span className="flywheel-arrow">→</span>
        <span className="flywheel-step">⑥ 复评 / 冻结</span>
      </div>

      {error && <div className="page-error" role="alert">{error}</div>}
      {notice && <div className="status-line status-success" role="status" aria-live="polite">{notice}</div>}

      <div className="feedback-filters" aria-label="审核筛选">
        <label>优先级
          <select value={filters.priority ?? ""} onChange={(event) => updateFilter("priority", (event.target.value || undefined) as FeedbackTriagePriority | undefined)}>
            <option value="">全部</option><option value="P0">P0</option><option value="P1">P1</option><option value="P2">P2</option><option value="P3">P3</option>
          </select>
        </label>
        <label>审核状态
          <select value={filters.review_status ?? ""} onChange={(event) => updateFilter("review_status", (event.target.value || undefined) as FeedbackReviewStatus | undefined)}>
            <option value="">全部</option><option value="unreviewed">未审核</option><option value="in_review">审核中</option><option value="approved">已通过</option><option value="rejected">已拒绝</option><option value="adjudication_required">需复议</option>
          </select>
        </label>
        <label>反馈原因
          <select value={filters.reason_code ?? ""} onChange={(event) => updateFilter("reason_code", event.target.value || undefined)}>
            <option value="">全部</option><option value="not_answered">未回答</option><option value="missing_step">缺少步骤</option><option value="version_mismatch">版本不匹配</option><option value="citation_mismatch">引用不匹配</option><option value="too_slow">响应过慢</option>
          </select>
        </label>
        <label>答案状态
          <select value={filters.answer_status ?? ""} onChange={(event) => updateFilter("answer_status", (event.target.value || undefined) as FeedbackCaseFilters["answer_status"]) }>
            <option value="">全部</option><option value="answered">已回答</option><option value="refused">拒答</option><option value="error">错误</option>
          </select>
        </label>
        <label>资料版本
          <input value={filters.source_version ?? ""} onChange={(event) => updateFilter("source_version", event.target.value || undefined)} placeholder="例如 2026-standard-manual-v1" />
        </label>
      </div>

      <div className="feedback-review-grid">
        <section className="feedback-queue" aria-label="审核队列">
          <h3>待处理案例</h3>
          {loading && <p role="status">正在加载审核队列…</p>}
          {!loading && cases.length === 0 && <p>当前没有符合筛选条件的案例。</p>}
          {!loading && cases.map((item) => (
            <button
              type="button"
              key={item.case_id}
              className={selectedId === item.case_id ? "feedback-queue-item active" : "feedback-queue-item"}
              onClick={() => setSelectedId(item.case_id)}
              aria-pressed={selectedId === item.case_id}
            >
              <span className="feedback-queue-meta"><strong>{item.triage_priority}</strong><span>{reviewStatusLabel(item.review_status)}</span></span>
              <span>{item.question_preview}</span>
              <small>{item.reason_code ?? "无原因标签"} · {item.source_version ?? "版本未知"}</small>
            </button>
          ))}
        </section>

        <section className="feedback-case-detail" aria-label="案例详情">
          {detailLoading && <p role="status">正在加载案例详情…</p>}
          {!detailLoading && !detail && <p>请选择左侧案例。</p>}
          {!detailLoading && detail && (
            <>
              <div className="feedback-case-heading"><span className={`priority-badge priority-${detail.case.triage_priority}`}>{detail.case.triage_priority}</span><span>{detail.case.review_status}</span></div>
              <h3>问题与回答</h3>
              <div className="feedback-question"><strong>问题</strong><p>{detail.case.question ?? "内容因隐私状态不可展示"}</p></div>
              <div className="feedback-answer"><strong>回答</strong><p>{detail.case.answer ?? "内容因隐私状态不可展示"}</p></div>
              <h3>证据与 Trace 摘要</h3>
              <div className={detail.evidence.status === "available" ? "evidence-card available" : "evidence-card blocked"}>
                <strong>{detail.evidence.status === "available" ? "证据可核验" : "证据不可核验"}</strong>
                {detail.evidence.reason_code && <span>{detail.evidence.reason_code}</span>}
                {detail.evidence.references.map((reference) => <p key={reference.chunk_id}>{reference.file_name} · {reference.heading_path.join(" / ")} · {reference.excerpt}</p>)}
              </div>
              <dl className="trace-summary"><div><dt>检索配置</dt><dd>{detail.case.retrieval_profile ?? "—"}</dd></div><div><dt>提示词版本</dt><dd>{detail.case.prompt_version ?? "—"}</dd></div><div><dt>引用数</dt><dd>{detail.case.citation_count}</dd></div></dl>
            </>
          )}
        </section>

        <section className="feedback-review-form" role="form" aria-label="审核表单">
          <h3>审核 Rubric</h3>
          {detail?.case.triage_priority && ["P0", "P1"].includes(detail.case.triage_priority) && <div className="review-warning" role="note">P0/P1 需要专业审核（clinical_reviewer）才能进入候选状态。</div>}
          <label>审核角色
            <select value={reviewInput.reviewer_role} onChange={(event) => setReviewInput({ ...reviewInput, reviewer_role: event.target.value as FeedbackReviewInput["reviewer_role"] })}>
              <option value="product">产品</option><option value="engineering">工程</option><option value="clinical_reviewer">专业审核</option>
            </select>
          </label>
          <label>审核决定
            <select value={reviewInput.decision} onChange={(event) => setReviewInput({ ...reviewInput, decision: event.target.value as FeedbackReviewInput["decision"] })}>
              <option value="approve">通过</option><option value="reject">拒绝</option><option value="needs_adjudication">标记需复议</option>
            </select>
          </label>
          <RubricSelect label="证据充分" value={reviewInput.evidence_ok} onChange={(value) => setReviewInput({ ...reviewInput, evidence_ok: value })} />
          <RubricSelect label="要点完整" value={reviewInput.points_ok} onChange={(value) => setReviewInput({ ...reviewInput, points_ok: value })} />
          <RubricSelect label="医疗安全合格" value={reviewInput.safety_ok} onChange={(value) => setReviewInput({ ...reviewInput, safety_ok: value })} />
          <button type="button" className="primary-button" disabled={!detail || saving} onClick={() => void submitReview()}>{saving ? "保存中…" : "保存并处理下一条"}</button>
          <div className="promotion-check"><strong>晋级检查</strong><p>{detail?.promotion.reasons.join("、") ?? "请选择案例"}</p><span>{detail?.promotion.status === "golden_v2_candidate" ? "当前为候选" : "尚未晋级"}</span></div>
          <section className="promotion-form" aria-label="测试集增量">
            <h3>有序增加测试集</h3>
            <p className="promotion-help">只有审核门通过后才能追加候选；不会直接修改当前黄金集或自动冻结。</p>
            <label>目标测试集
              <input aria-label="目标测试集" value={promotionInput.target_set_id} onChange={(event) => setPromotionInput({ ...promotionInput, target_set_id: event.target.value })} />
            </label>
            <label>候选版本
              <input aria-label="候选版本" value={promotionInput.target_version} onChange={(event) => setPromotionInput({ ...promotionInput, target_version: event.target.value })} />
            </label>
            <label>数据分层
              <select aria-label="数据分层" value={promotionInput.target_split} onChange={(event) => setPromotionInput({ ...promotionInput, target_split: event.target.value as FeedbackPromotionInput["target_split"] })}>
                <option value="dev">dev（调优集）</option>
                <option value="holdout">holdout（留出集）</option>
              </select>
            </label>
            <label>Manifest ID
              <input aria-label="Manifest ID" value={promotionInput.manifest_id} onChange={(event) => setPromotionInput({ ...promotionInput, manifest_id: event.target.value })} />
            </label>
            <button type="button" className="primary-button" disabled={!detail || detail.promotion.status !== "golden_v2_candidate" || promoting} onClick={() => void submitPromotion()}>
              {promoting ? "加入中…" : "加入 golden-v2 候选集"}
            </button>
          </section>
          <h3>审核历史</h3>
          <ol className="review-history">{detail?.reviews.map((review) => <li key={review.review_id}>{review.reviewed_at} · {review.reviewer_role} · {review.decision}</li>) ?? <li>暂无审核记录</li>}</ol>
          <h3>测试集增量历史</h3>
          <ol className="review-history">{detail?.promotions.map((promotion) => <li key={promotion.promotion_id}>{promotion.target_set_id ?? "目标集未知"} · {promotion.target_version ?? "版本未知"} · {promotion.target_split}</li>) ?? <li>尚未追加候选</li>}</ol>
        </section>
      </div>
    </section>
  );
}

function RubricSelect({ label, value, onChange }: { label: string; value: boolean | null; onChange: (value: boolean | null) => void }) {
  return <label>{label}<select value={value === true ? "yes" : value === false ? "no" : "unknown"} onChange={(event) => onChange(event.target.value === "yes" ? true : event.target.value === "no" ? false : null)}><option value="yes">是</option><option value="no">否</option><option value="unknown">无法判断</option></select></label>;
}

function reviewStatusLabel(value: string): string {
  return ({ unreviewed: "未审核", in_review: "审核中", approved: "已通过", rejected: "已拒绝", adjudication_required: "需复议" } as Record<string, string>)[value] ?? "状态未知";
}
