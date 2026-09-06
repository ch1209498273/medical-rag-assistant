import type {
  ChatEvent,
  Citation,
  AudienceScope,
  FeedbackCaseDetailResponse,
  FeedbackCaseFilters,
  FeedbackCaseSummary,
  FeedbackEvidence,
  FeedbackEvidenceReference,
  FeedbackEventView,
  FeedbackPromotion,
  FeedbackPromotionInput,
  FeedbackReviewInput,
  FeedbackReviewSummary,
  FeedbackReviewView,
  PromotionCheck,
  DocumentRow,
  DocumentBusinessMetadata,
  DocumentBusinessMetadataInput,
  Feedback,
  FeedbackReason,
  HealthResponse,
  ScanResult,
  SessionSummary,
  SessionResponse,
  WorkflowSummary,
} from "./types";

const API_PREFIX = "/api";
const CHAT_STAGES = new Set([
  "accepted",
  "rewriting",
  "preflight",
  "routing",
  "retrieving",
  "reranking",
  "generating",
  "validating",
  "verifying",
]);
const EVENT_TYPES = new Set(["status", "answer_delta", "final", "error"]);
const SAFE_ID = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/;
const SAFE_REFERENCE_ID = /^S[1-6]$/;
const SAFE_FILE_NAME = /^[^<>:"/\\|?*\u0000-\u001f]+\.(?:pdf|docx)$/i;
const SAFE_TIMESTAMP =
  /^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})?$/;
const SAFE_REASONS = new Set([
  "INSUFFICIENT_EVIDENCE",
  "MODEL_REFUSED",
  "ANSWER_NOT_VERIFIABLE",
  "RETRIEVAL_UNAVAILABLE",
  "GENERATION_UNAVAILABLE",
  "FOLLOW_UP_UNAVAILABLE",
  "CHAT_UNAVAILABLE",
  "INVALID_EVENT",
  "PROVIDER_UNAVAILABLE",
  "EVIDENCE_SCOPE_UNCLEAR",
  "DOCUMENT_BUSINESS_STATUS_UNKNOWN",
  "PROVIDER_TIMEOUT",
  "PROVIDER_TRANSPORT_ERROR",
  "PROVIDER_SCHEMA_INVALID",
  "OUTPUT_TRUNCATED",
  "ROUTER_UNAVAILABLE",
  "WORKFLOW_BUDGET_EXCEEDED",
  "WORKFLOW_UNAVAILABLE",
  "WORKFLOW_TIMEOUT",
  "QUESTION_NEEDS_CLARIFICATION",
  "OUT_OF_SCOPE",
]);
const SAFE_FEEDBACK_REASONS = new Set<FeedbackReason>([
  "not_answered",
  "missing_step",
  "version_mismatch",
  "citation_mismatch",
  "too_slow",
]);
const SAFE_AUDIENCE_SCOPES = new Set([
  "unspecified",
  "all_staff",
  "nurse",
  "doctor",
  "pharmacist",
  "administrator",
]);
const SAFE_MESSAGE_STATUSES = new Set(["submitted", "answered", "refused", "error"]);
const REFERENCE_REASONS = new Set([
  "INSUFFICIENT_EVIDENCE",
  "MODEL_REFUSED",
  "ANSWER_NOT_VERIFIABLE",
]);
const SAFE_DOCUMENT_STATUSES = new Set([
  "indexing",
  "active",
  "failed",
  "superseded",
  "inactive",
  "needs_ocr",
]);
const SAFE_FAILURE_REASONS = new Set([
  "PROVIDER_UNAVAILABLE",
  "SOURCE_FILE_MISSING",
  "SOURCE_FILE_UNREADABLE",
  "DOCUMENT_NOT_INDEXABLE",
  "INDEX_FAILED",
]);
const SAFE_ERROR_TEXT = "当前会话记录无法安全恢复。";
const SAFE_RUNTIME_MODES = new Set(["demo", "cloud"]);
const SAFE_PROVIDER_STATES = new Set([
  "configured",
  "missing",
  "optional_missing",
  "not_required",
]);
const HEALTH_PROVIDERS = ["deepseek", "siliconflow", "minimax"] as const;

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;

  constructor(status: number, code: string) {
    super(code);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

export class ClientStreamError extends Error {
  readonly code = "CLIENT_STREAM_INVALID" as const;

  constructor() {
    super("CLIENT_STREAM_INVALID");
    this.name = "ClientStreamError";
  }
}

export async function getHealth(): Promise<HealthResponse> {
  const payload = await requestJson(`${API_PREFIX}/health`, { method: "GET" }, "HEALTH_UNAVAILABLE");
  try {
    return normalizeHealthResponse(payload);
  } catch {
    throw new ApiError(200, "HEALTH_UNAVAILABLE");
  }
}

export async function streamChat(
  question: string,
  sessionId: string | null,
  onEvent: (event: ChatEvent) => void,
  audienceScope: AudienceScope = "unspecified",
): Promise<void> {
  const body: { question: string; session_id?: string; audience_scope?: AudienceScope } = { question };
  if (sessionId) body.session_id = sessionId;
  if (audienceScope !== "unspecified") body.audience_scope = audienceScope;

  let response: Response;
  try {
    response = await fetch(`${API_PREFIX}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, "CHAT_UNAVAILABLE");
  }

  if (!response.ok) {
    throw new ApiError(response.status, "CHAT_REQUEST_FAILED");
  }
  if (!response.body) {
    throw new ClientStreamError();
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    for (;;) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
      const frames = buffer.split(/\r?\n\r?\n/);
      buffer = frames.pop() ?? "";
      frames.forEach((frame) => {
        const event = parseFrame(frame);
        if (event) onEvent(event);
      });
      if (done) break;
    }
    buffer += decoder.decode();
    const finalFrame = parseFrame(buffer);
    if (finalFrame) onEvent(finalFrame);
  } catch (error) {
    if (error instanceof ClientStreamError) throw error;
    throw new ClientStreamError();
  } finally {
    reader.releaseLock();
  }
}

export async function getSession(sessionId: string): Promise<SessionResponse> {
  const payload = await requestJson(
    `${API_PREFIX}/chat/sessions/${encodeURIComponent(sessionId)}`,
    { method: "GET" },
    "SESSION_UNAVAILABLE",
  );
  try {
    return normalizeSessionResponse(payload);
  } catch {
    throw new ApiError(200, "SESSION_UNAVAILABLE");
  }
}

export async function listSessions(limit = 20): Promise<SessionSummary[]> {
  const payload = await requestJson(
    `${API_PREFIX}/chat/sessions?limit=${encodeURIComponent(String(limit))}`,
    { method: "GET" },
    "SESSION_LIST_UNAVAILABLE",
  );
  try {
    return normalizeSessionListResponse(payload);
  } catch {
    throw new ApiError(200, "SESSION_LIST_UNAVAILABLE");
  }
}

export async function saveFeedback(
  messageId: string,
  helpful: boolean,
  reason?: FeedbackReason | null,
): Promise<Feedback> {
  if (helpful && reason != null) throw new ApiError(400, "FEEDBACK_REASON_INVALID");
  if (reason != null && !SAFE_FEEDBACK_REASONS.has(reason)) {
    throw new ApiError(400, "FEEDBACK_REASON_INVALID");
  }
  const body: { helpful: boolean; reason?: FeedbackReason } = { helpful };
  if (reason != null) body.reason = reason;
  const payload = await requestJson(
    `${API_PREFIX}/chat/messages/${encodeURIComponent(messageId)}/feedback`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
    "FEEDBACK_UNAVAILABLE",
  );
  try {
    const feedback = normalizeFeedback(payload);
    if (!feedback) throw new Error("invalid feedback");
    return feedback;
  } catch {
    throw new ApiError(200, "FEEDBACK_UNAVAILABLE");
  }
}

export async function listDocuments(): Promise<DocumentRow[]> {
  const payload = await requestJson(
    documentsPath(),
    { method: "GET" },
    "DOCUMENTS_UNAVAILABLE",
  );
  if (!isRecord(payload) || !Array.isArray(payload.documents)) return [];
  return payload.documents
    .map((value) => normalizeDocumentRow(value))
    .filter((value): value is DocumentRow => value !== null);
}

export async function scanDocuments(): Promise<ScanResult> {
  const payload = await requestJson(
    documentsPath("scan"),
    { method: "POST" },
    "DOCUMENT_SCAN_FAILED",
  );
  try {
    return normalizeScanResult(payload);
  } catch {
    throw new ApiError(200, "DOCUMENT_SCAN_FAILED");
  }
}

export async function reindexDocument(documentId: number): Promise<DocumentRow> {
  const payload = await requestJson(
    documentsPath(String(documentId), "reindex"),
    { method: "POST" },
    "DOCUMENT_REINDEX_FAILED",
  );
  const row = normalizeDocumentRow(payload);
  if (!row) throw new ApiError(200, "DOCUMENT_REINDEX_FAILED");
  return row;
}

export async function saveDocumentBusinessMetadata(
  documentId: number,
  metadata: DocumentBusinessMetadataInput,
): Promise<DocumentBusinessMetadata> {
  const payload = await requestJson(
    documentsPath(String(documentId), "business-metadata"),
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(metadata),
    },
    "DOCUMENT_METADATA_SAVE_FAILED",
  );
  const result = normalizeDocumentBusinessMetadata(payload);
  if (!result) throw new ApiError(200, "DOCUMENT_METADATA_SAVE_FAILED");
  return result;
}

export async function getFeedbackReviewSummary(): Promise<FeedbackReviewSummary> {
  const payload = await requestJson(
    "/api/admin/feedback/summary",
    { method: "GET" },
    "FEEDBACK_REVIEW_UNAVAILABLE",
  );
  if (!isRecord(payload) || !isRecord(payload.summary)) {
    throw new ApiError(200, "FEEDBACK_REVIEW_UNAVAILABLE");
  }
  const summary: FeedbackReviewSummary = {};
  for (const [key, value] of Object.entries(payload.summary)) {
    if (!/^[A-Za-z0-9_]+$/.test(key) || typeof value !== "number" || !Number.isInteger(value) || value < 0) {
      throw new ApiError(200, "FEEDBACK_REVIEW_UNAVAILABLE");
    }
    summary[key] = value;
  }
  return summary;
}

export async function listFeedbackCases(filters: FeedbackCaseFilters = {}): Promise<FeedbackCaseSummary[]> {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(filters)) {
    if (value !== undefined) params.set(key, String(value));
  }
  const query = params.toString();
  const payload = await requestJson(
    `/api/admin/feedback/cases${query ? `?${query}` : ""}`,
    { method: "GET" },
    "FEEDBACK_REVIEW_UNAVAILABLE",
  );
  if (!isRecord(payload) || !Array.isArray(payload.cases)) {
    throw new ApiError(200, "FEEDBACK_REVIEW_UNAVAILABLE");
  }
  try {
    return payload.cases.map(normalizeFeedbackCaseSummary);
  } catch {
    throw new ApiError(200, "FEEDBACK_REVIEW_UNAVAILABLE");
  }
}

export async function getFeedbackCase(caseId: string): Promise<FeedbackCaseDetailResponse> {
  if (!isSafeId(caseId)) throw new ApiError(400, "FEEDBACK_REVIEW_UNAVAILABLE");
  const payload = await requestJson(
    `/api/admin/feedback/cases/${encodeURIComponent(caseId)}`,
    { method: "GET" },
    "FEEDBACK_REVIEW_UNAVAILABLE",
  );
  try {
    return normalizeFeedbackCaseDetailResponse(payload);
  } catch {
    throw new ApiError(200, "FEEDBACK_REVIEW_UNAVAILABLE");
  }
}

export async function saveFeedbackReview(
  caseId: string,
  input: FeedbackReviewInput,
): Promise<{ review: FeedbackReviewView; promotion: PromotionCheck }> {
  if (!isSafeId(caseId) || !isFeedbackReviewInput(input)) {
    throw new ApiError(400, "FEEDBACK_REVIEW_SAVE_FAILED");
  }
  const payload = await requestJson(
    `/api/admin/feedback/cases/${encodeURIComponent(caseId)}/reviews`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    },
    "FEEDBACK_REVIEW_SAVE_FAILED",
  );
  try {
    if (!isRecord(payload)) throw new Error("invalid review payload");
    return {
      review: normalizeFeedbackReview(payload.review),
      promotion: normalizePromotionCheck(payload.promotion),
    };
  } catch {
    throw new ApiError(200, "FEEDBACK_REVIEW_SAVE_FAILED");
  }
}

export async function getFeedbackPromotionCheck(caseId: string): Promise<PromotionCheck> {
  if (!isSafeId(caseId)) throw new ApiError(400, "FEEDBACK_REVIEW_UNAVAILABLE");
  const payload = await requestJson(
    `/api/admin/feedback/cases/${encodeURIComponent(caseId)}/promotion-check`,
    { method: "GET" },
    "FEEDBACK_REVIEW_UNAVAILABLE",
  );
  try {
    return normalizePromotionCheck(payload);
  } catch {
    throw new ApiError(200, "FEEDBACK_REVIEW_UNAVAILABLE");
  }
}

export async function promoteFeedbackCase(
  caseId: string,
  input: FeedbackPromotionInput,
): Promise<{ promotion: PromotionCheck; record: FeedbackPromotion }> {
  if (
    !isSafeId(caseId) ||
    !isSafeId(input.target_set_id) ||
    !isSafeId(input.target_version) ||
    !isSafeId(input.manifest_id) ||
    !isOneOf(input.target_split, ["dev", "holdout"])
  ) {
    throw new ApiError(400, "FEEDBACK_PROMOTION_INVALID");
  }
  const payload = await requestJson(
    `/api/admin/feedback/cases/${encodeURIComponent(caseId)}/promote`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    },
    "FEEDBACK_PROMOTION_FAILED",
  );
  try {
    if (!isRecord(payload)) throw new Error("invalid promotion payload");
    return {
      promotion: normalizePromotionCheck(payload.promotion),
      record: normalizeFeedbackPromotion(payload.record),
    };
  } catch {
    throw new ApiError(200, "FEEDBACK_PROMOTION_FAILED");
  }
}

function documentsPath(...segments: string[]): string {
  return [API_PREFIX, "documents", ...segments].join("/");
}

function normalizeFeedbackCaseSummary(value: unknown): FeedbackCaseSummary {
  if (!isRecord(value)) throw new Error("case summary is invalid");
  const priority = value.triage_priority;
  const reviewStatus = value.review_status;
  const promotionStatus = value.promotion_status;
  if (
    !isSafeId(value.case_id) ||
    !isOneOf(priority, ["P0", "P1", "P2", "P3"]) ||
    !isOneOf(reviewStatus, ["unreviewed", "in_review", "approved", "rejected", "adjudication_required"]) ||
    !isOneOf(promotionStatus, ["not_promoted", "silver", "golden_v2_candidate", "golden_v2"]) ||
    typeof value.question_preview !== "string" ||
    value.question_preview.length > 120 ||
    !isSafeText(value.question_preview) ||
    !isSafeTimestamp(value.collected_at)
  ) throw new Error("case summary is invalid");
  return {
    case_id: value.case_id,
    triage_priority: priority,
    reason_code: optionalSafeCode(value.reason_code),
    question_preview: value.question_preview,
    source_version: optionalSafeCode(value.source_version),
    review_status: reviewStatus,
    promotion_status: promotionStatus,
    collected_at: value.collected_at,
  };
}

function normalizeFeedbackCaseDetailResponse(value: unknown): FeedbackCaseDetailResponse {
  if (!isRecord(value)) throw new Error("case detail is invalid");
  if (!isRecord(value.case) || !Array.isArray(value.events) || !Array.isArray(value.reviews) || !isRecord(value.evidence) || !isRecord(value.promotion) || !Array.isArray(value.promotions)) {
    throw new Error("case detail is invalid");
  }
  const detail = value.case;
  if (
    !isSafeId(detail.case_id) ||
    !isOneOf(detail.redaction_status, ["passed", "review_required", "blocked"]) ||
    !isOneOf(detail.answer_status, ["answered", "refused", "error"]) ||
    (detail.question !== null && (typeof detail.question !== "string" || !isSafeText(detail.question))) ||
    (detail.answer !== null && (typeof detail.answer !== "string" || !isSafeText(detail.answer))) ||
    !isSafeTimestamp(detail.collected_at) ||
    !isOneOf(detail.triage_priority, ["P0", "P1", "P2", "P3"]) ||
    typeof detail.triage_score !== "number" || !Number.isInteger(detail.triage_score) || detail.triage_score < 0 ||
    !isOneOf(detail.review_status, ["unreviewed", "in_review", "approved", "rejected", "adjudication_required"]) ||
    !isOneOf(detail.promotion_status, ["not_promoted", "silver", "golden_v2_candidate", "golden_v2"]) ||
    !isOneOf(detail.audience_scope, ["unspecified", "all_staff", "nurse", "doctor", "pharmacist", "administrator"])
  ) throw new Error("case detail is invalid");
  const detailResult = {
    case_id: detail.case_id,
    question: detail.question,
    answer: detail.answer,
    redaction_status: detail.redaction_status,
    answer_status: detail.answer_status,
    reason_code: optionalSafeCode(detail.reason_code),
    source_version: optionalSafeCode(detail.source_version),
    retrieval_profile: optionalSafeCode(detail.retrieval_profile),
    citation_chunk_ids: safeIdArray(detail.citation_chunk_ids),
    citation_count: safeNonNegativeInteger(detail.citation_count),
    model_id: optionalSafeCode(detail.model_id),
    prompt_version: optionalSafeCode(detail.prompt_version),
    latency_bucket: optionalSafeCode(detail.latency_bucket),
    audience_scope: detail.audience_scope,
    collected_at: detail.collected_at,
    triage_priority: detail.triage_priority,
    triage_score: detail.triage_score,
    triage_reasons: safeCodeArray(detail.triage_reasons),
    review_status: detail.review_status,
    promotion_status: detail.promotion_status,
    event_count: safeNonNegativeInteger(detail.event_count),
  } satisfies FeedbackCaseDetailResponse["case"];
  return {
    case: detailResult,
    events: value.events.map(normalizeFeedbackEvent),
    reviews: value.reviews.map(normalizeFeedbackReview),
    evidence: normalizeFeedbackEvidence(value.evidence),
    promotion: normalizePromotionCheck(value.promotion),
    promotions: value.promotions.map(normalizeFeedbackPromotion),
  };
}

function normalizeFeedbackEvidence(value: unknown): FeedbackEvidence {
  if (!isRecord(value) || !isOneOf(value.status, ["available", "unavailable", "version_mismatch", "unsafe"]) || !Array.isArray(value.references)) throw new Error("evidence is invalid");
  return {
    status: value.status,
    reason_code: optionalSafeCode(value.reason_code),
    references: value.references.map(normalizeFeedbackEvidenceReference),
  };
}

function normalizeFeedbackEvidenceReference(value: unknown): FeedbackEvidenceReference {
  if (!isRecord(value) || !isSafeId(value.chunk_id) || !isSafeId(value.source_version) || !isSafeFileName(value.file_name) || !Array.isArray(value.heading_path) || !value.heading_path.every((item) => typeof item === "string" && isSafeHeading(item)) || typeof value.excerpt !== "string" || !isSafeText(value.excerpt) || value.excerpt.length > 300) throw new Error("evidence reference is invalid");
  return {
    chunk_id: value.chunk_id,
    source_version: value.source_version,
    file_name: value.file_name,
    heading_path: value.heading_path,
    page: nullableNumber(value.page) ?? null,
    page_end: nullableNumber(value.page_end) ?? null,
    paragraph_start: nullableNumber(value.paragraph_start) ?? null,
    paragraph_end: nullableNumber(value.paragraph_end) ?? null,
    table_id: typeof value.table_id === "string" ? value.table_id : null,
    excerpt: value.excerpt,
  };
}

function normalizeFeedbackEvent(value: unknown): FeedbackEventView {
  if (!isRecord(value) || !isSafeId(value.event_id) || typeof value.helpful !== "boolean" || !isSafeTimestamp(value.event_at) || !isOneOf(value.source, ["user_click", "admin_correction", "import"]) || typeof value.event_schema_version !== "number" || !Number.isInteger(value.event_schema_version) || value.event_schema_version < 1) throw new Error("event is invalid");
  return { event_id: value.event_id, helpful: value.helpful, feedback_reason: optionalSafeCode(value.feedback_reason), event_at: value.event_at, source: value.source, event_schema_version: value.event_schema_version };
}

function normalizeFeedbackReview(value: unknown): FeedbackReviewView {
  if (!isRecord(value) || !isSafeId(value.review_id) || !isOneOf(value.reviewer_role, ["product", "engineering", "clinical_reviewer"]) || !isOneOf(value.decision, ["approve", "reject", "needs_adjudication"]) || !nullableBoolean(value.evidence_ok) || !nullableBoolean(value.points_ok) || !nullableBoolean(value.safety_ok) || !isSafeTimestamp(value.reviewed_at) || !isSafeId(value.review_version)) throw new Error("review is invalid");
  return { review_id: value.review_id, reviewer_role: value.reviewer_role, decision: value.decision, evidence_ok: value.evidence_ok, points_ok: value.points_ok, safety_ok: value.safety_ok, note_code: optionalSafeCode(value.note_code), review_version: value.review_version, reviewed_at: value.reviewed_at };
}

function normalizeFeedbackPromotion(value: unknown): FeedbackPromotion {
  if (!isRecord(value) || !isSafeId(value.promotion_id) || !isOneOf(value.target_split, ["dev", "holdout"]) || !isSafeTimestamp(value.promoted_at)) throw new Error("promotion is invalid");
  return { promotion_id: value.promotion_id, target_set_id: optionalSafeCode(value.target_set_id), target_version: optionalSafeCode(value.target_version), target_split: value.target_split, promotion_reason: optionalSafeCode(value.promotion_reason), manifest_id: optionalSafeCode(value.manifest_id), promoted_at: value.promoted_at };
}

function normalizePromotionCheck(value: unknown): PromotionCheck {
  if (!isRecord(value) || !isOneOf(value.status, ["not_promoted", "golden_v2_candidate"]) || !Array.isArray(value.reasons) || !value.reasons.every((item) => typeof item === "string" && _safeCode(item))) throw new Error("promotion check is invalid");
  return { status: value.status, reasons: value.reasons };
}

function isFeedbackReviewInput(value: FeedbackReviewInput): boolean {
  return isOneOf(value.reviewer_role, ["product", "engineering", "clinical_reviewer"]) && isOneOf(value.decision, ["approve", "reject", "needs_adjudication"]) && nullableBoolean(value.evidence_ok) && nullableBoolean(value.points_ok) && nullableBoolean(value.safety_ok) && (value.note_code === null || _safeCode(value.note_code)) && isSafeId(value.review_version);
}

function nullableBoolean(value: unknown): value is boolean | null {
  return value === null || typeof value === "boolean";
}

function optionalSafeCode(value: unknown): string | null {
  if (value === undefined || value === null) return null;
  return typeof value === "string" && _safeCode(value) ? value : null;
}

function _safeCode(value: string): boolean {
  return /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}$/.test(value);
}

function safeIdArray(value: unknown): string[] {
  if (!Array.isArray(value) || !value.every(isSafeId)) throw new Error("ids are invalid");
  return value;
}

function safeCodeArray(value: unknown): string[] {
  if (!Array.isArray(value) || !value.every((item) => typeof item === "string" && _safeCode(item))) throw new Error("codes are invalid");
  return value;
}

function safeNonNegativeInteger(value: unknown): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 0) throw new Error("count is invalid");
  return value;
}

async function requestJson(url: string, init: RequestInit, code: string): Promise<unknown> {
  let response: Response;
  try {
    response = await fetch(url, init);
  } catch {
    throw new ApiError(0, code);
  }
  if (!response.ok) throw new ApiError(response.status, code);
  try {
    return await response.json();
  } catch {
    throw new ApiError(response.status, code);
  }
}

function parseFrame(frame: string): ChatEvent | null {
  let eventName = "";
  const dataLines: string[] = [];
  for (const line of frame.split(/\r?\n/)) {
    if (line.startsWith("event:")) eventName = line.slice("event:".length).trim();
    if (line.startsWith("data:")) dataLines.push(line.slice("data:".length).trimStart());
  }
  if (!eventName && dataLines.length === 0 && !frame.trim()) return null;
  if (!EVENT_TYPES.has(eventName) || dataLines.length === 0) throw new ClientStreamError();

  let rawData: unknown;
  try {
    rawData = JSON.parse(dataLines.join("\n"));
  } catch {
    throw new ClientStreamError();
  }
  if (!isRecord(rawData)) throw new ClientStreamError();
  return normalizeEvent(eventName, rawData);
}

function normalizeHealthResponse(value: unknown): HealthResponse {
  if (!isRecord(value) || !isRuntimeMode(value.runtime_mode) || !isRecord(value.providers)) {
    throw new Error("health response is invalid");
  }
  const rawProviders = value.providers;
  const providers = Object.fromEntries(
    HEALTH_PROVIDERS.map((provider) => {
      const status = rawProviders[provider];
      if (!isProviderState(status)) throw new Error("provider status is invalid");
      return [provider, status];
    }),
  ) as HealthResponse["providers"];
  return { runtime_mode: value.runtime_mode, providers };
}

function isRuntimeMode(value: unknown): value is HealthResponse["runtime_mode"] {
  return typeof value === "string" && SAFE_RUNTIME_MODES.has(value);
}

function isProviderState(value: unknown): value is HealthResponse["providers"][keyof HealthResponse["providers"]] {
  return typeof value === "string" && SAFE_PROVIDER_STATES.has(value);
}

const WORKFLOW_FIELDS = new Set([
  "workflow_version",
  "run_id",
  "route",
  "outcome",
  "verifier_status",
  "http_calls",
  "elapsed_ms",
  "reason_code",
]);
const WORKFLOW_ROUTES = new Set(["direct", "verify", "clarify", "out_of_scope"]);
const WORKFLOW_OUTCOMES = new Set(["answered", "refused", "error", "cancelled"]);
const WORKFLOW_VERIFIERS = new Set(["not_run", "passed", "failed", "unavailable"]);

export function normalizeWorkflowSummary(value: unknown): WorkflowSummary | null {
  if (value === undefined || value === null) return null;
  if (!isRecord(value) || Object.keys(value).some((key) => !WORKFLOW_FIELDS.has(key))) {
    throw new ClientStreamError();
  }
  if (
    value.workflow_version !== "baseline_v1" &&
    value.workflow_version !== "agent_workflow_v2a"
  ) {
    throw new ClientStreamError();
  }
  const runId = value.run_id;
  if (!isSafeId(runId)) throw new ClientStreamError();
  const route = value.route;
  if (route !== null && (typeof route !== "string" || !WORKFLOW_ROUTES.has(route))) {
    throw new ClientStreamError();
  }
  const outcome = value.outcome;
  if (typeof outcome !== "string" || !WORKFLOW_OUTCOMES.has(outcome)) {
    throw new ClientStreamError();
  }
  const verifierStatus = value.verifier_status;
  if (typeof verifierStatus !== "string" || !WORKFLOW_VERIFIERS.has(verifierStatus)) {
    throw new ClientStreamError();
  }
  if (
    typeof value.http_calls !== "number" ||
    !Number.isInteger(value.http_calls) ||
    value.http_calls < 0 ||
    value.http_calls > 7
  ) {
    throw new ClientStreamError();
  }
  if (
    value.elapsed_ms !== null &&
    (typeof value.elapsed_ms !== "number" ||
      !Number.isInteger(value.elapsed_ms) ||
      value.elapsed_ms < 0 ||
      value.elapsed_ms > 86_400_000)
  ) {
    throw new ClientStreamError();
  }
  const reasonCode = value.reason_code;
  if (reasonCode !== null && (typeof reasonCode !== "string" || !SAFE_REASONS.has(reasonCode))) {
    throw new ClientStreamError();
  }
  return {
    workflow_version: value.workflow_version,
    run_id: runId,
    route,
    outcome: outcome as WorkflowSummary["outcome"],
    verifier_status: verifierStatus as WorkflowSummary["verifier_status"],
    http_calls: value.http_calls,
    elapsed_ms: value.elapsed_ms,
    reason_code: reasonCode,
  } as WorkflowSummary;
}

function normalizeEvent(eventName: string, data: Record<string, unknown>): ChatEvent {
  if (eventName === "status") {
    if (typeof data.stage !== "string" || !CHAT_STAGES.has(data.stage)) {
      throw new ClientStreamError();
    }
    if (data.stage === "accepted" && (!isSafeId(data.session_id) || !isSafeId(data.message_id))) {
      throw new ClientStreamError();
    }
    if (data.stage === "rewriting" && !isSafeId(data.session_id)) {
      throw new ClientStreamError();
    }
    const result: Extract<ChatEvent, { type: "status" }>["data"] = {
      stage: data.stage as Extract<ChatEvent, { type: "status" }>["data"]["stage"],
    };
    if (data.session_id !== undefined) result.session_id = requireId(data.session_id);
    if (data.message_id !== undefined) result.message_id = requireId(data.message_id);
    return { type: "status", data: result };
  }

  if (eventName === "answer_delta") {
    if (typeof data.text !== "string" || !isSafeText(data.text)) {
      throw new ClientStreamError();
    }
    const sourceIds = data.source_ids === undefined ? [] : data.source_ids;
    if (!Array.isArray(sourceIds) || !sourceIds.every((id) => isSafeReferenceId(id))) {
      throw new ClientStreamError();
    }
    return { type: "answer_delta", data: { text: data.text, source_ids: sourceIds } };
  }

  if (eventName === "final") {
    if (
      !isSafeId(data.session_id) ||
      !isSafeId(data.message_id) ||
      typeof data.refused !== "boolean" ||
      !Array.isArray(data.citations)
    ) {
      throw new ClientStreamError();
    }
    const citations = data.citations.map(normalizeCitation);
    const reasonCode = data.reason_code === undefined ? undefined : requireReason(data.reason_code);
    let referenceAnswer: string | null | undefined;
    if (data.reference_answer !== undefined && data.reference_answer !== null) {
      const referenceAllowed = data.refused
        ? Boolean(reasonCode && REFERENCE_REASONS.has(reasonCode))
        : reasonCode === undefined;
      if (
        !referenceAllowed ||
        typeof data.reference_answer !== "string" ||
        !data.reference_answer.trim() ||
        data.reference_answer.length > 4000 ||
        !isSafeText(data.reference_answer)
      ) {
        throw new ClientStreamError();
      }
      referenceAnswer = data.reference_answer;
    }
    const result: Extract<ChatEvent, { type: "final" }>["data"] = {
      session_id: data.session_id,
      message_id: data.message_id,
      refused: data.refused,
      citations,
    };
    if (reasonCode !== undefined) result.reason_code = reasonCode;
    if (referenceAnswer !== undefined) result.reference_answer = referenceAnswer;
    if (data.workflow_summary !== undefined) {
      result.workflow_summary = normalizeWorkflowSummary(data.workflow_summary);
    }
    return { type: "final", data: result };
  }

  const result: Extract<ChatEvent, { type: "error" }>["data"] = {
    reason_code: requireReason(data.reason_code),
  };
  if (data.session_id !== undefined) result.session_id = requireId(data.session_id);
  if (data.message_id !== undefined) result.message_id = requireId(data.message_id);
  if (data.workflow_summary !== undefined) {
    result.workflow_summary = normalizeWorkflowSummary(data.workflow_summary);
  }
  return { type: "error", data: result };
}

export function normalizeCitation(value: unknown): Citation {
  if (!isRecord(value)) throw new ClientStreamError();
  if (!isSafeReferenceId(value.reference_id) || !isSafeFileName(value.file_name)) {
    throw new ClientStreamError();
  }
  if (!Array.isArray(value.heading_path) || value.heading_path.length === 0) {
    throw new ClientStreamError();
  }
  const headingPath = value.heading_path.map((item) => {
    if (typeof item !== "string" || !isSafeHeading(item)) throw new ClientStreamError();
    return item;
  });
  if (
    typeof value.excerpt !== "string" ||
    !value.excerpt.trim() ||
    value.excerpt.length > 300 ||
    !isSafeText(value.excerpt)
  ) {
    throw new ClientStreamError();
  }
  const page = nullableNumber(value.page) ?? null;
  const pageEnd = nullableNumber(value.page_end) ?? null;
  const paragraphStart = nullableNumber(value.paragraph_start) ?? null;
  const paragraphEnd = nullableNumber(value.paragraph_end) ?? null;
  const tableId = value.table_id === undefined || value.table_id === null ? null : value.table_id;
  if (tableId !== null && (typeof tableId !== "string" || !/^table-[1-9]\d*$/.test(tableId))) {
    throw new ClientStreamError();
  }
  if (value.file_name.toLowerCase().endsWith(".pdf")) {
    if (!validRange(page, pageEnd) || paragraphStart !== null || paragraphEnd !== null) {
      throw new ClientStreamError();
    }
  } else if (
    (!validRange(paragraphStart, paragraphEnd) && tableId === null) ||
    page !== null ||
    pageEnd !== null
  ) {
    throw new ClientStreamError();
  }
  return {
    reference_id: value.reference_id,
    file_name: value.file_name,
    heading_path: headingPath,
    page,
    page_end: pageEnd,
    paragraph_start: paragraphStart,
    paragraph_end: paragraphEnd,
    table_id: tableId,
    excerpt: value.excerpt,
  };
}

function nullableNumber(value: unknown): number | null | undefined {
  if (value === undefined || value === null) return value;
  if (typeof value !== "number" || !Number.isFinite(value)) throw new ClientStreamError();
  return value;
}

function validRange(start: number | null | undefined, end: number | null | undefined): boolean {
  return typeof start === "number" && typeof end === "number" && start > 0 && end >= start;
}

function requireId(value: unknown): string {
  if (!isSafeId(value)) throw new ClientStreamError();
  return value;
}

function requireReason(value: unknown): string {
  if (typeof value !== "string" || !SAFE_REASONS.has(value)) throw new ClientStreamError();
  return value;
}

function isSafeId(value: unknown): value is string {
  return typeof value === "string" && SAFE_ID.test(value);
}

function isSafeReferenceId(value: unknown): value is string {
  return typeof value === "string" && SAFE_REFERENCE_ID.test(value);
}

function isSafeFileName(value: unknown): value is string {
  return typeof value === "string" && SAFE_FILE_NAME.test(value) && isSafeText(value);
}

function isSafeHeading(value: string): boolean {
  return isSafeText(value) && Boolean(value.trim()) && !value.includes("/") && !value.includes("\\");
}

function isSafeText(value: string): boolean {
  return (
    !/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/.test(value) &&
    !/(?:(?:[A-Za-z]:[\\/])|(?<!\w)\/)[^\s]+/.test(value) &&
    !/(?:traceback|stack\s+trace|providererror)/i.test(value) &&
    !(/\b(?:sk|pk)-[A-Za-z0-9_-]+\b|\bbearer\s+[^\s,;]+|\b(?:api[_ -]?key|access[_ -]?token)\s*[:=]/i).test(value)
  );
}

export function normalizeSessionResponse(value: unknown): SessionResponse {
  if (!isRecord(value) || !isSafeId(value.session_id) || !Array.isArray(value.messages)) {
    throw new Error("invalid session response");
  }
  return {
    session_id: value.session_id,
    messages: value.messages.map((item, index) =>
      normalizeSessionMessage(item, value.session_id as string, index),
    ),
  };
}

export function normalizeSessionListResponse(value: unknown): SessionSummary[] {
  if (!isRecord(value) || !Array.isArray(value.sessions)) {
    throw new Error("invalid session list response");
  }
  return value.sessions.map((item) => {
    if (!isRecord(item)) throw new Error("session summary is not an object");
    if (
      !isSafeId(item.session_id) ||
      typeof item.title !== "string" ||
      !item.title.trim() ||
      item.title.length > 40 ||
      !isSafeText(item.title) ||
      !isSafeTimestamp(item.created_at) ||
      !isSafeTimestamp(item.last_activity_at) ||
      typeof item.message_count !== "number" ||
      !Number.isInteger(item.message_count) ||
      item.message_count < 0
    ) {
      throw new Error("session summary is invalid");
    }
    return {
      session_id: item.session_id,
      title: item.title,
      created_at: item.created_at,
      last_activity_at: item.last_activity_at,
      message_count: item.message_count,
    };
  });
}

function normalizeSessionMessage(
  value: unknown,
  sessionId: string,
  index: number,
): SessionResponse["messages"][number] {
  try {
    if (!isRecord(value)) throw new Error("message is not an object");
    if (!isSafeId(value.message_id) || !isSafeId(sessionId)) throw new Error("message id is invalid");
    if (value.role !== "user" && value.role !== "assistant") throw new Error("role is invalid");
    if (typeof value.status !== "string" || !SAFE_MESSAGE_STATUSES.has(value.status)) {
      throw new Error("status is invalid");
    }
    if (value.role === "user" && value.status !== "submitted") throw new Error("user status is invalid");
    if (value.role === "assistant" && !["answered", "refused", "error"].includes(value.status)) {
      throw new Error("assistant status is invalid");
    }
    if (typeof value.content !== "string" || !value.content.trim() || !isSafeText(value.content)) {
      throw new Error("content is unsafe");
    }
    if (!isSafeTimestamp(value.created_at)) throw new Error("timestamp is invalid");
    if (!Array.isArray(value.citations)) throw new Error("citations are invalid");
    const citations = value.citations.map(normalizeCitation);
    const reasonCode = normalizeReason(value.reason_code);
    const audienceScope = normalizeAudienceScope(value.audience_scope);
    const workflowSummary = normalizeWorkflowSummary(value.workflow_summary);
    if (value.role === "user" && reasonCode !== null) throw new Error("user reason is invalid");
    if (value.role === "user" && citations.length > 0) throw new Error("user citations are invalid");
    if (value.role === "assistant" && ["refused", "error"].includes(value.status) && citations.length > 0) {
      throw new Error("non-answer citations are invalid");
    }
    let referenceAnswer: string | null = null;
    if (value.reference_answer !== undefined && value.reference_answer !== null) {
      const referenceAllowed = value.status === "answered"
        ? reasonCode === null
        : value.status === "refused" && Boolean(reasonCode && REFERENCE_REASONS.has(reasonCode));
      if (
        value.role !== "assistant" ||
        !referenceAllowed ||
        typeof value.reference_answer !== "string" ||
        !value.reference_answer.trim() ||
        value.reference_answer.length > 4000 ||
        !isSafeText(value.reference_answer)
      ) {
        throw new Error("reference answer is invalid");
      }
      referenceAnswer = value.reference_answer;
    }
    let rewrittenQuestion: string | null = null;
    if (value.rewritten_question !== undefined && value.rewritten_question !== null) {
      if (
        value.role !== "user" ||
        typeof value.rewritten_question !== "string" ||
        !value.rewritten_question.trim() ||
        value.rewritten_question.length > 1000 ||
        !isSafeText(value.rewritten_question)
      ) {
        throw new Error("rewritten question is invalid");
      }
      rewrittenQuestion = value.rewritten_question;
    }
    return {
      message_id: value.message_id,
      role: value.role,
      content: value.status === "error" ? "当前服务暂时不可用，请稍后重试。" : value.content,
      status: value.status,
      rewritten_question: rewrittenQuestion,
      citations,
      reason_code: reasonCode,
      reference_answer: referenceAnswer,
      audience_scope: audienceScope,
      workflow_summary: workflowSummary,
      created_at: value.created_at,
      feedback: normalizeFeedback(value.feedback),
    };
  } catch {
    return safeErrorMessage(index);
  }
}

function safeErrorMessage(index: number): SessionResponse["messages"][number] {
  return {
    message_id: `invalid-message-${index}`,
    role: "assistant",
    content: SAFE_ERROR_TEXT,
    status: "error",
    rewritten_question: null,
    citations: [],
    reason_code: "CHAT_UNAVAILABLE",
    audience_scope: "unspecified",
    workflow_summary: null,
    created_at: "1970-01-01 00:00:00",
    feedback: null,
  };
}

function normalizeReason(value: unknown): string | null {
  if (value === undefined || value === null) return null;
  return typeof value === "string" && SAFE_REASONS.has(value) ? value : (() => { throw new Error("reason is invalid"); })();
}

function normalizeAudienceScope(value: unknown): AudienceScope {
  const scope = value === undefined || value === null ? "unspecified" : value;
  if (typeof scope !== "string" || !SAFE_AUDIENCE_SCOPES.has(scope)) {
    throw new Error("audience scope is invalid");
  }
  return scope as AudienceScope;
}

function normalizeFeedback(value: unknown): Feedback | null {
  if (value === undefined || value === null) return null;
  if (
    !isRecord(value) ||
    !isSafeId(value.message_id) ||
    typeof value.helpful !== "boolean" ||
    !isSafeTimestamp(value.created_at)
  ) {
    return null;
  }
  const reason = value.reason === undefined || value.reason === null
    ? null
    : typeof value.reason === "string" && SAFE_FEEDBACK_REASONS.has(value.reason as FeedbackReason)
      ? value.reason as FeedbackReason
      : null;
  if (value.reason !== undefined && value.reason !== null && reason === null) return null;
  if (value.helpful && reason !== null) return null;
  return {
    message_id: value.message_id,
    helpful: value.helpful,
    created_at: value.created_at,
    reason,
  };
}

export function normalizeDocumentRow(value: unknown): DocumentRow | null {
  if (!isRecord(value)) return null;
  if (
    typeof value.document_id !== "number" ||
    !Number.isInteger(value.document_id) ||
    value.document_id < 1 ||
    typeof value.file_name !== "string" ||
    !isSafeFileName(value.file_name) ||
    typeof value.status !== "string" ||
    !SAFE_DOCUMENT_STATUSES.has(value.status)
  ) {
    return null;
  }
  if (value.version_id !== undefined && value.version_id !== null && !isSafeId(value.version_id)) {
    return null;
  }
  if (value.updated_at !== undefined && value.updated_at !== null && !isSafeTimestamp(value.updated_at)) {
    return null;
  }
  let failureReason: string | null = null;
  if (value.failure_reason !== undefined && value.failure_reason !== null) {
    failureReason = typeof value.failure_reason === "string" && SAFE_FAILURE_REASONS.has(value.failure_reason)
      ? value.failure_reason
      : "INDEX_FAILED";
  }
  return {
    document_id: value.document_id,
    file_name: value.file_name,
    version_id: value.version_id === undefined ? null : value.version_id as string | null,
    status: value.status,
    updated_at: value.updated_at === undefined ? null : value.updated_at as string | null,
    failure_reason: failureReason,
    business_metadata: value.business_metadata === undefined
      ? null
      : normalizeDocumentBusinessMetadata(value.business_metadata),
  };
}

export function normalizeDocumentBusinessMetadata(value: unknown): DocumentBusinessMetadata | null {
  if (!isRecord(value) || !isSafeId(value.version_id) || !isSafeTimestamp(value.updated_at)) return null;
  if (
    !isOneOf(value.content_type, ["policy", "training", "procedure", "other"]) ||
    !isOneOf(value.applicable_scope, ["unspecified", "all_staff", "nurse", "doctor", "pharmacist", "administrator"]) ||
    !isOneOf(value.business_status, ["draft", "approved", "superseded", "retired", "unknown"])
  ) return null;
  if (!isOptionalIsoDate(value.effective_from) || !isOptionalIsoDate(value.review_due_at)) return null;
  if (value.owner_role !== null && value.owner_role !== undefined && (typeof value.owner_role !== "string" || !value.owner_role.trim() || value.owner_role.length > 80)) return null;
  if (value.supersedes_version_id !== null && value.supersedes_version_id !== undefined && !isSafeId(value.supersedes_version_id)) return null;
  return {
    version_id: value.version_id,
    content_type: value.content_type,
    applicable_scope: value.applicable_scope,
    effective_from: value.effective_from === undefined ? null : value.effective_from,
    review_due_at: value.review_due_at === undefined ? null : value.review_due_at,
    business_status: value.business_status,
    owner_role: value.owner_role === undefined ? null : value.owner_role,
    supersedes_version_id: value.supersedes_version_id === undefined ? null : value.supersedes_version_id,
    updated_at: value.updated_at,
  };
}

function normalizeScanResult(value: unknown): ScanResult {
  if (!isRecord(value)) throw new Error("scan response is invalid");
  const rawResults = Array.isArray(value.results) ? value.results : [];
  const deactivated = Array.isArray(value.deactivated)
    ? value.deactivated.filter((item): item is string => isSafeFileName(item))
    : [];
  return {
    results: rawResults
      .map((item) => normalizeDocumentRow(item))
      .filter((item): item is DocumentRow => item !== null),
    deactivated,
    snapshot_complete: typeof value.snapshot_complete === "boolean" ? value.snapshot_complete : false,
  };
}

function isSafeTimestamp(value: unknown): value is string {
  return (
    typeof value === "string" &&
    SAFE_TIMESTAMP.test(value) &&
    !Number.isNaN(Date.parse(value.replace(" ", "T")))
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isOneOf<T extends string>(value: unknown, values: readonly T[]): value is T {
  return typeof value === "string" && values.includes(value as T);
}

function isOptionalIsoDate(value: unknown): value is string | null | undefined {
  return value === null || value === undefined || (typeof value === "string" && /^\d{4}-\d{2}-\d{2}$/.test(value));
}
