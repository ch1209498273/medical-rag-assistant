import type {
  ChatEvent,
  Citation,
  DocumentRow,
  Feedback,
  HealthResponse,
  ScanResult,
  SessionSummary,
  SessionResponse,
} from "./types";

const API_PREFIX = "/api";
const CHAT_STAGES = new Set([
  "accepted",
  "rewriting",
  "retrieving",
  "reranking",
  "generating",
  "validating",
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
): Promise<void> {
  const body: { question: string; session_id?: string } = { question };
  if (sessionId) body.session_id = sessionId;

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

export async function saveFeedback(messageId: string, helpful: boolean): Promise<Feedback> {
  const payload = await requestJson(
    `${API_PREFIX}/chat/messages/${encodeURIComponent(messageId)}/feedback`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ helpful }),
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

function documentsPath(...segments: string[]): string {
  return [API_PREFIX, "documents", ...segments].join("/");
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
    return { type: "final", data: result };
  }

  const result: Extract<ChatEvent, { type: "error" }>["data"] = {
    reason_code: requireReason(data.reason_code),
  };
  if (data.session_id !== undefined) result.session_id = requireId(data.session_id);
  if (data.message_id !== undefined) result.message_id = requireId(data.message_id);
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
  if (value.file_name.toLowerCase().endsWith(".pdf")) {
    if (!validRange(page, pageEnd) || paragraphStart !== null || paragraphEnd !== null) {
      throw new ClientStreamError();
    }
  } else if (
    !validRange(paragraphStart, paragraphEnd) ||
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
    created_at: "1970-01-01 00:00:00",
    feedback: null,
  };
}

function normalizeReason(value: unknown): string | null {
  if (value === undefined || value === null) return null;
  return typeof value === "string" && SAFE_REASONS.has(value) ? value : (() => { throw new Error("reason is invalid"); })();
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
  return {
    message_id: value.message_id,
    helpful: value.helpful,
    created_at: value.created_at,
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
