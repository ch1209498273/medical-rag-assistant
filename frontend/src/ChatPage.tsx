import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";

import { ApiError, getSession, listSessions, streamChat } from "./api/client";
import type {
  AudienceScope,
  ChatEvent,
  ChatStage,
  SessionMessage,
  SessionSummary,
} from "./api/types";
import CitationList from "./CitationList";
import FeedbackButtons from "./FeedbackButtons";
import WorkflowProgress from "./WorkflowProgress";

export const CURRENT_SESSION_KEY = "hemodialysis.currentSessionId";

type UiMessage = SessionMessage;

const STAGE_LABELS: Record<ChatStage, string> = {
  accepted: "处理中",
  rewriting: "正在理解追问",
  preflight: "正在检查资料资格",
  routing: "正在选择处理路径",
  retrieving: "正在检索内部资料",
  reranking: "正在重排序依据",
  generating: "正在生成有依据的回答",
  validating: "正在核验引用",
  verifying: "正在核验回答",
};

const EMPTY_MESSAGE = "还没有问题，先从一条制度问题开始。";

const AUDIENCE_SCOPE_OPTIONS: Array<{ value: AudienceScope; label: string }> = [
  { value: "unspecified", label: "未指定" },
  { value: "all_staff", label: "全体医护" },
  { value: "nurse", label: "护士" },
  { value: "doctor", label: "医生" },
  { value: "pharmacist", label: "药师" },
  { value: "administrator", label: "管理员" },
];

export default function ChatPage() {
  const [sessionId, setSessionId] = useState<string | null>(() => readSessionId());
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [audienceScope, setAudienceScope] = useState<AudienceScope>("unspecified");
  const [streaming, setStreaming] = useState(false);
  const [restoring, setRestoring] = useState(() => Boolean(readSessionId()));
  const [activeStage, setActiveStage] = useState<ChatStage | null>(null);
  const [pageError, setPageError] = useState<string | null>(null);
  const [sessionSummaries, setSessionSummaries] = useState<SessionSummary[]>([]);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const restoreCancelledRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    void listSessions()
      .then((summaries) => {
        if (cancelled) return;
        setSessionSummaries(summaries);
        setHistoryError(null);
      })
      .catch(() => {
        if (cancelled) return;
        setHistoryError("历史会话暂时无法加载。");
      })
      .finally(() => {
        if (!cancelled) setHistoryLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    restoreCancelledRef.current = false;
    const storedSessionId = readSessionId();
    if (!storedSessionId) {
      setRestoring(false);
      return;
    }
    let cancelled = false;
    setRestoring(true);
    void getSession(storedSessionId)
      .then((response) => {
        if (cancelled || restoreCancelledRef.current) return;
        setSessionId(response.session_id);
        setMessages(response.messages.map(normalizeMessage));
        setPageError(null);
      })
      .catch((error: unknown) => {
        if (cancelled || restoreCancelledRef.current) return;
        if (error instanceof ApiError && error.status === 404) {
          window.localStorage.removeItem(CURRENT_SESSION_KEY);
          setSessionId(null);
          setMessages([]);
          setPageError(null);
        } else {
          setPageError("当前会话暂时无法恢复，请新建会话后重试。");
        }
      })
      .finally(() => {
        if (!cancelled && !restoreCancelledRef.current) setRestoring(false);
      });
    return () => {
      cancelled = true;
      restoreCancelledRef.current = true;
    };
  }, []);

  const startNewSession = () => {
    restoreCancelledRef.current = true;
    window.localStorage.removeItem(CURRENT_SESSION_KEY);
    setSessionId(null);
    setMessages([]);
    setDraft("");
    setActiveStage(null);
    setPageError(null);
  };

  const restoreSelectedSession = async (selectedSessionId: string) => {
    if (streaming || restoring || selectedSessionId === sessionId) return;
    restoreCancelledRef.current = false;
    setRestoring(true);
    setPageError(null);
    try {
      const response = await getSession(selectedSessionId);
      if (restoreCancelledRef.current) return;
      setSessionId(response.session_id);
      window.localStorage.setItem(CURRENT_SESSION_KEY, response.session_id);
      setMessages(response.messages.map(normalizeMessage));
    } catch {
      if (!restoreCancelledRef.current) {
        setPageError("所选会话暂时无法恢复，请稍后重试。");
      }
    } finally {
      if (!restoreCancelledRef.current) setRestoring(false);
    }
  };

  const sendQuestion = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const question = draft.trim();
    if (!question || streaming || restoring) return;

    const sessionAtStart = sessionId;
    const pendingUserId = `pending-user-${Date.now()}`;
    const pendingAssistantId = `pending-assistant-${Date.now()}`;
    let answerText = "";
    setDraft("");
    setPageError(null);
    setActiveStage("accepted");
    setStreaming(true);
    setMessages((current) => [
      ...current,
      makeMessage({
        message_id: pendingUserId,
        role: "user",
        content: question,
        status: "submitted",
        audience_scope: audienceScope,
      }),
    ]);

    const handleEvent = (eventValue: ChatEvent) => {
      if (eventValue.type === "status") {
        setActiveStage(eventValue.data.stage);
        if (eventValue.data.stage === "accepted" && eventValue.data.session_id) {
          const acceptedSessionId = eventValue.data.session_id;
          window.localStorage.setItem(CURRENT_SESSION_KEY, acceptedSessionId);
          setSessionId(acceptedSessionId);
          setMessages((current) =>
            current.map((message) =>
              message.message_id === pendingUserId
                ? { ...message, message_id: eventValue.data.message_id ?? pendingUserId }
                : message,
            ),
          );
          setSessionSummaries((current) => upsertSessionSummary(current, acceptedSessionId, question));
        }
        return;
      }

      if (eventValue.type === "answer_delta") {
        answerText += eventValue.data.text;
        setMessages((current) =>
          upsertMessage(current, makeMessage({
            message_id: pendingAssistantId,
            role: "assistant",
            content: answerText,
            status: "processing",
          })),
        );
        return;
      }

      if (eventValue.type === "final") {
        const content = eventValue.data.refused
          ? refusalText(eventValue.data.reason_code)
          : answerText;
        const finalMessage = makeMessage({
          message_id: eventValue.data.message_id,
          role: "assistant",
          content,
          status: eventValue.data.refused ? "refused" : "answered",
          citations: eventValue.data.citations,
          reason_code: eventValue.data.reason_code ?? null,
          reference_answer: eventValue.data.reference_answer ?? null,
          workflow_summary: eventValue.data.workflow_summary ?? null,
        });
        setMessages((current) => updatePendingAssistant(current, pendingAssistantId, finalMessage));
        setActiveStage(null);
        return;
      }

      const content = errorText(eventValue.data.reason_code);
      const errorMessage = makeMessage({
        message_id: eventValue.data.message_id ?? pendingAssistantId,
        role: "assistant",
        content,
        status: "error",
        reason_code: eventValue.data.reason_code,
        workflow_summary: eventValue.data.workflow_summary ?? null,
      });
      setMessages((current) => updatePendingAssistant(current, pendingAssistantId, errorMessage));
      setActiveStage(null);
    };

    try {
      await streamChat(question, sessionAtStart, handleEvent, audienceScope);
    } catch {
      setPageError("暂时无法提交问题，请稍后重试。");
      setMessages((current) =>
        updatePendingAssistant(
          current,
          pendingAssistantId,
          makeMessage({
            message_id: pendingAssistantId,
            role: "assistant",
            content: "暂时无法完成回答，请稍后重试。",
            status: "error",
            reason_code: "CHAT_UNAVAILABLE",
          }),
        ),
      );
      setActiveStage(null);
    } finally {
      setStreaming(false);
    }
  };

  const latestTerminalStatus = latestAssistantTerminalStatus(messages);

  return (
    <section className="page chat-page" aria-labelledby="chat-page-title">
      <div className="chat-shell">
        <aside className="session-sidebar" aria-labelledby="session-history-title">
          <div className="session-history-heading">
            <div>
              <p className="eyebrow">可随时切换</p>
              <h3 id="session-history-title">历史会话</h3>
            </div>
            <span className="session-history-note">按首个问题命名</span>
          </div>
          {historyLoading && <p className="history-status">正在加载历史会话…</p>}
          {!historyLoading && historyError && <p className="history-status history-error">{historyError}</p>}
          {!historyLoading && !historyError && sessionSummaries.length === 0 && (
            <p className="history-status">还没有已保存的会话。</p>
          )}
          {!historyLoading && !historyError && sessionSummaries.length > 0 && (
            <div className="session-history-list">
              {sessionSummaries.map((summary) => (
                <button
                  key={summary.session_id}
                  type="button"
                  className={summary.session_id === sessionId ? "session-item active" : "session-item"}
                  aria-label={`恢复会话：${summary.title}`}
                  aria-pressed={summary.session_id === sessionId}
                  disabled={streaming || restoring}
                  onClick={() => void restoreSelectedSession(summary.session_id)}
                >
                  <span className="session-item-title">{summary.title}</span>
                  <span className="session-item-meta">
                    {formatSessionDate(summary.last_activity_at)} · {summary.message_count} 条消息
                  </span>
                </button>
              ))}
            </div>
          )}
        </aside>

        <div className="chat-main">
          <header className="page-header">
            <div>
              <p className="eyebrow">制度问答</p>
              <h2 id="chat-page-title">当前会话</h2>
              <p className="page-description">围绕已扫描的内部制度提问，答案会标注可核验的资料依据。</p>
            </div>
            <button type="button" className="secondary-button" onClick={startNewSession} disabled={streaming || restoring}>
              新建会话
            </button>
          </header>

          {restoring && (
            <div className="status-line" aria-live="polite">
              正在恢复当前会话
            </div>
          )}
          {pageError && (
            <div className="page-error" role="alert">
              {pageError}
            </div>
          )}

          <div className="conversation" aria-live="polite">
            {!restoring && messages.length === 0 && <p className="empty-state">{EMPTY_MESSAGE}</p>}
            {messages.map((message) => (
              <MessageView key={message.message_id} message={message} />
            ))}
          </div>

          <div className="chat-composer-wrap">
            <div className="status-line" aria-live="polite">
              {activeStage ? (
                <>
                  状态：<span>{STAGE_LABELS[activeStage]}</span>
                </>
              ) : latestTerminalStatus ? (
                `状态：${messageStatusLabel(latestTerminalStatus)}`
              ) : ""}
            </div>
            <form className="chat-composer" onSubmit={(event) => void sendQuestion(event)}>
              <label htmlFor="question-input">问题</label>
              <div className="chat-scope-control">
                <label htmlFor="audience-scope">希望适用的人员范围</label>
                <select
                  id="audience-scope"
                  value={audienceScope}
                  onChange={(event) => setAudienceScope(event.target.value as AudienceScope)}
                  disabled={streaming || restoring}
                >
                  {AUDIENCE_SCOPE_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
                <small className="chat-scope-help">仅影响资料检索，不等于登录权限</small>
              </div>
              <textarea
                id="question-input"
                name="question"
                rows={3}
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                placeholder="例如：透析中低血压应如何处理？"
                disabled={streaming || restoring}
              />
              <button type="submit" className="primary-button" disabled={streaming || restoring || !draft.trim()}>
                {streaming ? "处理中…" : "发送问题"}
              </button>
            </form>
          </div>
        </div>
      </div>
    </section>
  );
}

function MessageView({ message }: { message: UiMessage }) {
  const isAssistant = message.role === "assistant";
  return (
    <article className={`message ${isAssistant ? "message-assistant" : "message-user"}`}>
      <div className="message-label">{isAssistant ? "助手" : "提问"}</div>
      {isAssistant && <div className="message-status" aria-live="polite">状态：{messageStatusLabel(message.status)}</div>}
      {isAssistant && message.status === "answered" && (
        <div className="grounded-answer-title">资料依据答案（可核验）</div>
      )}
      <div className={`message-content status-${message.status}`}>
        {message.content || (message.status === "processing" ? "正在整理有依据的回答…" : "")}
      </div>
      {isAssistant && message.status === "error" && (
        <div className="message-error" role="alert">
          {message.content}
        </div>
      )}
      {isAssistant && message.status !== "processing" && (
        <WorkflowProgress summary={message.workflow_summary ?? null} />
      )}
      {isAssistant && message.status !== "error" && message.status !== "processing" && (
        <>
          {message.reference_answer && <ReferenceAnswerCard answer={message.reference_answer} />}
          <CitationList citations={message.citations} />
          <FeedbackButtons
            messageId={message.message_id}
            initialHelpful={message.feedback?.helpful ?? message.helpful ?? null}
            initialReason={message.feedback?.reason ?? null}
          />
        </>
      )}
    </article>
  );
}

function ReferenceAnswerCard({ answer }: { answer: string }) {
  return (
    <aside className="reference-answer" aria-label="模型通用参考（未核验）">
      <div className="reference-answer-title">模型通用参考（未核验）</div>
      <p className="reference-answer-text">{answer}</p>
      <p className="reference-answer-disclaimer">
        这段内容由大模型根据通用知识生成，未经过本单位制度或医保资料核验，不代表正式结论。医疗决策请以本单位制度和专业人员意见为准。
      </p>
    </aside>
  );
}

function messageStatusLabel(status: string): string {
  switch (status) {
    case "answered":
      return "完成";
    case "refused":
      return "依据不足";
    case "error":
      return "失败";
    case "processing":
      return "处理中";
    default:
      return "处理中";
  }
}

function latestAssistantTerminalStatus(messages: UiMessage[]): string | null {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role === "assistant" && ["answered", "refused", "error"].includes(message.status)) {
      return message.status;
    }
  }
  return null;
}

function makeMessage(input: Partial<UiMessage> & Pick<UiMessage, "message_id" | "role" | "content" | "status">): UiMessage {
  return {
    message_id: input.message_id,
    role: input.role,
    content: input.content,
    status: input.status,
    rewritten_question: input.rewritten_question ?? null,
    citations: input.citations ?? [],
    reason_code: input.reason_code ?? null,
    reference_answer: input.reference_answer ?? null,
    audience_scope: input.audience_scope ?? "unspecified",
    workflow_summary: input.workflow_summary ?? null,
    created_at: input.created_at ?? new Date().toISOString(),
    feedback: input.feedback ?? null,
  };
}

function normalizeMessage(message: SessionMessage): SessionMessage {
  return makeMessage(message);
}

function upsertSessionSummary(
  summaries: SessionSummary[],
  sessionId: string,
  firstQuestion?: string,
): SessionSummary[] {
  const now = new Date().toISOString();
  const fallbackTitle = firstQuestion ? deriveSessionTitle(firstQuestion) : "新会话";
  const existing = summaries.find((summary) => summary.session_id === sessionId);
  if (existing) {
    return [
      {
        ...existing,
        title: existing.title === "新会话" ? fallbackTitle : existing.title,
        last_activity_at: now,
        message_count: existing.message_count + 1,
      },
      ...summaries.filter((summary) => summary.session_id !== sessionId),
    ];
  }
  return [
    {
      session_id: sessionId,
      title: fallbackTitle,
      created_at: now,
      last_activity_at: now,
      message_count: 1,
    },
    ...summaries,
  ].slice(0, 20);
}

function deriveSessionTitle(question: string): string {
  const compact = question.replace(/\s+/g, " ").trim();
  if (compact.length <= 40) return compact || "新会话";
  return `${compact.slice(0, 39)}…`;
}

function formatSessionDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "时间未知";
  return new Intl.DateTimeFormat("zh-CN", {
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function upsertMessage(messages: UiMessage[], next: UiMessage): UiMessage[] {
  const index = messages.findIndex((message) => message.message_id === next.message_id);
  if (index < 0) return [...messages, next];
  return messages.map((message, messageIndex) => (messageIndex === index ? { ...message, ...next } : message));
}

function updatePendingAssistant(messages: UiMessage[], pendingId: string, finalMessage: UiMessage): UiMessage[] {
  const hasPending = messages.some((message) => message.message_id === pendingId);
  if (!hasPending) return upsertMessage(messages, finalMessage);
  return messages.map((message) =>
    message.message_id === pendingId ? finalMessage : message,
  );
}

function readSessionId(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(CURRENT_SESSION_KEY);
}

function refusalText(reasonCode?: string): string {
  if (reasonCode === "EVIDENCE_SCOPE_UNCLEAR") {
    return "希望适用的人员范围尚未确认，暂时无法提供经过核验的回答。请选择合适的人员范围后重试。";
  }
  if (reasonCode === "DOCUMENT_BUSINESS_STATUS_UNKNOWN") {
    return "资料尚未完成业务确认，暂时无法作为正式依据。请联系资料管理员确认后重试。";
  }
  if (reasonCode === "INSUFFICIENT_EVIDENCE" || reasonCode === "ANSWER_NOT_VERIFIABLE") {
    return "现有资料不足，暂时无法提供有依据的回答。";
  }
  return "这条问题暂时无法形成经过核验的回答。";
}

function errorText(reasonCode: string): string {
  switch (reasonCode) {
    case "FOLLOW_UP_UNAVAILABLE":
      return "未能理解这次追问，请补充完整问题后重试。";
    case "RETRIEVAL_UNAVAILABLE":
      return "内部资料检索暂时不可用，请稍后重试。";
    case "GENERATION_UNAVAILABLE":
    case "CHAT_UNAVAILABLE":
    case "INVALID_EVENT":
    case "ANSWER_NOT_VERIFIABLE":
    default:
      return "暂时无法完成回答，请稍后重试。";
  }
}
