"""Value objects used by the stateful chat backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.domain.ports import AudienceScope, FeedbackReason

AssistantMessageStatus = Literal["answered", "refused", "error"]


@dataclass(frozen=True)
class ChatMessage:
    """A safe, persisted user or assistant message."""

    message_id: str
    session_id: str
    role: Literal["user", "assistant"]
    content: str
    status: str
    rewritten_question: str | None
    citations: tuple[dict[str, object], ...]
    reason_code: str | None
    created_at: str
    reply_to_message_id: str | None = None
    reference_answer: str | None = None
    audience_scope: AudienceScope = "unspecified"
    # Optional, safe execution facts from the candidate workflow.  Keeping
    # this at the end preserves positional construction of legacy messages.
    workflow_summary: dict[str, object] | None = None


@dataclass(frozen=True)
class ChatSessionSummary:
    """Safe metadata used to browse stored chat sessions."""

    session_id: str
    created_at: str
    last_activity_at: str
    message_count: int
    title: str = "新会话"


# The design brief calls the repository-facing rows ``StoredMessage`` in its
# history examples.  Keep the explicit alias so callers do not need to care
# whether a row came from SQLite or a test repository.
StoredMessage = ChatMessage


@dataclass(frozen=True)
class ChatTurn:
    """One complete user question and terminal assistant response."""

    user_message_id: str
    user_question: str
    rewritten_question: str | None
    assistant_message_id: str
    answer: str
    status: AssistantMessageStatus
    citations: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class HistoryContext:
    """Bounded context, ordered newest turn first."""

    turns: tuple[ChatTurn, ...]


@dataclass(frozen=True)
class Feedback:
    """The current feedback value for one final assistant message."""

    message_id: str
    helpful: bool
    created_at: str
    reason: FeedbackReason | None = None
