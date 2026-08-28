"""Storage-facing domain contracts for policy indexing and retrieval."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from app.domain.models import Chunk

if TYPE_CHECKING:
    from app.chat.models import (
        AssistantMessageStatus,
        ChatMessage,
        ChatSessionSummary,
        Feedback,
    )

VersionStatus = Literal[
    "indexing",
    "active",
    "failed",
    "superseded",
    "inactive",
    "needs_ocr",
]
IndexAction = Literal["index", "skip", "needs_ocr", "in_progress"]


@dataclass(frozen=True)
class DocumentVersion:
    """The lifecycle state of a single imported-file version."""

    version_id: str
    file_name: str
    sha256: str
    status: VersionStatus
    action: IndexAction
    failure_reason: str | None = None


@dataclass(frozen=True)
class DocumentSummary:
    """Safe document metadata exposed to management APIs."""

    document_id: int
    file_name: str
    version_id: str | None
    status: VersionStatus | None
    sha256: str | None
    failure_reason: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class VectorPoint:
    """A chunk and its embedding, ready to be persisted by a vector store."""

    version_id: str
    chunk: Chunk
    vector: Sequence[float]


@dataclass(frozen=True)
class SearchHit:
    """A locatable retrieved chunk with its vector-similarity score."""

    version_id: str
    chunk: Chunk
    score: float


class DocumentRepository(Protocol):
    """Durable document-version lifecycle boundary."""

    def begin_index(
        self,
        file_name: str,
        sha256: str,
        *,
        needs_ocr: bool = False,
        force: bool = False,
    ) -> DocumentVersion: ...

    def activate(self, version_id: str) -> None: ...

    def begin_ocr_indexing(self, version_id: str) -> DocumentVersion: ...

    def mark_needs_ocr(self, version_id: str) -> DocumentVersion: ...

    def fail(self, version_id: str, reason: str) -> None: ...

    def get_active(self, file_name: str) -> DocumentVersion | None: ...

    def active_version_ids(self) -> set[str]: ...

    def deactivate_missing(self, present_file_names: set[str]) -> list[str]: ...

    def list_documents(self) -> list[DocumentSummary]: ...

    def get_document(self, document_id: int) -> DocumentSummary | None: ...

    def create_chat_session(self) -> str: ...

    def chat_session_exists(self, session_id: str) -> bool: ...

    def list_chat_sessions(self, limit: int = 20) -> list[ChatSessionSummary]: ...

    def append_user_message(self, session_id: str, content: str) -> ChatMessage: ...

    def set_rewritten_question(self, message_id: str, rewritten_question: str) -> None: ...

    def append_assistant_message(
        self,
        session_id: str,
        reply_to_message_id: str,
        content: str,
        status: AssistantMessageStatus,
        citations: Sequence[Mapping[str, object]],
        reason_code: str | None,
        reference_answer: str | None = None,
    ) -> ChatMessage: ...

    def list_chat_messages(self, session_id: str) -> list[ChatMessage]: ...

    def upsert_feedback(self, message_id: str, helpful: bool) -> Feedback: ...

    def get_feedback(self, message_id: str) -> Feedback | None: ...

    def schema_version(self) -> int: ...


class VectorStore(Protocol):
    """Version-filtered semantic retrieval boundary."""

    def upsert(
        self,
        version_id: str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> None: ...

    def search(
        self,
        query_vector: Sequence[float],
        active_versions: set[str],
        limit: int,
    ) -> list[SearchHit]: ...
