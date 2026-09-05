"""Storage-facing domain contracts for policy indexing and retrieval."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
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
BusinessStatus = Literal["draft", "approved", "superseded", "retired", "unknown"]
AudienceScope = Literal[
    "unspecified", "all_staff", "nurse", "doctor", "pharmacist", "administrator"
]
ContentType = Literal["policy", "training", "procedure", "other"]
FeedbackReason = Literal[
    "not_answered",
    "missing_step",
    "version_mismatch",
    "citation_mismatch",
    "too_slow",
]
BadCaseStatus = Literal[
    "new", "triaged", "fixed", "regression_checked", "closed"
]


@dataclass(frozen=True)
class BadCaseRecord:
    """A safe local bad-case record without question or answer text."""

    case_id: str
    code: str
    status: BadCaseStatus
    document_version_id: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class DocumentBusinessMetadata:
    """Business governance fields for one technical document version."""

    version_id: str
    content_type: ContentType
    applicable_scope: AudienceScope
    effective_from: str | None
    review_due_at: str | None
    business_status: BusinessStatus
    owner_role: str | None
    supersedes_version_id: str | None
    updated_at: str


@dataclass(frozen=True)
class DocumentBusinessMetadataInput:
    """Controlled admin input; version and timestamp remain repository-owned."""

    content_type: ContentType
    applicable_scope: AudienceScope
    effective_from: str | None
    review_due_at: str | None
    business_status: BusinessStatus
    owner_role: str | None
    supersedes_version_id: str | None


@dataclass(frozen=True)
class EligibilityResult:
    """The formal-answer allowlist and its safe explanation, when empty."""

    version_ids: frozenset[str]
    exclusion_reason: str | None


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
    business_metadata: DocumentBusinessMetadata | None = None


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

    def get_document_business_metadata(
        self, document_id: int
    ) -> DocumentBusinessMetadata | None: ...

    def upsert_document_business_metadata(
        self, document_id: int, metadata: DocumentBusinessMetadataInput
    ) -> DocumentBusinessMetadata: ...

    def eligible_version_ids(
        self, audience_scope: AudienceScope, as_of: date
    ) -> EligibilityResult: ...

    def create_chat_session(self) -> str: ...

    def chat_session_exists(self, session_id: str) -> bool: ...

    def list_chat_sessions(self, limit: int = 20) -> list[ChatSessionSummary]: ...

    def append_user_message(
        self,
        session_id: str,
        content: str,
        audience_scope: AudienceScope = "unspecified",
    ) -> ChatMessage: ...

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
        workflow_summary: Mapping[str, object] | None = None,
    ) -> ChatMessage: ...

    def list_chat_messages(self, session_id: str) -> list[ChatMessage]: ...

    def upsert_feedback(
        self,
        message_id: str,
        helpful: bool,
        reason: FeedbackReason | None = None,
    ) -> Feedback: ...

    def create_bad_case(self, message_id: str, code: str) -> BadCaseRecord: ...

    def transition_bad_case(
        self, case_id: str, target_status: BadCaseStatus
    ) -> BadCaseRecord: ...

    def list_bad_case_aggregates(self) -> dict[str, dict[str, int]]: ...

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
