"""Immutable, JSON-safe contracts for the local feedback projection."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

RedactionStatus = Literal["passed", "review_required", "blocked"]
AnswerStatus = Literal["answered", "refused", "error"]
AudienceScope = Literal[
    "unspecified", "all_staff", "nurse", "doctor", "pharmacist", "administrator"
]
TriagePriority = Literal["P0", "P1", "P2", "P3"]
ReviewStatus = Literal[
    "unreviewed", "in_review", "approved", "rejected", "adjudication_required"
]
PromotionStatus = Literal[
    "not_promoted", "silver", "golden_v2_candidate", "golden_v2"
]
ReviewerRole = Literal["product", "engineering", "clinical_reviewer"]
ReviewDecisionKind = Literal["approve", "reject", "needs_adjudication"]
PromotionSplit = Literal["dev", "holdout"]
FeedbackSource = Literal["user_click", "admin_correction", "import"]

REDACTION_STATUSES = frozenset({"passed", "review_required", "blocked"})
ANSWER_STATUSES = frozenset({"answered", "refused", "error"})
AUDIENCE_SCOPES = frozenset(
    {"unspecified", "all_staff", "nurse", "doctor", "pharmacist", "administrator"}
)
TRIAGE_PRIORITIES = frozenset({"P0", "P1", "P2", "P3"})
REVIEW_STATUSES = frozenset(
    {"unreviewed", "in_review", "approved", "rejected", "adjudication_required"}
)
PROMOTION_STATUSES = frozenset(
    {"not_promoted", "silver", "golden_v2_candidate", "golden_v2"}
)
REVIEWER_ROLES = frozenset({"product", "engineering", "clinical_reviewer"})
REVIEW_DECISIONS = frozenset({"approve", "reject", "needs_adjudication"})
PROMOTION_SPLITS = frozenset({"dev", "holdout"})
FEEDBACK_SOURCES = frozenset({"user_click", "admin_correction", "import"})


def _require_text(value: object, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"{field} must not be empty")
    return value


def _require_iso_like(value: object, field: str) -> str:
    text = _require_text(value, field)
    if "T" not in text:
        raise ValueError(f"{field} must be an ISO timestamp")
    return text


def _require_tuple_strings(value: object, field: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (tuple, list)):
        raise TypeError(f"{field} must be a sequence of strings")
    values = tuple(_require_text(item, field) for item in value)
    return values


def _json_value(value: object) -> object:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class CaseIdentity:
    """Non-reversible identity material used to build a feedback case."""

    case_id: str
    message_id_hash: str
    session_id_hash: str


def _hash_identifier(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_case_identity(message_id: str, session_id: str) -> CaseIdentity:
    """Return deterministic, one-way identifiers without retaining source IDs."""

    message = _require_text(message_id, "message_id")
    session = _require_text(session_id, "session_id")
    message_hash = _hash_identifier(message)
    session_hash = _hash_identifier(session)
    return CaseIdentity(
        case_id=f"fc_{message_hash[:32]}",
        message_id_hash=message_hash,
        session_id_hash=session_hash,
    )


@dataclass(frozen=True)
class FeedbackCase:
    """A redacted final assistant turn and its safe review metadata."""

    case_id: str
    message_id_hash: str
    session_id_hash: str
    collected_at: str
    question_redacted: str
    answer_redacted: str
    redaction_status: RedactionStatus
    redaction_version: str
    redaction_flags: tuple[str, ...]
    answer_status: AnswerStatus
    reason_code: str | None
    source_version: str | None
    retrieval_profile: str | None
    citation_chunk_ids: tuple[str, ...]
    citation_count: int
    model_id: str | None
    prompt_version: str | None
    latency_bucket: str | None
    audience_scope: AudienceScope
    retention_expires_at: str
    triage_priority: TriagePriority
    triage_score: int
    triage_reasons: tuple[str, ...]
    review_status: ReviewStatus
    promotion_status: PromotionStatus
    event_count: int = 0

    def __post_init__(self) -> None:
        _require_text(self.case_id, "case_id")
        for field in ("message_id_hash", "session_id_hash"):
            value = _require_text(getattr(self, field), field)
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
                raise ValueError(f"{field} must be a SHA-256 hex digest")
        _require_iso_like(self.collected_at, "collected_at")
        _require_text(self.question_redacted, "question_redacted")
        _require_text(self.answer_redacted, "answer_redacted")
        if self.redaction_status not in REDACTION_STATUSES:
            raise ValueError("redaction_status is invalid")
        _require_text(self.redaction_version, "redaction_version")
        object.__setattr__(
            self, "redaction_flags", _require_tuple_strings(self.redaction_flags, "redaction_flags")
        )
        if self.answer_status not in ANSWER_STATUSES:
            raise ValueError("answer_status is invalid")
        for field in (
            "reason_code",
            "source_version",
            "retrieval_profile",
            "model_id",
            "prompt_version",
            "latency_bucket",
        ):
            value = getattr(self, field)
            if value is not None:
                _require_text(value, field)
        object.__setattr__(
            self,
            "citation_chunk_ids",
            _require_tuple_strings(self.citation_chunk_ids, "citation_chunk_ids"),
        )
        if not isinstance(self.citation_count, int) or isinstance(self.citation_count, bool):
            raise TypeError("citation_count must be an integer")
        if self.citation_count < 0:
            raise ValueError("citation_count must be non-negative")
        if self.audience_scope not in AUDIENCE_SCOPES:
            raise ValueError("audience_scope is invalid")
        _require_iso_like(self.retention_expires_at, "retention_expires_at")
        if self.triage_priority not in TRIAGE_PRIORITIES:
            raise ValueError("triage_priority is invalid")
        if not isinstance(self.triage_score, int) or isinstance(self.triage_score, bool):
            raise TypeError("triage_score must be an integer")
        if self.triage_score < 0:
            raise ValueError("triage_score must be non-negative")
        object.__setattr__(
            self, "triage_reasons", _require_tuple_strings(self.triage_reasons, "triage_reasons")
        )
        if self.review_status not in REVIEW_STATUSES:
            raise ValueError("review_status is invalid")
        if self.promotion_status not in PROMOTION_STATUSES:
            raise ValueError("promotion_status is invalid")
        if not isinstance(self.event_count, int) or isinstance(self.event_count, bool):
            raise TypeError("event_count must be an integer")
        if self.event_count < 0:
            raise ValueError("event_count must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {field: _json_value(getattr(self, field)) for field in self.__dataclass_fields__}


@dataclass(frozen=True)
class FeedbackEvent:
    event_id: str
    case_id: str
    helpful: bool
    feedback_reason: str | None
    event_at: str
    source: FeedbackSource
    event_schema_version: int

    def __post_init__(self) -> None:
        _require_text(self.event_id, "event_id")
        _require_text(self.case_id, "case_id")
        if not isinstance(self.helpful, bool):
            raise TypeError("helpful must be a boolean")
        if self.feedback_reason is not None:
            _require_text(self.feedback_reason, "feedback_reason")
        _require_iso_like(self.event_at, "event_at")
        if self.source not in FEEDBACK_SOURCES:
            raise ValueError("source is invalid")
        if not isinstance(self.event_schema_version, int) or self.event_schema_version < 1:
            raise ValueError("event_schema_version is invalid")

    def to_dict(self) -> dict[str, object]:
        return {field: _json_value(getattr(self, field)) for field in self.__dataclass_fields__}


@dataclass(frozen=True)
class ReviewDecision:
    review_id: str
    case_id: str
    reviewer_role: ReviewerRole
    decision: ReviewDecisionKind
    evidence_ok: bool | None
    points_ok: bool | None
    safety_ok: bool | None
    note_code: str | None
    review_version: str
    reviewed_at: str

    def __post_init__(self) -> None:
        _require_text(self.review_id, "review_id")
        _require_text(self.case_id, "case_id")
        if self.reviewer_role not in REVIEWER_ROLES:
            raise ValueError("reviewer_role is invalid")
        if self.decision not in REVIEW_DECISIONS:
            raise ValueError("decision is invalid")
        for field in ("evidence_ok", "points_ok", "safety_ok"):
            value = getattr(self, field)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{field} must be a boolean or null")
        if self.note_code is not None:
            _require_text(self.note_code, "note_code")
        _require_text(self.review_version, "review_version")
        _require_iso_like(self.reviewed_at, "reviewed_at")

    def to_dict(self) -> dict[str, object]:
        return {field: _json_value(getattr(self, field)) for field in self.__dataclass_fields__}


@dataclass(frozen=True)
class PromotionRecord:
    promotion_id: str
    case_id: str
    target_set_id: str
    target_version: str
    target_split: PromotionSplit
    promotion_reason: str
    manifest_id: str
    promoted_at: str

    def __post_init__(self) -> None:
        for field in (
            "promotion_id",
            "case_id",
            "target_set_id",
            "target_version",
            "promotion_reason",
            "manifest_id",
        ):
            _require_text(getattr(self, field), field)
        if self.target_split not in PROMOTION_SPLITS:
            raise ValueError("target_split is invalid")
        _require_iso_like(self.promoted_at, "promoted_at")

    def to_dict(self) -> dict[str, object]:
        return {field: _json_value(getattr(self, field)) for field in self.__dataclass_fields__}
