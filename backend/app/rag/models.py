"""Stable application-owned values passed between RAG layers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from app.domain.ports import SearchHit

RetrievalStatus = Literal["ready", "refused"]
EventType = Literal["status", "answer_delta", "final", "error"]
AnswerFailureCode = Literal[
    "ANSWER_EVIDENCE_INVALID",
    "ANSWER_SCHEMA_INVALID",
    "ANSWER_TEXT_UNSAFE",
    "ANSWER_CITATION_INVALID",
    "ANSWER_VERIFICATION_FAILED",
]
AnswerFailureDetail = Literal[
    "SOURCE_MARKER",
    "SECRET_OR_PATH",
    "SYSTEM_PROMPT",
    "UNKNOWN",
]
ANSWER_FAILURE_DETAILS = frozenset(
    {"SOURCE_MARKER", "SECRET_OR_PATH", "SYSTEM_PROMPT", "UNKNOWN"}
)
ANSWER_FAILURE_CODES = frozenset(
    {
        "ANSWER_EVIDENCE_INVALID",
        "ANSWER_SCHEMA_INVALID",
        "ANSWER_TEXT_UNSAFE",
        "ANSWER_CITATION_INVALID",
        "ANSWER_VERIFICATION_FAILED",
    }
)


@dataclass(frozen=True)
class Evidence:
    """One ranked, locatable source that can support an answer."""

    reference_id: str
    hit: SearchHit
    rerank_score: float


@dataclass
class EphemeralEvaluationCapture:
    """Evaluation-only in-memory evidence handoff with no serialization API."""

    evidence: tuple[Evidence, ...] = ()

    def record_evidence(self, evidence: Sequence[Evidence]) -> None:
        self.evidence = tuple(evidence)

    def clear(self) -> None:
        self.evidence = ()


@dataclass(frozen=True)
class RetrievalDiagnosticCandidate:
    """Safe, candidate-level facts retained only for private diagnostics."""

    source_id: str
    candidate_rank: int
    retrieval_score: float | None
    rerank_score: float | None
    selected: bool
    selected_rank: int | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize only the ID/rank/score/selection fields."""

        return {
            "source_id": self.source_id,
            "candidate_rank": self.candidate_rank,
            "retrieval_score": self.retrieval_score,
            "rerank_score": self.rerank_score,
            "selected": self.selected,
            "selected_rank": self.selected_rank,
        }


@dataclass(frozen=True)
class RetrievalDiagnosticMetadata:
    """Aggregate configuration facts for a private retrieval trace."""

    schema_version: str
    retrieval_strategy: str
    selector_name: str
    selector_version: str
    retrieval_limit: int
    rerank_limit: int
    relevance_threshold: float
    rrf_k: int
    all_candidate_scores_captured: bool
    # Weighted fusion facts are emitted for hybrid traces only.  Keeping the
    # fields optional preserves the existing vector-only diagnostic shape.
    fusion_profile: str | None = None
    vector_weight: float | None = None
    lexical_weight: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "retrieval_strategy": self.retrieval_strategy,
            "selector_name": self.selector_name,
            "selector_version": self.selector_version,
            "retrieval_limit": self.retrieval_limit,
            "rerank_limit": self.rerank_limit,
            "relevance_threshold": self.relevance_threshold,
            "rrf_k": self.rrf_k,
            "all_candidate_scores_captured": self.all_candidate_scores_captured,
        }
        if self.fusion_profile is not None:
            payload["fusion_profile"] = self.fusion_profile
            payload["vector_weight"] = self.vector_weight
            payload["lexical_weight"] = self.lexical_weight
        return payload


@dataclass(frozen=True)
class RetrievalDiagnosticTrace:
    """Typed private trace for one retained retrieval candidate pool."""

    metadata: RetrievalDiagnosticMetadata
    candidates: tuple[RetrievalDiagnosticCandidate, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "candidates": [item.to_dict() for item in self.candidates],
        }


@dataclass(frozen=True)
class RetrievalResult:
    """The result of retrieval and relevance gating."""

    status: RetrievalStatus
    evidence: tuple[Evidence, ...] = ()
    reason_code: str | None = None
    diagnostic_trace: RetrievalDiagnosticTrace | None = None


@dataclass(frozen=True)
class CandidateSet:
    """Candidates retained between vector recall and reranking."""

    question: str
    hits: tuple[SearchHit, ...] = ()
    # A non-null value means the business allowlist deliberately excluded all
    # technical candidates.  RagService turns this into a safe terminal
    # refusal before reranking or generation can run.
    reason_code: str | None = None


@dataclass(frozen=True)
class ChatEvent:
    """A safe, provider-independent event sent to the API layer."""

    type: EventType
    data: dict[str, object]


__all__ = [
    "ANSWER_FAILURE_CODES",
    "AnswerFailureCode",
    "CandidateSet",
    "ChatEvent",
    "EphemeralEvaluationCapture",
    "EventType",
    "Evidence",
    "RetrievalDiagnosticCandidate",
    "RetrievalDiagnosticMetadata",
    "RetrievalDiagnosticTrace",
    "RetrievalResult",
    "RetrievalStatus",
]
