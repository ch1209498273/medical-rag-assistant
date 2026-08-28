"""Stable application-owned values passed between RAG layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.domain.ports import SearchHit

RetrievalStatus = Literal["ready", "refused"]
EventType = Literal["status", "answer_delta", "final", "error"]


@dataclass(frozen=True)
class Evidence:
    """One ranked, locatable source that can support an answer."""

    reference_id: str
    hit: SearchHit
    rerank_score: float


@dataclass(frozen=True)
class RetrievalResult:
    """The result of retrieval and relevance gating."""

    status: RetrievalStatus
    evidence: tuple[Evidence, ...] = ()
    reason_code: str | None = None


@dataclass(frozen=True)
class CandidateSet:
    """Candidates retained between vector recall and reranking."""

    question: str
    hits: tuple[SearchHit, ...] = ()


@dataclass(frozen=True)
class ChatEvent:
    """A safe, provider-independent event sent to the API layer."""

    type: EventType
    data: dict[str, object]


__all__ = [
    "CandidateSet",
    "ChatEvent",
    "EventType",
    "Evidence",
    "RetrievalResult",
    "RetrievalStatus",
]
