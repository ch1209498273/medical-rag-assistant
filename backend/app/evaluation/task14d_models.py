"""Private, text-free facts used by the Task 14D retrieval diagnosis."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

BranchName = Literal["vector", "lexical"]
CandidateMembership = Literal["vector_only", "lexical_only", "shared"]
DedupDropReason = Literal[
    "rrf_content_hash_merge",
    "exact_duplicate",
    "near_duplicate",
    "heading_cap",
    "top20_truncation",
    "active_version_filter",
    "business_eligibility_filter",
]
Task14DArmName = Literal["vector", "lexical", "hybrid", "hybrid_audit"]
Task14DCaseType = Literal["answerable", "no_answer"]


def _validate_candidate_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("candidate_id must be a non-empty string")
    return value.strip()


def _validate_rank(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_score(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number or None")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number or None")
    return result


@dataclass(frozen=True)
class BranchCandidateFact:
    """One candidate as observed in an individual recall branch."""

    candidate_id: str
    branch: BranchName
    rank: int
    retrieval_score: float | None
    content_hash: str
    relevant: bool | None = None
    source_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _validate_candidate_id(self.candidate_id))
        if self.branch not in {"vector", "lexical"}:
            raise ValueError("unsupported branch")
        object.__setattr__(self, "rank", _validate_rank(self.rank, "rank"))
        object.__setattr__(
            self,
            "retrieval_score",
            _validate_score(self.retrieval_score, "retrieval_score"),
        )
        if not isinstance(self.content_hash, str) or not self.content_hash.strip():
            raise ValueError("content_hash must be a non-empty string")
        if self.relevant is not None and not isinstance(self.relevant, bool):
            raise ValueError("relevant must be a boolean or None")
        if self.source_id is not None:
            object.__setattr__(self, "source_id", _validate_candidate_id(self.source_id))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "branch": self.branch,
            "rank": self.rank,
            "retrieval_score": self.retrieval_score,
            "content_hash": self.content_hash,
            "relevant": self.relevant,
            "source_id": self.source_id,
        }


@dataclass(frozen=True)
class FusionCandidateFact:
    """A candidate's branch membership and rank movement through RRF."""

    candidate_id: str
    membership: CandidateMembership
    vector_rank: int | None
    lexical_rank: int | None
    fused_rank: int
    rrf_score: float
    relevant: bool | None = None
    source_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _validate_candidate_id(self.candidate_id))
        if self.membership not in {"vector_only", "lexical_only", "shared"}:
            raise ValueError("unsupported candidate membership")
        for name in ("vector_rank", "lexical_rank"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _validate_rank(value, name))
        object.__setattr__(self, "fused_rank", _validate_rank(self.fused_rank, "fused_rank"))
        object.__setattr__(self, "rrf_score", _validate_score(self.rrf_score, "rrf_score"))
        if self.relevant is not None and not isinstance(self.relevant, bool):
            raise ValueError("relevant must be a boolean or None")
        if self.source_id is not None:
            object.__setattr__(self, "source_id", _validate_candidate_id(self.source_id))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "membership": self.membership,
            "vector_rank": self.vector_rank,
            "lexical_rank": self.lexical_rank,
            "fused_rank": self.fused_rank,
            "rrf_score": self.rrf_score,
            "relevant": self.relevant,
            "source_id": self.source_id,
        }


@dataclass(frozen=True)
class DedupDropFact:
    """A candidate removed by a diagnostic stage, without its content."""

    candidate_id: str
    reason: DedupDropReason
    stage: Literal["rrf", "post_rrf"]
    kept_candidate_id: str | None = None
    relevant: bool | None = None
    source_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _validate_candidate_id(self.candidate_id))
        if self.reason not in {
            "rrf_content_hash_merge",
            "exact_duplicate",
            "near_duplicate",
            "heading_cap",
            "top20_truncation",
            "active_version_filter",
            "business_eligibility_filter",
        }:
            raise ValueError("unsupported dedup drop reason")
        if self.stage not in {"rrf", "post_rrf"}:
            raise ValueError("unsupported diagnostic stage")
        if self.kept_candidate_id is not None:
            object.__setattr__(
                self,
                "kept_candidate_id",
                _validate_candidate_id(self.kept_candidate_id),
            )
        if self.relevant is not None and not isinstance(self.relevant, bool):
            raise ValueError("relevant must be a boolean or None")
        if self.source_id is not None:
            object.__setattr__(self, "source_id", _validate_candidate_id(self.source_id))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "reason": self.reason,
            "stage": self.stage,
            "kept_candidate_id": self.kept_candidate_id,
            "relevant": self.relevant,
            "source_id": self.source_id,
        }


@dataclass(frozen=True)
class Task14DRetrievalTrace:
    """Text-free branch, fusion, and post-fusion facts for one question."""

    vector: tuple[BranchCandidateFact, ...] = ()
    lexical: tuple[BranchCandidateFact, ...] = ()
    fusion: tuple[FusionCandidateFact, ...] = ()
    drops: tuple[DedupDropFact, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "vector", tuple(self.vector))
        object.__setattr__(self, "lexical", tuple(self.lexical))
        object.__setattr__(self, "fusion", tuple(self.fusion))
        object.__setattr__(self, "drops", tuple(self.drops))

    def to_dict(self) -> dict[str, Any]:
        return {
            "vector": [item.to_dict() for item in self.vector],
            "lexical": [item.to_dict() for item in self.lexical],
            "fusion": [item.to_dict() for item in self.fusion],
            "drops": [item.to_dict() for item in self.drops],
        }


def _validate_id_tuple(values: Any, name: str) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{name} must be a sequence of identifiers")
    result: list[str] = []
    for value in values:
        result.append(_validate_candidate_id(value))
    return tuple(result)


@dataclass(frozen=True)
class Task14DArmCaseFact:
    """Private per-question outcome for one diagnostic arm.

    Only stable identifiers and counts are retained.  The model deliberately
    has no question text, answer text, evidence excerpt, prompt, or provider
    response field.
    """

    arm: Task14DArmName
    retrieved_top20: tuple[str, ...] = ()
    evidence_top6: tuple[str, ...] = ()
    answered: bool = False
    refused: bool = True
    trace: Task14DRetrievalTrace | None = None
    duplicate_noise_count: int = 0

    def __post_init__(self) -> None:
        if self.arm not in {"vector", "lexical", "hybrid", "hybrid_audit"}:
            raise ValueError("unsupported Task 14D arm")
        object.__setattr__(
            self, "retrieved_top20", _validate_id_tuple(self.retrieved_top20, "retrieved_top20")
        )
        object.__setattr__(
            self, "evidence_top6", _validate_id_tuple(self.evidence_top6, "evidence_top6")
        )
        if len(self.retrieved_top20) > 20:
            raise ValueError("retrieved_top20 exceeds the Top-20 contract")
        if len(self.evidence_top6) > 6:
            raise ValueError("evidence_top6 exceeds the Top-6 contract")
        if not isinstance(self.answered, bool) or not isinstance(self.refused, bool):
            raise TypeError("answered and refused must be booleans")
        if self.trace is not None and not isinstance(self.trace, Task14DRetrievalTrace):
            raise TypeError("trace must be a Task14DRetrievalTrace or None")
        if (
            isinstance(self.duplicate_noise_count, bool)
            or not isinstance(self.duplicate_noise_count, int)
            or self.duplicate_noise_count < 0
        ):
            raise ValueError("duplicate_noise_count must be a non-negative integer")


@dataclass(frozen=True)
class Task14DCaseFact:
    """Private, aligned outcome for one frozen evaluation question."""

    case_id: str
    case_type: Task14DCaseType
    expected_ids: tuple[str, ...]
    arms: Mapping[str, Task14DArmCaseFact]

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _validate_candidate_id(self.case_id))
        if self.case_type not in {"answerable", "no_answer"}:
            raise ValueError("unsupported Task 14D case type")
        expected_ids = _validate_id_tuple(self.expected_ids, "expected_ids")
        if self.case_type == "answerable" and not expected_ids:
            raise ValueError("answerable case must have expected identifiers")
        if self.case_type == "no_answer" and expected_ids:
            raise ValueError("no_answer case cannot have expected identifiers")
        object.__setattr__(self, "expected_ids", expected_ids)
        if not isinstance(self.arms, Mapping):
            raise TypeError("arms must be a mapping")
        required = {"vector", "lexical", "hybrid", "hybrid_audit"}
        if set(self.arms) != required:
            raise ValueError("Task 14D case must contain all four arms")
        normalised: dict[str, Task14DArmCaseFact] = {}
        for arm, value in self.arms.items():
            if not isinstance(value, Task14DArmCaseFact) or value.arm != arm:
                raise ValueError("arm fact does not match its mapping key")
            normalised[arm] = value
        object.__setattr__(self, "arms", normalised)


__all__ = [
    "BranchCandidateFact",
    "BranchName",
    "CandidateMembership",
    "DedupDropFact",
    "DedupDropReason",
    "FusionCandidateFact",
    "Task14DArmCaseFact",
    "Task14DArmName",
    "Task14DCaseFact",
    "Task14DCaseType",
    "Task14DRetrievalTrace",
]
