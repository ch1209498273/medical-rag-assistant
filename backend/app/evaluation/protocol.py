"""Frozen protocol metadata for the Task 12A evaluation run."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class EvaluationProtocol:
    """Explicit evaluation constants so a run cannot silently drift."""

    version: str = "task12a-v1"
    case_count: int = 30
    cases_per_type: int = 5
    retrieval_k: int = 20
    evidence_k: int = 6
    relevance_threshold: float = 0.35
    max_retries: int = 0
    candidate_generator: str = "deepseek"
    formal_answer: str = "deepseek"
    embedding: str = "siliconflow"
    reranker: str = "siliconflow"


TASK12A_PROTOCOL = EvaluationProtocol()


@dataclass(frozen=True)
class EvaluationRunMetadata:
    """Reproducibility facts for one private run; never stores credentials."""

    set_id: str
    source_snapshot_hash: str
    active_version_count: int
    chunk_count: int
    embedding_model: str
    reranker_model: str
    answer_model: str
    prompt_fingerprint: str
    provider_requests: int = 0
    retry_count: int = 0
    minimax_calls: int = 0
    protocol_version: str = TASK12A_PROTOCOL.version
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.set_id.strip() or not self.source_snapshot_hash.strip():
            raise ValueError("run metadata identifiers are required")
        for name in (
            "active_version_count",
            "chunk_count",
            "provider_requests",
            "retry_count",
            "minimax_calls",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.retry_count != 0:
            raise ValueError("Task 12A does not permit retries")
        if self.minimax_calls != 0:
            raise ValueError("Task 12A does not permit MiniMax calls")
        if not self.created_at:
            object.__setattr__(
                self,
                "created_at",
                datetime.now(UTC).isoformat(timespec="seconds"),
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def assert_deepseek_only(providers: Sequence[str]) -> None:
    """Reject any candidate-generation or answer chain that is not DeepSeek-only."""

    normalised = tuple(
        provider.strip().casefold()
        for provider in providers
        if isinstance(provider, str) and provider.strip()
    )
    if not normalised or any(provider != "deepseek" for provider in normalised):
        raise ValueError("Task 12A requires a DeepSeek-only provider chain")


def assert_generator_deepseek_only(generator: Any) -> None:
    """Inspect a direct generator or fallback chain without invoking it."""

    candidates = getattr(generator, "generators", (generator,))
    providers: list[str] = []
    for candidate in candidates:
        provider = getattr(candidate, "provider", None)
        if not isinstance(provider, str):
            client = getattr(candidate, "client", None)
            provider = getattr(client, "provider", None)
        if isinstance(provider, str):
            providers.append(provider)
    assert_deepseek_only(providers)


__all__ = [
    "TASK12A_PROTOCOL",
    "EvaluationProtocol",
    "EvaluationRunMetadata",
    "assert_deepseek_only",
    "assert_generator_deepseek_only",
]
