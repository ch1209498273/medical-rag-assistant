"""Non-secret, frozen contracts for Task 14D four-arm diagnostics.

The protocol module intentionally contains no provider construction, file I/O,
credential loading, or network access.  It turns a named diagnostic arm into
an explicit configuration and validates the exact authorization envelope
before a runner is allowed to send any material out of the machine.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

Task14DArm = Literal["vector", "lexical", "hybrid", "hybrid_audit"]
Task14DRetrievalStrategy = Literal["vector", "lexical", "hybrid"]

EXPECTED_SET_ID = "30824bb1e67349c3b7d96579d3b59249-corrected-v2"
EXPECTED_SET_CHECKSUM = (
    "80e6e58c690e5d497762324e8b25be47c32b74aa3104c55cbd2aff66ab32be01"
)
EXPECTED_SNAPSHOT_HASH = (
    "bd8b00f1e517374cb86d5f82cfacf29316fdda655ac323a8ece5f0a98cd5572d"
)
EXPECTED_CASE_COUNT = 30
EXPECTED_ACTIVE_VERSION_COUNT = 7
EXPECTED_ACTIVE_CHUNK_COUNT = 8021
RETRIEVAL_LIMIT = 20
RERANK_LIMIT = 6
RELEVANCE_THRESHOLD = 0.35
RRF_K = 60
ANSWER_PROMPT_PROFILE = "c1"

SUPPORTED_ARMS = frozenset({"vector", "lexical", "hybrid", "hybrid_audit"})
_ARM_BUDGETS: dict[str, dict[str, int]] = {
    "vector": {
        "siliconflow_embedding": 30,
        "siliconflow_reranker": 30,
        "deepseek_answer": 30,
    },
    "lexical": {
        "siliconflow_embedding": 0,
        "siliconflow_reranker": 30,
        "deepseek_answer": 30,
    },
    "hybrid": {
        "siliconflow_embedding": 30,
        "siliconflow_reranker": 30,
        "deepseek_answer": 30,
    },
    "hybrid_audit": {
        "siliconflow_embedding": 30,
        "siliconflow_reranker": 30,
        "deepseek_answer": 30,
    },
}
TOTAL_CALL_BUDGET = sum(
    sum(budget.values()) for budget in _ARM_BUDGETS.values()
)


@dataclass(frozen=True)
class Task14DArmConfig:
    """A complete, non-secret description of one diagnostic arm."""

    arm: Task14DArm
    retrieval_strategy: Task14DRetrievalStrategy
    post_fusion_dedup: bool
    embedding_max_calls: int
    reranker_max_calls: int
    answer_max_calls: int

    @property
    def call_budget(self) -> dict[str, int]:
        """Return a fresh provider budget mapping for this arm."""

        return {
            "siliconflow_embedding": self.embedding_max_calls,
            "siliconflow_reranker": self.reranker_max_calls,
            "deepseek_answer": self.answer_max_calls,
        }


def resolve_task14d_arm(arm: str) -> Task14DArmConfig:
    """Resolve a named arm without inferring settings from another run."""

    if arm not in SUPPORTED_ARMS:
        raise ValueError("unsupported Task 14D arm")
    if arm == "vector":
        strategy: Task14DRetrievalStrategy = "vector"
        post_fusion_dedup = True
    elif arm == "lexical":
        strategy = "lexical"
        post_fusion_dedup = True
    elif arm == "hybrid":
        strategy = "hybrid"
        post_fusion_dedup = True
    else:
        strategy = "hybrid"
        post_fusion_dedup = False
    budget = _ARM_BUDGETS[arm]
    return Task14DArmConfig(
        arm=arm,  # type: ignore[arg-type]
        retrieval_strategy=strategy,
        post_fusion_dedup=post_fusion_dedup,
        embedding_max_calls=budget["siliconflow_embedding"],
        reranker_max_calls=budget["siliconflow_reranker"],
        answer_max_calls=budget["deepseek_answer"],
    )


def validate_task14d_identity(
    *,
    set_id: str,
    set_checksum: str,
    snapshot_hash: str,
    case_count: int,
) -> None:
    """Reject any run whose frozen input identity differs from the protocol."""

    if (
        set_id != EXPECTED_SET_ID
        or set_checksum != EXPECTED_SET_CHECKSUM
        or snapshot_hash != EXPECTED_SNAPSHOT_HASH
        or case_count != EXPECTED_CASE_COUNT
    ):
        raise ValueError("TASK14D_PREFLIGHT_FAILED")


def build_task14d_authorization() -> dict[str, object]:
    """Build the exact non-secret scope for the four-arm cloud run."""

    return {
        "schema_version": "task14d-four-arm-authorization-v1",
        "authorization_status": "approved",
        "set_id": EXPECTED_SET_ID,
        "set_checksum": EXPECTED_SET_CHECKSUM,
        "case_count": EXPECTED_CASE_COUNT,
        "source_snapshot": {
            "active_version_count": EXPECTED_ACTIVE_VERSION_COUNT,
            "active_chunk_count": EXPECTED_ACTIVE_CHUNK_COUNT,
            "source_snapshot_hash": EXPECTED_SNAPSHOT_HASH,
        },
        "sqlite_snapshot": "approved-business-metadata-migration-20260830",
        "retrieval": {
            "retrieval_limit": RETRIEVAL_LIMIT,
            "rerank_limit": RERANK_LIMIT,
            "relevance_threshold": RELEVANCE_THRESHOLD,
            "rrf_k": RRF_K,
            "fusion_profile": "balanced",
        },
        "answer": {"prompt_profile": ANSWER_PROMPT_PROFILE},
        "models": {
            "embedding": {
                "provider": "siliconflow_platform",
                "model": "BAAI/bge-m3",
            },
            "reranker": {
                "provider": "siliconflow_platform",
                "model": "BAAI/bge-reranker-v2-m3",
            },
            "answer": {"provider": "deepseek", "model": "deepseek-v4-flash"},
        },
        "providers": {
            "embedding": "siliconflow_platform",
            "reranker": "siliconflow_platform",
            "answer": "deepseek",
        },
        "arms": ["vector", "lexical", "hybrid", "hybrid_audit"],
        "call_budget": {
            arm: dict(budget) for arm, budget in _ARM_BUDGETS.items()
        },
        "total_call_budget": TOTAL_CALL_BUDGET,
        "max_retries": 0,
        "allow_network": True,
        "save_policy": {
            "raw_provider_responses": False,
            "output_scope": "local_private",
        },
        "excluded_providers": ["minimax", "taotoken", "sol"],
    }


def validate_task14d_authorization(payload: Mapping[str, object]) -> None:
    """Validate the full authorization envelope before any Provider exists."""

    if not isinstance(payload, Mapping):
        raise TypeError("authorization must be an object")
    expected = build_task14d_authorization()
    if set(payload) != set(expected):
        raise ValueError("authorization fields are invalid")
    if payload != expected:
        raise ValueError("authorization does not match Task 14D protocol")
    if payload.get("allow_network") is not True or payload.get("max_retries") != 0:
        raise ValueError("authorization network scope is invalid")


__all__ = [
    "ANSWER_PROMPT_PROFILE",
    "EXPECTED_ACTIVE_CHUNK_COUNT",
    "EXPECTED_ACTIVE_VERSION_COUNT",
    "EXPECTED_CASE_COUNT",
    "EXPECTED_SET_CHECKSUM",
    "EXPECTED_SET_ID",
    "EXPECTED_SNAPSHOT_HASH",
    "RELEVANCE_THRESHOLD",
    "RERANK_LIMIT",
    "RETRIEVAL_LIMIT",
    "RRF_K",
    "SUPPORTED_ARMS",
    "TOTAL_CALL_BUDGET",
    "Task14DArm",
    "Task14DArmConfig",
    "Task14DRetrievalStrategy",
    "build_task14d_authorization",
    "resolve_task14d_arm",
    "validate_task14d_authorization",
    "validate_task14d_identity",
]
