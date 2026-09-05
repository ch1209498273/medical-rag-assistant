from __future__ import annotations

import json

import pytest
from app.evaluation.task14d_protocol import (
    EXPECTED_CASE_COUNT,
    EXPECTED_SET_CHECKSUM,
    EXPECTED_SET_ID,
    EXPECTED_SNAPSHOT_HASH,
    TOTAL_CALL_BUDGET,
    build_task14d_authorization,
    resolve_task14d_arm,
    validate_task14d_authorization,
    validate_task14d_identity,
)


def test_four_arm_mapping_changes_only_the_declared_retrieval_dimension() -> None:
    vector = resolve_task14d_arm("vector")
    lexical = resolve_task14d_arm("lexical")
    hybrid = resolve_task14d_arm("hybrid")
    audit = resolve_task14d_arm("hybrid_audit")

    assert vector.retrieval_strategy == "vector"
    assert vector.post_fusion_dedup is True
    assert lexical.retrieval_strategy == "lexical"
    assert lexical.post_fusion_dedup is True
    assert hybrid.retrieval_strategy == "hybrid"
    assert hybrid.post_fusion_dedup is True
    assert audit.retrieval_strategy == "hybrid"
    assert audit.post_fusion_dedup is False

    assert lexical.embedding_max_calls == 0
    assert vector.embedding_max_calls == hybrid.embedding_max_calls == audit.embedding_max_calls == 30
    assert TOTAL_CALL_BUDGET == 330


def test_unknown_arm_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported Task 14D arm"):
        resolve_task14d_arm("rrf_guess")


def test_identity_rejects_changed_set_snapshot_or_case_count() -> None:
    with pytest.raises(ValueError, match="TASK14D_PREFLIGHT_FAILED"):
        validate_task14d_identity(
            set_id=EXPECTED_SET_ID + "-wrong",
            set_checksum=EXPECTED_SET_CHECKSUM,
            snapshot_hash=EXPECTED_SNAPSHOT_HASH,
            case_count=EXPECTED_CASE_COUNT,
        )

    with pytest.raises(ValueError, match="TASK14D_PREFLIGHT_FAILED"):
        validate_task14d_identity(
            set_id=EXPECTED_SET_ID,
            set_checksum=EXPECTED_SET_CHECKSUM,
            snapshot_hash="changed-snapshot",
            case_count=EXPECTED_CASE_COUNT,
        )

    with pytest.raises(ValueError, match="TASK14D_PREFLIGHT_FAILED"):
        validate_task14d_identity(
            set_id=EXPECTED_SET_ID,
            set_checksum=EXPECTED_SET_CHECKSUM,
            snapshot_hash=EXPECTED_SNAPSHOT_HASH,
            case_count=29,
        )


def test_authorization_is_non_secret_and_contains_four_arm_budget() -> None:
    payload = build_task14d_authorization()

    assert payload["set_id"] == EXPECTED_SET_ID
    assert payload["set_checksum"] == EXPECTED_SET_CHECKSUM
    assert payload["case_count"] == EXPECTED_CASE_COUNT
    assert payload["source_snapshot"]["source_snapshot_hash"] == EXPECTED_SNAPSHOT_HASH
    assert payload["call_budget"] == {
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
    assert payload["total_call_budget"] == 330
    assert payload["max_retries"] == 0
    assert payload["allow_network"] is True
    assert payload["save_policy"] == {
        "raw_provider_responses": False,
        "output_scope": "local_private",
    }
    serialized = json.dumps(payload, ensure_ascii=False).casefold()
    assert "api_key" not in serialized
    assert "secret" not in serialized
    assert '"access_token"' not in serialized
    assert "sk-" not in serialized


def test_authorization_validation_rejects_changed_arm_budget() -> None:
    payload = build_task14d_authorization()
    payload["call_budget"]["lexical"]["siliconflow_embedding"] = 1

    with pytest.raises(ValueError, match="authorization"):
        validate_task14d_authorization(payload)


def test_authorization_validation_rejects_network_or_retry_relaxation() -> None:
    payload = build_task14d_authorization()
    payload["allow_network"] = False

    with pytest.raises(ValueError, match="authorization"):
        validate_task14d_authorization(payload)

    payload = build_task14d_authorization()
    payload["max_retries"] = 1

    with pytest.raises(ValueError, match="authorization"):
        validate_task14d_authorization(payload)
