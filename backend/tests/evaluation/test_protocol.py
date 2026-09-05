from __future__ import annotations

import pytest
from app.evaluation.protocol import (
    TASK12A_PROTOCOL,
    EvaluationRunMetadata,
    assert_deepseek_only,
    assert_generator_deepseek_only,
)


def test_task12a_protocol_is_frozen() -> None:
    assert TASK12A_PROTOCOL.version == "task12a-v1"
    assert TASK12A_PROTOCOL.case_count == 30
    assert TASK12A_PROTOCOL.cases_per_type == 5
    assert TASK12A_PROTOCOL.retrieval_k == 20
    assert TASK12A_PROTOCOL.evidence_k == 6
    assert TASK12A_PROTOCOL.relevance_threshold == 0.35
    assert TASK12A_PROTOCOL.max_retries == 0
    assert TASK12A_PROTOCOL.candidate_generator == "deepseek"
    assert TASK12A_PROTOCOL.formal_answer == "deepseek"
    assert TASK12A_PROTOCOL.embedding == "siliconflow"
    assert TASK12A_PROTOCOL.reranker == "siliconflow"


def test_task12a_rejects_minimax_or_taotoken() -> None:
    assert_deepseek_only(("deepseek",))
    with pytest.raises(ValueError, match="DeepSeek-only"):
        assert_deepseek_only(("deepseek", "minimax"))
    with pytest.raises(ValueError, match="DeepSeek-only"):
        assert_deepseek_only(("taotoken",))


def test_generator_policy_inspects_a_fallback_chain() -> None:
    class Provider:
        def __init__(self, name: str) -> None:
            self.provider = name

    class Chain:
        generators = (Provider("deepseek"),)

    assert_generator_deepseek_only(Chain())
    Chain.generators = (Provider("deepseek"), Provider("minimax"))
    with pytest.raises(ValueError, match="DeepSeek-only"):
        assert_generator_deepseek_only(Chain())


def test_run_metadata_contains_only_reproducibility_fields() -> None:
    metadata = EvaluationRunMetadata(
        set_id="set-1",
        source_snapshot_hash="a" * 64,
        active_version_count=2,
        chunk_count=100,
        embedding_model="BAAI/bge-m3",
        reranker_model="BAAI/bge-reranker-v2-m3",
        answer_model="deepseek-v4-flash",
        prompt_fingerprint="b" * 64,
        provider_requests=90,
    )

    payload = metadata.to_dict()

    assert payload["protocol_version"] == "task12a-v1"
    assert payload["provider_requests"] == 90
    assert "api_key" not in payload
    assert "source_ids" not in payload
