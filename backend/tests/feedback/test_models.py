from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError

import pytest
from app.feedback.models import (
    FeedbackCase,
    FeedbackEvent,
    PromotionRecord,
    ReviewDecision,
    build_case_identity,
)


def _case(**overrides: object) -> FeedbackCase:
    values: dict[str, object] = {
        "case_id": "case-1",
        "message_id_hash": "a" * 64,
        "session_id_hash": "b" * 64,
        "collected_at": "2026-09-03T10:00:00+08:00",
        "question_redacted": "透析中低血压怎么处理？",
        "answer_redacted": "请按本单位流程处理。",
        "redaction_status": "passed",
        "redaction_version": "redactor-v1",
        "redaction_flags": (),
        "answer_status": "answered",
        "reason_code": None,
        "source_version": "2026-standard-manual-v1",
        "retrieval_profile": "baseline_v1+vector+c1",
        "citation_chunk_ids": ("chunk-1",),
        "citation_count": 1,
        "model_id": "deepseek-v4-flash",
        "prompt_version": "c1",
        "latency_bucket": "lt_5s",
        "audience_scope": "nurse",
        "retention_expires_at": "2026-12-02T10:00:00+08:00",
        "triage_priority": "P2",
        "triage_score": 3,
        "triage_reasons": ("user_unhelpful",),
        "review_status": "unreviewed",
        "promotion_status": "not_promoted",
    }
    values.update(overrides)
    return FeedbackCase(**values)


def test_feedback_case_is_frozen_and_json_safe() -> None:
    case = _case(redaction_flags=("phone",), triage_reasons=("slow",))

    with pytest.raises(FrozenInstanceError):
        case.question_redacted = "changed"  # type: ignore[misc]

    payload = case.to_dict()
    assert payload["redaction_flags"] == ["phone"]
    assert payload["triage_reasons"] == ["slow"]
    json.dumps(payload, ensure_ascii=False)


def test_feedback_case_rejects_invalid_enums_and_empty_redacted_text() -> None:
    with pytest.raises(ValueError, match="redaction_status"):
        _case(redaction_status="unknown")
    with pytest.raises(ValueError, match="review_status"):
        _case(review_status="unknown")
    with pytest.raises(ValueError, match="question_redacted"):
        _case(question_redacted=" ")
    with pytest.raises(ValueError, match="citation_count"):
        _case(citation_count=-1)


def test_case_identity_is_deterministic_and_hashes_source_ids() -> None:
    first = build_case_identity("message-123", "session-456")
    second = build_case_identity("message-123", "session-456")
    other = build_case_identity("message-999", "session-456")

    assert first == second
    assert first.case_id != other.case_id
    assert first.message_id_hash == hashlib.sha256(b"message-123").hexdigest()


def test_feedback_event_review_and_promotion_contracts() -> None:
    event = FeedbackEvent(
        event_id="event-1",
        case_id="case-1",
        helpful=False,
        feedback_reason="citation_mismatch",
        event_at="2026-09-03T10:00:00+08:00",
        source="user_click",
        event_schema_version=1,
    )
    review = ReviewDecision(
        review_id="review-1",
        case_id="case-1",
        reviewer_role="clinical_reviewer",
        decision="approve",
        evidence_ok=True,
        points_ok=True,
        safety_ok=True,
        note_code="evidence_checked",
        review_version="review-v1",
        reviewed_at="2026-09-03T10:00:00+08:00",
    )
    promotion = PromotionRecord(
        promotion_id="promotion-1",
        case_id="case-1",
        target_set_id="golden-v2-dev",
        target_version="v2.0.0",
        target_split="dev",
        promotion_reason="review_approved",
        manifest_id="manifest-1",
        promoted_at="2026-09-03T10:00:00+08:00",
    )

    assert event.to_dict()["helpful"] is False
    assert review.to_dict()["reviewer_role"] == "clinical_reviewer"
    assert promotion.to_dict()["target_split"] == "dev"


def test_model_constructor_does_not_accept_raw_provider_fields() -> None:
    with pytest.raises(TypeError):
        _case(raw_provider_response="secret")  # type: ignore[call-arg]
