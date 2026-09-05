from __future__ import annotations

import pytest
from app.feedback.models import FeedbackCase


@pytest.fixture
def sample_case() -> FeedbackCase:
    return FeedbackCase(
        case_id="case-1",
        message_id_hash="a" * 64,
        session_id_hash="b" * 64,
        collected_at="2026-09-03T10:00:00+08:00",
        question_redacted="透析中低血压怎么处理？",
        answer_redacted="请按本单位流程处理。",
        redaction_status="passed",
        redaction_version="redactor-v1",
        redaction_flags=(),
        answer_status="answered",
        reason_code=None,
        source_version="2026-standard-manual-v1",
        retrieval_profile="baseline_v1+vector+c1",
        citation_chunk_ids=("chunk-1",),
        citation_count=1,
        model_id="deepseek-v4-flash",
        prompt_version="c1",
        latency_bucket="lt_5s",
        audience_scope="nurse",
        retention_expires_at="2026-12-02T10:00:00+08:00",
        triage_priority="P2",
        triage_score=3,
        triage_reasons=("user_unhelpful",),
        review_status="unreviewed",
        promotion_status="not_promoted",
    )
