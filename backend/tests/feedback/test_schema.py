from __future__ import annotations

import sqlite3

import pytest
from app.feedback.models import FeedbackCase, FeedbackEvent
from app.feedback.schema import FeedbackProjectionSchema


def _case() -> FeedbackCase:
    return FeedbackCase(
        case_id="case-1",
        message_id_hash="a" * 64,
        session_id_hash="b" * 64,
        collected_at="2026-09-03T10:00:00+08:00",
        question_redacted="问题",
        answer_redacted="回答",
        redaction_status="passed",
        redaction_version="redactor-v1",
        redaction_flags=(),
        answer_status="answered",
        reason_code=None,
        source_version=None,
        retrieval_profile=None,
        citation_chunk_ids=(),
        citation_count=0,
        model_id=None,
        prompt_version=None,
        latency_bucket=None,
        audience_scope="nurse",
        retention_expires_at="2026-12-02T10:00:00+08:00",
        triage_priority="P3",
        triage_score=0,
        triage_reasons=(),
        review_status="unreviewed",
        promotion_status="not_promoted",
    )


def test_schema_creation_is_idempotent_and_has_required_tables() -> None:
    connection = sqlite3.connect(":memory:")
    FeedbackProjectionSchema.create(connection)
    FeedbackProjectionSchema.create(connection)

    names = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert {
        "feedback_cases",
        "feedback_events",
        "review_decisions",
        "promotion_records",
        "schema_version",
    }.issubset(names)
    assert connection.execute("SELECT version FROM schema_version").fetchone()[0] == 1


def test_schema_enforces_case_identity_and_append_only_event_identity() -> None:
    connection = sqlite3.connect(":memory:")
    FeedbackProjectionSchema.create(connection)
    case = _case()
    FeedbackProjectionSchema.insert_case(connection, case)
    with pytest.raises(sqlite3.IntegrityError):
        FeedbackProjectionSchema.insert_case(connection, case)

    event = FeedbackEvent(
        event_id="event-1",
        case_id=case.case_id,
        helpful=False,
        feedback_reason="not_answered",
        event_at="2026-09-03T10:00:00+08:00",
        source="user_click",
        event_schema_version=1,
    )
    FeedbackProjectionSchema.insert_event(connection, event)
    with pytest.raises(sqlite3.IntegrityError):
        FeedbackProjectionSchema.insert_event(connection, event)
