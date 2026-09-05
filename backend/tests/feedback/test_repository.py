from __future__ import annotations

from datetime import datetime, timezone

import pytest
from app.feedback.models import FeedbackEvent, PromotionRecord, ReviewDecision
from app.feedback.repository import (
    FeedbackProjectionRepository,
    validate_private_feedback_path,
)


def _path(tmp_path):
    path = tmp_path / "data" / "private" / "feedback" / "feedback.db"
    path.parent.mkdir(parents=True)
    return path


def test_private_path_must_be_below_data_private_feedback(tmp_path) -> None:
    path = _path(tmp_path)
    assert validate_private_feedback_path(path) == path.resolve()
    with pytest.raises(ValueError, match="private feedback"):
        validate_private_feedback_path(tmp_path / "feedback.db")


def test_repository_upsert_is_idempotent_and_rejects_conflicting_identity(tmp_path, sample_case):
    repository = FeedbackProjectionRepository(_path(tmp_path))
    try:
        repository.upsert_case(sample_case)
        repository.upsert_case(sample_case)
        assert len(repository.list_cases()) == 1

        conflicting = sample_case.__class__(
            **{**sample_case.to_dict(), "message_id_hash": "c" * 64}
        )
        with pytest.raises(ValueError, match="identity"):
            repository.upsert_case(conflicting)
    finally:
        repository.close()


def test_events_reviews_promotions_are_append_only_and_queryable(tmp_path, sample_case):
    repository = FeedbackProjectionRepository(_path(tmp_path))
    try:
        repository.upsert_case(sample_case)
        event = FeedbackEvent(
            event_id="event-1",
            case_id=sample_case.case_id,
            helpful=False,
            feedback_reason="not_answered",
            event_at="2026-09-03T10:00:00+08:00",
            source="user_click",
            event_schema_version=1,
        )
        repository.append_event(event)
        repository.append_event(event)
        assert repository.event_count(sample_case.case_id) == 1

        review = ReviewDecision(
            review_id="review-1",
            case_id=sample_case.case_id,
            reviewer_role="product",
            decision="approve",
            evidence_ok=True,
            points_ok=True,
            safety_ok=True,
            note_code="ok",
            review_version="v1",
            reviewed_at="2026-09-03T10:00:00+08:00",
        )
        repository.save_review(review)
        promotion = PromotionRecord(
            promotion_id="promotion-1",
            case_id=sample_case.case_id,
            target_set_id="golden-v2-dev",
            target_version="v2.0.0",
            target_split="dev",
            promotion_reason="review_approved",
            manifest_id="manifest-1",
            promoted_at="2026-09-03T10:00:00+08:00",
        )
        repository.save_promotion(promotion)
        assert len(repository.list_reviews(sample_case.case_id)) == 1
        assert len(repository.list_promotions(sample_case.case_id)) == 1
        assert len(repository.list_cases(priority="P2", status="unreviewed")) == 1
    finally:
        repository.close()


def test_retention_deletes_expired_children_and_returns_counts(tmp_path, sample_case):
    repository = FeedbackProjectionRepository(_path(tmp_path))
    try:
        expired = sample_case.__class__(
            **{
                **sample_case.to_dict(),
                "retention_expires_at": "2026-01-01T00:00:00+08:00",
            }
        )
        repository.upsert_case(expired)
        summary = repository.delete_expired(datetime(2026, 2, 1, tzinfo=timezone.utc))
        assert summary.cases_deleted == 1
        assert summary.events_deleted == 0
        assert repository.list_cases() == []
    finally:
        repository.close()
