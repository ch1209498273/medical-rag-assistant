from __future__ import annotations

from app.feedback.repository import FeedbackProjectionRepository
from app.feedback.review_service import FeedbackQueueFilters, FeedbackReviewService


def _path(tmp_path):
    path = tmp_path / "data" / "private" / "feedback" / "feedback.db"
    path.parent.mkdir(parents=True)
    return path


def test_review_service_returns_safe_sorted_queue_and_summary(tmp_path, sample_case):
    repository = FeedbackProjectionRepository(_path(tmp_path))
    try:
        repository.upsert_case(sample_case)
        repository.upsert_case(
            sample_case.__class__(
                **{
                    **sample_case.to_dict(),
                    "case_id": "case-p0",
                    "message_id_hash": "c" * 64,
                    "session_id_hash": "d" * 64,
                    "triage_priority": "P0",
                    "triage_score": 8,
                    "collected_at": "2026-09-03T09:00:00+08:00",
                }
            )
        )
        repository.upsert_case(
            sample_case.__class__(
                **{
                    **sample_case.to_dict(),
                    "case_id": "case-p1",
                    "message_id_hash": "e" * 64,
                    "session_id_hash": "f" * 64,
                    "triage_priority": "P1",
                    "triage_score": 7,
                    "collected_at": "2026-09-03T09:30:00+08:00",
                }
            )
        )
        service = FeedbackReviewService(repository)

        summary = service.summary()
        rows = service.list_cases(
            FeedbackQueueFilters(review_status="unreviewed"), limit=10
        )

        assert summary["cases_total"] == 3
        assert [row.triage_priority for row in rows] == ["P0", "P1", "P2"]
        assert "message_id" not in rows[0].to_dict()
        assert "source_path" not in rows[0].to_dict()
        assert rows[0].question_preview == "透析中低血压怎么处理？"
    finally:
        repository.close()


def test_review_service_hides_text_for_redaction_failures(tmp_path, sample_case):
    repository = FeedbackProjectionRepository(_path(tmp_path))
    try:
        blocked = sample_case.__class__(
            **{
                **sample_case.to_dict(),
                "redaction_status": "blocked",
                "question_redacted": "不应展示的内容",
                "answer_redacted": "不应展示的答案",
            }
        )
        repository.upsert_case(blocked)
        detail = FeedbackReviewService(repository).get_case(sample_case.case_id)

        assert detail is not None
        assert detail.to_dict()["question"] is None
        assert detail.to_dict()["answer"] is None
        assert detail.to_dict()["redaction_status"] == "blocked"
    finally:
        repository.close()
