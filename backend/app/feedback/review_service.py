"""Safe, local-only views for the administrator feedback workbench."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.feedback.models import (
    FeedbackCase,
    FeedbackEvent,
    PromotionRecord,
    ReviewDecision,
)

_PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
_SAFE_CODE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}$")
_SAFE_TEXT_LIMIT = 120


def _safe_code(value: str | None) -> str | None:
    if value is None:
        return None
    return value if _SAFE_CODE.fullmatch(value) else None


def _preview(value: str) -> str:
    first_line = value.splitlines()[0].strip() if value.splitlines() else value.strip()
    return first_line[:_SAFE_TEXT_LIMIT]


def _safe_event(event: FeedbackEvent) -> dict[str, object]:
    return {
        "event_id": event.event_id,
        "helpful": event.helpful,
        "feedback_reason": _safe_code(event.feedback_reason),
        "event_at": event.event_at,
        "source": event.source,
        "event_schema_version": event.event_schema_version,
    }


def _safe_review(review: ReviewDecision) -> dict[str, object]:
    return {
        "review_id": review.review_id,
        "reviewer_role": review.reviewer_role,
        "decision": review.decision,
        "evidence_ok": review.evidence_ok,
        "points_ok": review.points_ok,
        "safety_ok": review.safety_ok,
        "note_code": _safe_code(review.note_code),
        "review_version": _safe_code(review.review_version),
        "reviewed_at": review.reviewed_at,
    }


def _safe_promotion(record: PromotionRecord) -> dict[str, object]:
    return {
        "promotion_id": record.promotion_id,
        "target_set_id": _safe_code(record.target_set_id),
        "target_version": _safe_code(record.target_version),
        "target_split": record.target_split,
        "promotion_reason": _safe_code(record.promotion_reason),
        "manifest_id": _safe_code(record.manifest_id),
        "promoted_at": record.promoted_at,
    }


@dataclass(frozen=True)
class FeedbackQueueFilters:
    priority: str | None = None
    review_status: str | None = None
    reason_code: str | None = None
    answer_status: str | None = None
    source_version: str | None = None


@dataclass(frozen=True)
class FeedbackCaseSummaryView:
    case_id: str
    triage_priority: str
    reason_code: str | None
    question_preview: str
    source_version: str | None
    review_status: str
    promotion_status: str
    collected_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "triage_priority": self.triage_priority,
            "reason_code": self.reason_code,
            "question_preview": self.question_preview,
            "source_version": self.source_version,
            "review_status": self.review_status,
            "promotion_status": self.promotion_status,
            "collected_at": self.collected_at,
        }


@dataclass(frozen=True)
class FeedbackCaseDetailView:
    case_id: str
    question: str | None
    answer: str | None
    redaction_status: str
    answer_status: str
    reason_code: str | None
    source_version: str | None
    retrieval_profile: str | None
    citation_chunk_ids: tuple[str, ...]
    citation_count: int
    model_id: str | None
    prompt_version: str | None
    latency_bucket: str | None
    audience_scope: str
    collected_at: str
    triage_priority: str
    triage_score: int
    triage_reasons: tuple[str, ...]
    review_status: str
    promotion_status: str
    event_count: int
    events: tuple[dict[str, object], ...]
    reviews: tuple[dict[str, object], ...]
    promotions: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "answer": self.answer,
            "redaction_status": self.redaction_status,
            "answer_status": self.answer_status,
            "reason_code": self.reason_code,
            "source_version": self.source_version,
            "retrieval_profile": self.retrieval_profile,
            "citation_chunk_ids": list(self.citation_chunk_ids),
            "citation_count": self.citation_count,
            "model_id": self.model_id,
            "prompt_version": self.prompt_version,
            "latency_bucket": self.latency_bucket,
            "audience_scope": self.audience_scope,
            "collected_at": self.collected_at,
            "triage_priority": self.triage_priority,
            "triage_score": self.triage_score,
            "triage_reasons": list(self.triage_reasons),
            "review_status": self.review_status,
            "promotion_status": self.promotion_status,
            "event_count": self.event_count,
            "events": list(self.events),
            "reviews": list(self.reviews),
            "promotions": list(self.promotions),
        }


class FeedbackReviewService:
    """Project repository records into a strict administrator-safe view."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def _cases(self) -> list[FeedbackCase]:
        return self.repository.list_cases(limit=1000)

    def summary(self) -> dict[str, int]:
        cases = self._cases()
        summary: dict[str, int] = {"cases_total": len(cases)}
        for case in cases:
            for key, value in (
                (f"priority_{case.triage_priority}", 1),
                (f"review_{case.review_status}", 1),
                (f"promotion_{case.promotion_status}", 1),
            ):
                summary[key] = summary.get(key, 0) + value
        return summary

    def list_cases(
        self, filters: FeedbackQueueFilters, limit: int = 50
    ) -> list[FeedbackCaseSummaryView]:
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        candidates = self.repository.list_cases(
            status=filters.review_status,
            priority=filters.priority,
            limit=1000,
        )
        if filters.reason_code is not None:
            candidates = [item for item in candidates if item.reason_code == filters.reason_code]
        if filters.answer_status is not None:
            candidates = [item for item in candidates if item.answer_status == filters.answer_status]
        if filters.source_version is not None:
            candidates = [item for item in candidates if item.source_version == filters.source_version]
        candidates.sort(
            key=lambda item: (
                _PRIORITY_ORDER[item.triage_priority],
                item.collected_at,
                item.case_id,
            )
        )
        return [self._summary_view(item) for item in candidates[:limit]]

    def get_case(self, case_id: str) -> FeedbackCaseDetailView | None:
        if not isinstance(case_id, str) or not case_id.strip():
            return None
        case = next((item for item in self._cases() if item.case_id == case_id), None)
        if case is None:
            return None
        events = tuple(_safe_event(item) for item in self.repository.list_events(case_id))
        reviews = tuple(_safe_review(item) for item in self.repository.list_reviews(case_id))
        promotions = tuple(_safe_promotion(item) for item in self.repository.list_promotions(case_id))
        visible = case.redaction_status == "passed"
        return FeedbackCaseDetailView(
            case_id=case.case_id,
            question=case.question_redacted if visible else None,
            answer=case.answer_redacted if visible else None,
            redaction_status=case.redaction_status,
            answer_status=case.answer_status,
            reason_code=_safe_code(case.reason_code),
            source_version=_safe_code(case.source_version),
            retrieval_profile=_safe_code(case.retrieval_profile),
            citation_chunk_ids=case.citation_chunk_ids if visible else (),
            citation_count=case.citation_count if visible else 0,
            model_id=_safe_code(case.model_id),
            prompt_version=_safe_code(case.prompt_version),
            latency_bucket=_safe_code(case.latency_bucket),
            audience_scope=case.audience_scope,
            collected_at=case.collected_at,
            triage_priority=case.triage_priority,
            triage_score=case.triage_score,
            triage_reasons=tuple(_safe_code(item) for item in case.triage_reasons if _safe_code(item)),
            review_status=case.review_status,
            promotion_status=case.promotion_status,
            event_count=case.event_count,
            events=events,
            reviews=reviews,
            promotions=promotions,
        )

    @staticmethod
    def _summary_view(case: FeedbackCase) -> FeedbackCaseSummaryView:
        visible = case.redaction_status == "passed"
        return FeedbackCaseSummaryView(
            case_id=case.case_id,
            triage_priority=case.triage_priority,
            reason_code=_safe_code(case.reason_code),
            question_preview=_preview(case.question_redacted) if visible else "内容因隐私状态不可展示",
            source_version=_safe_code(case.source_version),
            review_status=case.review_status,
            promotion_status=case.promotion_status,
            collected_at=case.collected_at,
        )


__all__ = [
    "FeedbackCaseDetailView",
    "FeedbackCaseSummaryView",
    "FeedbackQueueFilters",
    "FeedbackReviewService",
]
