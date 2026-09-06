"""Local administrator feedback review API with a strict safe boundary."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from app.feedback.evidence import EvidenceBundle
from app.feedback.models import ReviewDecision
from app.feedback.promotion import (
    EvidenceCheck,
    build_promotion_record,
    evaluate_promotion,
)
from app.feedback.review_service import FeedbackQueueFilters

_SAFE_CODE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}$")
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr


class ReviewSubmissionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    reviewer_role: Literal["product", "engineering", "clinical_reviewer"]
    decision: Literal["approve", "reject", "needs_adjudication"]
    evidence_ok: StrictBool | None = None
    points_ok: StrictBool | None = None
    safety_ok: StrictBool | None = None
    note_code: StrictStr | None = None
    review_version: StrictStr


class PromotionSubmissionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    target_set_id: StrictStr
    target_version: StrictStr
    target_split: Literal["dev", "holdout"]
    manifest_id: StrictStr


def create_feedback_review_router() -> APIRouter:
    router = APIRouter(prefix="/api/admin/feedback", tags=["admin-feedback"])

    @router.get("/summary")
    async def summary(request: Request) -> dict[str, dict[str, int]]:
        service = _service(request)
        return {"summary": service.summary()}

    @router.get("/cases")
    async def cases(
        request: Request,
        priority: str | None = None,
        review_status: str | None = None,
        reason_code: str | None = None,
        answer_status: str | None = None,
        source_version: str | None = None,
        limit: int = Query(default=50, ge=1, le=100),
    ) -> dict[str, object]:
        service = _service(request)
        filters = FeedbackQueueFilters(
            priority=priority,
            review_status=review_status,
            reason_code=reason_code,
            answer_status=answer_status,
            source_version=source_version,
        )
        try:
            rows = service.list_cases(filters, limit=limit)
        except ValueError as error:
            raise HTTPException(status_code=400, detail="FEEDBACK_REVIEW_INVALID_FILTER") from error
        return {"cases": [row.to_dict() for row in rows], "limit": limit}

    @router.get("/cases/{case_id}")
    async def case_detail(case_id: str, request: Request) -> dict[str, object]:
        service = _service(request)
        detail = service.get_case(case_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="FEEDBACK_CASE_NOT_FOUND")
        evidence = _resolve_evidence(request, detail)
        promotion = _promotion_check(request, case_id, evidence)
        payload = detail.to_dict()
        events = payload.pop("events", [])
        reviews = payload.pop("reviews", [])
        promotions = payload.pop("promotions", [])
        return {
            "case": payload,
            "events": events,
            "reviews": reviews,
            "evidence": evidence.to_dict(),
            "promotion": promotion,
            "promotions": promotions,
        }

    @router.post("/cases/{case_id}/reviews")
    async def save_review(
        case_id: str,
        payload: ReviewSubmissionRequest,
        request: Request,
    ) -> dict[str, object]:
        service = _service(request)
        detail = service.get_case(case_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="FEEDBACK_CASE_NOT_FOUND")
        if not _is_safe_code(payload.review_version) or (
            payload.note_code is not None and not _is_safe_code(payload.note_code)
        ):
            raise HTTPException(status_code=400, detail="FEEDBACK_REVIEW_SAVE_FAILED")
        repository = getattr(service, "repository", None)
        if repository is None:
            raise HTTPException(status_code=503, detail="FEEDBACK_REVIEW_UNAVAILABLE")
        review = ReviewDecision(
            review_id=f"rv_{uuid4().hex}",
            case_id=case_id,
            reviewer_role=payload.reviewer_role,
            decision=payload.decision,
            evidence_ok=payload.evidence_ok,
            points_ok=payload.points_ok,
            safety_ok=payload.safety_ok,
            note_code=payload.note_code,
            review_version=payload.review_version,
            reviewed_at=datetime.now(timezone.utc).isoformat(),
        )
        try:
            repository.save_review(review)
            status = {
                "approve": "approved",
                "reject": "rejected",
                "needs_adjudication": "adjudication_required",
            }[payload.decision]
            repository.update_review_status(case_id, status)
        except (KeyError, ValueError) as error:
            raise HTTPException(status_code=400, detail="FEEDBACK_REVIEW_SAVE_FAILED") from error
        evidence = _resolve_evidence(request, detail)
        return {
            "review": _safe_review_response(review),
            "promotion": _promotion_check(request, case_id, evidence),
        }

    @router.post("/cases/{case_id}/promote")
    async def promote_case(
        case_id: str,
        payload: PromotionSubmissionRequest,
        request: Request,
    ) -> dict[str, object]:
        service = _service(request)
        detail = service.get_case(case_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="FEEDBACK_CASE_NOT_FOUND")
        if not all(
            _is_safe_code(value)
            for value in (
                payload.target_set_id,
                payload.target_version,
                payload.manifest_id,
            )
        ):
            raise HTTPException(status_code=400, detail="FEEDBACK_PROMOTION_INVALID")
        repository = getattr(service, "repository", None)
        if repository is None:
            raise HTTPException(status_code=503, detail="FEEDBACK_REVIEW_UNAVAILABLE")
        evidence = _resolve_evidence(request, detail)
        decision = _promotion_decision(request, case_id, evidence, payload.target_split)
        if decision.status != "golden_v2_candidate":
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "FEEDBACK_PROMOTION_BLOCKED",
                    "promotion": decision.to_dict(),
                },
            )
        record = build_promotion_record(
            next(
                item for item in repository.list_cases(limit=1000) if item.case_id == case_id
            ),
            decision,
            target_set_id=payload.target_set_id,
            target_version=payload.target_version,
            target_split=payload.target_split,
            manifest_id=payload.manifest_id,
            promoted_at=datetime.now(timezone.utc).isoformat(),
        )
        try:
            repository.save_promotion(record)
            repository.update_promotion_status(case_id, decision.status)
        except ValueError as error:
            raise HTTPException(status_code=409, detail="FEEDBACK_PROMOTION_ALREADY_RECORDED") from error
        except KeyError as error:
            raise HTTPException(status_code=404, detail="FEEDBACK_CASE_NOT_FOUND") from error
        return {
            "promotion": decision.to_dict(),
            "record": _safe_promotion_response(record),
        }

    @router.get("/cases/{case_id}/promotion-check")
    async def promotion_check(case_id: str, request: Request) -> dict[str, object]:
        _service(request)
        evidence = _resolve_evidence(request, case_id)
        return _promotion_check(request, case_id, evidence)

    return router


def _service(request: Request):
    if not getattr(request.app.state, "feedback_review_ui_enabled", False):
        raise HTTPException(status_code=503, detail="FEEDBACK_REVIEW_DISABLED")
    service = getattr(request.app.state, "feedback_review_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="FEEDBACK_REVIEW_UNAVAILABLE")
    return service


def _resolve_evidence(request: Request, detail_or_case) -> EvidenceBundle:
    resolver = getattr(request.app.state, "feedback_evidence_resolver", None)
    if resolver is None:
        return EvidenceBundle("unavailable", reason_code="evidence_resolver_unavailable")
    if isinstance(detail_or_case, str):
        detail = request.app.state.feedback_review_service.get_case(detail_or_case)
        if detail is None:
            raise HTTPException(status_code=404, detail="FEEDBACK_CASE_NOT_FOUND")
    else:
        detail = detail_or_case
    return resolver.resolve(
        source_version=detail.source_version,
        citation_chunk_ids=tuple(detail.citation_chunk_ids),
    )


def _promotion_check(request: Request, case_id: str, evidence: EvidenceBundle) -> dict[str, object]:
    return _promotion_decision(request, case_id, evidence, "dev").to_dict()


def _promotion_decision(
    request: Request,
    case_id: str,
    evidence: EvidenceBundle,
    target_split: str,
):
    service = request.app.state.feedback_review_service
    detail = service.get_case(case_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="FEEDBACK_CASE_NOT_FOUND")
    repository = getattr(service, "repository", None)
    reviews = repository.list_reviews(case_id) if repository is not None else []
    case = next(
        (item for item in repository.list_cases(limit=1000) if item.case_id == case_id),
        None,
    ) if repository is not None else None
    if case is None:
        raise HTTPException(status_code=404, detail="FEEDBACK_CASE_NOT_FOUND")
    if case.answer_status in {"refused", "error"}:
        decision = evaluate_promotion(
            case,
            evidence_check=None,
            corpus_check=None,
            review_decisions=reviews,
            target_split=target_split,
        )
    else:
        evidence_check = EvidenceCheck(
            valid=evidence.status == "available",
            necessary_chunk_ids=tuple(item.chunk_id for item in evidence.references),
            atomic_points_ok=evidence.status == "available",
            traceable=evidence.status == "available",
            reason=evidence.reason_code,
        )
        decision = evaluate_promotion(
            case,
            evidence_check=evidence_check,
            corpus_check=None,
            review_decisions=reviews,
            target_split=target_split,
        )
    return decision


def _safe_promotion_response(record) -> dict[str, object]:
    return {
        "promotion_id": record.promotion_id,
        "target_set_id": record.target_set_id,
        "target_version": record.target_version,
        "target_split": record.target_split,
        "promotion_reason": record.promotion_reason,
        "manifest_id": record.manifest_id,
        "promoted_at": record.promoted_at,
    }


def _safe_review_response(review: ReviewDecision) -> dict[str, object]:
    return {
        "review_id": review.review_id,
        "reviewer_role": review.reviewer_role,
        "decision": review.decision,
        "evidence_ok": review.evidence_ok,
        "points_ok": review.points_ok,
        "safety_ok": review.safety_ok,
        "note_code": review.note_code if _is_safe_code(review.note_code) else None,
        "review_version": review.review_version if _is_safe_code(review.review_version) else "unknown",
        "reviewed_at": review.reviewed_at,
    }


def _is_safe_code(value: str | None) -> bool:
    return value is not None and _SAFE_CODE.fullmatch(value) is not None


__all__ = ["ReviewSubmissionRequest", "create_feedback_review_router"]
