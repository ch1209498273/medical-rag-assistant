"""Explicit evidence and review gates for incremental golden-v2 promotion."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from app.feedback.models import FeedbackCase, PromotionRecord, ReviewDecision


@dataclass(frozen=True)
class EvidenceCheck:
    valid: bool
    necessary_chunk_ids: tuple[str, ...]
    atomic_points_ok: bool
    traceable: bool
    reason: str | None = None


@dataclass(frozen=True)
class NegativeCorpusCheck:
    full_corpus_checked: bool
    no_supporting_evidence: bool
    corpus_version: str | None
    reason: str | None = None


@dataclass(frozen=True)
class PromotionDecision:
    status: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"status": self.status, "reasons": list(self.reasons)}


def _blocked(*reasons: str) -> PromotionDecision:
    return PromotionDecision("not_promoted", tuple(dict.fromkeys(reasons)))


def _review_gate(case: FeedbackCase, reviews: Sequence[ReviewDecision]) -> PromotionDecision:
    if not reviews:
        return _blocked("review_missing")
    if any(review.decision == "needs_adjudication" for review in reviews):
        return _blocked("review_adjudication_required")
    if any(review.decision == "reject" for review in reviews):
        return _blocked("review_rejected")
    approvals = [
        review
        for review in reviews
        if review.decision == "approve"
        and review.evidence_ok is True
        and review.points_ok is True
        and review.safety_ok is True
    ]
    if not approvals:
        return _blocked("review_approval_missing")
    if case.triage_priority in {"P0", "P1"} and not any(
        review.reviewer_role == "clinical_reviewer" for review in approvals
    ):
        return _blocked("clinical_review_required")
    return PromotionDecision("golden_v2_candidate", ("review_approved",))


def evaluate_answerable(
    case: FeedbackCase,
    *,
    evidence_check: EvidenceCheck,
    review_decisions: Sequence[ReviewDecision],
) -> PromotionDecision:
    if case.redaction_status != "passed":
        return _blocked("redaction_not_passed")
    if not evidence_check.valid:
        return _blocked(evidence_check.reason or "evidence_invalid")
    if not evidence_check.necessary_chunk_ids:
        return _blocked("necessary_evidence_missing")
    if not evidence_check.atomic_points_ok:
        return _blocked("atomic_points_missing")
    if not evidence_check.traceable:
        return _blocked("traceability_missing")
    if not case.source_version:
        return _blocked("source_version_missing")
    return _review_gate(case, review_decisions)


def evaluate_no_answer(
    case: FeedbackCase,
    *,
    corpus_check: NegativeCorpusCheck,
    review_decisions: Sequence[ReviewDecision],
) -> PromotionDecision:
    if case.redaction_status != "passed":
        return _blocked("redaction_not_passed")
    if not corpus_check.full_corpus_checked:
        return _blocked(corpus_check.reason or "full_corpus_check_missing")
    if not corpus_check.no_supporting_evidence:
        return _blocked("supporting_evidence_found")
    if not corpus_check.corpus_version:
        return _blocked("corpus_version_missing")
    return _review_gate(case, review_decisions)


def evaluate_promotion(
    case: FeedbackCase,
    *,
    evidence_check: EvidenceCheck | None,
    corpus_check: NegativeCorpusCheck | None,
    review_decisions: Sequence[ReviewDecision],
    target_split: str,
) -> PromotionDecision:
    if target_split not in {"dev", "holdout"}:
        return _blocked("invalid_target_split")
    if target_split == "holdout" and any(
        reason in {"tuning_input", "used_for_tuning"} for reason in case.triage_reasons
    ):
        return _blocked("holdout_cannot_be_tuning_input")
    is_no_answer = case.answer_status in {"refused", "error"} or case.reason_code in {
        "no_answer",
        "not_answered",
        "ANSWER_NOT_VERIFIABLE",
    }
    if is_no_answer:
        if corpus_check is None:
            return _blocked("negative_corpus_check_missing")
        return evaluate_no_answer(
            case, corpus_check=corpus_check, review_decisions=review_decisions
        )
    if evidence_check is None:
        return _blocked("evidence_check_missing")
    return evaluate_answerable(
        case, evidence_check=evidence_check, review_decisions=review_decisions
    )


def build_promotion_record(
    case: FeedbackCase,
    decision: PromotionDecision,
    *,
    target_set_id: str,
    target_version: str,
    target_split: str,
    manifest_id: str,
    promoted_at: str,
) -> PromotionRecord:
    if decision.status not in {"golden_v2_candidate", "golden_v2"}:
        raise ValueError("only a promotion decision can create a record")
    if target_split not in {"dev", "holdout"}:
        raise ValueError("target_split is invalid")
    raw_id = f"{case.case_id}|{target_set_id}|{target_version}|{target_split}|{manifest_id}"
    promotion_id = f"pr_{hashlib.sha256(raw_id.encode('utf-8')).hexdigest()[:32]}"
    return PromotionRecord(
        promotion_id=promotion_id,
        case_id=case.case_id,
        target_set_id=target_set_id,
        target_version=target_version,
        target_split=target_split,
        promotion_reason=";".join(decision.reasons) or "approved",
        manifest_id=manifest_id,
        promoted_at=promoted_at,
    )
