from __future__ import annotations

from dataclasses import replace

from app.feedback.models import ReviewDecision
from app.feedback.promotion import (
    EvidenceCheck,
    NegativeCorpusCheck,
    build_promotion_record,
    evaluate_answerable,
    evaluate_no_answer,
    evaluate_promotion,
)


def _review(case_id: str, role: str = "product", decision: str = "approve", review_id: str = "r1") -> ReviewDecision:
    return ReviewDecision(
        review_id=review_id,
        case_id=case_id,
        reviewer_role=role,
        decision=decision,
        evidence_ok=True,
        points_ok=True,
        safety_ok=True,
        note_code="checked",
        review_version="review-v1",
        reviewed_at="2026-09-03T10:00:00+08:00",
    )


def test_answerable_requires_evidence_traceability_and_review(sample_case) -> None:
    evidence = EvidenceCheck(
        valid=True,
        necessary_chunk_ids=("chunk-1",),
        atomic_points_ok=True,
        traceable=True,
    )
    decision = evaluate_answerable(sample_case, evidence_check=evidence, review_decisions=(_review(sample_case.case_id),))
    assert decision.status == "golden_v2_candidate"

    missing = evaluate_answerable(
        sample_case,
        evidence_check=EvidenceCheck(False, (), False, False, "missing_evidence"),
        review_decisions=(_review(sample_case.case_id),),
    )
    assert missing.status == "not_promoted"
    assert "evidence" in " ".join(missing.reasons)


def test_no_answer_requires_full_corpus_negative_check(sample_case) -> None:
    case = replace(sample_case, answer_status="refused", reason_code="not_answered")
    review = _review(case.case_id)
    not_checked = evaluate_no_answer(
        case,
        corpus_check=NegativeCorpusCheck(False, False, None, "top_k_only"),
        review_decisions=(review,),
    )
    assert not_checked.status == "not_promoted"

    checked = evaluate_no_answer(
        case,
        corpus_check=NegativeCorpusCheck(True, True, "2026-standard-manual-v1", None),
        review_decisions=(review,),
    )
    assert checked.status == "golden_v2_candidate"


def test_high_risk_requires_clinical_review_and_disagreement_blocks(sample_case) -> None:
    high_risk = replace(sample_case, triage_priority="P0")
    evidence = EvidenceCheck(True, ("chunk-1",), True, True)
    no_clinical = evaluate_answerable(
        high_risk,
        evidence_check=evidence,
        review_decisions=(_review(high_risk.case_id),),
    )
    assert no_clinical.status == "not_promoted"
    assert "clinical" in " ".join(no_clinical.reasons)

    disputed = evaluate_answerable(
        sample_case,
        evidence_check=evidence,
        review_decisions=(_review(sample_case.case_id, decision="needs_adjudication"),),
    )
    assert disputed.status == "not_promoted"
    assert "adjudication" in " ".join(disputed.reasons)


def test_holdout_cannot_be_tuning_input_and_record_is_traceable(sample_case) -> None:
    case = replace(sample_case, triage_reasons=("tuning_input",))
    decision = evaluate_promotion(
        case,
        evidence_check=EvidenceCheck(True, ("chunk-1",), True, True),
        corpus_check=None,
        review_decisions=(_review(case.case_id),),
        target_split="holdout",
    )
    assert decision.status == "not_promoted"
    assert "holdout" in " ".join(decision.reasons)

    record = build_promotion_record(
        sample_case,
        decision=type(decision)("golden_v2_candidate", ("approved",)),
        target_set_id="golden-v2-dev",
        target_version="v2.0.0",
        target_split="dev",
        manifest_id="manifest-1",
        promoted_at="2026-09-03T10:00:00+08:00",
    )
    assert record.target_set_id == "golden-v2-dev"


def test_redaction_failure_and_helpful_feedback_do_not_promote(sample_case) -> None:
    case = replace(sample_case, redaction_status="review_required")
    decision = evaluate_promotion(
        case,
        evidence_check=EvidenceCheck(True, ("chunk-1",), True, True),
        corpus_check=None,
        review_decisions=(_review(case.case_id),),
        target_split="dev",
    )
    assert decision.status == "not_promoted"
    assert "redaction" in " ".join(decision.reasons)
