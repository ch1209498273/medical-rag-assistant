from __future__ import annotations

from dataclasses import replace

from app.feedback.triage import (
    add_random_success_marker,
    deduplicate_cases,
    normalise_question,
    question_fingerprint,
    score_case,
)


def test_question_normalization_and_fingerprint_are_deterministic() -> None:
    first = normalise_question("  透析中低血压？ ")
    second = normalise_question("透析中低血压")

    assert first == second
    assert question_fingerprint(first) == question_fingerprint(second)


def test_score_case_applies_explainable_weights(sample_case) -> None:
    case = replace(
        sample_case,
        answer_status="refused",
        reason_code="not_answered",
        latency_bucket="slow",
    )
    decision = score_case(
        case,
        repeated_followup=True,
        high_risk=True,
        citation_mismatch=True,
        version_mismatch=True,
    )

    assert decision.priority == "P0"
    assert decision.score == 23
    assert {"high_risk", "citation_mismatch", "version_mismatch", "answer_refused", "user_unhelpful", "repeated_followup", "slow_response"}.issubset(decision.reasons)
    assert all("correct" not in reason for reason in decision.reasons)


def test_score_case_maps_non_high_risk_scores_to_p1_p2_p3(sample_case) -> None:
    assert score_case(replace(sample_case, reason_code="not_answered"), high_risk=False).priority == "P2"
    assert score_case(
        replace(sample_case, reason_code="not_answered"),
        repeated_followup=True,
        version_mismatch=True,
    ).priority == "P1"
    assert score_case(replace(sample_case, reason_code=None), high_risk=False).priority == "P3"


def test_duplicate_questions_keep_deterministic_representative_and_event_count(sample_case) -> None:
    duplicate = replace(
        sample_case,
        case_id="case-2",
        message_id_hash="c" * 64,
        session_id_hash="d" * 64,
        question_redacted="透析中低血压 怎么处理",
        event_count=3,
        triage_priority="P1",
        triage_score=7,
    )

    result = deduplicate_cases((sample_case, duplicate))

    assert len(result) == 1
    assert result[0].case_id == "case-2"
    assert result[0].event_count == 3
    assert "duplicate_question" in result[0].triage_reasons


def test_random_success_sampling_is_explicit_and_not_a_correctness_verdict(sample_case) -> None:
    marked = add_random_success_marker(sample_case)

    assert "random_success_sample" in marked.triage_reasons
    assert "correct" not in " ".join(marked.triage_reasons)
