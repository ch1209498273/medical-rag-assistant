"""Deterministic deduplication and review prioritisation."""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, replace

from app.feedback.models import FeedbackCase

TRIAGE_WEIGHTS = {
    "high_risk": 5,
    "version_mismatch": 4,
    "citation_mismatch": 4,
    "answer_failure": 4,
    "user_unhelpful": 3,
    "repeated_followup": 2,
    "slow_response": 1,
}
_PRIORITY_RANK = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
_UNHELPFUL_REASONS = frozenset({"not_answered", "missing_step"})
_SLOW_BUCKETS = frozenset({"slow", "gt_30s", "gt_60s", "over_30s"})


@dataclass(frozen=True)
class TriageDecision:
    priority: str
    score: int
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "priority": self.priority,
            "score": self.score,
            "reasons": list(self.reasons),
        }


def normalise_question(text: str) -> str:
    """Canonicalise a question without embeddings or network calls."""

    if not isinstance(text, str):
        raise TypeError("question must be text")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    compact = "".join(char for char in normalized if char.isalnum())
    return compact


def question_fingerprint(text: str) -> str:
    return hashlib.sha256(normalise_question(text).encode("utf-8")).hexdigest()


def score_case(
    case: FeedbackCase,
    *,
    repeated_followup: bool = False,
    high_risk: bool = False,
    citation_mismatch: bool = False,
    version_mismatch: bool = False,
) -> TriageDecision:
    score = 0
    reasons: list[str] = []

    def add(reason: str, weight_key: str) -> None:
        nonlocal score
        if reason not in reasons:
            reasons.append(reason)
            score += TRIAGE_WEIGHTS[weight_key]

    if high_risk:
        add("high_risk", "high_risk")
    if version_mismatch or case.reason_code == "version_mismatch":
        add("version_mismatch", "version_mismatch")
    if citation_mismatch or case.reason_code == "citation_mismatch":
        add("citation_mismatch", "citation_mismatch")
    if case.answer_status in {"error", "refused"}:
        add(
            "answer_error" if case.answer_status == "error" else "answer_refused",
            "answer_failure",
        )
    elif case.reason_code in {"ANSWER_NOT_VERIFIABLE", "unverifiable"}:
        add("answer_unverifiable", "answer_failure")
    if case.reason_code in _UNHELPFUL_REASONS or case.reason_code == "not_helpful":
        add("user_unhelpful", "user_unhelpful")
    if repeated_followup:
        add("repeated_followup", "repeated_followup")
    if case.reason_code == "too_slow" or case.latency_bucket in _SLOW_BUCKETS:
        add("slow_response", "slow_response")

    if high_risk:
        priority = "P0"
    elif score >= 7:
        priority = "P1"
    elif score >= 3:
        priority = "P2"
    else:
        priority = "P3"
    return TriageDecision(priority=priority, score=score, reasons=tuple(reasons))


def add_random_success_marker(case: FeedbackCase) -> FeedbackCase:
    """Mark a sampled successful case without implying correctness."""

    reasons = list(case.triage_reasons)
    if "random_success_sample" not in reasons:
        reasons.append("random_success_sample")
    return replace(case, triage_reasons=tuple(reasons))


def deduplicate_cases(cases: Sequence[FeedbackCase]) -> tuple[FeedbackCase, ...]:
    """Collapse normalized-question duplicates into stable representatives."""

    groups: dict[str, list[FeedbackCase]] = {}
    for case in cases:
        groups.setdefault(question_fingerprint(case.question_redacted), []).append(case)

    representatives: list[FeedbackCase] = []
    for group in groups.values():
        representative = min(
            group,
            key=lambda item: (
                _PRIORITY_RANK[item.triage_priority],
                -item.triage_score,
                item.case_id,
            ),
        )
        total_events = sum(item.event_count for item in group)
        reasons = list(representative.triage_reasons)
        if len(group) > 1 and "duplicate_question" not in reasons:
            reasons.append("duplicate_question")
        representatives.append(
            replace(
                representative,
                event_count=total_events,
                triage_reasons=tuple(reasons),
            )
        )

    representatives.sort(
        key=lambda item: (
            _PRIORITY_RANK[item.triage_priority],
            -item.triage_score,
            item.case_id,
        )
    )
    return tuple(representatives)
