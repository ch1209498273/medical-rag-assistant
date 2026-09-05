"""Offline metrics and root-cause classification for Task 14D.

This module consumes private, text-free facts produced by the four-arm
runner.  It never calls a provider and its returned aggregate contains counts,
rates, and diagnostic labels rather than question/answer/evidence content.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from math import log2
from typing import Any

from app.evaluation.task14d_models import (
    DedupDropFact,
    Task14DCaseFact,
)

ARM_NAMES = ("vector", "lexical", "hybrid", "hybrid_audit")
_ROOT_CAUSE_THRESHOLD = 2
_DEDUP_REASONS = (
    "rrf_content_hash_merge",
    "exact_duplicate",
    "near_duplicate",
    "heading_cap",
    "top20_truncation",
    "active_version_filter",
    "business_eligibility_filter",
)


def aggregate_task14d(cases: Sequence[Task14DCaseFact]) -> dict[str, Any]:
    """Aggregate four-arm retrieval, fusion, dedup, and answer facts.

    Ranking metrics use answerable cases only.  Recall is micro-averaged over
    expected identifiers; precision is averaged over returned Top-6 evidence
    positions; MRR and nDCG are macro-averaged over answerable cases.  The
    private ``_case_signals`` list contains booleans/counts only so the root
    cause classifier can remain deterministic without retaining identifiers.
    """

    normalised_cases = tuple(cases)
    if any(not isinstance(case, Task14DCaseFact) for case in normalised_cases):
        raise TypeError("Task 14D cases are invalid")
    answerable = tuple(case for case in normalised_cases if case.case_type == "answerable")
    no_answer = tuple(case for case in normalised_cases if case.case_type == "no_answer")
    report: dict[str, Any] = {
        "schema_version": "task14d-metrics-v1",
        "case_count": len(normalised_cases),
        "answerable_case_count": len(answerable),
        "no_answer_case_count": len(no_answer),
        "arms": {},
        "rrf": _aggregate_rrf(answerable),
        "dedup": _aggregate_dedup(normalised_cases),
        "dedup_comparison": _aggregate_dedup_comparison(normalised_cases),
    }
    for arm in ARM_NAMES:
        report["arms"][arm] = _aggregate_arm(arm, answerable, no_answer)
    report["_case_signals"] = _build_case_signals(normalised_cases)
    return report


def classify_root_causes(aggregate: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return deterministic, multi-label degradation hypotheses.

    A signal supported by at least two cases is marked ``high`` confidence;
    one case is ``medium``.  With no positive signal the result contains an
    explicit ``cannot_attribute`` label instead of inventing a cause.
    """

    if not isinstance(aggregate, Mapping):
        raise TypeError("Task 14D aggregate is invalid")
    signals = aggregate.get("_case_signals", ())
    if not isinstance(signals, Sequence) or isinstance(signals, (str, bytes)):
        raise TypeError("Task 14D case signals are invalid")
    codes = (
        "vector_recall_missing",
        "lexical_contamination",
        "rrf_rank_displacement",
        "dedup_false_drop",
        "duplicate_noise",
    )
    counts = {
        code: sum(bool(item.get(code)) for item in signals if isinstance(item, Mapping))
        for code in codes
    }
    labels: list[dict[str, Any]] = []
    for code in codes:
        count = counts[code]
        if count == 0:
            continue
        labels.append(
            {
                "code": code,
                "confidence": _confidence(count),
                "supporting_case_count": count,
                "threshold_case_count": _ROOT_CAUSE_THRESHOLD,
                "decision": "investigate_in_separate_tuning_task",
            }
        )
    if not labels:
        labels.append(
            {
                "code": "cannot_attribute",
                "confidence": "low",
                "supporting_case_count": 0,
                "threshold_case_count": _ROOT_CAUSE_THRESHOLD,
                "decision": "keep_baseline_and_collect_more_diagnostics",
            }
        )
    return labels


def _aggregate_arm(
    arm: str,
    answerable: Sequence[Task14DCaseFact],
    no_answer: Sequence[Task14DCaseFact],
) -> dict[str, Any]:
    answerable_facts = [case.arms[arm] for case in answerable]
    no_answer_facts = [case.arms[arm] for case in no_answer]
    gold_count = sum(len(case.expected_ids) for case in answerable)
    recall20_num = sum(
        _relevant_count(case.expected_ids, fact.retrieved_top20)
        for case, fact in zip(answerable, answerable_facts, strict=True)
    )
    recall6_num = sum(
        _relevant_count(case.expected_ids, fact.evidence_top6)
        for case, fact in zip(answerable, answerable_facts, strict=True)
    )
    precision6_num = sum(
        _relevant_count(case.expected_ids, fact.evidence_top6)
        for case, fact in zip(answerable, answerable_facts, strict=True)
    )
    precision6_den = sum(len(fact.evidence_top6) for fact in answerable_facts)
    mrr_values = [
        _reciprocal_rank(case.expected_ids, fact.retrieved_top20)
        for case, fact in zip(answerable, answerable_facts, strict=True)
    ]
    ndcg_values = [
        _ndcg_at_k(case.expected_ids, fact.evidence_top6, 6)
        for case, fact in zip(answerable, answerable_facts, strict=True)
    ]
    answered = sum(fact.answered for fact in answerable_facts)
    grounded = sum(
        bool(fact.answered and _relevant_count(case.expected_ids, fact.evidence_top6))
        for case, fact in zip(answerable, answerable_facts, strict=True)
    )
    no_answer_candidates = sum(bool(fact.retrieved_top20) for fact in no_answer_facts)
    no_answer_evidence = sum(bool(fact.evidence_top6) for fact in no_answer_facts)
    correct_refusal = sum(
        bool(fact.refused and not fact.answered) for fact in no_answer_facts
    )
    return {
        "retrieval": {
            "recall_at_20": _rate(recall20_num, gold_count),
            "recall_at_6": _rate(recall6_num, gold_count),
            "precision_at_6": _rate(precision6_num, precision6_den),
            "mrr_at_20": _mean_rate(mrr_values, len(answerable)),
            "ndcg_at_6": _mean_rate(ndcg_values, len(answerable)),
        },
        "answer": {
            "availability": _rate(answered, len(answerable)),
            "grounded_success": _rate(grounded, len(answerable)),
        },
        "no_answer": {
            "candidate_contamination": _rate(
                no_answer_candidates, len(no_answer)
            ),
            "evidence_contamination": _rate(no_answer_evidence, len(no_answer)),
            "correct_refusal": _rate(correct_refusal, len(no_answer)),
        },
    }


def _aggregate_rrf(answerable: Sequence[Task14DCaseFact]) -> dict[str, Any]:
    membership = Counter()
    relevant_survival_num = 0
    relevant_total = 0
    pushed_out = 0
    displacement_values: list[float] = []
    for case in answerable:
        trace = case.arms["hybrid"].trace
        relevant_total += len(case.expected_ids)
        if trace is None:
            continue
        for fact in trace.fusion:
            membership[fact.membership] += 1
            if _fact_matches(fact, case.expected_ids):
                if fact.fused_rank <= 20:
                    relevant_survival_num += 1
                branch_ranks = [
                    rank
                    for rank in (fact.vector_rank, fact.lexical_rank)
                    if rank is not None
                ]
                if branch_ranks:
                    displacement_values.append(fact.fused_rank - min(branch_ranks))
                if fact.fused_rank > 20:
                    pushed_out += 1
    return {
        "membership_counts": {
            "shared": membership["shared"],
            "vector_only": membership["vector_only"],
            "lexical_only": membership["lexical_only"],
        },
        "relevant_survival": _rate(relevant_survival_num, relevant_total),
        "relevant_rank_displacement": {
            "count": len(displacement_values),
            "mean": _mean(displacement_values),
        },
        "relevant_pushed_out_top20_case_count": pushed_out,
    }


def _aggregate_dedup(cases: Sequence[Task14DCaseFact]) -> dict[str, Any]:
    by_arm: dict[str, Counter[str]] = {arm: Counter() for arm in ARM_NAMES}
    relevant_drop_by_arm: dict[str, int] = {arm: 0 for arm in ARM_NAMES}
    for case in cases:
        for arm in ARM_NAMES:
            trace = case.arms[arm].trace
            if trace is None:
                continue
            for drop in trace.drops:
                by_arm[arm][drop.reason] += 1
                if drop.relevant:
                    relevant_drop_by_arm[arm] += 1
    # The primary drop-rate headline is the production H arm.  V/L do not
    # execute RRF and H-A intentionally skips post-RRF production dedup, so
    # summing all arms would count the same diagnostic trace multiple times
    # and obscure the exact comparison we are trying to explain.
    aggregate_counts = by_arm["hybrid"]
    drop_counts = {reason: aggregate_counts[reason] for reason in _DEDUP_REASONS}
    return {
        "drop_counts": drop_counts,
        "relevant_drop_counts": relevant_drop_by_arm,
        "by_arm": {
            arm: {reason: by_arm[arm][reason] for reason in _DEDUP_REASONS}
            for arm in ARM_NAMES
        },
    }


def _aggregate_dedup_comparison(cases: Sequence[Task14DCaseFact]) -> dict[str, Any]:
    h = [case.arms["hybrid"] for case in cases]
    audit = [case.arms["hybrid_audit"] for case in cases]
    h_evidence = sum(len(fact.evidence_top6) for fact in h)
    audit_evidence = sum(len(fact.evidence_top6) for fact in audit)
    h_candidates = sum(len(fact.retrieved_top20) for fact in h)
    audit_candidates = sum(len(fact.retrieved_top20) for fact in audit)
    return {
        "h_vs_h_a": {
            "candidate_count_delta": h_candidates - audit_candidates,
            "evidence_count_delta": h_evidence - audit_evidence,
            "duplicate_noise_count_h_a": sum(
                fact.duplicate_noise_count for fact in audit
            ),
            "false_drop_case_count": sum(
                _is_false_drop_case(case) for case in cases
            ),
        }
    }


def _build_case_signals(cases: Sequence[Task14DCaseFact]) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    for case in cases:
        vector = case.arms["vector"]
        lexical = case.arms["lexical"]
        hybrid = case.arms["hybrid"]
        audit = case.arms["hybrid_audit"]
        expected = case.expected_ids
        vector_has = _has_expected(expected, vector.retrieved_top20)
        lexical_has = _has_expected(expected, lexical.retrieved_top20)
        hybrid_has = _has_expected(expected, hybrid.retrieved_top20)
        audit_has = _has_expected(expected, audit.retrieved_top20)
        hybrid_trace = hybrid.trace
        dedup_drop = bool(
            hybrid_trace
            and any(
                drop.stage == "post_rrf"
                and drop.reason in {"exact_duplicate", "near_duplicate", "heading_cap"}
                and bool(drop.relevant)
                for drop in hybrid_trace.drops
            )
        )
        signals.append(
            {
                "vector_recall_missing": bool(
                    case.case_type == "answerable"
                    and not vector_has
                    and (lexical_has or hybrid_has)
                ),
                "lexical_contamination": bool(
                    case.case_type == "no_answer"
                    and bool(lexical.retrieved_top20)
                    and not bool(vector.retrieved_top20)
                ),
                "rrf_rank_displacement": bool(
                    case.case_type == "answerable"
                    and vector_has
                    and lexical_has
                    and not hybrid_has
                ),
                "dedup_false_drop": bool(
                    case.case_type == "answerable"
                    and dedup_drop
                    and audit_has
                    and not hybrid_has
                ),
                "duplicate_noise": bool(
                    case.case_type == "answerable"
                    and audit.duplicate_noise_count > 0
                    and not audit_has
                    and (not hybrid_has or len(audit.evidence_top6) == 0)
                ),
            }
        )
    return signals


def _is_false_drop_case(case: Task14DCaseFact) -> bool:
    trace = case.arms["hybrid"].trace
    if trace is None or case.case_type != "answerable":
        return False
    relevant_drop = any(
        drop.stage == "post_rrf"
        and drop.reason in {"exact_duplicate", "near_duplicate", "heading_cap"}
        and bool(drop.relevant)
        for drop in trace.drops
    )
    return relevant_drop and _has_expected(
        case.expected_ids, case.arms["hybrid_audit"].retrieved_top20
    ) and not _has_expected(case.expected_ids, case.arms["hybrid"].retrieved_top20)


def _fact_matches(fact: Any, expected: Sequence[str]) -> bool:
    identifiers = {getattr(fact, "candidate_id", None), getattr(fact, "source_id", None)}
    return bool(identifiers.intersection(expected))


def _has_expected(expected: Sequence[str], observed: Sequence[str]) -> bool:
    return bool(set(expected).intersection(observed))


def _relevant_count(expected: Sequence[str], observed: Sequence[str]) -> int:
    return len(set(expected).intersection(observed))


def _reciprocal_rank(expected: Sequence[str], observed: Sequence[str]) -> float:
    expected_set = set(expected)
    for index, identifier in enumerate(observed, start=1):
        if identifier in expected_set:
            return 1.0 / index
    return 0.0


def _ndcg_at_k(expected: Sequence[str], observed: Sequence[str], k: int) -> float:
    expected_set = set(expected)
    gains = [1.0 if identifier in expected_set else 0.0 for identifier in observed[:k]]
    dcg = sum(gain / log2(index + 2) for index, gain in enumerate(gains))
    ideal_length = min(len(expected_set), k)
    ideal = sum(1.0 / log2(index + 2) for index in range(ideal_length))
    return dcg / ideal if ideal else 0.0


def _rate(numerator: float, denominator: int) -> dict[str, Any]:
    if denominator < 0:
        raise ValueError("metric denominator cannot be negative")
    rate = float(numerator) / denominator if denominator else 0.0
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": round(rate, 6),
    }


def _mean_rate(values: Sequence[float], denominator: int) -> dict[str, Any]:
    numerator = round(sum(values), 6)
    return _rate(numerator, denominator)


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _confidence(count: int) -> str:
    if count >= _ROOT_CAUSE_THRESHOLD:
        return "high"
    if count == 1:
        return "medium"
    return "low"


def _drop_is_relevant(drop: DedupDropFact) -> bool:
    return bool(drop.relevant)


__all__ = ["ARM_NAMES", "aggregate_task14d", "classify_root_causes"]
