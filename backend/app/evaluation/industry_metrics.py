"""Pure, provider-independent metrics used by the Task 12A evaluator."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

_WILSON_Z_95 = 1.959963984540054


@dataclass(frozen=True)
class RateMetric:
    """A rate with an explicit sample size and Wilson 95% interval."""

    numerator: int
    denominator: int
    value: float
    ci95_low: float
    ci95_high: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalMetricReport:
    recall_at_20: RateMetric
    recall_at_6: RateMetric
    precision_at_6: RateMetric
    mrr_at_20: float
    mrr_denominator: int
    ndcg_at_6: float | None
    ndcg_denominator: int
    negative_evidence_rate: RateMetric

    def to_dict(self) -> dict[str, object]:
        return {
            "recall_at_20": self.recall_at_20.to_dict(),
            "recall_at_6": self.recall_at_6.to_dict(),
            "precision_at_6": self.precision_at_6.to_dict(),
            "mrr_at_20": self.mrr_at_20,
            "mrr_denominator": self.mrr_denominator,
            "ndcg_at_6": self.ndcg_at_6,
            "ndcg_denominator": self.ndcg_denominator,
            "negative_evidence_rate": self.negative_evidence_rate.to_dict(),
        }


@dataclass(frozen=True)
class GenerationMetricReport:
    answer_availability: RateMetric
    grounded_answer_success: RateMetric
    answerable_over_refusal: RateMetric
    citation_precision: RateMetric
    citation_recall: RateMetric

    def to_dict(self) -> dict[str, object]:
        return {
            "answer_availability": self.answer_availability.to_dict(),
            "grounded_answer_success": self.grounded_answer_success.to_dict(),
            "answerable_over_refusal": self.answerable_over_refusal.to_dict(),
            "citation_precision": self.citation_precision.to_dict(),
            "citation_recall": self.citation_recall.to_dict(),
        }


@dataclass(frozen=True)
class SafetyMetricReport:
    unsafe_answer_rate: RateMetric
    runtime_citation_visibility: RateMetric
    no_answer_refusal: RateMetric

    def to_dict(self) -> dict[str, object]:
        return {
            "unsafe_answer_rate": self.unsafe_answer_rate.to_dict(),
            "runtime_citation_visibility": self.runtime_citation_visibility.to_dict(),
            "no_answer_refusal": self.no_answer_refusal.to_dict(),
        }


@dataclass(frozen=True)
class IndustryMetricReport:
    protocol_version: str
    retrieval: RetrievalMetricReport
    generation: GenerationMetricReport
    safety: SafetyMetricReport

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "retrieval": self.retrieval.to_dict(),
            "generation": self.generation.to_dict(),
            "safety": self.safety.to_dict(),
        }


def wilson_interval(
    successes: int,
    trials: int,
    z: float = _WILSON_Z_95,
) -> tuple[float, float]:
    """Return a Wilson score interval for a binomial proportion."""

    _validate_counts(successes, trials)
    if trials == 0:
        return 0.0, 0.0
    if not isinstance(z, (int, float)) or isinstance(z, bool) or not math.isfinite(float(z)) or z < 0:
        raise ValueError("z must be a finite non-negative number")
    z_value = float(z)
    n = float(trials)
    p = float(successes) / n
    z2 = z_value * z_value
    denominator = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denominator
    margin = z_value * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def rate_metric(successes: int, trials: int) -> RateMetric:
    """Build a rate metric from integer counts."""

    _validate_counts(successes, trials)
    value = successes / trials if trials else 0.0
    low, high = wilson_interval(successes, trials)
    return RateMetric(
        numerator=successes,
        denominator=trials,
        value=value,
        ci95_low=low,
        ci95_high=high,
    )


def binary_rate(outcomes: Sequence[bool]) -> RateMetric:
    """Build a rate metric from boolean case outcomes."""

    if isinstance(outcomes, (str, bytes)):
        raise TypeError("outcomes must be a sequence of booleans")
    values = list(outcomes)
    if any(not isinstance(item, bool) for item in values):
        raise TypeError("outcomes must contain only booleans")
    return rate_metric(sum(values), len(values))


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
) -> float:
    """Return item recall at rank ``k`` after stable ID de-duplication."""

    _validate_k(k)
    relevant = _normalise_ids(relevant_ids)
    if not relevant:
        return 0.0
    retrieved = _normalise_ids(retrieved_ids)[:k]
    return len(set(retrieved) & set(relevant)) / len(set(relevant))


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
) -> float:
    """Return item precision at rank ``k`` after stable ID de-duplication."""

    _validate_k(k)
    retrieved = _normalise_ids(retrieved_ids)[:k]
    if not retrieved:
        return 0.0
    relevant = set(_normalise_ids(relevant_ids))
    return len(set(retrieved) & relevant) / len(retrieved)


def reciprocal_rank(
    retrieved_ids: Sequence[str], relevant_ids: Sequence[str]
) -> float:
    """Return reciprocal rank of the first relevant retrieved item."""

    relevant = set(_normalise_ids(relevant_ids))
    if not relevant:
        return 0.0
    for rank, item in enumerate(_normalise_ids(retrieved_ids), start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    graded_relevance: Mapping[str, float],
    k: int,
) -> float:
    """Return normalised discounted cumulative gain at rank ``k``."""

    _validate_k(k)
    if not isinstance(graded_relevance, Mapping):
        raise TypeError("graded_relevance must be a mapping")
    relevance: dict[str, float] = {}
    for key, value in graded_relevance.items():
        if not isinstance(key, str) or not key.strip():
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("graded relevance values must be finite numbers")
        relevance[key.strip()] = max(0.0, float(value))
    ranked = _normalise_ids(retrieved_ids)[:k]
    if not relevance:
        return 0.0
    dcg = sum(
        _gain(relevance.get(item, 0.0)) / math.log2(rank + 1)
        for rank, item in enumerate(ranked, start=1)
    )
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(_gain(value) / math.log2(rank + 1) for rank, value in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def _gain(value: float) -> float:
    return (2.0**value) - 1.0


def _normalise_ids(values: Sequence[str]) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise TypeError("IDs must be a sequence, not a string")
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        item = value.strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")


def _validate_counts(successes: int, trials: int) -> None:
    if (
        isinstance(successes, bool)
        or not isinstance(successes, int)
        or isinstance(trials, bool)
        or not isinstance(trials, int)
        or successes < 0
        or trials < 0
        or successes > trials
    ):
        raise ValueError("successes and trials must be integers with 0 <= successes <= trials")


__all__ = [
    "GenerationMetricReport",
    "IndustryMetricReport",
    "RateMetric",
    "RetrievalMetricReport",
    "SafetyMetricReport",
    "binary_rate",
    "ndcg_at_k",
    "precision_at_k",
    "rate_metric",
    "recall_at_k",
    "reciprocal_rank",
    "wilson_interval",
]
