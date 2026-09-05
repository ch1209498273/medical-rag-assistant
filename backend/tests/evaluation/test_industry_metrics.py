from __future__ import annotations

import math
from pathlib import Path

import pytest

from app.evaluation.industry_metrics import (
    binary_rate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

EVALUATION_DOC = Path(__file__).resolve().parents[3] / "docs" / "evaluation.md"


def test_binary_rate_contains_wilson_95_ci() -> None:
    metric = binary_rate([True, True, False, False])
    assert metric.numerator == 2
    assert metric.denominator == 4
    assert metric.value == 0.5
    assert 0.15 < metric.ci95_low < 0.16
    assert 0.84 < metric.ci95_high < 0.85


def test_binary_rate_uses_zero_interval_for_empty_sample() -> None:
    metric = binary_rate([])
    assert metric.numerator == 0
    assert metric.denominator == 0
    assert metric.value == 0.0
    assert metric.ci95_low == 0.0
    assert metric.ci95_high == 0.0


def test_retrieval_metrics_deduplicate_ids_and_respect_k() -> None:
    assert recall_at_k(["a", "a", "b"], ["a", "c"], 2) == 0.5
    assert precision_at_k(["a", "a", "b"], ["a", "c"], 2) == 0.5
    assert reciprocal_rank(["x", "b", "a"], ["a"]) == 1 / 3
    assert ndcg_at_k(["a", "b"], {"a": 2, "b": 1}, 2) == 1.0


def test_retrieval_metrics_handle_empty_or_invalid_inputs() -> None:
    assert recall_at_k([], ["a"], 6) == 0.0
    assert precision_at_k(["a"], [], 6) == 0.0
    assert reciprocal_rank(["a"], []) == 0.0
    assert ndcg_at_k(["a"], {}, 6) == 0.0
    with pytest.raises(ValueError, match="k"):
        recall_at_k(["a"], ["a"], 0)


def test_ndcg_returns_zero_for_zero_ideal_gain() -> None:
    assert ndcg_at_k(["a", "b"], {"a": 0, "b": 0}, 2) == 0.0
    assert math.isfinite(ndcg_at_k(["a"], {"a": 2}, 1))


def test_public_metric_dictionary_names_denominator_and_limitations() -> None:
    """Public metric claims must be auditable instead of bare percentages."""

    text = EVALUATION_DOC.read_text(encoding="utf-8")
    assert "## 指标字典（分子、分母、置信区间与限制）" in text
    for metric in (
        "正式回答可用性",
        "无答案拒答率",
        "引用可见率",
        "检索 Recall@6",
    ):
        assert metric in text
    assert text.count("分子") >= 3
    assert text.count("分母") >= 3
    assert text.count("限制") >= 4
    assert "C1/C2 与 9F-B 使用不同评测协议，不能横向比较" in text
