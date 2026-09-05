from __future__ import annotations

from app.evaluation.task14d_metrics import aggregate_task14d, classify_root_causes
from app.evaluation.task14d_models import (
    BranchCandidateFact,
    DedupDropFact,
    FusionCandidateFact,
    Task14DArmCaseFact,
    Task14DCaseFact,
    Task14DRetrievalTrace,
)


def _arm(
    arm: str,
    candidates: tuple[str, ...],
    evidence: tuple[str, ...],
    *,
    answered: bool = False,
    refused: bool = True,
    trace: Task14DRetrievalTrace | None = None,
    duplicate_noise_count: int = 0,
) -> Task14DArmCaseFact:
    return Task14DArmCaseFact(
        arm=arm,
        retrieved_top20=candidates,
        evidence_top6=evidence,
        answered=answered,
        refused=refused,
        trace=trace,
        duplicate_noise_count=duplicate_noise_count,
    )


def test_aggregate_reports_standard_ranking_and_no_answer_metrics() -> None:
    cases = (
        Task14DCaseFact(
            case_id="c1",
            case_type="answerable",
            expected_ids=("gold-a",),
            arms={
                "vector": _arm("vector", ("gold-a", "noise"), ("gold-a",)),
                "lexical": _arm("lexical", ("gold-a",), ("gold-a",)),
                "hybrid": _arm("hybrid", ("gold-a",), ("gold-a",)),
                "hybrid_audit": _arm("hybrid_audit", ("gold-a",), ("gold-a",)),
            },
        ),
        Task14DCaseFact(
            case_id="c2",
            case_type="answerable",
            expected_ids=("gold-b",),
            arms={
                "vector": _arm("vector", ("noise", "gold-b"), ("noise",)),
                "lexical": _arm("lexical", ("gold-b",), ("gold-b",)),
                "hybrid": _arm("hybrid", ("noise",), ("noise",)),
                "hybrid_audit": _arm("hybrid_audit", ("noise",), ("noise",)),
            },
        ),
        Task14DCaseFact(
            case_id="n1",
            case_type="no_answer",
            expected_ids=(),
            arms={
                "vector": _arm("vector", ("noise",), (), refused=True),
                "lexical": _arm("lexical", ("noise",), (), refused=True),
                "hybrid": _arm("hybrid", (), (), refused=True),
                "hybrid_audit": _arm("hybrid_audit", (), (), refused=True),
            },
        ),
    )

    report = aggregate_task14d(cases)
    vector = report["arms"]["vector"]

    assert report["case_count"] == 3
    assert report["answerable_case_count"] == 2
    assert vector["retrieval"]["recall_at_20"] == {
        "numerator": 2,
        "denominator": 2,
        "rate": 1.0,
    }
    assert vector["retrieval"]["recall_at_6"]["rate"] == 0.5
    assert vector["retrieval"]["mrr_at_20"]["rate"] == 0.75
    assert vector["retrieval"]["ndcg_at_6"]["rate"] == 0.5
    assert vector["no_answer"]["candidate_contamination"]["numerator"] == 1
    assert vector["no_answer"]["correct_refusal"]["rate"] == 1.0


def test_aggregate_explains_rrf_membership_rank_displacement_and_dedup() -> None:
    trace = Task14DRetrievalTrace(
        vector=(
            BranchCandidateFact("gold", "vector", 1, 0.9, "hv", True),
            BranchCandidateFact("v-only", "vector", 2, 0.8, "vv", False),
        ),
        lexical=(
            BranchCandidateFact("gold", "lexical", 1, 0.8, "hg", True),
            BranchCandidateFact("l-only", "lexical", 2, 0.7, "ll", False),
        ),
        fusion=(
            FusionCandidateFact("gold", "shared", 1, 1, 2, 0.03, True),
            FusionCandidateFact("v-only", "vector_only", 2, None, 1, 0.02, False),
            FusionCandidateFact("l-only", "lexical_only", None, 2, 3, 0.01, False),
        ),
        drops=(
            DedupDropFact("gold-copy", "near_duplicate", "post_rrf", "gold", True),
        ),
    )
    case = Task14DCaseFact(
        case_id="c1",
        case_type="answerable",
        expected_ids=("gold",),
        arms={
            "vector": _arm("vector", ("gold",), ("gold",), trace=trace),
            "lexical": _arm("lexical", ("gold",), ("gold",), trace=trace),
            "hybrid": _arm("hybrid", ("v-only", "gold"), ("gold",), trace=trace),
            "hybrid_audit": _arm(
                "hybrid_audit", ("v-only", "gold"), ("gold",), trace=trace
            ),
        },
    )

    report = aggregate_task14d((case,))
    rrf = report["rrf"]
    assert rrf["membership_counts"] == {
        "shared": 1,
        "vector_only": 1,
        "lexical_only": 1,
    }
    assert rrf["relevant_survival"]["rate"] == 1.0
    assert rrf["relevant_rank_displacement"]["mean"] == 1.0
    assert report["dedup"]["drop_counts"]["near_duplicate"] == 1
    assert report["dedup_comparison"]["h_vs_h_a"]["evidence_count_delta"] == 0


def test_root_cause_classifier_uses_two_case_signal_and_can_multilabel() -> None:
    cases: list[Task14DCaseFact] = []
    for index in range(2):
        cases.append(
            Task14DCaseFact(
                case_id=f"vector-miss-{index}",
                case_type="answerable",
                expected_ids=(f"gold-v-{index}",),
                arms={
                    "vector": _arm("vector", ("noise",), ()),
                    "lexical": _arm("lexical", (f"gold-v-{index}",), (f"gold-v-{index}",)),
                    "hybrid": _arm("hybrid", (f"gold-v-{index}",), (f"gold-v-{index}",)),
                    "hybrid_audit": _arm("hybrid_audit", (f"gold-v-{index}",), (f"gold-v-{index}",)),
                },
            )
        )
        cases.append(
            Task14DCaseFact(
                case_id=f"lexical-noise-{index}",
                case_type="no_answer",
                expected_ids=(),
                arms={
                    "vector": _arm("vector", (), (), refused=True),
                    "lexical": _arm("lexical", ("noise",), (), refused=True),
                    "hybrid": _arm("hybrid", (), (), refused=True),
                    "hybrid_audit": _arm("hybrid_audit", (), (), refused=True),
                },
            )
        )
        trace = Task14DRetrievalTrace(
            drops=(
                DedupDropFact(
                    f"gold-dropped-{index}",
                    "near_duplicate",
                    "post_rrf",
                    "noise",
                    True,
                ),
            )
        )
        cases.append(
            Task14DCaseFact(
                case_id=f"dedup-{index}",
                case_type="answerable",
                expected_ids=(f"gold-d-{index}",),
                arms={
                    "vector": _arm("vector", (f"gold-d-{index}",), (f"gold-d-{index}",)),
                    "lexical": _arm("lexical", (f"gold-d-{index}",), (f"gold-d-{index}",)),
                    "hybrid": _arm("hybrid", ("noise",), (), trace=trace),
                    "hybrid_audit": _arm("hybrid_audit", (f"gold-d-{index}",), (f"gold-d-{index}",), trace=trace),
                },
            )
        )
        cases.append(
            Task14DCaseFact(
                case_id=f"rrf-{index}",
                case_type="answerable",
                expected_ids=(f"gold-r-{index}",),
                arms={
                    "vector": _arm("vector", (f"gold-r-{index}",), (f"gold-r-{index}",)),
                    "lexical": _arm("lexical", (f"gold-r-{index}",), (f"gold-r-{index}",)),
                    "hybrid": _arm("hybrid", ("noise",), (), refused=True),
                    "hybrid_audit": _arm("hybrid_audit", ("noise",), (), refused=True),
                },
            )
        )
    # A duplicate-heavy audit pair without any new relevant evidence.
    for index in range(2):
        cases.append(
            Task14DCaseFact(
                case_id=f"dup-noise-{index}",
                case_type="answerable",
                expected_ids=(f"gold-n-{index}",),
                arms={
                    "vector": _arm("vector", (f"gold-n-{index}",), (f"gold-n-{index}",)),
                    "lexical": _arm("lexical", (f"gold-n-{index}",), (f"gold-n-{index}",)),
                    "hybrid": _arm("hybrid", (f"gold-n-{index}",), (f"gold-n-{index}",)),
                    "hybrid_audit": _arm("hybrid_audit", ("dup-1", "dup-2"), (), refused=True, duplicate_noise_count=2),
                },
            )
        )

    labels = classify_root_causes(aggregate_task14d(tuple(cases)))
    by_code = {item["code"]: item for item in labels}
    assert by_code["vector_recall_missing"]["supporting_case_count"] >= 2
    assert by_code["lexical_contamination"]["supporting_case_count"] >= 2
    assert by_code["rrf_rank_displacement"]["supporting_case_count"] >= 2
    assert by_code["dedup_false_drop"]["supporting_case_count"] >= 2
    assert by_code["duplicate_noise"]["supporting_case_count"] >= 2
    assert all(item["confidence"] in {"high", "medium", "low"} for item in labels)
