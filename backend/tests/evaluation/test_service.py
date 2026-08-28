from __future__ import annotations

from pathlib import Path

import pytest
from app.evaluation.service import (
    EvaluationCaseResult,
    EvaluationService,
    calculate_report,
)
from app.providers.minimax import EvaluationCase, EvaluationGeneration
from app.rag.models import ChatEvent


def valid_case(source_id: str = "S1") -> EvaluationCase:
    return EvaluationCase(
        question="问题",
        reference_points=["要点"],
        source_ids=[source_id],
        type="fact",
    )


class FakeGenerator:
    result: EvaluationGeneration

    async def generate(self, context):
        return self.result


@pytest.mark.asyncio
async def test_generation_filters_cases_with_unknown_sources(tmp_path: Path):
    generator = FakeGenerator()
    generator.result = EvaluationGeneration(
        cases=(valid_case("S1"), valid_case("MISSING")),
        provider="minimax",
        model_id="MiniMax-M2.7",
    )
    service = EvaluationService(
        generator=generator,
        source_provider=lambda: {"S1": {"text": "source"}},
        storage_dir=tmp_path,
    )

    result = await service.generate(count=1)

    assert len(result.cases) == 1
    assert result.cases[0].source_ids == ["S1"]


@pytest.mark.asyncio
async def test_generation_rejects_thirty_cases_without_exact_type_quota_before_persisting(
    tmp_path: Path,
):
    case_types = (
        "fact",
        "fact",
        "fact",
        "fact",
        "fact",
        "fact",
        "procedure",
        "procedure",
        "procedure",
        "procedure",
        "procedure",
        "condition",
        "condition",
        "condition",
        "condition",
        "condition",
        "prohibition",
        "prohibition",
        "prohibition",
        "prohibition",
        "prohibition",
        "synthesis",
        "synthesis",
        "synthesis",
        "synthesis",
        "synthesis",
        "no_answer",
        "no_answer",
        "no_answer",
        "no_answer",
    )
    cases = tuple(
        EvaluationCase(
            question=f"问题 {index}",
            reference_points=[f"要点 {index}"],
            source_ids=["S1"],
            type=case_type,
        )
        for index, case_type in enumerate(case_types, start=1)
    )
    generator = FakeGenerator()
    generator.result = EvaluationGeneration(
        cases=cases,
        provider="minimax",
        model_id="MiniMax-M2.7",
    )
    service = EvaluationService(
        generator=generator,
        source_provider=lambda: {"S1": {"text": "source"}},
        storage_dir=tmp_path,
    )

    with pytest.raises(
        ValueError,
        match="^evaluation generation did not satisfy exact type quotas$",
    ):
        await service.generate(count=30)

    assert list(tmp_path.glob("*.json")) == []


def test_report_calculates_acceptance_metrics():
    report = calculate_report(
        [
            {
                "answered": True,
                "reference_points_covered": True,
                "citation_valid": True,
                "source_visibility": True,
                "refused": False,
                "no_answer": False,
            },
            {
                "answered": False,
                "reference_points_covered": False,
                "citation_valid": True,
                "source_visibility": False,
                "refused": True,
                "no_answer": True,
            },
            {
                "answered": True,
                "reference_points_covered": True,
                "citation_valid": False,
                "source_visibility": False,
                "refused": False,
                "no_answer": False,
            },
        ]
    )

    assert report.total == 3
    assert report.citation_accuracy == pytest.approx(2 / 3)
    assert report.answer_availability_rate == pytest.approx(2 / 3)
    assert report.runtime_citation_visibility_rate == pytest.approx(1 / 2)


class EventRagService:
    def __init__(self, events: list[ChatEvent]) -> None:
        self.events = events

    async def stream(self, question: str):
        del question
        for event in self.events:
            yield event


async def run_case(tmp_path: Path, case: EvaluationCase, *events: ChatEvent):
    service = EvaluationService(
        generator=object(),
        rag_service=EventRagService(list(events)),
        storage_dir=tmp_path,
    )
    return await service._run_case("1", case)


@pytest.mark.asyncio
async def test_paraphrase_is_generation_available_and_source_visible_without_reference_match(
    tmp_path: Path,
):
    case = EvaluationCase(
        question="请问请假怎么申请？",
        reference_points=["员工必须先填写纸质申请单"],
        source_ids=["S1"],
        type="procedure",
    )
    result = await run_case(
        tmp_path,
        case,
        ChatEvent(type="answer_delta", data={"text": "先提交申请，再等待负责人审批。", "source_ids": ["S1"]}),
        ChatEvent(
            type="final",
            data={
                "refused": False,
                "citations": [{"reference_id": "S1"}],
            },
        ),
    )

    assert result.answered is True
    assert result.reference_points_covered is False
    assert result.generation_available is True
    assert result.source_visibility is True
    assert result.reason_code is None


@pytest.mark.asyncio
async def test_run_case_captures_private_retrieval_diagnostics_without_raw_content(
    tmp_path: Path,
):
    result = await run_case(
        tmp_path,
        valid_case(),
        ChatEvent(type="final", data={"refused": True, "reason_code": "MODEL_REFUSED", "citations": [], "retrieval_diagnostics": {
            "candidate_count": 4,
            "evidence_count": 2,
            "highest_rerank_score": 0.34,
            "status": "refused",
            "reason_code": "INSUFFICIENT_EVIDENCE",
        }}),
    )

    assert result.candidate_count == 4
    assert result.evidence_count == 2
    assert result.highest_rerank_score == pytest.approx(0.34)
    assert result.retrieval_status == "refused"
    assert result.retrieval_reason_code == "INSUFFICIENT_EVIDENCE"


@pytest.mark.asyncio
async def test_explicit_model_refusal_is_generation_available_and_preserves_no_answer_refusal(
    tmp_path: Path,
):
    result = await run_case(
        tmp_path,
        EvaluationCase(
            question="没有资料的问题",
            reference_points=["不存在的制度"],
            source_ids=["S1"],
            type="no_answer",
        ),
        ChatEvent(
            type="final",
            data={"refused": True, "reason_code": "MODEL_REFUSED", "citations": []},
        ),
    )

    assert result.generation_available is True
    assert result.refused is True
    assert result.answered is False
    assert result.no_answer is True
    assert result.citation_valid is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("events", "reason_code"),
    [
        (
            [
                ChatEvent(
                    type="error",
                    data={"stage": "generating", "reason_code": "GENERATION_UNAVAILABLE"},
                )
            ],
            "GENERATION_UNAVAILABLE",
        ),
        (
            [
                ChatEvent(
                    type="error",
                    data={"stage": "retrieving", "reason_code": "RETRIEVAL_UNAVAILABLE"},
                )
            ],
            "RETRIEVAL_UNAVAILABLE",
        ),
        (
            [
                ChatEvent(
                    type="final",
                    data={"refused": False, "citations": []},
                )
            ],
            "RAG_UNAVAILABLE",
        ),
        (
            [
                ChatEvent(
                    type="final",
                    data={
                        "refused": True,
                        "reason_code": "ANSWER_NOT_VERIFIABLE",
                        "citations": [],
                    },
                )
            ],
            "ANSWER_NOT_VERIFIABLE",
        ),
    ],
)
async def test_generation_unavailable_for_errors_empty_stream_and_malformed_answer(
    tmp_path: Path,
    events: list[ChatEvent],
    reason_code: str,
):
    result = await run_case(tmp_path, valid_case(), *events)

    assert result.generation_available is False
    assert result.reason_code == reason_code


@pytest.mark.asyncio
@pytest.mark.parametrize("reason_code", [None, "INSUFFICIENT_EVIDENCE", "TIMEOUT", "UNKNOWN_FAILURE"])
async def test_refusal_requires_explicit_model_refused_reason(
    tmp_path: Path,
    reason_code: str | None,
):
    data: dict[str, object] = {"refused": True, "citations": []}
    if reason_code is not None:
        data["reason_code"] = reason_code

    result = await run_case(
        tmp_path,
        valid_case(),
        ChatEvent(type="final", data=data),
    )

    assert result.generation_available is False


@pytest.mark.asyncio
@pytest.mark.parametrize("reason_code", ["TIMEOUT", "UNKNOWN_FAILURE", "INSUFFICIENT_EVIDENCE"])
async def test_answer_with_any_non_success_reason_is_not_generation_available(
    tmp_path: Path,
    reason_code: str,
):
    result = await run_case(
        tmp_path,
        valid_case(),
        ChatEvent(
            type="answer_delta",
            data={"text": "答案", "source_ids": ["S1"]},
        ),
        ChatEvent(
            type="final",
            data={
                "refused": False,
                "reason_code": reason_code,
                "citations": [{"reference_id": "S1"}],
            },
        ),
    )

    assert result.generation_available is False


def test_coerced_legacy_results_use_strict_generation_availability_rule():
    report = calculate_report(
        [
            {
                "answered": True,
                "refused": False,
                "reason_code": "TIMEOUT",
            },
            {
                "answered": False,
                "refused": True,
            },
            {
                "answered": False,
                "refused": True,
                "reason_code": "MODEL_REFUSED",
            },
        ]
    )

    assert [item.generation_available for item in report.results] == [False, False, True]


def test_coerce_ignores_inconsistent_explicit_generation_flags_in_mappings():
    report = calculate_report(
        [
            {
                "answered": True,
                "refused": False,
                "reason_code": "TIMEOUT",
                "generation_available": True,
            },
            {
                "answered": False,
                "refused": True,
                "reason_code": None,
                "generation_available": True,
            },
        ]
    )

    assert [item.generation_available for item in report.results] == [False, False]


def test_coerce_normalizes_inconsistent_evaluation_case_result_objects():
    inconsistent = EvaluationCaseResult(
        answered=True,
        refused=False,
        reason_code="TIMEOUT",
        generation_available=True,
    )

    report = calculate_report([inconsistent])

    assert report.results[0].generation_available is False
    assert report.results[0] is not inconsistent


@pytest.mark.asyncio
async def test_source_visibility_rejects_invalid_emitted_citation_ids(
    tmp_path: Path,
):
    result = await run_case(
        tmp_path,
        valid_case("S1"),
        ChatEvent(
            type="answer_delta",
            data={"text": "答案中提到了 S1，但没有后端引用。", "source_ids": ["S9"]},
        ),
        ChatEvent(
            type="final",
            data={"refused": False, "citations": [{"reference_id": "S9"}]},
        ),
    )

    assert result.answered is True
    assert result.citation_source_ids == ("S9",)
    assert result.citation_valid is False
    assert result.source_visibility is False


@pytest.mark.asyncio
async def test_source_visibility_accepts_public_runtime_citation_for_stable_case_source_id(
    tmp_path: Path,
):
    result = await run_case(
        tmp_path,
        valid_case("a" * 64),
        ChatEvent(
            type="answer_delta",
            data={"text": "答案", "source_ids": ["S1"]},
        ),
        ChatEvent(
            type="final",
            data={"refused": False, "citations": [{"reference_id": "S1"}]},
        ),
    )

    assert result.citation_source_ids == ("S1",)
    assert result.citation_valid is False
    assert result.source_visibility is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answer_source_ids", "final_citations"),
    [
        (["S9"], [{"reference_id": "S9"}]),
        ([], []),
    ],
    ids=["invalid-runtime-id", "no-emitted-id"],
)
async def test_source_visibility_rejects_invalid_or_missing_runtime_citations(
    tmp_path: Path,
    answer_source_ids: list[str],
    final_citations: list[dict[str, str]],
):
    result = await run_case(
        tmp_path,
        valid_case("a" * 64),
        ChatEvent(
            type="answer_delta",
            data={"text": "答案", "source_ids": answer_source_ids},
        ),
        ChatEvent(
            type="final",
            data={"refused": False, "citations": final_citations},
        ),
    )

    assert result.answered is True
    assert result.citation_valid is False
    assert result.source_visibility is False


def test_report_and_case_serialization_preserve_legacy_metrics_and_only_add_safe_booleans():
    result = EvaluationCaseResult(
        case_id="1",
        question="问题不应新增外泄字段",
        type="fact",
        answered=True,
        refused=False,
        no_answer=False,
        reference_points_covered=False,
        citation_valid=True,
        citation_source_ids=("S1",),
        generation_available=True,
        source_visibility=True,
    )
    report = calculate_report([result])

    result_payload = result.to_dict()
    report_payload = report.to_dict()

    assert result_payload["answered"] is True
    assert result_payload["reference_points_covered"] is False
    assert result_payload["citation_valid"] is True
    assert result_payload["generation_available"] is True
    assert result_payload["source_visibility"] is True
    assert report_payload["results"][0]["generation_available"] is True
    assert report_payload["results"][0]["source_visibility"] is True
    assert "answer" not in result_payload
    assert "provider_response" not in result_payload
