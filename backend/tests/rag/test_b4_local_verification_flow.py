from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace

import pytest
from app.domain.models import Chunk, SourceRef
from app.domain.ports import SearchHit
from app.rag.answering import AnswerService
from app.rag.deepseek_verifier import DeepSeekClaimVerifier
from app.rag.models import Evidence, RetrievalResult


@dataclass
class FixedAnswerProvider:
    raw: str

    def __post_init__(self) -> None:
        self.calls = 0

    async def stream_answer(self, messages, *, json_output=False):
        self.calls += 1
        midpoint = max(1, len(self.raw) // 2)
        yield self.raw[:midpoint]
        yield self.raw[midpoint:]


@dataclass
class FixedVerifierProvider:
    payload: object

    def __post_init__(self) -> None:
        self.calls = 0

    async def complete_json(self, messages):
        self.calls += 1
        return self.payload


class FailingVerifierProvider:
    def __init__(self, detail: str) -> None:
        self.detail = detail
        self.calls = 0

    async def complete_json(self, messages):
        self.calls += 1
        raise RuntimeError(self.detail)


class SlowVerifierProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def complete_json(self, messages):
        self.calls += 1
        await asyncio.sleep(0.05)
        return {
            "claims": [
                {
                    "claim": "先提交申请",
                    "verdict": "supported",
                    "source_ids": ["S1"],
                }
            ]
        }


def ready_retrieval() -> RetrievalResult:
    first = Evidence(
        reference_id="S1",
        hit=SearchHit(
            version_id="fictional-v1",
            chunk=Chunk(
                chunk_id="fictional-c1",
                text="员工请假应先申请，再由负责人审批。",
                source=SourceRef(
                    file_name="虚构制度.docx",
                    heading_path=("请假制度",),
                    paragraph_start=2,
                    paragraph_end=2,
                ),
                content_hash="fictional-hash-1",
            ),
            score=0.9,
        ),
        rerank_score=0.9,
    )
    second = replace(first, reference_id="S2")
    return RetrievalResult(status="ready", evidence=(first, second))


def make_answerer(
    answer: str,
    verifier_payload: object,
) -> tuple[AnswerService, FixedAnswerProvider, FixedVerifierProvider]:
    answer_provider = FixedAnswerProvider(
        '{"refused":false,"answer":"' + answer + '"}'
    )
    verifier_provider = FixedVerifierProvider(verifier_payload)
    verifier = DeepSeekClaimVerifier(verifier_provider)
    return (
        AnswerService(deepseek=answer_provider, verifier=verifier),
        answer_provider,
        verifier_provider,
    )


@pytest.mark.asyncio
async def test_b4_fixed_supported_claim_allows_paraphrased_answer_and_binds_citation():
    answerer, answer_provider, verifier_provider = make_answerer(
        "办理时先提交申请，再等待负责人审批。",
        {
            "claims": [
                {
                    "claim": "办理时先提交申请，再等待负责人审批",
                    "verdict": "supported",
                    "source_ids": ["S1"],
                }
            ]
        },
    )

    events = [event async for event in answerer.stream("请假流程？", ready_retrieval())]

    assert [event.type for event in events].count("answer_delta") == 1
    assert events[-1].data["refused"] is False
    assert [item["reference_id"] for item in events[-1].data["citations"]] == ["S1"]
    assert answer_provider.calls == 1
    assert verifier_provider.calls == 1


@pytest.mark.asyncio
async def test_b4_fixed_unsupported_claim_refuses_without_partial_answer():
    answerer, _, verifier_provider = make_answerer(
        "办理时先提交申请，再等待负责人审批。",
        {
            "claims": [
                {
                    "claim": "办理时先提交申请",
                    "verdict": "unsupported",
                    "source_ids": [],
                }
            ]
        },
    )

    events = [event async for event in answerer.stream("请假流程？", ready_retrieval())]

    assert all(event.type != "answer_delta" for event in events)
    assert events[-1].data == {
        "refused": True,
        "reason_code": "ANSWER_NOT_VERIFIABLE",
        "citations": [],
    }
    assert verifier_provider.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "case_id"),
    [
        (
            {
                "claims": [
                    {
                        "claim": "回答中不存在的事实",
                        "verdict": "supported",
                        "source_ids": ["S1"],
                    }
                ]
            },
            "claim-mismatch",
        ),
        (
            {
                "claims": [
                    {
                        "claim": "办理时先提交申请",
                        "verdict": "supported",
                        "source_ids": ["S99"],
                    }
                ]
            },
            "unknown-source",
        ),
        ([], "non-object-root"),
        (
            {
                "claims": [
                    {
                        "claim": "办理时先提交申请",
                        "verdict": "maybe",
                        "source_ids": ["S1"],
                    }
                ]
            },
            "unknown-verdict",
        ),
    ],
    ids=["claim-mismatch", "unknown-source", "non-object-root", "unknown-verdict"],
)
async def test_b4_invalid_fixed_verifier_payload_refuses_whole_answer(
    payload: object, case_id: str
):
    answerer, _, verifier_provider = make_answerer(
        "办理时先提交申请，再等待负责人审批。", payload
    )

    events = [event async for event in answerer.stream("请假流程？", ready_retrieval())]

    assert all(event.type != "answer_delta" for event in events), case_id
    assert events[-1].data["reason_code"] == "ANSWER_NOT_VERIFIABLE"
    assert "S99" not in str(events[-1].data)
    assert verifier_provider.calls == 1


@pytest.mark.asyncio
async def test_b4_verifier_provider_failure_hides_detail_and_does_not_retry():
    answer_provider = FixedAnswerProvider(
        '{"refused":false,"answer":"办理时先提交申请。"}'
    )
    verifier_provider = FailingVerifierProvider(
        "Authorization: " + "Bearer " + "local-secret"
    )
    answerer = AnswerService(
        deepseek=answer_provider,
        verifier=DeepSeekClaimVerifier(verifier_provider),
    )

    events = [event async for event in answerer.stream("请假流程？", ready_retrieval())]

    assert all(event.type != "answer_delta" for event in events)
    assert events[-1].data["reason_code"] == "ANSWER_NOT_VERIFIABLE"
    assert "local-secret" not in str(events[-1].data)
    assert verifier_provider.calls == 1


@pytest.mark.asyncio
async def test_b4_verifier_timeout_fails_closed_without_retry():
    answer_provider = FixedAnswerProvider(
        '{"refused":false,"answer":"办理时先提交申请。"}'
    )
    verifier_provider = SlowVerifierProvider()
    answerer = AnswerService(
        deepseek=answer_provider,
        verifier=DeepSeekClaimVerifier(verifier_provider),
        answer_timeout_seconds=0.001,
    )

    events = [event async for event in answerer.stream("请假流程？", ready_retrieval())]

    assert all(event.type != "answer_delta" for event in events)
    assert events[-1].data["reason_code"] == "ANSWER_NOT_VERIFIABLE"
    assert verifier_provider.calls == 1
