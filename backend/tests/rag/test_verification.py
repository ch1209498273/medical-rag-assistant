from __future__ import annotations

from dataclasses import dataclass

import pytest
from app.domain.models import Chunk, SourceRef
from app.domain.ports import SearchHit
from app.rag.deepseek_verifier import DeepSeekClaimVerifier
from app.rag.models import Evidence
from app.rag.verification import ClaimVerdict, VerificationResult
from pydantic import ValidationError


def _windows_path(*parts: str) -> str:
    return "C:" + chr(92) + chr(92).join(parts)


def test_supported_claim_requires_a_non_empty_source_id():
    with pytest.raises(ValidationError):
        ClaimVerdict(claim="先申请", verdict="supported", source_ids=[])


def test_verification_rejects_unknown_verdict_and_extra_fields():
    with pytest.raises(ValidationError):
        ClaimVerdict(
            claim="先申请", verdict="maybe", source_ids=["S1"], extra="拒绝"
        )


def test_verification_result_requires_at_least_one_claim():
    with pytest.raises(ValidationError):
        VerificationResult(claims=[])


def test_unsupported_claim_may_have_no_source_id():
    result = ClaimVerdict(claim="资料没有提到这一点", verdict="unsupported", source_ids=[])

    assert result.verdict == "unsupported"
    assert result.source_ids == []


def test_valid_verification_result_preserves_strict_claims():
    result = VerificationResult(
        claims=[
            ClaimVerdict(
                claim="先申请，再审批", verdict="supported", source_ids=["S1"]
            ),
            ClaimVerdict(claim="没有提到线上提交", verdict="unsupported", source_ids=[]),
        ]
    )

    assert [claim.verdict for claim in result.claims] == ["supported", "unsupported"]


def test_claim_and_source_ids_do_not_coerce_non_strings():
    with pytest.raises(ValidationError):
        ClaimVerdict(claim=123, verdict="supported", source_ids=["S1"])
    with pytest.raises(ValidationError):
        ClaimVerdict(claim="先申请", verdict="supported", source_ids=[1])


def ready_evidence() -> tuple[Evidence, ...]:
    hit = SearchHit(
        version_id="v1",
        chunk=Chunk(
            chunk_id="c1",
            text="员工请假应先申请，再由负责人审批。",
            source=SourceRef(
                file_name="虚构制度.docx",
                heading_path=("请假制度",),
                paragraph_start=2,
                paragraph_end=2,
            ),
            content_hash="fictional-hash",
        ),
        score=0.9,
    )
    return (Evidence(reference_id="S1", hit=hit, rerank_score=0.9),)


@dataclass
class RecordingJsonProvider:
    payload: object

    def __post_init__(self) -> None:
        self.messages = None

    async def complete_json(self, messages):
        self.messages = messages
        return self.payload


@pytest.mark.asyncio
async def test_deepseek_claim_verifier_sends_question_answer_and_evidence_as_data():
    provider = RecordingJsonProvider(
        {
            "claims": [
                {"claim": "先申请", "verdict": "supported", "source_ids": ["S1"]}
            ]
        }
    )
    verifier = DeepSeekClaimVerifier(provider)

    result = await verifier.verify("请假流程？", "先申请，再审批。", ready_evidence())

    assert result.claims[0].source_ids == ["S1"]
    assert "请假流程" in str(provider.messages)
    assert "先申请，再审批" in str(provider.messages)
    assert "员工请假应先申请" in str(provider.messages)
    assert "claim 必须直接复制回答中的连续文字" in str(provider.messages)
    assert "只返回包含 claims 数组的 JSON 对象" in str(provider.messages)
    assert "claim、verdict、source_ids 三个字段" in str(provider.messages)
    assert "verdict 只能是字符串 supported 或 unsupported" in str(provider.messages)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"claims": [{"claim": "先申请", "verdict": "supported", "source_ids": ["S1"], "extra": "拒绝"}]},
        [],
        {"claims": [{"claim": "先申请", "verdict": "supported", "source_ids": ["S9"]}]},
    ],
    ids=["extra-field", "non-object-root", "unknown-source"],
)
async def test_deepseek_claim_verifier_rejects_invalid_provider_payload(payload):
    verifier = DeepSeekClaimVerifier(RecordingJsonProvider(payload))

    with pytest.raises(ValueError, match="^verification unavailable$"):
        await verifier.verify("问题", "答案", ready_evidence())


@pytest.mark.asyncio
async def test_deepseek_claim_verifier_hides_provider_failure_details():
    class FailingProvider:
        async def complete_json(self, messages):
            raise RuntimeError("Authorization: " + "Bearer " + "secret-provider-response")

    verifier = DeepSeekClaimVerifier(FailingProvider())

    with pytest.raises(ValueError) as error:
        await verifier.verify("问题", "答案", ready_evidence())

    assert str(error.value) == "verification unavailable"
    assert "Bearer" not in str(error.value)


@pytest.mark.asyncio
async def test_deepseek_claim_verifier_redacts_prompt_boundary_secrets_and_paths():
    provider = RecordingJsonProvider(
        {"claims": [{"claim": "没有依据", "verdict": "unsupported", "source_ids": []}]}
    )
    verifier = DeepSeekClaimVerifier(provider)

    await verifier.verify(
        "问题 " + "sk-test-token" + " " + _windows_path("private", "question.txt"),
        "Bearer " + "answer-secret",
        ready_evidence(),
    )

    rendered = str(provider.messages)
    assert "sk-test-token" not in rendered
    assert "Bearer answer-secret" not in rendered
    assert _windows_path("private", "question.txt") not in rendered
