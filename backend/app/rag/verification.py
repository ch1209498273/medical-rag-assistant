"""Provider-independent contracts for claim-level answer verification."""

from __future__ import annotations

from collections.abc import Awaitable, Sequence
from typing import Literal, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    field_validator,
    model_validator,
)

from app.rag.models import Evidence

ClaimVerdictType = Literal["supported", "unsupported"]


class ClaimVerdict(BaseModel):
    """One independently verifiable claim from a generated answer."""

    model_config = ConfigDict(extra="forbid", strict=True)

    claim: StrictStr
    verdict: ClaimVerdictType
    source_ids: list[StrictStr] = Field(default_factory=list)

    @field_validator("claim")
    @classmethod
    def _claim_must_be_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("claim must not be empty")
        return value

    @field_validator("source_ids")
    @classmethod
    def _source_ids_must_be_non_empty(cls, value: list[str]) -> list[str]:
        if any(not source_id.strip() for source_id in value):
            raise ValueError("source IDs must not be empty")
        return value

    @model_validator(mode="after")
    def _supported_claim_requires_source(self) -> ClaimVerdict:
        if self.verdict == "supported" and not self.source_ids:
            raise ValueError("supported claims require at least one source ID")
        return self


class VerificationResult(BaseModel):
    """Strict, non-empty result returned by a claim verifier."""

    model_config = ConfigDict(extra="forbid", strict=True)

    claims: list[ClaimVerdict] = Field(min_length=1)


class ClaimVerifier(Protocol):
    """Async provider boundary used by the optional Stage B answer gate."""

    def verify(
        self,
        question: str,
        answer: str,
        evidence: Sequence[Evidence],
    ) -> Awaitable[VerificationResult]:
        ...


__all__ = ["ClaimVerdict", "ClaimVerdictType", "ClaimVerifier", "VerificationResult"]
