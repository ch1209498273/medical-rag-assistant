"""Strict, provider-independent contracts for the candidate workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    model_validator,
)

from app.rag.models import ChatEvent

Route = Literal["direct", "verify", "clarify", "out_of_scope"]
RouteReasonCode = Literal[
    "single_fact", "multi_condition", "ambiguous", "unsupported_scope"
]
WorkflowVersion = Literal["baseline_v1", "agent_workflow_v2a"]
AgentVariant = Literal[
    "router_direct_fallback",
    "router_conditional_verify_fallback",
]
VerifierMode = Literal["off", "verify_route"]
Stage = Literal[
    "accepted",
    "rewriting",
    "routing",
    "retrieving",
    "reranking",
    "generating",
    "verifying",
    "validating",
    "reference",
]
Outcome = Literal["answered", "refused", "error", "cancelled"]
VerifierStatus = Literal["not_run", "passed", "failed", "unavailable"]

_AGENT_VARIANT_CONFIG: dict[str, dict[str, str]] = {
    "router_direct_fallback": {
        "verifier_mode": "off",
        "fallback": "baseline_v1",
    },
    "router_conditional_verify_fallback": {
        "verifier_mode": "verify_route",
        "fallback": "baseline_v1",
    },
}


def validate_agent_variant(value: str) -> AgentVariant:
    """Validate an experiment variant before any provider is constructed."""

    if value not in _AGENT_VARIANT_CONFIG:
        raise ValueError("INVALID_AGENT_VARIANT")
    return cast(AgentVariant, value)


def variant_config(variant: AgentVariant) -> dict[str, object]:
    """Return the fixed, non-secret configuration for an experiment variant."""

    validated = validate_agent_variant(variant)
    return {
        "variant": validated,
        **_AGENT_VARIANT_CONFIG[validated],
    }

ROUTE_REASON_BY_ROUTE: dict[str, str] = {
    "direct": "single_fact",
    "verify": "multi_condition",
    "clarify": "ambiguous",
    "out_of_scope": "unsupported_scope",
}


class RouteDecision(BaseModel):
    """The only output accepted from the Router role."""

    model_config = ConfigDict(extra="forbid", strict=True)

    route: Route
    reason_code: RouteReasonCode

    @model_validator(mode="after")
    def route_reason_must_match(self) -> RouteDecision:
        if ROUTE_REASON_BY_ROUTE[self.route] != self.reason_code:
            raise ValueError("route and reason_code do not match")
        return self


class WorkflowSummary(BaseModel):
    """Safe execution facts; no prompt, question, evidence or answer text."""

    model_config = ConfigDict(extra="forbid", strict=True)

    workflow_version: WorkflowVersion = "agent_workflow_v2a"
    run_id: StrictStr
    route: Route | None = None
    outcome: Outcome
    verifier_status: VerifierStatus = "not_run"
    http_calls: StrictInt = Field(default=0, ge=0, le=7)
    elapsed_ms: StrictInt | None = Field(default=None, ge=0, le=86_400_000)
    reason_code: StrictStr | None = None

    @model_validator(mode="after")
    def validate_summary(self) -> WorkflowSummary:
        if len(self.run_id) < 8 or len(self.run_id) > 64:
            raise ValueError("run_id is invalid")
        if self.reason_code is not None and (
            not self.reason_code.strip() or len(self.reason_code) > 64
        ):
            raise ValueError("reason_code is invalid")
        return self


@dataclass(frozen=True)
class BufferedAnswer:
    """An in-memory AnswerService result, never persisted by this module."""

    events: tuple[ChatEvent, ...]

    def __post_init__(self) -> None:
        if any(not isinstance(event, ChatEvent) for event in self.events):
            raise TypeError("events must contain ChatEvent values")
        object.__setattr__(self, "events", tuple(self.events))

    @property
    def terminal(self) -> ChatEvent | None:
        terminals = [event for event in self.events if event.type in {"final", "error"}]
        return terminals[0] if len(terminals) == 1 else None

    @property
    def answer_text(self) -> str:
        if self.terminal is None or self.terminal.type != "final" or self.terminal.data.get("refused") is not False:
            return ""
        return "".join(
            str(event.data.get("text", ""))
            for event in self.events
            if event.type == "answer_delta" and isinstance(event.data.get("text"), str)
        )

    @property
    def citations(self) -> list[dict[str, object]]:
        terminal = self.terminal
        citations = terminal.data.get("citations", []) if terminal and terminal.type == "final" else []
        return [dict(item) for item in citations] if isinstance(citations, list) else []

    @property
    def refused(self) -> bool:
        terminal = self.terminal
        return terminal is None or terminal.type != "final" or terminal.data.get("refused") is not False


async def collect_answer(service: object, question: str, retrieval: object) -> BufferedAnswer:
    """Consume an AnswerService stream before releasing any answer delta."""

    stream = getattr(service, "stream", None)
    if not callable(stream):
        raise TypeError("answer service does not provide stream")
    events: list[ChatEvent] = []
    async for event in stream(question, retrieval):
        if not isinstance(event, ChatEvent):
            raise TypeError("answer service returned an invalid event")
        events.append(event)
    return BufferedAnswer(tuple(events))


__all__ = [
    "ROUTE_REASON_BY_ROUTE",
    "AgentVariant",
    "BufferedAnswer",
    "Outcome",
    "Route",
    "RouteDecision",
    "RouteReasonCode",
    "Stage",
    "VerifierMode",
    "VerifierStatus",
    "WorkflowSummary",
    "WorkflowVersion",
    "collect_answer",
    "validate_agent_variant",
    "variant_config",
]
