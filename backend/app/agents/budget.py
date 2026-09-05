"""Per-request physical provider-call budgets.

The counter lives at the HTTP transport boundary.  A function that happens to
call a provider is not counted until a physical request is about to leave the
process.
"""

from __future__ import annotations

import asyncio
import contextvars
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx


class WorkflowBudgetExceeded(RuntimeError):
    """Raised before a request that would exceed the workflow budget."""


class BudgetConfigurationError(RuntimeError):
    """Raised when a candidate request is not associated with a fixed stage."""


_VALID_STAGES = frozenset(
    {
        "accepted",
        "rewriting",
        "routing",
        "retrieving",
        "reranking",
        "generating",
        "verifying",
        "validating",
        "reference",
    }
)


@dataclass(frozen=True)
class UsageRecord:
    """Safe numeric usage facts observed at a provider boundary."""

    provider: str
    model: str
    operation: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    usage_complete: bool = False

    def __post_init__(self) -> None:
        for name in ("provider", "model", "operation"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or len(value) > 128:
                raise ValueError(f"{name} is invalid")
        for name in ("input_tokens", "output_tokens"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} is invalid")
        if not isinstance(self.usage_complete, bool):
            raise TypeError("usage_complete is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "operation": self.operation,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "usage_complete": self.usage_complete,
        }


@dataclass
class RequestBudget:
    max_calls: int = 7
    timeout_seconds: float = 180.0
    started_monotonic: float | None = None
    http_calls: int = 0
    stage_counts: dict[str, int] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.max_calls, bool) or not isinstance(self.max_calls, int) or not 1 <= self.max_calls <= 7:
            raise ValueError("max_calls must be between 1 and 7")
        if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float)) or not 0 < self.timeout_seconds <= 300:
            raise ValueError("timeout_seconds must be between 0 and 300 seconds")

    def consume(self, stage: str) -> None:
        if stage not in _VALID_STAGES:
            raise BudgetConfigurationError("request stage is not configured")
        if self.http_calls >= self.max_calls:
            raise WorkflowBudgetExceeded("workflow request budget exceeded")
        self.http_calls += 1
        self.stage_counts[stage] = self.stage_counts.get(stage, 0) + 1

    def record_usage(self, values: dict[str, object]) -> None:
        """Accumulate only non-negative numeric token totals."""

        for key in ("input_tokens", "output_tokens", "total_tokens"):
            value = values.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                self.usage[key] = self.usage.get(key, 0) + value


_BUDGET: contextvars.ContextVar[RequestBudget | None] = contextvars.ContextVar(
    "agent_workflow_budget", default=None
)
_STAGE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "agent_workflow_stage", default="unknown"
)


def current_budget() -> RequestBudget | None:
    return _BUDGET.get()


def current_stage() -> str:
    return _STAGE.get()


def consume_current_request(stage: str | None = None) -> None:
    budget = _BUDGET.get()
    if budget is not None:
        budget.consume(stage or _STAGE.get())


@asynccontextmanager
async def request_scope(budget: RequestBudget) -> AsyncIterator[RequestBudget]:
    """Set one request budget and enforce its total deadline."""

    if not isinstance(budget, RequestBudget):
        raise TypeError("request_scope requires RequestBudget")
    if _BUDGET.get() is not None:
        raise RuntimeError("nested request scopes are not allowed")
    budget.started_monotonic = time.monotonic()
    token = _BUDGET.set(budget)
    try:
        async with asyncio.timeout(budget.timeout_seconds):
            yield budget
    finally:
        _BUDGET.reset(token)


@contextmanager
def stage_scope(stage: str) -> Any:
    if stage not in _VALID_STAGES:
        raise BudgetConfigurationError("request stage is not configured")
    token = _STAGE.set(stage)
    try:
        yield
    finally:
        _STAGE.reset(token)


class BudgetTransport(httpx.AsyncBaseTransport):
    """Count a request immediately before delegating to another transport."""

    _counts_budget = True

    def __init__(self, delegate: httpx.AsyncBaseTransport, budget: RequestBudget | None = None) -> None:
        self.delegate = delegate
        self.budget = budget

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        budget = self.budget or _BUDGET.get()
        if budget is not None:
            budget.consume(_STAGE.get())
        return await self.delegate.handle_async_request(request)

    async def aclose(self) -> None:
        close = getattr(self.delegate, "aclose", None)
        if callable(close):
            result = close()
            if hasattr(result, "__await__"):
                await result


__all__ = [
    "BudgetConfigurationError",
    "BudgetTransport",
    "RequestBudget",
    "UsageRecord",
    "WorkflowBudgetExceeded",
    "consume_current_request",
    "current_budget",
    "current_stage",
    "request_scope",
    "stage_scope",
]
