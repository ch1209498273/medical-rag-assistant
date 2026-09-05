from __future__ import annotations

import asyncio

import httpx
import pytest
from app.agents.budget import (
    BudgetConfigurationError,
    BudgetTransport,
    RequestBudget,
    UsageRecord,
    WorkflowBudgetExceeded,
    consume_current_request,
    request_scope,
    stage_scope,
)


def test_eighth_request_is_rejected_before_sending():
    budget = RequestBudget(max_calls=7)
    for _ in range(7):
        budget.consume("generating")
    with pytest.raises(WorkflowBudgetExceeded):
        budget.consume("generating")
    assert budget.http_calls == 7


def test_unconfigured_stage_is_rejected_without_consuming_budget():
    budget = RequestBudget()

    with pytest.raises(BudgetConfigurationError):
        budget.consume("deepseek")

    assert budget.http_calls == 0
    assert budget.stage_counts == {}


def test_unconfigured_stage_scope_is_rejected():
    with pytest.raises(BudgetConfigurationError), stage_scope("provider"):
        pass


def test_usage_record_accepts_only_safe_numeric_fields():
    record = UsageRecord(
        provider="deepseek",
        model="deepseek-v4-flash",
        operation="complete",
        input_tokens=12,
        output_tokens=8,
        usage_complete=True,
    )

    assert record.to_dict()["input_tokens"] == 12
    with pytest.raises(ValueError):
        UsageRecord("deepseek", "model", "complete", input_tokens=-1)


@pytest.mark.asyncio
async def test_budget_transport_rejects_eighth_physical_request_before_delegate():
    calls = 0

    class Delegate(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            nonlocal calls
            calls += 1
            return httpx.Response(200, request=request)

    budget = RequestBudget(max_calls=7)
    transport = BudgetTransport(Delegate(), budget)
    async with request_scope(budget):
        with stage_scope("retrieving"):
            for _ in range(7):
                await transport.handle_async_request(httpx.Request("GET", "http://test"))
            with pytest.raises(WorkflowBudgetExceeded):
                await transport.handle_async_request(httpx.Request("GET", "http://test"))
    assert calls == 7
    assert budget.stage_counts == {"retrieving": 7}


@pytest.mark.asyncio
async def test_request_scope_resets_after_cancellation():
    budget = RequestBudget(timeout_seconds=1)
    with pytest.raises(asyncio.CancelledError):
        async with request_scope(budget):
            raise asyncio.CancelledError
    from app.agents.budget import current_budget

    assert current_budget() is None


@pytest.mark.asyncio
async def test_nested_request_scope_is_rejected():
    outer = RequestBudget()
    async with request_scope(outer):
        with pytest.raises(RuntimeError):
            async with request_scope(RequestBudget()):
                pass


@pytest.mark.asyncio
async def test_concurrent_request_scopes_keep_counts_isolated():
    async def consume_twice(budget: RequestBudget) -> int:
        async with request_scope(budget):
            await asyncio.sleep(0)
            consume_current_request("generating")
            await asyncio.sleep(0)
            consume_current_request("generating")
            return budget.http_calls

    first, second = await asyncio.gather(
        consume_twice(RequestBudget(max_calls=2)),
        consume_twice(RequestBudget(max_calls=2)),
    )

    assert first == 2
    assert second == 2
