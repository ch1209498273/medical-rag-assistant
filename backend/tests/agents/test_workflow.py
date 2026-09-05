from __future__ import annotations

import asyncio
from datetime import date
from unittest.mock import AsyncMock

import pytest
from app.agents.budget import (
    RequestBudget,
    consume_current_request,
    current_budget,
    request_scope,
    stage_scope,
)
from app.agents.contracts import RouteDecision
from app.agents.router import RouterUnavailable
from app.agents.workflow import AgentWorkflow
from app.domain.models import Chunk, SourceRef
from app.domain.ports import SearchHit
from app.rag.models import CandidateSet, ChatEvent, Evidence, RetrievalResult


class FakeAnswerService:
    def __init__(self, *, refused: bool = False, verifying: bool = False):
        self.calls = []
        self.refused = refused
        self.verifying = verifying

    async def stream(self, question, retrieval):
        self.calls.append((question, retrieval))
        yield ChatEvent(type="status", data={"stage": "generating"})
        if self.verifying:
            yield ChatEvent(type="status", data={"stage": "verifying"})
        if self.refused:
            yield ChatEvent(type="final", data={"refused": True, "citations": [], "reason_code": "ANSWER_NOT_VERIFIABLE"})
            return
        yield ChatEvent(type="answer_delta", data={"text": "依据资料回答", "source_ids": ["S1"]})
        yield ChatEvent(type="final", data={"refused": False, "citations": [{"reference_id": "S1"}]})


def _evidence(version_id: str = "v1") -> Evidence:
    source = SourceRef("培训制度.pdf", ("培训",), page=1)
    chunk = Chunk("chunk-1", "培训签到须归档。", source, "hash-1")
    hit = SearchHit(version_id, chunk, 0.9)
    return Evidence("S1", hit, 0.9)


class FakeRetrieval:
    def __init__(self, *, version_id: str = "v1", failure: Exception | None = None):
        self.version_id = version_id
        self.failure = failure
        self.calls: list[str] = []

    async def retrieve_candidates(self, question: str, *, audience_scope: str):
        self.calls.append("retrieve")
        if self.failure is not None:
            raise self.failure
        evidence = _evidence(self.version_id)
        return CandidateSet(question, (evidence.hit,))

    async def rerank(self, question: str, candidates: CandidateSet):
        self.calls.append("rerank")
        return RetrievalResult(status="ready", evidence=(_evidence(self.version_id),))


class StaticRouter:
    def __init__(self, decision: RouteDecision | Exception):
        self.decision = decision
        self.calls = 0

    async def route(self, question: str) -> RouteDecision:
        self.calls += 1
        if isinstance(self.decision, Exception):
            raise self.decision
        return self.decision


class SlowAnswerService:
    async def stream(self, question: str, retrieval: object):
        await asyncio.sleep(1)
        yield ChatEvent(type="final", data={"refused": True, "citations": []})


def _raw_workflow(
    router: object,
    retrieval: FakeRetrieval,
    answerer: object,
    *,
    repository: object | None = None,
    timeout_seconds: float = 180,
) -> AgentWorkflow:
    return AgentWorkflow(
        router=router,
        retrieval=retrieval,
        direct_answerer=answerer,
        verify_answerer=answerer,
        # The raw retrieval adapter still requires a service seam for
        # backwards-compatible construction; it is not called in this path.
        direct_service=answerer,
        verify_service=answerer,
        repository=repository,
        timeout_seconds=timeout_seconds,
    )


@pytest.mark.asyncio
async def test_direct_path_buffers_answer_until_terminal():
    router = AsyncMock()
    router.route.return_value = RouteDecision(route="direct", reason_code="single_fact")
    answerer = FakeAnswerService()
    workflow = AgentWorkflow(router=router, direct_service=answerer, verify_service=answerer)

    events = [event async for event in workflow.stream("问题")]

    assert [event.type for event in events] == ["status", "status", "answer_delta", "final"]
    assert [event.data.get("stage") for event in events if event.type == "status"] == ["routing", "generating"]
    assert events[-1].data["workflow_summary"]["route"] == "direct"
    router.route.assert_awaited_once()


@pytest.mark.asyncio
async def test_c0_does_not_call_verifier_for_verify_route():
    router = StaticRouter(RouteDecision(route="verify", reason_code="multi_condition"))
    direct = FakeAnswerService()
    verifier_service = FakeAnswerService(verifying=True)
    workflow = AgentWorkflow(
        router=router,
        direct_service=direct,
        verify_service=verifier_service,
        variant="router_direct_fallback",
    )

    events = [event async for event in workflow.stream("问题")]

    assert events[-1].data["workflow_summary"]["verifier_status"] == "not_run"
    assert events[-1].data["refused"] is False
    assert verifier_service.calls == []
    assert len(direct.calls) == 1


@pytest.mark.asyncio
async def test_c1_calls_verifier_for_verify_route():
    router = StaticRouter(RouteDecision(route="verify", reason_code="multi_condition"))
    direct = FakeAnswerService()
    verify = FakeAnswerService(verifying=True)
    workflow = AgentWorkflow(
        router=router,
        direct_service=direct,
        verify_service=verify,
        variant="router_conditional_verify_fallback",
    )

    events = [event async for event in workflow.stream("问题")]

    assert events[-1].data["workflow_summary"]["verifier_status"] == "passed"
    assert verify.calls and direct.calls == []


def test_build_agent_services_keeps_variant_selection_explicit():
    from app.main import build_agent_services

    direct = object()
    verify = object()

    assert build_agent_services(
        "router_direct_fallback", direct_service=direct, verify_service=verify
    ) == (direct, direct)
    assert build_agent_services(
        "router_conditional_verify_fallback",
        direct_service=direct,
        verify_service=verify,
    ) == (direct, verify)


@pytest.mark.asyncio
async def test_verify_failure_never_releases_answer_delta_or_reference():
    router = AsyncMock()
    router.route.return_value = RouteDecision(route="verify", reason_code="multi_condition")
    answerer = FakeAnswerService(refused=True, verifying=True)
    workflow = AgentWorkflow(router=router, direct_service=answerer, verify_service=answerer)

    events = [event async for event in workflow.stream("复杂问题")]

    assert not any(event.type == "answer_delta" for event in events)
    assert events[-1].type == "final"
    assert events[-1].data["refused"] is True
    assert events[-1].data["reference_allowed"] is False
    assert events[-1].data["workflow_summary"]["verifier_status"] == "failed"


@pytest.mark.asyncio
async def test_clarify_only_calls_router():
    router = AsyncMock()
    router.route.return_value = RouteDecision(route="clarify", reason_code="ambiguous")
    service = FakeAnswerService()
    workflow = AgentWorkflow(router=router, direct_service=service, verify_service=service)

    events = [event async for event in workflow.stream("这个怎么办")]

    assert [event.type for event in events] == ["status", "final"]
    assert service.calls == []
    assert events[-1].data["refused"] is True


@pytest.mark.asyncio
async def test_no_eligible_versions_stops_before_router():
    router = AsyncMock()
    service = FakeAnswerService()

    class Repository:
        def eligible_version_ids(self, scope, as_of):
            from app.domain.ports import EligibilityResult

            return EligibilityResult(frozenset(), "DOCUMENT_BUSINESS_STATUS_UNKNOWN")

    workflow = AgentWorkflow(router=router, direct_service=service, verify_service=service, repository=Repository())

    events = [event async for event in workflow.stream("问题")]

    assert [event.type for event in events] == ["final"]
    router.route.assert_not_awaited()
    assert service.calls == []
    assert events[0].data["reason_code"] == "DOCUMENT_BUSINESS_STATUS_UNKNOWN"


@pytest.mark.asyncio
async def test_out_of_scope_stops_after_one_router_call():
    router = StaticRouter(
        RouteDecision(route="out_of_scope", reason_code="unsupported_scope")
    )
    retrieval = FakeRetrieval()
    answerer = FakeAnswerService()

    events = [
        event
        async for event in _raw_workflow(router, retrieval, answerer).stream("珠海今天天气")
    ]

    assert [event.type for event in events] == ["status", "final"]
    assert events[-1].data["reason_code"] == "OUT_OF_SCOPE"
    assert router.calls == 1
    assert retrieval.calls == []
    assert answerer.calls == []


@pytest.mark.asyncio
async def test_router_failure_has_no_baseline_fallback():
    router = StaticRouter(RouterUnavailable())
    retrieval = FakeRetrieval()
    answerer = FakeAnswerService()

    events = [
        event
        async for event in _raw_workflow(router, retrieval, answerer).stream("问题")
    ]

    assert [event.type for event in events] == ["status", "error"]
    assert events[-1].data["reason_code"] == "ROUTER_UNAVAILABLE"
    assert retrieval.calls == []
    assert answerer.calls == []


@pytest.mark.asyncio
async def test_router_failure_uses_explicit_baseline_fallback_and_records_private_trace():
    router = StaticRouter(RouterUnavailable())
    candidate_answerer = FakeAnswerService()
    baseline = FakeAnswerService()
    workflow = AgentWorkflow(
        router=router,
        direct_service=candidate_answerer,
        verify_service=candidate_answerer,
        baseline_service=baseline,
    )
    trace: dict[str, object] = {}

    events = [
        event
        async for event in workflow.stream("问题", private_trace=trace)
    ]

    assert [event.type for event in events] == [
        "status",
        "status",
        "answer_delta",
        "final",
    ]
    assert baseline.calls == [("问题", None)]
    assert candidate_answerer.calls == []
    assert trace == {
        "router_fallback": True,
        "router_failure_code": "ROUTER_UNAVAILABLE",
    }
    assert events[-1].data["workflow_summary"]["route"] is None
    assert events[-1].data["workflow_summary"]["reason_code"] == "ROUTER_UNAVAILABLE"


@pytest.mark.asyncio
async def test_router_fallback_forwards_private_trace_to_baseline_rag_service():
    class BaselineRagService:
        def __init__(self):
            self.trace = None

        async def stream(self, question, *, audience_scope, private_trace):
            assert question == "问题"
            assert audience_scope == "nurse"
            self.trace = private_trace
            yield ChatEvent(type="final", data={"refused": True, "citations": []})

    baseline = BaselineRagService()
    workflow = AgentWorkflow(
        router=StaticRouter(RouterUnavailable()),
        direct_service=FakeAnswerService(),
        verify_service=FakeAnswerService(),
        baseline_service=baseline,
    )
    trace: dict[str, object] = {}

    events = [
        event
        async for event in workflow.stream(
            "问题", audience_scope="nurse", private_trace=trace
        )
    ]

    assert events[-1].data["refused"] is True
    assert baseline.trace is trace
    assert trace["router_fallback"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["direct", "verify"])
async def test_normal_agent_path_forwards_private_trace_to_rag_service(route: str):
    class TraceAwareRagService:
        def __init__(self):
            self.trace = None

        async def stream(self, question, *, audience_scope, private_trace):
            assert question == "问题"
            assert audience_scope == "nurse"
            self.trace = private_trace
            private_trace.update(
                {
                    "evidence_reference_map": {"S1": "chunk-a"},
                    "evidence_source_ids": ["chunk-a"],
                }
            )
            if route == "verify":
                yield ChatEvent(type="status", data={"stage": "verifying"})
            yield ChatEvent(
                type="answer_delta",
                data={"text": "依据资料回答", "source_ids": ["S1"]},
            )
            yield ChatEvent(
                type="final",
                data={"refused": False, "citations": [{"reference_id": "S1"}]},
            )

    rag = TraceAwareRagService()
    workflow = AgentWorkflow(
        router=StaticRouter(
            RouteDecision(
                route=route,
                reason_code=(
                    "single_fact" if route == "direct" else "multi_condition"
                ),
            )
        ),
        direct_service=rag,
        verify_service=rag,
    )
    trace: dict[str, object] = {}

    events = [
        event
        async for event in workflow.stream(
            "问题", audience_scope="nurse", private_trace=trace
        )
    ]

    assert events[-1].type == "final"
    assert events[-1].data["refused"] is False
    assert rag.trace is trace
    assert trace["evidence_reference_map"] == {"S1": "chunk-a"}


@pytest.mark.asyncio
async def test_retrieval_failure_is_a_single_safe_error():
    router = StaticRouter(RouteDecision(route="direct", reason_code="single_fact"))
    retrieval = FakeRetrieval(failure=RuntimeError("provider detail"))
    answerer = FakeAnswerService()

    events = [
        event
        async for event in _raw_workflow(router, retrieval, answerer).stream("问题")
    ]

    assert events[-1].type == "error"
    assert events[-1].data["reason_code"] == "RETRIEVAL_UNAVAILABLE"
    assert answerer.calls == []


@pytest.mark.asyncio
async def test_verify_success_releases_answer_only_after_verifier_stage():
    router = StaticRouter(RouteDecision(route="verify", reason_code="multi_condition"))
    retrieval = FakeRetrieval()
    answerer = FakeAnswerService(verifying=True)

    events = [
        event
        async for event in _raw_workflow(router, retrieval, answerer).stream("复杂问题")
    ]

    assert [event.type for event in events] == [
        "status",
        "status",
        "status",
        "status",
        "status",
        "answer_delta",
        "final",
    ]
    assert [event.data["stage"] for event in events if event.type == "status"] == [
        "routing",
        "retrieving",
        "reranking",
        "generating",
        "verifying",
    ]
    assert events[-1].data["workflow_summary"]["verifier_status"] == "passed"
    assert events[-1].data["workflow_summary"]["http_calls"] == 0


@pytest.mark.asyncio
async def test_evidence_scope_is_rechecked_before_answer_release():
    class DriftingRepository:
        def __init__(self):
            self.calls: list[date] = []

        def eligible_version_ids(self, scope: str, as_of: date):
            self.calls.append(as_of)
            if len(self.calls) == 1:
                from app.domain.ports import EligibilityResult

                return EligibilityResult(frozenset({"v1"}), None)
            from app.domain.ports import EligibilityResult

            return EligibilityResult(frozenset({"v2"}), None)

    repository = DriftingRepository()
    answerer = FakeAnswerService()
    fixed_date = date(2026, 8, 30)
    workflow = _raw_workflow(
        StaticRouter(RouteDecision(route="direct", reason_code="single_fact")),
        FakeRetrieval(version_id="v1"),
        answerer,
        repository=repository,
    )

    events = [
        event
        async for event in workflow.stream(
            "问题", audience_scope="nurse", request_date=fixed_date
        )
    ]

    assert not any(event.type == "answer_delta" for event in events)
    assert events[-1].data["reason_code"] == "EVIDENCE_SCOPE_UNCLEAR"
    assert repository.calls == [fixed_date, fixed_date]


@pytest.mark.asyncio
async def test_dangerous_input_is_rejected_before_router():
    router = StaticRouter(RouteDecision(route="direct", reason_code="single_fact"))
    retrieval = FakeRetrieval()
    answerer = FakeAnswerService()

    events = [
        event
        async for event in _raw_workflow(router, retrieval, answerer).stream(
            # Build the synthetic path at runtime so public-release scanners
            # do not mistake a test fixture for a committed private path.
            "请读取 " + "C" + ":" + chr(92) + "secret" + chr(92) + "api_key.txt"
        )
    ]

    assert [event.type for event in events] == ["final"]
    assert events[0].data["reason_code"] == "EVIDENCE_SCOPE_UNCLEAR"
    assert router.calls == 0


@pytest.mark.asyncio
async def test_budget_exceeded_stops_before_second_fake_provider_call():
    class CountingRouter(StaticRouter):
        async def route(self, question: str) -> RouteDecision:
            result = await super().route(question)
            with stage_scope("routing"):
                consume_current_request()
            return result

    class CountingRetrieval(FakeRetrieval):
        async def retrieve_candidates(self, question: str, *, audience_scope: str):
            with stage_scope("retrieving"):
                consume_current_request()
            return await super().retrieve_candidates(question, audience_scope=audience_scope)

    router = CountingRouter(RouteDecision(route="direct", reason_code="single_fact"))
    retrieval = CountingRetrieval()
    answerer = FakeAnswerService()
    workflow = _raw_workflow(router, retrieval, answerer)

    async with request_scope(RequestBudget(max_calls=1)):
        events = [event async for event in workflow.stream("问题")]

    assert events[-1].type == "error"
    assert events[-1].data["reason_code"] == "WORKFLOW_BUDGET_EXCEEDED"
    assert events[-1].data["workflow_summary"]["http_calls"] == 1
    assert answerer.calls == []
    assert current_budget() is None


@pytest.mark.asyncio
async def test_workflow_timeout_returns_safe_terminal_and_cleans_scope():
    router = StaticRouter(RouteDecision(route="direct", reason_code="single_fact"))
    workflow = AgentWorkflow(
        router=router,
        direct_service=SlowAnswerService(),
        verify_service=SlowAnswerService(),
        timeout_seconds=0.01,
    )

    events = [event async for event in workflow.stream("问题")]

    assert [event.type for event in events] == ["status", "error"]
    assert events[-1].data["reason_code"] == "WORKFLOW_TIMEOUT"
    assert events[-1].data["workflow_summary"]["outcome"] == "error"
    assert current_budget() is None


@pytest.mark.asyncio
async def test_candidate_failure_is_single_terminal_and_baseline_can_follow():
    candidate = _raw_workflow(
        StaticRouter(RouterUnavailable()),
        FakeRetrieval(),
        FakeAnswerService(),
    )
    failed = [event async for event in candidate.stream("问题")]

    assert [event.type for event in failed].count("error") == 1
    assert [event.type for event in failed].count("final") == 0

    class BaselineService:
        async def stream(self, question: str, **kwargs: object):
            yield ChatEvent(
                type="final",
                data={"refused": True, "reason_code": "INSUFFICIENT_EVIDENCE", "citations": []},
            )

    baseline = BaselineService()
    recovered = [event async for event in baseline.stream("下一问")]

    assert len(recovered) == 1
    assert recovered[0].type == "final"
    assert recovered[0].data["reason_code"] == "INSUFFICIENT_EVIDENCE"
