"""Bounded Router → retrieval → Answer/Verifier workflow adapter."""

from __future__ import annotations

import inspect
import secrets
import time
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime
from typing import Any

from app.agents.budget import (
    RequestBudget,
    WorkflowBudgetExceeded,
    current_budget,
    request_scope,
    stage_scope,
)
from app.agents.contracts import (
    AgentVariant,
    BufferedAnswer,
    RouteDecision,
    WorkflowSummary,
    validate_agent_variant,
)
from app.agents.router import RouterUnavailable
from app.chat.safety import ABSOLUTE_PATH_RE
from app.domain.ports import AudienceScope, EligibilityResult
from app.rag.models import ChatEvent, RetrievalResult
from app.rag.query import clean_question
from app.rag.verification import ClaimVerdict, VerificationResult

_DANGEROUS_QUERY_MARKERS = ("api_key", "access_token", "bearer ", "sk-")


class RetrievalUnavailableError(RuntimeError):
    """The existing retrieval seam could not produce a safe result."""


class AgentWorkflow:
    """Select one pre-wired service; never dynamically executes tools."""

    requires_request_scope = True

    def __init__(
        self,
        *,
        router: Any,
        direct_service: Any | None = None,
        verify_service: Any | None = None,
        retrieval: Any | None = None,
        direct_answerer: Any | None = None,
        verify_answerer: Any | None = None,
        rag_service: Any | None = None,
        baseline_service: Any | None = None,
        repository: Any | None = None,
        variant: AgentVariant = "router_conditional_verify_fallback",
        max_calls: int = 7,
        timeout_seconds: float = 180.0,
    ) -> None:
        self.variant = validate_agent_variant(variant)
        self.router = router
        self.retrieval = retrieval
        self.direct_answerer = direct_answerer
        self.verify_answerer = verify_answerer or direct_answerer
        self.direct_service = direct_service or rag_service
        self.verify_service = verify_service or rag_service or direct_service
        if self.variant == "router_direct_fallback":
            self.verify_answerer = self.direct_answerer
            self.verify_service = self.direct_service
        self.baseline_service = baseline_service
        if (
            self.retrieval is None
            and (self.direct_service is None or self.verify_service is None)
        ):
            raise ValueError("at least one RAG service is required")
        if self.retrieval is not None and (
            self.direct_answerer is None or self.verify_answerer is None
        ) and (self.direct_service is None or self.verify_service is None):
            raise ValueError("retrieval workflows require answerers")
        self.repository = repository
        self.max_calls = max_calls
        self.timeout_seconds = timeout_seconds

    def _verifier_enabled(self, route: str | None) -> bool:
        return (
            route == "verify"
            and self.variant == "router_conditional_verify_fallback"
        )

    async def preflight(
        self,
        question: str,
        *,
        audience_scope: AudienceScope = "unspecified",
        request_date: date | None = None,
    ) -> EligibilityResult:
        """Run local input and business-eligibility checks without providers."""

        cleaned = clean_question(question)
        if _dangerous_question(cleaned):
            return EligibilityResult(frozenset(), "EVIDENCE_SCOPE_UNCLEAR")
        if self.repository is None:
            return EligibilityResult(frozenset({"local-test"}), None)
        method = getattr(self.repository, "eligible_version_ids", None)
        if not callable(method):
            return EligibilityResult(frozenset(), "DOCUMENT_BUSINESS_STATUS_UNKNOWN")
        result = method(
            audience_scope,
            request_date or datetime.now(UTC).date(),
        )
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, EligibilityResult):
            raise TypeError("eligibility provider returned an invalid result")
        return result

    async def stream(
        self,
        question: str,
        *,
        audience_scope: AudienceScope = "unspecified",
        request_date: date | None = None,
        private_trace: dict[str, object] | None = None,
    ) -> AsyncIterator[ChatEvent]:
        """Run one bounded path and release answer text only after its gate."""

        started = time.monotonic()
        run_id = secrets.token_hex(12)
        # Freeze the business date once.  A request that crosses midnight must
        # not re-evaluate eligibility against a different date at the release
        # gate.
        effective_request_date = request_date or datetime.now(UTC).date()
        budget = current_budget()
        owns_scope = budget is None
        selected_budget = budget or RequestBudget(self.max_calls, self.timeout_seconds)
        if owns_scope:
            try:
                async with request_scope(selected_budget):
                    async for event in self._stream_inner(
                        question,
                        audience_scope,
                        effective_request_date,
                        private_trace,
                        started,
                        selected_budget,
                        run_id,
                    ):
                        yield event
            except TimeoutError:
                summary = _make_summary(
                    run_id,
                    route=None,
                    outcome="error",
                    verifier_status="unavailable",
                    budget=selected_budget,
                    started=started,
                    reason_code="WORKFLOW_TIMEOUT",
                )
                yield ChatEvent(
                    type="error",
                    data={
                        "stage": "generating",
                        "reason_code": "WORKFLOW_TIMEOUT",
                        "workflow_summary": summary.model_dump(),
                        "reference_allowed": False,
                    },
                )
            return
        async for event in self._stream_inner(
            question,
            audience_scope,
            effective_request_date,
            private_trace,
            started,
            selected_budget,
            run_id,
        ):
            yield event

    async def _stream_inner(
        self,
        question: str,
        audience_scope: AudienceScope,
        request_date: date | None,
        private_trace: dict[str, object] | None,
        started: float,
        budget: RequestBudget,
        run_id: str,
    ) -> AsyncIterator[ChatEvent]:
        try:
            eligibility = await self.preflight(question, audience_scope=audience_scope, request_date=request_date)
        except Exception:  # noqa: BLE001 - preflight fails closed
            yield ChatEvent(type="error", data={"stage": "preflight", "reason_code": "EVIDENCE_SCOPE_UNCLEAR"})
            return
        if not eligibility.version_ids:
            reason = eligibility.exclusion_reason or "INSUFFICIENT_EVIDENCE"
            summary = _make_summary(
                run_id,
                route=None,
                outcome="refused",
                verifier_status="not_run",
                budget=budget,
                started=started,
                reason_code=reason,
            )
            yield ChatEvent(
                type="final",
                data={
                    "refused": True,
                    "citations": [],
                    "reason_code": reason,
                    "workflow_summary": summary.model_dump(),
                    "reference_allowed": False,
                },
            )
            return

        yield ChatEvent(type="status", data={"stage": "routing"})
        decision: RouteDecision | None = None
        buffered: BufferedAnswer | None = None
        fallback_reason: str | None = None
        try:
            with stage_scope("routing"):
                decision = await self.router.route(question)
            if not isinstance(decision, RouteDecision):
                decision = RouteDecision.model_validate(decision)
        except WorkflowBudgetExceeded:
            summary = _make_summary(
                run_id,
                route=None,
                outcome="error",
                verifier_status="not_run",
                budget=budget,
                started=started,
                reason_code="WORKFLOW_BUDGET_EXCEEDED",
            )
            yield ChatEvent(
                type="error",
                data={
                    "stage": "routing",
                    "reason_code": "WORKFLOW_BUDGET_EXCEEDED",
                    "workflow_summary": summary.model_dump(),
                    "reference_allowed": False,
                },
            )
            return
        except RouterUnavailable:
            if self.baseline_service is None:
                summary = _make_summary(
                    run_id,
                    route=None,
                    outcome="error",
                    verifier_status="not_run",
                    budget=budget,
                    started=started,
                    reason_code="ROUTER_UNAVAILABLE",
                )
                yield ChatEvent(type="error", data={"stage": "routing", "reason_code": "ROUTER_UNAVAILABLE", "workflow_summary": summary.model_dump(), "reference_allowed": False})
                return
            fallback_reason = "ROUTER_UNAVAILABLE"
            _record_router_fallback(private_trace, fallback_reason)
            try:
                buffered = await _collect_service(
                    self.baseline_service,
                    question,
                    audience_scope,
                    private_trace=private_trace,
                )
            except WorkflowBudgetExceeded:
                summary = _make_summary(
                    run_id,
                    route=None,
                    outcome="error",
                    verifier_status="not_run",
                    budget=budget,
                    started=started,
                    reason_code="WORKFLOW_BUDGET_EXCEEDED",
                )
                yield ChatEvent(type="error", data={"stage": "generating", "reason_code": "WORKFLOW_BUDGET_EXCEEDED", "workflow_summary": summary.model_dump(), "reference_allowed": False})
                return
            except Exception:  # noqa: BLE001 - fallback fails closed
                summary = _make_summary(
                    run_id,
                    route=None,
                    outcome="error",
                    verifier_status="not_run",
                    budget=budget,
                    started=started,
                    reason_code="GENERATION_UNAVAILABLE",
                )
                yield ChatEvent(type="error", data={"stage": "generating", "reason_code": "GENERATION_UNAVAILABLE", "workflow_summary": summary.model_dump(), "reference_allowed": False})
                return
        except Exception:  # noqa: BLE001 - router fails closed
            if self.baseline_service is None:
                summary = _make_summary(
                    run_id,
                    route=None,
                    outcome="error",
                    verifier_status="not_run",
                    budget=budget,
                    started=started,
                    reason_code="ROUTER_UNAVAILABLE",
                )
                yield ChatEvent(type="error", data={"stage": "routing", "reason_code": "ROUTER_UNAVAILABLE", "workflow_summary": summary.model_dump(), "reference_allowed": False})
                return
            fallback_reason = "ROUTER_UNAVAILABLE"
            _record_router_fallback(private_trace, fallback_reason)
            try:
                buffered = await _collect_service(
                    self.baseline_service,
                    question,
                    audience_scope,
                    private_trace=private_trace,
                )
            except WorkflowBudgetExceeded:
                summary = _make_summary(
                    run_id,
                    route=None,
                    outcome="error",
                    verifier_status="not_run",
                    budget=budget,
                    started=started,
                    reason_code="WORKFLOW_BUDGET_EXCEEDED",
                )
                yield ChatEvent(type="error", data={"stage": "generating", "reason_code": "WORKFLOW_BUDGET_EXCEEDED", "workflow_summary": summary.model_dump(), "reference_allowed": False})
                return
            except Exception:  # noqa: BLE001 - fallback fails closed
                summary = _make_summary(
                    run_id,
                    route=None,
                    outcome="error",
                    verifier_status="not_run",
                    budget=budget,
                    started=started,
                    reason_code="GENERATION_UNAVAILABLE",
                )
                yield ChatEvent(type="error", data={"stage": "generating", "reason_code": "GENERATION_UNAVAILABLE", "workflow_summary": summary.model_dump(), "reference_allowed": False})
                return

        route = decision.route if decision is not None else None
        if route in {"clarify", "out_of_scope"}:
            final_reason = (
                "QUESTION_NEEDS_CLARIFICATION"
                if route == "clarify"
                else "OUT_OF_SCOPE"
            )
            summary = _make_summary(
                run_id,
                route=route,
                outcome="refused",
                verifier_status="not_run",
                budget=budget,
                started=started,
                reason_code=final_reason,
            )
            yield ChatEvent(
                type="final",
                data={"refused": True, "citations": [], "reason_code": final_reason, "workflow_summary": summary.model_dump(), "reference_allowed": False},
            )
            return

        if buffered is None:
            service = self.direct_service if route == "direct" else self.verify_service
            try:
                if self.retrieval is not None and (
                    self.direct_answerer is not None or self.verify_answerer is not None
                ):
                    answerer = (
                        self.direct_answerer
                        if route == "direct"
                        else self.verify_answerer
                    )
                    yield ChatEvent(type="status", data={"stage": "retrieving"})
                    yield ChatEvent(type="status", data={"stage": "reranking"})
                    retrieval, evidence_version_ids = await self._retrieve(
                        question,
                        audience_scope,
                        eligibility.version_ids,
                    )
                    with stage_scope("generating"):
                        buffered = await _collect_answerer(answerer, question, retrieval)
                    if evidence_version_ids and not await self._evidence_still_eligible(
                        evidence_version_ids,
                        audience_scope,
                        request_date,
                    ):
                        buffered = BufferedAnswer(
                            (
                                ChatEvent(
                                    type="final",
                                    data={
                                        "refused": True,
                                        "citations": [],
                                        "reason_code": "EVIDENCE_SCOPE_UNCLEAR",
                                    },
                                ),
                            )
                        )
                else:
                    buffered = await _collect_service(
                        service,
                        question,
                        audience_scope,
                        private_trace=private_trace,
                    )
            except WorkflowBudgetExceeded:
                summary = _make_summary(
                    run_id,
                    route=route,
                    outcome="error",
                    verifier_status="not_run",
                    budget=budget,
                    started=started,
                    reason_code="WORKFLOW_BUDGET_EXCEEDED",
                )
                yield ChatEvent(
                    type="error",
                    data={
                        "stage": "generating",
                        "reason_code": "WORKFLOW_BUDGET_EXCEEDED",
                        "workflow_summary": summary.model_dump(),
                        "reference_allowed": False,
                    },
                )
                return
            except RetrievalUnavailableError:
                summary = _make_summary(
                    run_id,
                    route=route,
                    outcome="error",
                    verifier_status="not_run",
                    budget=budget,
                    started=started,
                    reason_code="RETRIEVAL_UNAVAILABLE",
                )
                yield ChatEvent(
                    type="error",
                    data={
                        "stage": "retrieving",
                        "reason_code": "RETRIEVAL_UNAVAILABLE",
                        "workflow_summary": summary.model_dump(),
                        "reference_allowed": False,
                    },
                )
                return

            except Exception:  # noqa: BLE001 - workflow fails closed
                summary = _make_summary(
                    run_id,
                    route=route,
                    outcome="error",
                    verifier_status=(
                        "failed" if self._verifier_enabled(route) else "not_run"
                    ),
                    budget=budget,
                    started=started,
                    reason_code="GENERATION_UNAVAILABLE",
                )
                yield ChatEvent(type="error", data={"stage": "generating", "reason_code": "GENERATION_UNAVAILABLE", "workflow_summary": summary.model_dump(), "reference_allowed": False})
                return
        if buffered is None:
            # The branches above either collect a result or emit a terminal
            # error.  Keep this guard explicit for type checkers and future
            # changes to the fallback path.
            return
        terminal = buffered.terminal
        if terminal is None:
            summary = _make_summary(
                run_id,
                route=route,
                outcome="error",
                verifier_status=(
                    "failed" if self._verifier_enabled(route) else "not_run"
                ),
                budget=budget,
                started=started,
                reason_code=fallback_reason or "GENERATION_UNAVAILABLE",
            )
            yield ChatEvent(type="error", data={"stage": "generating", "reason_code": "GENERATION_UNAVAILABLE", "workflow_summary": summary.model_dump(), "reference_allowed": False})
            return
        for event in buffered.events:
            if event.type == "status":
                yield event

        verifier_status = (
            _verifier_status(buffered)
            if self._verifier_enabled(route)
            else "not_run"
        )
        data = dict(terminal.data)
        if (
            self._verifier_enabled(route)
            and verifier_status != "passed"
            and data.get("refused") is False
        ):
            data = {"refused": True, "citations": [], "reason_code": "ANSWER_NOT_VERIFIABLE"}
        if terminal.type == "error":
            data["reference_allowed"] = False
            data["workflow_summary"] = _make_summary(
                run_id,
                route=route,
                outcome="error",
                verifier_status=verifier_status,
                budget=budget,
                started=started,
                reason_code=data.get("reason_code")
                if isinstance(data.get("reason_code"), str)
                else "GENERATION_UNAVAILABLE",
            ).model_dump()
            yield ChatEvent(type="error", data=data)
            return
        final_refused = data.get("refused") is not False
        summary = _make_summary(
            run_id,
            route=route,
            outcome="refused" if final_refused else "answered",
            verifier_status=verifier_status,
            budget=budget,
            started=started,
            reason_code=data.get("reason_code")
            if isinstance(data.get("reason_code"), str)
            else fallback_reason,
        )
        data["workflow_summary"] = summary.model_dump()
        data["reference_allowed"] = (
            final_refused and bool(data.get("reference_allowed", True))
            if fallback_reason is not None
            else final_refused and route == "direct"
        )
        if data.get("refused") is False:
            for event in buffered.events:
                if event.type == "answer_delta":
                    yield event
        yield ChatEvent(type="final", data=data)

    async def _retrieve(
        self,
        question: str,
        audience_scope: AudienceScope,
        eligible_version_ids: frozenset[str],
    ) -> tuple[RetrievalResult, set[str]]:
        """Use the existing two-step retrieval boundary with a stable allowlist."""

        retrieval = self.retrieval
        retrieve_candidates = getattr(retrieval, "retrieve_candidates", None)
        try:
            if callable(retrieve_candidates):
                with stage_scope("retrieving"):
                    candidates = await _call_audience(
                        retrieve_candidates, question, audience_scope
                    )
                if getattr(candidates, "reason_code", None):
                    return RetrievalResult(
                        status="refused",
                        reason_code=candidates.reason_code,
                    ), set()
            else:
                retrieve = getattr(retrieval, "retrieve", None)
                if not callable(retrieve):
                    raise TypeError("retrieval service does not provide retrieve")
                with stage_scope("retrieving"):
                    candidates = await _call_audience(retrieve, question, audience_scope)
            rerank = getattr(retrieval, "rerank", None)
            if callable(rerank):
                with stage_scope("reranking"):
                    result = await rerank(question, candidates)
            elif isinstance(candidates, RetrievalResult):
                result = candidates
            else:
                raise TypeError("retrieval service does not provide rerank")
        except Exception as error:
            if isinstance(error, WorkflowBudgetExceeded):
                raise
            raise RetrievalUnavailableError("retrieval unavailable") from error
        if not isinstance(result, RetrievalResult):
            raise TypeError("retrieval result is invalid")
        evidence_ids = {
            item.hit.version_id
            for item in result.evidence
            if isinstance(item.hit.version_id, str)
        }
        if not evidence_ids.issubset(eligible_version_ids):
            return RetrievalResult(
                status="refused",
                reason_code="EVIDENCE_SCOPE_UNCLEAR",
            ), evidence_ids
        return result, evidence_ids

    async def _evidence_still_eligible(
        self,
        version_ids: set[str],
        audience_scope: AudienceScope,
        request_date: date | None,
    ) -> bool:
        if self.repository is None:
            return True
        current = await self.preflight(
            "资格复核",
            audience_scope=audience_scope,
            request_date=request_date,
        )
        return version_ids.issubset(current.version_ids)


def _verifier_status(buffered: BufferedAnswer) -> str:
    if any(event.type == "status" and event.data.get("stage") == "verifying" for event in buffered.events):
        terminal = buffered.terminal
        if terminal and terminal.type == "final" and terminal.data.get("refused") is False:
            return "passed"
        return "failed"
    return "failed" if buffered.refused else "not_run"


def _make_summary(
    run_id: str,
    *,
    route: str | None,
    outcome: str,
    verifier_status: str,
    budget: RequestBudget,
    started: float,
    reason_code: object,
) -> WorkflowSummary:
    return WorkflowSummary(
        run_id=run_id,
        route=route,
        outcome=outcome,
        verifier_status=verifier_status,
        http_calls=budget.http_calls,
        elapsed_ms=max(0, int((time.monotonic() - started) * 1000)),
        reason_code=reason_code if isinstance(reason_code, str) else None,
    )


async def _collect_service(
    service: Any,
    question: str,
    audience_scope: AudienceScope,
    *,
    private_trace: dict[str, object] | None = None,
) -> BufferedAnswer:
    """Buffer either an AnswerService-style or a RagService-style stream."""

    method = getattr(service, "stream", None)
    if not callable(method):
        raise TypeError("workflow service does not provide stream")
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        parameters = {}
    required_positional = [
        parameter
        for parameter in parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and parameter.default is inspect.Parameter.empty
    ]
    if len(required_positional) >= 2:
        stream = method(question, None)
    else:
        kwargs: dict[str, object] = {}
        if "audience_scope" in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            kwargs["audience_scope"] = audience_scope
        if private_trace is not None and (
            "private_trace" in parameters
            or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
        ):
            kwargs["private_trace"] = private_trace
        stream = method(question, **kwargs)
    events: list[ChatEvent] = []
    async for event in stream:
        if not isinstance(event, ChatEvent):
            raise TypeError("workflow service returned an invalid event")
        events.append(event)
    return BufferedAnswer(tuple(events))


async def _collect_answerer(
    answerer: Any,
    question: str,
    retrieval: RetrievalResult,
) -> BufferedAnswer:
    stream = getattr(answerer, "stream", None)
    if not callable(stream):
        raise TypeError("answerer does not provide stream")
    events: list[ChatEvent] = []
    async for event in stream(question, retrieval):
        if not isinstance(event, ChatEvent):
            raise TypeError("answerer returned an invalid event")
        events.append(event)
    return BufferedAnswer(tuple(events))


async def _call_audience(method: Any, question: str, audience_scope: AudienceScope) -> Any:
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        result = method(question)
        return await result if inspect.isawaitable(result) else result
    audience = parameters.get("audience_scope")
    if audience is not None and audience.kind is inspect.Parameter.POSITIONAL_ONLY:
        result = method(question, audience_scope)
        return await result if inspect.isawaitable(result) else result
    if audience is not None or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        result = method(question, audience_scope=audience_scope)
        return await result if inspect.isawaitable(result) else result
    result = method(question)
    return await result if inspect.isawaitable(result) else result


class DeterministicVerifier:
    """Offline verifier seam for the demo; it never constructs a provider."""

    async def verify(self, question: str, answer: str, evidence: Any) -> VerificationResult:
        values = tuple(evidence)
        if not values or not isinstance(answer, str) or not answer.strip():
            raise ValueError("demo verification requires an answer and evidence")
        return VerificationResult(
            claims=[
                ClaimVerdict(
                    claim=answer,
                    verdict="supported",
                    source_ids=[values[0].reference_id],
                )
            ]
        )


def _dangerous_question(question: str) -> bool:
    lowered = question.casefold()
    return bool(ABSOLUTE_PATH_RE.search(question)) or any(marker in lowered for marker in _DANGEROUS_QUERY_MARKERS)


def _record_router_fallback(
    private_trace: dict[str, object] | None,
    reason_code: str,
) -> None:
    """Record bounded fallback facts without retaining provider output."""

    if private_trace is None:
        return
    private_trace["router_fallback"] = True
    private_trace["router_failure_code"] = reason_code


__all__ = ["AgentWorkflow", "DeterministicVerifier", "RetrievalUnavailableError"]
