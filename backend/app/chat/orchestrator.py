"""Coordinate sessions, bounded history, RAG and terminal persistence."""

from __future__ import annotations

import inspect
import math
from collections.abc import AsyncIterator, Mapping
from typing import Any

from app.agents.budget import (
    RequestBudget,
    WorkflowBudgetExceeded,
    current_budget,
    request_scope,
    stage_scope,
)
from app.chat.history import HistoryContextBuilder
from app.chat.safety import (
    REFERENCE_REASON_CODES,
    SAFE_ERROR_TEXT,
    is_safe_answer_text,
    is_safe_reference_id,
    normalize_workflow_summary,
    validate_audience_scope,
    validate_candidate_sources,
    validate_citations,
    validate_reason_code,
)
from app.domain.ports import AudienceScope
from app.rag.models import ChatEvent
from app.rag.query import clean_question

_RAG_EVENT_TYPES = frozenset({"status", "answer_delta", "final", "error"})
_STATUS_STAGES = frozenset(
    {
        "preflight",
        "routing",
        "retrieving",
        "reranking",
        "generating",
        "validating",
        "verifying",
    }
)
_RETRIEVAL_DIAGNOSTIC_FIELDS = frozenset(
    {
        "candidate_count",
        "evidence_count",
        "highest_rerank_score",
        "status",
        "reason_code",
    }
)
_RETRIEVAL_STATUSES = frozenset({"ready", "refused", "unavailable"})
_RETRIEVAL_REASON_CODES = frozenset(
    {
        "INSUFFICIENT_EVIDENCE",
        "RETRIEVAL_UNAVAILABLE",
        "EVIDENCE_SCOPE_UNCLEAR",
        "DOCUMENT_BUSINESS_STATUS_UNKNOWN",
    }
)
_REASON_CODES = frozenset(
    {
        "INSUFFICIENT_EVIDENCE",
        "MODEL_REFUSED",
        "ANSWER_NOT_VERIFIABLE",
        "RETRIEVAL_UNAVAILABLE",
        "GENERATION_UNAVAILABLE",
        "FOLLOW_UP_UNAVAILABLE",
        "CHAT_UNAVAILABLE",
        "INVALID_EVENT",
        "PROVIDER_UNAVAILABLE",
    }
)
_REFUSED_TEXT = "当前资料不足，无法提供经过核验的回答。请补充更具体的问题后重试。"
_ERROR_TEXT = SAFE_ERROR_TEXT


class ChatOrchestrator:
    """Keep persistence and conversational policy outside the RAG service."""

    def __init__(
        self,
        repository: Any,
        rewriter: Any,
        rag_service: Any,
        history_builder: HistoryContextBuilder | None = None,
        reference_generator: Any | None = None,
    ) -> None:
        self.repository = repository
        self.rewriter = rewriter
        self.rag_service = rag_service
        self.history_builder = history_builder or HistoryContextBuilder()
        self.reference_generator = reference_generator

    async def stream(
        self,
        question: str,
        session_id: str | None = None,
        *,
        audience_scope: AudienceScope = "unspecified",
    ) -> AsyncIterator[ChatEvent]:
        """Stream one request, adding a budget only for the candidate workflow."""

        if getattr(self.rag_service, "requires_request_scope", False) and current_budget() is None:
            async with request_scope(RequestBudget()):
                async for event in self._stream_legacy(
                    question, session_id, audience_scope=audience_scope
                ):
                    yield event
            return
        async for event in self._stream_legacy(
            question, session_id, audience_scope=audience_scope
        ):
            yield event

    async def _stream_legacy(
        self,
        question: str,
        session_id: str | None = None,
        *,
        audience_scope: AudienceScope = "unspecified",
    ) -> AsyncIterator[ChatEvent]:
        """Stream safe events while ensuring every terminal result is stored."""

        cleaned_question = clean_question(question)
        audience_scope = validate_audience_scope(audience_scope)
        if session_id is None:
            session_id = self.repository.create_chat_session()
        elif not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id is invalid")
        elif not self.repository.chat_session_exists(session_id):
            raise KeyError(f"Unknown chat session: {session_id}")

        user_message = _append_user_message(
            self.repository,
            session_id,
            cleaned_question,
            audience_scope,
        )
        yield ChatEvent(
            type="status",
            data={
                "stage": "accepted",
                "session_id": session_id,
                "message_id": user_message.message_id,
            },
        )

        effective_question = cleaned_question
        history = self.history_builder.build(
            self.repository.list_chat_messages(session_id)
        )
        if history.turns:
            yield ChatEvent(
                type="status",
                data={"stage": "rewriting", "session_id": session_id},
            )
            try:
                with stage_scope("rewriting"):
                    effective_question = await self.rewriter.rewrite(
                        cleaned_question, history
                    )
                effective_question = clean_question(effective_question)
                self.repository.set_rewritten_question(
                    user_message.message_id, effective_question
                )
            except Exception as error:
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                assistant = self.repository.append_assistant_message(
                    session_id,
                    user_message.message_id,
                    _ERROR_TEXT,
                    "error",
                    (),
                    "FOLLOW_UP_UNAVAILABLE",
                )
                yield self._error_event(
                    session_id, assistant.message_id, "FOLLOW_UP_UNAVAILABLE"
                )
                return

        answer_parts: list[str] = []
        terminal = False
        try:
            rag_stream = _stream_with_optional_audience(
                self.rag_service,
                effective_question,
                audience_scope,
            )
            async for event in rag_stream:
                if not isinstance(event, ChatEvent) or event.type not in _RAG_EVENT_TYPES:
                    assistant = self._persist_error(
                        session_id, user_message.message_id, "INVALID_EVENT"
                    )
                    yield self._error_event(
                        session_id, assistant.message_id, "INVALID_EVENT"
                    )
                    terminal = True
                    return

                safe_data = self._validate_event_data(event)
                if safe_data is None:
                    assistant = self._persist_error(
                        session_id, user_message.message_id, "INVALID_EVENT"
                    )
                    yield self._error_event(
                        session_id, assistant.message_id, "INVALID_EVENT"
                    )
                    terminal = True
                    return
                event = ChatEvent(type=event.type, data=safe_data)

                if event.type == "status":
                    stage = event.data["stage"]
                    yield ChatEvent(
                        type="status",
                        data={"stage": stage},
                    )
                    continue

                if event.type == "answer_delta":
                    text = event.data["text"]
                    if not isinstance(text, str):
                        raise AssertionError("validated answer delta text is not text")
                    if not is_safe_answer_text("".join(answer_parts) + text):
                        assistant = self._persist_error(
                            session_id, user_message.message_id, "INVALID_EVENT"
                        )
                        yield self._error_event(
                            session_id, assistant.message_id, "INVALID_EVENT"
                        )
                        terminal = True
                        return
                    answer_parts.append(text)
                    delta_data: dict[str, object] = {
                        "text": text,
                        "source_ids": event.data["source_ids"],
                    }
                    yield ChatEvent(type="answer_delta", data=delta_data)
                    continue

                if event.type == "final":
                    assistant, final_event = await self._persist_final(
                        session_id,
                        user_message.message_id,
                        effective_question,
                        answer_parts,
                        event.data,
                    )
                    terminal = True
                    yield final_event
                    return

                # ``error`` is terminal and never includes provider payloads.
                reason_code = event.data["reason_code"]
                if not isinstance(reason_code, str):
                    raise TypeError("validated reason code is not text")
                workflow_summary = event.data.get("workflow_summary")
                assistant = self._persist_error(
                    session_id,
                    user_message.message_id,
                    reason_code,
                    workflow_summary=workflow_summary,
                )
                terminal = True
                yield self._error_event(
                    session_id,
                    assistant.message_id,
                    reason_code,
                    workflow_summary=workflow_summary,
                )
                return
        except TimeoutError:
            if not terminal:
                assistant = self._persist_error(
                    session_id, user_message.message_id, "WORKFLOW_TIMEOUT"
                )
                yield self._error_event(
                    session_id, assistant.message_id, "WORKFLOW_TIMEOUT"
                )
                return
            raise
        except WorkflowBudgetExceeded:
            if not terminal:
                assistant = self._persist_error(
                    session_id, user_message.message_id, "WORKFLOW_BUDGET_EXCEEDED"
                )
                yield self._error_event(
                    session_id, assistant.message_id, "WORKFLOW_BUDGET_EXCEEDED"
                )
                return
            raise
        except Exception:
            if not terminal:
                assistant = self._persist_error(
                    session_id, user_message.message_id, "CHAT_UNAVAILABLE"
                )
                yield self._error_event(
                    session_id, assistant.message_id, "CHAT_UNAVAILABLE"
                )
                return
            raise

        if not terminal:
            assistant = self._persist_error(
                session_id, user_message.message_id, "CHAT_UNAVAILABLE"
            )
            yield self._error_event(
                session_id, assistant.message_id, "CHAT_UNAVAILABLE"
            )

    async def _persist_final(
        self,
        session_id: str,
        user_message_id: str,
        question: str,
        answer_parts: list[str],
        data: Mapping[str, object],
    ) -> tuple[Any, ChatEvent]:
        refused = data.get("refused")
        citations = data.get("citations", [])
        reason_code = data.get("reason_code")
        if not isinstance(refused, bool):
            refused = True
            reason_code = "ANSWER_NOT_VERIFIABLE"
        if not isinstance(citations, list):
            refused = True
            citations = []
            reason_code = "ANSWER_NOT_VERIFIABLE"
        if not refused and not citations:
            refused = True
            reason_code = "ANSWER_NOT_VERIFIABLE"
        safe_reason = _safe_reason_code(
            reason_code,
            fallback="INSUFFICIENT_EVIDENCE" if refused else None,
        )
        status = "refused" if refused else "answered"
        content = _REFUSED_TEXT if refused else "".join(answer_parts)
        reference_question = data.get("reference_question")
        reference_allowed = data.get("reference_allowed", True)
        if not isinstance(reference_allowed, bool):
            reference_allowed = False
        reference_answer: str | None = None
        if (
            reference_allowed
            and (
                isinstance(reference_question, str)
                or (refused and safe_reason in REFERENCE_REASON_CODES)
            )
            and self.reference_generator is not None
        ):
            try:
                with stage_scope("reference"):
                    reference_answer = await self.reference_generator.generate(
                        reference_question if isinstance(reference_question, str) else question
                    )
            except WorkflowBudgetExceeded:
                raise
            except Exception:  # noqa: BLE001 - optional enrichment must not fail chat
                reference_answer = None
        workflow_summary = data.get("workflow_summary")
        if workflow_summary is not None:
            workflow_summary = _validate_workflow_summary(workflow_summary)
        assistant = _append_assistant_message(
            self.repository,
            session_id,
            user_message_id,
            content,
            status,
            [] if refused else [dict(item) for item in citations],
            safe_reason,
            reference_answer,
            workflow_summary,
        )
        final_data: dict[str, object] = {
            "refused": refused,
            "citations": [] if refused else [dict(item) for item in citations],
            "session_id": session_id,
            "message_id": assistant.message_id,
        }
        if safe_reason is not None:
            final_data["reason_code"] = safe_reason
        if reference_answer is not None:
            final_data["reference_answer"] = reference_answer
        if workflow_summary is not None:
            final_data["workflow_summary"] = workflow_summary
        return assistant, ChatEvent(type="final", data=final_data)

    def _persist_error(
        self,
        session_id: str,
        user_message_id: str,
        reason_code: str,
        *,
        workflow_summary: object | None = None,
    ) -> Any:
        summary = (
            _validate_workflow_summary(workflow_summary)
            if workflow_summary is not None
            else None
        )
        return _append_assistant_message(
            self.repository,
            session_id,
            user_message_id,
            _ERROR_TEXT,
            "error",
            (),
            reason_code,
            None,
            summary,
        )

    @staticmethod
    def _validate_event_data(event: ChatEvent) -> dict[str, object] | None:
        """Normalize one RAG event before it can affect persistence or SSE."""

        if not isinstance(event.data, Mapping):
            return None
        data = dict(event.data)
        if event.type == "status":
            if set(data) != {"stage"} or data.get("stage") not in _STATUS_STAGES:
                return None
            return {"stage": data["stage"]}
        if event.type == "answer_delta":
            if not set(data).issubset({"text", "source_ids"}):
                return None
            text = data.get("text")
            if not isinstance(text, str) or not is_safe_answer_text(text):
                return None
            source_ids = data.get("source_ids", [])
            if not isinstance(source_ids, list) or not all(
                is_safe_reference_id(item) for item in source_ids
            ):
                return None
            return {"text": text, "source_ids": list(source_ids)}
        if event.type == "final":
            if not set(data).issubset(
                {
                    "refused",
                    "citations",
                    "reason_code",
                    "reference_question",
                    "retrieval_diagnostics",
                    "workflow_summary",
                    "reference_allowed",
                }
            ):
                return None
            if not isinstance(data.get("refused"), bool):
                return None
            try:
                citations = validate_citations(data.get("citations", []))
                reason_code = validate_reason_code(data.get("reason_code"), allow_none=True)
            except (TypeError, ValueError):
                return None
            result: dict[str, object] = {
                "refused": data["refused"],
                "citations": citations,
            }
            if "retrieval_diagnostics" in data:
                diagnostics = _validate_retrieval_diagnostics(
                    data["retrieval_diagnostics"]
                )
                if diagnostics is None:
                    return None
                result["retrieval_diagnostics"] = diagnostics
            if "workflow_summary" in data:
                summary = _validate_workflow_summary(data["workflow_summary"])
                if summary is None:
                    return None
                result["workflow_summary"] = summary
            if "reference_allowed" in data:
                if not isinstance(data["reference_allowed"], bool):
                    return None
                result["reference_allowed"] = data["reference_allowed"]
            if reason_code is not None:
                result["reason_code"] = reason_code
            reference_question = data.get("reference_question")
            if reference_question is not None:
                if not isinstance(reference_question, str):
                    return None
                try:
                    result["reference_question"] = clean_question(reference_question)
                except (TypeError, ValueError):
                    return None
            return result
        if event.type == "error":
            if not set(data).issubset(
                {
                    "stage",
                    "reason_code",
                    "candidate_sources",
                    "retrieval_diagnostics",
                    "workflow_summary",
                    "reference_allowed",
                }
            ):
                return None
            stage = data.get("stage", "chat")
            if stage not in _STATUS_STAGES and stage != "chat":
                return None
            try:
                reason_code = validate_reason_code(data.get("reason_code"))
                candidate_sources = (
                    validate_candidate_sources(data["candidate_sources"])
                    if "candidate_sources" in data
                    else None
                )
            except (TypeError, ValueError):
                return None
            result = {"stage": stage, "reason_code": reason_code}
            if candidate_sources is not None:
                result["candidate_sources"] = candidate_sources
            if "retrieval_diagnostics" in data:
                diagnostics = _validate_retrieval_diagnostics(
                    data["retrieval_diagnostics"]
                )
                if diagnostics is None:
                    return None
                result["retrieval_diagnostics"] = diagnostics
            if "workflow_summary" in data:
                summary = _validate_workflow_summary(data["workflow_summary"])
                if summary is None:
                    return None
                result["workflow_summary"] = summary
            if "reference_allowed" in data:
                if not isinstance(data["reference_allowed"], bool):
                    return None
                result["reference_allowed"] = data["reference_allowed"]
            return result
        return None

    @staticmethod
    def _error_event(
        session_id: str,
        message_id: str,
        reason_code: str,
        *,
        workflow_summary: object | None = None,
    ) -> ChatEvent:
        summary = (
            _validate_workflow_summary(workflow_summary)
            if workflow_summary is not None
            else None
        )
        data: dict[str, object] = {
            "stage": "chat",
            "reason_code": reason_code,
            "session_id": session_id,
            "message_id": message_id,
        }
        if summary is not None:
            data["workflow_summary"] = summary
        return ChatEvent(
            type="error",
            data=data,
        )


def _safe_reason_code(value: object, *, fallback: str = "CHAT_UNAVAILABLE") -> str:
    try:
        normalized = validate_reason_code(value)
    except (TypeError, ValueError):
        return fallback
    return normalized


def _validate_workflow_summary(value: object) -> dict[str, object] | None:
    try:
        return normalize_workflow_summary(value)
    except (TypeError, ValueError):
        return None


def _append_assistant_message(
    repository: Any,
    session_id: str,
    reply_to_message_id: str,
    content: str,
    status: str,
    citations: object,
    reason_code: str | None,
    reference_answer: str | None = None,
    workflow_summary: Mapping[str, object] | None = None,
) -> Any:
    """Pass optional workflow facts while retaining old repository fakes."""

    method = repository.append_assistant_message
    args = [
        session_id,
        reply_to_message_id,
        content,
        status,
        citations,
        reason_code,
    ]
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "workflow_summary" in parameters:
        return method(
            *args,
            reference_answer=reference_answer,
            workflow_summary=workflow_summary,
        )
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return method(
            *args,
            reference_answer=reference_answer,
            workflow_summary=workflow_summary,
        )
    if reference_answer is not None:
        return method(*args, reference_answer=reference_answer)
    return method(*args)


def _append_user_message(
    repository: Any,
    session_id: str,
    question: str,
    audience_scope: AudienceScope,
) -> Any:
    """Pass the new scope field while retaining compatibility with old fakes."""

    method = repository.append_user_message
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return method(session_id, question)
    audience = parameters.get("audience_scope")
    if audience is not None and audience.kind is inspect.Parameter.POSITIONAL_ONLY:
        return method(session_id, question, audience_scope)
    if audience is not None or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return method(session_id, question, audience_scope=audience_scope)
    return method(session_id, question)


def _stream_with_optional_audience(
    service: Any,
    question: str,
    audience_scope: AudienceScope,
):
    """Call audience-aware RAG services without breaking legacy injections."""

    method = service.stream
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return method(question)
    audience = parameters.get("audience_scope")
    if audience is not None and audience.kind is inspect.Parameter.POSITIONAL_ONLY:
        return method(question, audience_scope)
    if audience is not None or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return method(question, audience_scope=audience_scope)
    return method(question)


def _validate_retrieval_diagnostics(
    value: object,
) -> dict[str, object] | None:
    """Validate private retrieval metrics before they cross the chat boundary."""

    if not isinstance(value, Mapping) or set(value) != _RETRIEVAL_DIAGNOSTIC_FIELDS:
        return None
    candidate_count = value.get("candidate_count")
    evidence_count = value.get("evidence_count")
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count < 0
        or isinstance(evidence_count, bool)
        or not isinstance(evidence_count, int)
        or evidence_count < 0
    ):
        return None
    highest_score = value.get("highest_rerank_score")
    if highest_score is not None:
        if (
            isinstance(highest_score, bool)
            or not isinstance(highest_score, (int, float))
        ):
            return None
        try:
            highest_score = float(highest_score)
        except (OverflowError, ValueError):
            return None
        if not math.isfinite(highest_score):
            return None
    status = value.get("status")
    if not isinstance(status, str) or status not in _RETRIEVAL_STATUSES:
        return None
    reason_code = value.get("reason_code")
    if reason_code is not None and (
        not isinstance(reason_code, str) or reason_code not in _RETRIEVAL_REASON_CODES
    ):
        return None
    return {
        "candidate_count": candidate_count,
        "evidence_count": evidence_count,
        "highest_rerank_score": highest_score,
        "status": status,
        "reason_code": reason_code,
    }


__all__ = ["ChatOrchestrator"]
