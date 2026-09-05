"""Streaming chat endpoint with a narrow, non-sensitive SSE contract."""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import AsyncIterator
from typing import Any

from app.chat.safety import (
    REFERENCE_REASON_CODES,
    is_safe_answer_text,
    is_safe_id,
    is_safe_reference_id,
    normalize_chat_message,
    normalize_feedback,
    normalize_workflow_summary,
    validate_candidate_sources,
    validate_citation,
    validate_citations,
    validate_reason_code,
)
from app.domain.ports import AudienceScope, FeedbackReason
from app.rag.models import ChatEvent
from app.rag.query import clean_question
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StrictStr,
    field_validator,
    model_validator,
)

_ALLOWED_EVENT_TYPES = frozenset({"status", "answer_delta", "final", "error"})
_ALLOWED_STATUS_STAGES = frozenset(
    {
        "accepted",
        "rewriting",
        "preflight",
        "routing",
        "retrieving",
        "reranking",
        "generating",
        "validating",
        "verifying",
    }
)
_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: StrictStr
    session_id: StrictStr | None = None
    audience_scope: AudienceScope = "unspecified"

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str | None) -> str | None:
        if value is not None and not _SESSION_ID.fullmatch(value):
            raise ValueError("session_id is invalid")
        return value


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    helpful: StrictBool
    reason: FeedbackReason | None = None

    @model_validator(mode="after")
    def validate_reason_for_helpful_feedback(self) -> FeedbackRequest:
        if self.helpful and self.reason is not None:
            raise ValueError("helpful feedback cannot include a reason")
        return self


def create_chat_router() -> APIRouter:
    router = APIRouter(prefix="/api/chat", tags=["chat"])

    @router.post("/stream")
    async def stream_chat(payload: ChatRequest, request: Request) -> StreamingResponse:
        try:
            question = clean_question(payload.question)
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail="question is invalid") from error

        orchestrator = getattr(request.app.state, "chat_orchestrator", None)
        if orchestrator is not None:
            repository = _repository(request, orchestrator)
            if payload.session_id is not None and not repository.chat_session_exists(
                payload.session_id
            ):
                raise HTTPException(status_code=404, detail="chat session not found")
            return StreamingResponse(
                _safe_events(
                    orchestrator,
                    question,
                    payload.session_id,
                    audience_scope=payload.audience_scope,
                    require_terminal_metadata=True,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Content-Type-Options": "nosniff",
                },
            )

        rag_service = getattr(request.app.state, "rag_service", None)
        if rag_service is None:
            raise HTTPException(status_code=503, detail="chat service is not configured")

        return StreamingResponse(
            _safe_events(rag_service, question, audience_scope=payload.audience_scope),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @router.get("/sessions")
    async def list_sessions(
        request: Request,
        limit: int = Query(default=20, ge=1, le=50),
    ) -> dict[str, object]:
        repository = _repository(request)
        if repository is None:
            raise HTTPException(status_code=503, detail="chat service is not configured")
        try:
            rows = repository.list_chat_sessions(limit)
        except ValueError as error:
            raise HTTPException(status_code=422, detail="session limit is invalid") from error
        return {
            "sessions": [
                {
                    "session_id": row.session_id,
                    "title": row.title,
                    "created_at": row.created_at,
                    "last_activity_at": row.last_activity_at,
                    "message_count": row.message_count,
                }
                for row in rows
            ]
        }

    @router.get("/sessions/{session_id}")
    async def restore_session(session_id: str, request: Request) -> dict[str, object]:
        _validate_session_id_or_422(session_id)
        repository = _repository(request)
        if repository is None or not repository.chat_session_exists(session_id):
            raise HTTPException(status_code=404, detail="chat session not found")
        rows = repository.list_chat_messages(session_id)
        messages: list[dict[str, object]] = []
        for row in rows:
            safe_row = normalize_chat_message(row, fallback_session_id=session_id)
            feedback = repository.get_feedback(safe_row.message_id)
            messages.append(
                {
                    "message_id": safe_row.message_id,
                    "role": safe_row.role,
                    "content": safe_row.content,
                    "status": safe_row.status,
                    "rewritten_question": safe_row.rewritten_question,
                    "citations": [dict(item) for item in safe_row.citations],
                    "reason_code": safe_row.reason_code,
                    "reference_answer": safe_row.reference_answer,
                    "audience_scope": safe_row.audience_scope,
                    "workflow_summary": safe_row.workflow_summary,
                    "created_at": safe_row.created_at,
                    "feedback": _feedback_dict(feedback),
                }
            )
        return {"session_id": session_id, "messages": messages}

    @router.put("/messages/{message_id}/feedback")
    async def save_feedback(
        message_id: str, payload: FeedbackRequest, request: Request
    ) -> dict[str, object]:
        repository = _repository(request)
        if repository is None:
            raise HTTPException(status_code=503, detail="chat service is not configured")
        try:
            feedback = _upsert_feedback_with_optional_reason(
                repository, message_id, payload.helpful, payload.reason
            )
            if inspect.isawaitable(feedback):
                feedback = await feedback
            return _feedback_dict(feedback) or {}
        except KeyError as error:
            raise HTTPException(status_code=404, detail="chat message not found") from error
        except (TypeError, ValueError) as error:
            raise HTTPException(
                status_code=400, detail="feedback requires a final assistant message"
            ) from error

    return router


async def _safe_events(
    service: Any,
    question: str,
    session_id: str | None = None,
    *,
    audience_scope: AudienceScope = "unspecified",
    require_terminal_metadata: bool = False,
) -> AsyncIterator[str]:
    try:
        stream = _stream_with_optional_audience(
            service,
            question,
            session_id,
            audience_scope,
            require_terminal_metadata,
        )
        async for event in stream:
            if not isinstance(event, ChatEvent):
                raise TypeError("chat service returned an invalid event")
            if not isinstance(event.type, str) or event.type not in _ALLOWED_EVENT_TYPES:
                yield _encode_sse(
                    ChatEvent(
                        type="error",
                        data={"stage": "chat", "reason_code": "INVALID_EVENT"},
                    )
                )
                return
            safe_event = _safe_event(event, require_terminal_metadata=require_terminal_metadata)
            if safe_event is None:
                yield _encode_sse(
                    ChatEvent(
                        type="error",
                        data={"stage": "chat", "reason_code": "INVALID_EVENT"},
                    )
                )
                return
            yield _encode_sse(safe_event)
    except Exception:  # noqa: BLE001 - stream boundary must fail closed
        # Streaming errors happen after the HTTP status has been sent.  Keep
        # the final frame typed and safe instead of exposing provider details.
        yield _encode_sse(
            ChatEvent(
                type="error",
                data={"stage": "chat", "reason_code": "CHAT_UNAVAILABLE"},
            )
        )


async def _legacy_safe_events(service: Any, question: str) -> AsyncIterator[str]:
    """Compatibility wrapper for direct single-turn test/service injection."""

    async for frame in _safe_events(service, question):
        yield frame


def _stream_with_optional_audience(
    service: Any,
    question: str,
    session_id: str | None,
    audience_scope: AudienceScope,
    require_terminal_metadata: bool,
):
    """Call new scope-aware services while preserving legacy test doubles."""

    method = service.stream
    args: list[object] = [question]
    if require_terminal_metadata:
        args.append(session_id)
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return method(*args)
    audience = parameters.get("audience_scope")
    if audience is not None and audience.kind is inspect.Parameter.POSITIONAL_ONLY:
        args.append(audience_scope)
        return method(*args)
    if audience is not None or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return method(*args, audience_scope=audience_scope)
    return method(*args)


def _safe_event(event: ChatEvent, *, require_terminal_metadata: bool) -> ChatEvent | None:
    if event.type not in _ALLOWED_EVENT_TYPES or not isinstance(event.data, dict):
        return None
    data = event.data
    if event.type == "status":
        stage = data.get("stage")
        if stage not in _ALLOWED_STATUS_STAGES:
            return None
        if require_terminal_metadata and stage == "accepted":
            if not _has_ids(data):
                return None
            return ChatEvent(
                type="status",
                data={
                    "stage": stage,
                    "session_id": data["session_id"],
                    "message_id": data["message_id"],
                },
            )
        if require_terminal_metadata and stage == "rewriting":
            session = data.get("session_id")
            if not isinstance(session, str) or not _SESSION_ID.fullmatch(session):
                return None
            return ChatEvent(type="status", data={"stage": stage, "session_id": session})
        return ChatEvent(type="status", data={"stage": stage})
    if event.type == "answer_delta":
        text = data.get("text")
        if not isinstance(text, str) or not is_safe_answer_text(text):
            return None
        safe_data: dict[str, object] = {"text": text}
        source_ids = data.get("source_ids")
        if source_ids is not None and (
            not isinstance(source_ids, list)
            or not all(is_safe_reference_id(item) for item in source_ids)
        ):
            return None
        if source_ids is not None:
            safe_data["source_ids"] = list(source_ids)
        return ChatEvent(type="answer_delta", data=safe_data)
    if event.type == "final":
        refused = data.get("refused")
        citations = data.get("citations", [])
        if not set(data).issubset(
            {
                "refused",
                "citations",
                "reason_code",
                "reference_answer",
                # Retrieval diagnostics stay internal; the workflow summary
                # is separately validated and reduced to fixed safe facts.
                "retrieval_diagnostics",
                "session_id",
                "message_id",
                "workflow_summary",
                "reference_allowed",
            }
        ):
            return None
        if not isinstance(refused, bool) or not isinstance(citations, list):
            return None
        try:
            safe_citations = validate_citations(citations)
        except (TypeError, ValueError):
            return None
        safe_data: dict[str, object] = {"refused": refused, "citations": safe_citations}
        reason_code = data.get("reason_code")
        try:
            safe_reason = validate_reason_code(reason_code, allow_none=True)
        except (TypeError, ValueError):
            return None
        reference_answer = data.get("reference_answer")
        if reference_answer is not None and (
            not isinstance(reference_answer, str)
            or not reference_answer.strip()
            or len(reference_answer) > 4000
            or not is_safe_answer_text(reference_answer)
        ):
            return None
        if reference_answer is not None:
            allowed = (
                not refused and safe_reason is None
            ) or (
                refused
                and safe_reason in REFERENCE_REASON_CODES | {"MODEL_REFUSED"}
            )
            if not allowed:
                return None
        if safe_reason is not None:
            safe_data["reason_code"] = safe_reason
        if reference_answer is not None:
            safe_data["reference_answer"] = reference_answer
        if "workflow_summary" in data:
            try:
                summary = normalize_workflow_summary(data["workflow_summary"])
            except (TypeError, ValueError):
                return None
            if summary is not None:
                safe_data["workflow_summary"] = summary
        if "reference_allowed" in data and not isinstance(
            data["reference_allowed"], bool
        ):
            return None
        if require_terminal_metadata:
            if not _has_ids(data):
                return None
            safe_data.update(
                {"session_id": data["session_id"], "message_id": data["message_id"]}
            )
        return ChatEvent(type="final", data=safe_data)
    reason_code = data.get("reason_code")
    try:
        safe_reason = validate_reason_code(reason_code)
    except ValueError:
        return None
    stage = data.get("stage", "chat")
    if stage not in _ALLOWED_STATUS_STAGES and stage != "chat":
        return None
    safe_data = {"stage": stage, "reason_code": safe_reason}
    if "candidate_sources" in data:
        try:
            safe_data["candidate_sources"] = validate_candidate_sources(
                data["candidate_sources"]
            )
        except (TypeError, ValueError):
            return None
    if "workflow_summary" in data:
        try:
            summary = normalize_workflow_summary(data["workflow_summary"])
        except (TypeError, ValueError):
            return None
        if summary is not None:
            safe_data["workflow_summary"] = summary
    if require_terminal_metadata:
        if not _has_ids(data):
            return None
        safe_data.update(
            {"session_id": data["session_id"], "message_id": data["message_id"]}
        )
    return ChatEvent(type="error", data=safe_data)


def _has_ids(data: dict[str, object]) -> bool:
    return (
        is_safe_id(data.get("session_id"))
        and is_safe_id(data.get("message_id"))
    )


def _safe_citation(value: dict[str, object]) -> dict[str, object] | None:
    try:
        return validate_citation(value)
    except (TypeError, ValueError):
        return None


def _validate_session_id_or_422(session_id: str) -> None:
    if not _SESSION_ID.fullmatch(session_id):
        raise HTTPException(status_code=422, detail="session_id is invalid")


def _repository(request: Request, orchestrator: Any | None = None) -> Any | None:
    repository = getattr(request.app.state, "chat_repository", None)
    if repository is not None:
        return repository
    selected = orchestrator or getattr(request.app.state, "chat_orchestrator", None)
    return getattr(selected, "repository", None)


def _feedback_dict(value: Any) -> dict[str, object] | None:
    return normalize_feedback(value)


def _upsert_feedback_with_optional_reason(
    repository: Any,
    message_id: str,
    helpful: bool,
    reason: FeedbackReason | None,
) -> Any:
    """Call new repositories while preserving old two-argument test doubles."""

    method = repository.upsert_feedback
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return method(message_id, helpful, reason)
    if len(parameters) >= 3 or any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL
        for parameter in parameters.values()
    ):
        return method(message_id, helpful, reason)
    return method(message_id, helpful)


def _encode_sse(event: ChatEvent) -> str:
    data = json.dumps(event.data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event.type}\ndata: {data}\n\n"


__all__ = ["ChatRequest", "FeedbackRequest", "create_chat_router"]
