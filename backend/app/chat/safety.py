"""Shared safe-shape validators for persisted and streamed chat metadata."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import datetime

SAFE_REASON_CODES = frozenset(
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
REFERENCE_REASON_CODES = frozenset({"INSUFFICIENT_EVIDENCE", "ANSWER_NOT_VERIFIABLE"})

_REFERENCE_ID = re.compile(r"^S[1-6]$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SAFE_BASENAME = re.compile(r"^[^<>:\"/\\|?*\x00-\x1f]+\.(?:pdf|docx)$", re.IGNORECASE)
_SAFE_TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})?$"
)
_DANGEROUS_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_CITATION_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_PROVIDER_STACK = re.compile(r"(?i)\b(?:traceback|stack\s+trace|providererror)\b")
_ABSOLUTE_PATH_PATTERN = (
    r"(?:"
    # Windows drive paths, including both slash styles.
    r"(?:[A-Za-z]:[\\/])(?:[^\\/\s]+[\\/])*[^\\/\s]+"
    r"|"
    # UNC paths require a server and share component.
    r"(?:\\\\|//)[^\\/\s]+[\\/][^\\/\s]+(?:[\\/][^\\/\s]+)*"
    r"|"
    # Unix paths require at least one directory separator after the root.
    r"(?<![A-Za-z0-9_:/])/(?:[^/\s]+/)+[^/\s]+"
    r")"
)
ABSOLUTE_PATH_RE = re.compile(_ABSOLUTE_PATH_PATTERN)
_CITATION_ABSOLUTE_PATH = ABSOLUTE_PATH_RE
_ABSOLUTE_PATH = _CITATION_ABSOLUTE_PATH
_ANSWER_ABSOLUTE_PATH = _CITATION_ABSOLUTE_PATH
_CITATION_WHITESPACE = re.compile(r"[\r\n\t]+")
_SECRET_VALUE = re.compile(
    r"(?i)(?:\b(?:sk|pk)-[A-Za-z0-9_-]+\b|"
    r"\bbearer(?:[\s:=_-]+)[^\s,;]+|"
    r"\b(?:api[_ -]?key|access[_ -]?token)\s*[:=]\s*[^\s,;]+)"
)
_CITATION_KEYS = frozenset(
    {
        "reference_id",
        "file_name",
        "heading_path",
        "page",
        "page_end",
        "paragraph_start",
        "paragraph_end",
        "excerpt",
    }
)
_CANDIDATE_KEYS = frozenset({"label", "reference_id", "file_name", "heading_path"})
SAFE_CANDIDATE_LABEL = "可能相关的原文，尚未生成答案"
SAFE_MESSAGE_ROLES = frozenset({"user", "assistant"})
SAFE_MESSAGE_STATUSES = frozenset({"submitted", "answered", "refused", "error"})
SAFE_DOCUMENT_STATUSES = frozenset(
    {
        "indexing",
        "active",
        "skipped",
        "failed",
        "superseded",
        "inactive",
        "needs_ocr",
    }
)
SAFE_FAILURE_REASONS = frozenset(
    {
        "PROVIDER_UNAVAILABLE",
        "SOURCE_FILE_MISSING",
        "SOURCE_FILE_UNREADABLE",
        "DOCUMENT_NOT_INDEXABLE",
        "INDEX_FAILED",
    }
)
SAFE_ERROR_TEXT = "当前服务暂时不可用，请稍后重试。"
SAFE_CORRUPTED_MESSAGE_TEXT = "当前会话记录无法安全恢复。"
SAFE_EPOCH = "1970-01-01 00:00:00"


def validate_reason_code(value: object, *, allow_none: bool = False) -> str | None:
    """Validate a public reason code against the finite application contract."""

    if value is None and allow_none:
        return None
    if not isinstance(value, str) or value not in SAFE_REASON_CODES:
        raise ValueError("reason_code is not allowed")
    return value


def is_safe_reference_id(value: object) -> bool:
    return isinstance(value, str) and _REFERENCE_ID.fullmatch(value) is not None


def is_safe_id(value: object) -> bool:
    return isinstance(value, str) and _SAFE_ID.fullmatch(value) is not None


def is_safe_timestamp(value: object) -> bool:
    if not isinstance(value, str) or _SAFE_TIMESTAMP.fullmatch(value) is None:
        return False
    try:
        datetime.fromisoformat(
            value[:-1] + "+00:00" if value.endswith("Z") else value
        )
    except ValueError:
        return False
    return True


def is_safe_file_name(value: object) -> bool:
    return (
        isinstance(value, str)
        and _SAFE_BASENAME.fullmatch(value) is not None
        and _safe_citation_text(value)
    )


def is_safe_answer_text(value: str) -> bool:
    """Allow ordinary whitespace while rejecting control/credential/path data."""

    return (
        isinstance(value, str)
        and not _DANGEROUS_CONTROL_CHARS.search(value)
        and not _ANSWER_ABSOLUTE_PATH.search(value)
        and not _SECRET_VALUE.search(value)
        and not _PROVIDER_STACK.search(value)
    )


def validate_citations(values: object) -> list[dict[str, object]]:
    """Return normalized citation snapshots or reject them fail-closed."""

    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise TypeError("citations must be a sequence")
    normalized: list[dict[str, object]] = []
    for value in values:
        normalized.append(validate_citation(value))
    return normalized


def validate_citation(value: object) -> dict[str, object]:
    """Validate the safe citation DTO emitted by ``AnswerService``."""

    if not isinstance(value, Mapping):
        raise TypeError("citation must be a mapping")
    if not set(value).issubset(_CITATION_KEYS):
        raise ValueError("citation contains unknown fields")
    reference_id = value.get("reference_id")
    file_name = value.get("file_name")
    if not is_safe_reference_id(reference_id):
        raise ValueError("citation reference_id is invalid")
    if not isinstance(file_name, str) or not _SAFE_BASENAME.fullmatch(file_name):
        raise ValueError("citation file_name is unsafe")
    if not _safe_citation_text(file_name):
        raise ValueError("citation file_name contains unsafe text")

    result: dict[str, object] = {
        "reference_id": reference_id,
        "file_name": file_name,
    }
    heading_path = value.get("heading_path")
    if not isinstance(heading_path, Sequence) or isinstance(
        heading_path, (str, bytes, bytearray)
    ):
        raise TypeError("citation heading_path is invalid")
    cleaned_heading = []
    for item in heading_path:
        if not isinstance(item, str) or not item.strip() or not _safe_citation_text(item):
            raise ValueError("citation heading_path contains unsafe text")
        if "/" in item or "\\" in item:
            raise ValueError("citation heading_path contains a path")
        cleaned_heading.append(item)
    if not cleaned_heading:
        raise ValueError("citation heading_path is empty")
    result["heading_path"] = cleaned_heading

    locations: dict[str, int | None] = {}
    for key in ("page", "page_end", "paragraph_start", "paragraph_end"):
        item = value.get(key)
        if item is not None and (isinstance(item, bool) or not isinstance(item, int) or item < 1):
            raise ValueError(f"citation {key} is invalid")
        locations[key] = item
        result[key] = item

    suffix = file_name.casefold()
    if suffix.endswith(".pdf"):
        if not _valid_range(locations["page"], locations["page_end"]):
            raise ValueError("pdf citation page range is required")
        if locations["paragraph_start"] is not None or locations["paragraph_end"] is not None:
            raise ValueError("pdf citation must not contain paragraph range")
    elif suffix.endswith(".docx"):
        if not _valid_range(
            locations["paragraph_start"], locations["paragraph_end"]
        ):
            raise ValueError("docx citation paragraph range is required")
        if locations["page"] is not None or locations["page_end"] is not None:
            raise ValueError("docx citation must not contain page range")
    else:
        raise ValueError("citation file type is unsupported")

    excerpt = normalize_citation_excerpt(value.get("excerpt"), redact_paths=False)
    if excerpt is None or len(excerpt) > 300:
        raise ValueError("citation excerpt is unsafe")
    result["excerpt"] = excerpt
    return result


def validate_candidate_sources(value: object) -> list[dict[str, object]]:
    """Validate the optional safe candidate-source list on legacy errors."""

    if not isinstance(value, list):
        raise TypeError("candidate_sources must be a list")
    result: list[dict[str, object]] = []
    for candidate in value:
        if not isinstance(candidate, Mapping) or not set(candidate).issubset(_CANDIDATE_KEYS):
            raise ValueError("candidate source fields are unsafe")
        if candidate.get("label") != SAFE_CANDIDATE_LABEL:
            raise ValueError("candidate source label is invalid")
        reference_id = candidate.get("reference_id")
        file_name = candidate.get("file_name")
        heading_path = candidate.get("heading_path")
        if not is_safe_reference_id(reference_id):
            raise ValueError("candidate source reference_id is invalid")
        if not isinstance(file_name, str) or not _SAFE_BASENAME.fullmatch(file_name):
            raise ValueError("candidate source file_name is unsafe")
        if not _safe_citation_text(file_name):
            raise ValueError("candidate source file_name contains unsafe text")
        if not isinstance(heading_path, Sequence) or isinstance(
            heading_path, (str, bytes, bytearray)
        ):
            raise TypeError("candidate source heading_path is invalid")
        cleaned_heading = []
        for item in heading_path:
            if (
                not isinstance(item, str)
                or not item.strip()
                or not _safe_citation_text(item)
                or "/" in item
                or "\\" in item
            ):
                raise ValueError("candidate source heading_path is unsafe")
            cleaned_heading.append(item)
        if not cleaned_heading:
            raise ValueError("candidate source heading_path is empty")
        result.append(
            {
                "label": SAFE_CANDIDATE_LABEL,
                "reference_id": reference_id,
                "file_name": file_name,
                "heading_path": cleaned_heading,
            }
        )
    return result


def _safe_citation_text(value: str) -> bool:
    return (
        not _CITATION_CONTROL_CHARS.search(value)
        and not _CITATION_ABSOLUTE_PATH.search(value)
        and not _SECRET_VALUE.search(value)
    )


def normalize_citation_excerpt(
    value: object, *, redact_paths: bool
) -> str | None:
    """Apply the shared citation-display text policy at a boundary."""

    if not isinstance(value, str):
        return None
    if any(
        (ord(character) < 32 or ord(character) == 127)
        and character not in "\r\n\t"
        for character in value
    ):
        return None
    excerpt = _CITATION_WHITESPACE.sub(" ", value).strip()
    if not excerpt or _SECRET_VALUE.search(excerpt):
        return None
    if _CITATION_ABSOLUTE_PATH.search(excerpt):
        if not redact_paths:
            return None
        excerpt = _CITATION_ABSOLUTE_PATH.sub("[path redacted]", excerpt)
    return excerpt


def _valid_range(start: object, end: object) -> bool:
    return (
        isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(end, int)
        and not isinstance(end, bool)
        and start > 0
        and end >= start
    )


def _record(value: object, fields: Sequence[str]) -> dict[str, object] | None:
    if isinstance(value, Mapping):
        return dict(value)
    keys = getattr(value, "keys", None)
    if callable(keys):
        available = set(keys())
        return {field: value[field] for field in fields if field in available}
    result: dict[str, object] = {}
    for field in fields:
        if hasattr(value, field):
            result[field] = getattr(value, field)
    return result or None


def normalize_chat_message(
    value: object, *, fallback_session_id: str = "invalid-session"
):
    """Return a strictly safe chat DTO, degrading damaged rows to an error."""

    from app.chat.models import ChatMessage

    fields = (
        "message_id",
        "session_id",
        "role",
        "content",
        "status",
        "rewritten_question",
        "citations",
        "reason_code",
        "created_at",
        "reply_to_message_id",
        "reference_answer",
    )
    raw = _record(value, fields) or {}
    safe_session = (
        fallback_session_id if is_safe_id(fallback_session_id) else "invalid-session"
    )
    message_id = raw.get("message_id")
    safe_message_id = message_id if is_safe_id(message_id) else "invalid-message"
    row_session = raw.get("session_id")
    if is_safe_id(row_session):
        if safe_session == "invalid-session":
            safe_session = row_session
        elif row_session != safe_session:
            return ChatMessage(
                message_id=safe_message_id,
                session_id=safe_session,
                role="assistant",
                content=SAFE_CORRUPTED_MESSAGE_TEXT,
                status="error",
                rewritten_question=None,
                citations=(),
                reason_code="CHAT_UNAVAILABLE",
                created_at=SAFE_EPOCH,
                reply_to_message_id=None,
            )
    try:
        role = raw.get("role")
        status = raw.get("status")
        if not is_safe_id(message_id) or not is_safe_id(row_session):
            raise ValueError("message/session id is invalid")
        if role not in SAFE_MESSAGE_ROLES or status not in SAFE_MESSAGE_STATUSES:
            raise ValueError("message role/status is invalid")
        if not is_safe_timestamp(raw.get("created_at")):
            raise ValueError("message timestamp is invalid")
        content = raw.get("content")
        if not isinstance(content, str) or not content.strip() or not is_safe_answer_text(content):
            raise ValueError("message content is unsafe")
        citations = tuple(validate_citations(raw.get("citations", ())))
        reason_code = raw.get("reason_code")
        reason_code = validate_reason_code(reason_code, allow_none=True)
        reference_answer = raw.get("reference_answer")
        if reference_answer is not None and (
            not isinstance(reference_answer, str)
            or not reference_answer.strip()
            or len(reference_answer) > 4000
            or not is_safe_answer_text(reference_answer)
        ):
            raise ValueError("reference answer is unsafe")
        rewritten = raw.get("rewritten_question")
        if rewritten is not None and (
            role != "user"
            or not isinstance(rewritten, str)
            or len(rewritten) > 1000
            or not rewritten.strip()
            or not is_safe_answer_text(rewritten)
        ):
            raise ValueError("rewritten question is unsafe")
        reply_to = raw.get("reply_to_message_id")
        if role == "user":
            if (
                status != "submitted"
                or citations
                or reason_code is not None
                or reply_to is not None
                or reference_answer is not None
            ):
                raise ValueError("user message metadata is invalid")
            if not is_safe_id(safe_message_id):
                raise ValueError("user message id is invalid")
        else:
            if status not in {"answered", "refused", "error"} or not is_safe_id(reply_to):
                raise ValueError("assistant message metadata is invalid")
            if status in {"refused", "error"} and citations:
                raise ValueError("terminal non-answer must not contain citations")
            if status == "error":
                content = SAFE_ERROR_TEXT
                reference_answer = None
            if reference_answer is not None:
                allowed = (
                    status == "answered" and reason_code is None
                ) or (
                    status == "refused"
                    and reason_code in REFERENCE_REASON_CODES | {"MODEL_REFUSED"}
                )
                if not allowed:
                    raise ValueError("reference answer metadata is invalid")
        return ChatMessage(
            message_id=safe_message_id,
            session_id=safe_session,
            role=role,
            content=content,
            status=status,
            rewritten_question=rewritten,
            citations=citations,
            reason_code=reason_code,
            created_at=str(raw["created_at"]),
            reply_to_message_id=reply_to,
            reference_answer=reference_answer,
        )
    except (TypeError, ValueError):
        return ChatMessage(
            message_id=safe_message_id,
            session_id=safe_session,
            role="assistant",
            content=SAFE_CORRUPTED_MESSAGE_TEXT,
            status="error",
            rewritten_question=None,
            citations=(),
            reason_code="CHAT_UNAVAILABLE",
            created_at=(
                str(raw.get("created_at"))
                if is_safe_timestamp(raw.get("created_at"))
                else SAFE_EPOCH
            ),
            reply_to_message_id=None,
        )


def normalize_feedback(value: object) -> dict[str, object] | None:
    fields = ("message_id", "helpful", "created_at")
    raw = _record(value, fields)
    if raw is None:
        return None
    helpful = raw.get("helpful")
    if not is_safe_id(raw.get("message_id")) or not is_safe_timestamp(raw.get("created_at")):
        return None
    if type(helpful) is bool:
        safe_helpful = helpful
    elif isinstance(helpful, int) and helpful in (0, 1):
        safe_helpful = bool(helpful)
    else:
        return None
    return {
        "message_id": raw["message_id"],
        "helpful": safe_helpful,
        "created_at": raw["created_at"],
    }


__all__ = [
    "REFERENCE_REASON_CODES",
    "SAFE_CANDIDATE_LABEL",
    "SAFE_CORRUPTED_MESSAGE_TEXT",
    "SAFE_DOCUMENT_STATUSES",
    "SAFE_ERROR_TEXT",
    "SAFE_FAILURE_REASONS",
    "SAFE_MESSAGE_ROLES",
    "SAFE_MESSAGE_STATUSES",
    "SAFE_REASON_CODES",
    "is_safe_answer_text",
    "is_safe_file_name",
    "is_safe_id",
    "is_safe_reference_id",
    "is_safe_timestamp",
    "normalize_chat_message",
    "normalize_citation_excerpt",
    "normalize_feedback",
    "validate_candidate_sources",
    "validate_citation",
    "validate_citations",
    "validate_reason_code",
]
