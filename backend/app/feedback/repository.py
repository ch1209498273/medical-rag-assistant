"""Private SQLite projection repository for redacted production feedback."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.chat.models import ChatMessage, Feedback
from app.feedback.models import (
    AUDIENCE_SCOPES,
    FeedbackCase,
    FeedbackEvent,
    PromotionRecord,
    ReviewDecision,
    build_case_identity,
)
from app.feedback.redaction import redact_interaction
from app.feedback.schema import FeedbackProjectionSchema

_TERMINAL_STATUSES = frozenset({"answered", "refused", "error"})
_PRIVATE_PATH_PARTS = ("data", "private", "feedback")
_DEFAULT_RETENTION_DAYS = 90


def validate_private_feedback_path(path: Path) -> Path:
    """Resolve and validate a path under a ``data/private/feedback`` segment."""

    if not isinstance(path, Path):
        path = Path(path)
    resolved = path.expanduser().resolve()
    parts = tuple(part.casefold() for part in resolved.parts)
    required = tuple(part.casefold() for part in _PRIVATE_PATH_PARTS)
    if not any(parts[index : index + len(required)] == required for index in range(len(parts))):
        raise ValueError("path must be below the private feedback directory data/private/feedback")
    if resolved.name in {"", ".", ".."}:
        raise ValueError("private feedback path must name a file")
    return resolved


@dataclass(frozen=True)
class RetentionSummary:
    cases_deleted: int
    events_deleted: int
    reviews_deleted: int
    promotions_deleted: int
    as_of: str

    def to_dict(self) -> dict[str, object]:
        return {
            "cases_deleted": self.cases_deleted,
            "events_deleted": self.events_deleted,
            "reviews_deleted": self.reviews_deleted,
            "promotions_deleted": self.promotions_deleted,
            "as_of": self.as_of,
        }


def _json_loads(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    try:
        raw = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return ()
    if not isinstance(raw, list):
        return ()
    return tuple(str(item) for item in raw if isinstance(item, str) and item.strip())


def _optional_text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _iso_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def _normalise_timestamp(value: object, fallback: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return fallback
    text = value.strip().replace(" ", "T", 1)
    return text if "T" in text else fallback


def _safe_trace_text(trace: Mapping[str, object], key: str) -> str | None:
    value = trace.get(key)
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        return None
    return value.strip()


def _citation_ids(message: ChatMessage, trace: Mapping[str, object]) -> tuple[str, ...]:
    values: list[object] = []
    trace_ids = trace.get("citation_chunk_ids")
    if isinstance(trace_ids, (list, tuple)):
        values.extend(trace_ids)
    if not values:
        for citation in message.citations:
            if not isinstance(citation, Mapping):
                continue
            values.append(citation.get("chunk_id", citation.get("reference_id")))
    result: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        clean = value.strip()
        if not clean or len(clean) > 160 or any(char.isspace() for char in clean):
            continue
        if clean not in result:
            result.append(clean)
    return tuple(result)


def _event_id(case_id: str, feedback: Feedback) -> str:
    raw = f"{case_id}|{feedback.created_at}|{int(feedback.helpful)}|{feedback.reason or ''}"
    return f"fe_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:32]}"


class FeedbackProjectionRepository:
    """Persist only derived redacted feedback data in a private database."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = validate_private_feedback_path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        FeedbackProjectionSchema.create(self._connection)

    def upsert_case(self, case: FeedbackCase) -> None:
        existing = self._connection.execute(
            "SELECT message_id_hash, session_id_hash FROM feedback_cases WHERE case_id = ?",
            (case.case_id,),
        ).fetchone()
        if existing is not None:
            if (
                existing["message_id_hash"] != case.message_id_hash
                or existing["session_id_hash"] != case.session_id_hash
            ):
                raise ValueError("conflicting case identity")
            return
        FeedbackProjectionSchema.insert_case(self._connection, case)
        self._connection.commit()

    def append_event(self, event: FeedbackEvent) -> None:
        existing = self._connection.execute(
            "SELECT * FROM feedback_events WHERE event_id = ?", (event.event_id,)
        ).fetchone()
        if existing is not None:
            if (
                existing["case_id"] != event.case_id
                or bool(existing["helpful"]) != event.helpful
                or existing["feedback_reason"] != event.feedback_reason
            ):
                raise ValueError("conflicting event identity")
            return
        try:
            FeedbackProjectionSchema.insert_event(self._connection, event)
            self._connection.execute(
                "UPDATE feedback_cases SET event_count = "
                "(SELECT COUNT(*) FROM feedback_events WHERE case_id = ?) WHERE case_id = ?",
                (event.case_id, event.case_id),
            )
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    def save_review(self, decision: ReviewDecision) -> None:
        if self._connection.execute(
            "SELECT 1 FROM review_decisions WHERE review_id = ?", (decision.review_id,)
        ).fetchone():
            raise ValueError("review_id already exists")
        FeedbackProjectionSchema.insert_review(self._connection, decision)
        self._connection.commit()

    def update_review_status(self, case_id: str, review_status: str) -> None:
        """Update only the controlled review state on an existing case."""

        allowed = {
            "unreviewed",
            "in_review",
            "approved",
            "rejected",
            "adjudication_required",
        }
        if review_status not in allowed:
            raise ValueError("review status is invalid")
        cursor = self._connection.execute(
            "UPDATE feedback_cases SET review_status = ? WHERE case_id = ?",
            (review_status, case_id),
        )
        if cursor.rowcount != 1:
            self._connection.rollback()
            raise KeyError(case_id)
        self._connection.commit()

    def save_promotion(self, record: PromotionRecord) -> None:
        if self._connection.execute(
            "SELECT 1 FROM promotion_records WHERE promotion_id = ?", (record.promotion_id,)
        ).fetchone():
            raise ValueError("promotion_id already exists")
        FeedbackProjectionSchema.insert_promotion(self._connection, record)
        self._connection.commit()

    def update_promotion_status(self, case_id: str, promotion_status: str) -> None:
        """Record the controlled candidate state without rewriting review history."""

        allowed = {"not_promoted", "silver", "golden_v2_candidate", "golden_v2"}
        if promotion_status not in allowed:
            raise ValueError("promotion status is invalid")
        cursor = self._connection.execute(
            "UPDATE feedback_cases SET promotion_status = ? WHERE case_id = ?",
            (promotion_status, case_id),
        )
        if cursor.rowcount != 1:
            self._connection.rollback()
            raise KeyError(case_id)
        self._connection.commit()

    def update_triage(self, case_id: str, priority: str, score: int, reasons: tuple[str, ...]) -> None:
        """Update only deterministic triage fields on an existing projection case."""

        self._connection.execute(
            """
            UPDATE feedback_cases
            SET triage_priority = ?, triage_score = ?, triage_reasons_json = ?
            WHERE case_id = ?
            """,
            (priority, score, json.dumps(reasons, ensure_ascii=False), case_id),
        )
        self._connection.commit()

    def list_cases(
        self,
        *,
        status: str | None = None,
        priority: str | None = None,
        limit: int = 100,
    ) -> list[FeedbackCase]:
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        clauses: list[str] = []
        values: list[str | int] = []
        if status is not None:
            clauses.append("review_status = ?")
            values.append(status)
        if priority is not None:
            clauses.append("triage_priority = ?")
            values.append(priority)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._connection.execute(
            f"SELECT * FROM feedback_cases {where} "
            "ORDER BY CASE triage_priority WHEN 'P0' THEN 0 WHEN 'P1' THEN 1 "
            "WHEN 'P2' THEN 2 ELSE 3 END, triage_score DESC, case_id LIMIT ?",
            (*values, limit),
        ).fetchall()
        return [self._case_from_row(row) for row in rows]

    def list_reviews(self, case_id: str | None = None) -> list[ReviewDecision]:
        where = "WHERE case_id = ?" if case_id is not None else ""
        args = (case_id,) if case_id is not None else ()
        rows = self._connection.execute(
            f"SELECT * FROM review_decisions {where} ORDER BY reviewed_at, review_id", args
        ).fetchall()
        return [
            ReviewDecision(
                review_id=str(row["review_id"]),
                case_id=str(row["case_id"]),
                reviewer_role=row["reviewer_role"],
                decision=row["decision"],
                evidence_ok=None if row["evidence_ok"] is None else bool(row["evidence_ok"]),
                points_ok=None if row["points_ok"] is None else bool(row["points_ok"]),
                safety_ok=None if row["safety_ok"] is None else bool(row["safety_ok"]),
                note_code=row["note_code"],
                review_version=str(row["review_version"]),
                reviewed_at=str(row["reviewed_at"]),
            )
            for row in rows
        ]

    def list_events(self, case_id: str | None = None) -> list[FeedbackEvent]:
        """Return projected feedback events in stable chronological order."""

        where = "WHERE case_id = ?" if case_id is not None else ""
        args = (case_id,) if case_id is not None else ()
        rows = self._connection.execute(
            f"SELECT * FROM feedback_events {where} ORDER BY event_at, event_id", args
        ).fetchall()
        return [
            FeedbackEvent(
                event_id=str(row["event_id"]),
                case_id=str(row["case_id"]),
                helpful=bool(row["helpful"]),
                feedback_reason=row["feedback_reason"],
                event_at=str(row["event_at"]),
                source=row["source"],
                event_schema_version=int(row["event_schema_version"]),
            )
            for row in rows
        ]

    def list_promotions(self, case_id: str | None = None) -> list[PromotionRecord]:
        where = "WHERE case_id = ?" if case_id is not None else ""
        args = (case_id,) if case_id is not None else ()
        rows = self._connection.execute(
            f"SELECT * FROM promotion_records {where} ORDER BY promoted_at, promotion_id", args
        ).fetchall()
        return [
            PromotionRecord(
                promotion_id=str(row["promotion_id"]),
                case_id=str(row["case_id"]),
                target_set_id=str(row["target_set_id"]),
                target_version=str(row["target_version"]),
                target_split=row["target_split"],
                promotion_reason=str(row["promotion_reason"]),
                manifest_id=str(row["manifest_id"]),
                promoted_at=str(row["promoted_at"]),
            )
            for row in rows
        ]

    def event_count(self, case_id: str) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) FROM feedback_events WHERE case_id = ?", (case_id,)
        ).fetchone()
        return int(row[0]) if row else 0

    def delete_expired(self, now: datetime) -> RetentionSummary:
        as_of = _iso_datetime(now)
        candidates = self._connection.execute(
            "SELECT case_id, retention_expires_at FROM feedback_cases"
        ).fetchall()
        expired: list[str] = []
        for row in candidates:
            try:
                expires = datetime.fromisoformat(str(row["retention_expires_at"]))
            except ValueError:
                continue
            if expires.tzinfo is None:
                expires = expires.replace(tzinfo=timezone.utc)
            current = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
            if expires <= current:
                expired.append(str(row["case_id"]))
        counts = {"cases": 0, "events": 0, "reviews": 0, "promotions": 0}
        try:
            for case_id in expired:
                for key, table in (
                    ("events", "feedback_events"),
                    ("reviews", "review_decisions"),
                    ("promotions", "promotion_records"),
                ):
                    cursor = self._connection.execute(
                        f"DELETE FROM {table} WHERE case_id = ?", (case_id,)
                    )
                    counts[key] += cursor.rowcount
                cursor = self._connection.execute(
                    "DELETE FROM feedback_cases WHERE case_id = ?", (case_id,)
                )
                counts["cases"] += cursor.rowcount
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise
        return RetentionSummary(
            cases_deleted=counts["cases"],
            events_deleted=counts["events"],
            reviews_deleted=counts["reviews"],
            promotions_deleted=counts["promotions"],
            as_of=as_of,
        )

    @staticmethod
    def build_case_from_message(
        message: ChatMessage,
        feedback: Feedback | None,
        *,
        trace: Mapping[str, object],
        now: datetime,
    ) -> tuple[FeedbackCase, FeedbackEvent | None]:
        if message.role != "assistant" or message.status not in _TERMINAL_STATUSES:
            raise ValueError("feedback projection requires a final assistant message")
        if feedback is not None and feedback.message_id != message.message_id:
            raise ValueError("feedback belongs to another message")
        identity = build_case_identity(message.message_id, message.session_id)
        question = trace.get("question")
        if not isinstance(question, str) or not question.strip():
            question = message.rewritten_question or "[问题未提供]"
        question_result, answer_result, flags = redact_interaction(question, message.content)
        if "blocked" in {question_result.status, answer_result.status}:
            redaction_status = "blocked"
        elif "review_required" in {question_result.status, answer_result.status}:
            redaction_status = "review_required"
        else:
            redaction_status = "passed"
        citation_ids = _citation_ids(message, trace)
        audience = _safe_trace_text(trace, "audience_scope") or message.audience_scope
        if audience not in AUDIENCE_SCOPES:
            audience = "unspecified"
        collected_at = _iso_datetime(now)
        event = None
        if feedback is not None:
            event = FeedbackEvent(
                event_id=_event_id(identity.case_id, feedback),
                case_id=identity.case_id,
                helpful=feedback.helpful,
                feedback_reason=feedback.reason,
                event_at=_normalise_timestamp(feedback.created_at, collected_at),
                source="user_click",
                event_schema_version=1,
            )
        case = FeedbackCase(
            case_id=identity.case_id,
            message_id_hash=identity.message_id_hash,
            session_id_hash=identity.session_id_hash,
            collected_at=collected_at,
            question_redacted=question_result.text,
            answer_redacted=answer_result.text,
            redaction_status=redaction_status,
            redaction_version=question_result.version,
            redaction_flags=flags,
            answer_status=message.status,
            reason_code=message.reason_code,
            source_version=_safe_trace_text(trace, "source_version"),
            retrieval_profile=_safe_trace_text(trace, "retrieval_profile"),
            citation_chunk_ids=citation_ids,
            citation_count=len(citation_ids),
            model_id=_safe_trace_text(trace, "model_id"),
            prompt_version=_safe_trace_text(trace, "prompt_version"),
            latency_bucket=_safe_trace_text(trace, "latency_bucket"),
            audience_scope=audience,
            retention_expires_at=_iso_datetime(now + timedelta(days=_DEFAULT_RETENTION_DAYS)),
            triage_priority="P3",
            triage_score=0,
            triage_reasons=(),
            review_status="unreviewed",
            promotion_status="not_promoted",
            event_count=1 if event is not None else 0,
        )
        return case, event

    @staticmethod
    def _case_from_row(row: sqlite3.Row) -> FeedbackCase:
        return FeedbackCase(
            case_id=str(row["case_id"]),
            message_id_hash=str(row["message_id_hash"]),
            session_id_hash=str(row["session_id_hash"]),
            collected_at=str(row["collected_at"]),
            question_redacted=str(row["question_redacted"]),
            answer_redacted=str(row["answer_redacted"]),
            redaction_status=row["redaction_status"],
            redaction_version=str(row["redaction_version"]),
            redaction_flags=_json_loads(row["redaction_flags_json"]),
            answer_status=row["answer_status"],
            reason_code=row["reason_code"],
            source_version=row["source_version"],
            retrieval_profile=row["retrieval_profile"],
            citation_chunk_ids=_json_loads(row["citation_chunk_ids_json"]),
            citation_count=int(row["citation_count"]),
            model_id=row["model_id"],
            prompt_version=row["prompt_version"],
            latency_bucket=row["latency_bucket"],
            audience_scope=row["audience_scope"],
            retention_expires_at=str(row["retention_expires_at"]),
            triage_priority=row["triage_priority"],
            triage_score=int(row["triage_score"]),
            triage_reasons=_json_loads(row["triage_reasons_json"]),
            review_status=row["review_status"],
            promotion_status=row["promotion_status"],
            event_count=int(row["event_count"]),
        )

    def close(self) -> None:
        self._connection.close()
