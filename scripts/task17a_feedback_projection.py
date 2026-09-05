"""Build or inspect the local-only, redacted Task17A feedback projection."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_BACKEND_ROOT = _PROJECT_ROOT / "backend"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.chat.models import ChatMessage, Feedback
from app.feedback.repository import (
    FeedbackProjectionRepository,
    validate_private_feedback_path,
)
from app.feedback.triage import score_case

_TERMINAL_STATUSES = ("answered", "refused", "error")


def _parse_json(value: object, default: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _read_only_connection(path: Path) -> sqlite3.Connection:
    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError("source database is unavailable")
    return sqlite3.connect(f"file:{resolved.as_posix()}?mode=ro", uri=True)


def _message_from_row(row: sqlite3.Row) -> ChatMessage:
    citations = _parse_json(row["citations_json"], [])
    if not isinstance(citations, list):
        citations = []
    citations = tuple(item for item in citations if isinstance(item, dict))
    summary = _parse_json(row["workflow_summary_json"], None)
    if not isinstance(summary, dict):
        summary = None
    return ChatMessage(
        message_id=str(row["message_id"]),
        session_id=str(row["session_id"]),
        role="assistant",
        content=str(row["content"]),
        status=str(row["status"]),
        rewritten_question=row["rewritten_question"],
        citations=citations,
        reason_code=row["reason_code"],
        created_at=str(row["created_at"]),
        reply_to_message_id=row["reply_to_message_id"],
        reference_answer=row["reference_answer"],
        audience_scope=row["audience_scope"] or "unspecified",
        workflow_summary=summary,
    )


def _load_feedback(connection: sqlite3.Connection, message_id: str) -> Feedback | None:
    row = connection.execute(
        "SELECT message_id, helpful, reason, created_at FROM feedback "
        "WHERE message_id = ? ORDER BY created_at DESC, rowid DESC LIMIT 1",
        (message_id,),
    ).fetchone()
    if row is None:
        return None
    return Feedback(
        message_id=str(row["message_id"]),
        helpful=bool(row["helpful"]),
        created_at=str(row["created_at"]),
        reason=row["reason"],
    )


def _build(args: argparse.Namespace) -> int:
    output = validate_private_feedback_path(Path(args.output_db))
    source = Path(args.source_db).expanduser().resolve()
    if source == output:
        raise ValueError("source and private output database must differ")
    if not isinstance(args.limit, int) or not 1 <= args.limit <= 10000:
        raise ValueError("limit is invalid")

    repository = FeedbackProjectionRepository(output)
    connection = _read_only_connection(source)
    counts = {
        "cases_seen": 0,
        "redaction_passed": 0,
        "redaction_review_required": 0,
        "redaction_blocked": 0,
        "events_seen": 0,
        "p0": 0,
        "p1": 0,
        "p2": 0,
        "p3": 0,
    }
    now = datetime.now(timezone.utc)
    try:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM messages WHERE role = 'assistant' "
            "AND status IN ('answered', 'refused', 'error') "
            "ORDER BY created_at ASC, rowid ASC LIMIT ?",
            (args.limit,),
        ).fetchall()
        for row in rows:
            counts["cases_seen"] += 1
            message = _message_from_row(row)
            feedback = _load_feedback(connection, message.message_id)
            trace = dict(message.workflow_summary or {})
            user_row = connection.execute(
                "SELECT content FROM messages WHERE message_id = ? AND role = 'user'",
                (message.reply_to_message_id,),
            ).fetchone()
            if user_row is not None:
                trace["question"] = str(user_row["content"])
            case, event = repository.build_case_from_message(
                message, feedback, trace=trace, now=now
            )
            repository.upsert_case(case)
            if event is not None:
                repository.append_event(event)
                counts["events_seen"] += 1
            high_risk = bool(trace.get("high_risk", False))
            decision = score_case(
                case,
                high_risk=high_risk,
                citation_mismatch=feedback is not None and feedback.reason == "citation_mismatch",
                version_mismatch=feedback is not None and feedback.reason == "version_mismatch",
            )
            repository.update_triage(case.case_id, decision.priority, decision.score, decision.reasons)
            counts[f"redaction_{case.redaction_status}"] += 1
            counts[decision.priority.casefold()] += 1
    finally:
        connection.close()
        repository.close()
    summary = {
        **counts,
        "network_calls": 0,
        "provider_constructed": False,
        "api_key_read": False,
        "raw_provider_responses_saved": False,
        "source_read_only": True,
        "private_output": True,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def _list(args: argparse.Namespace) -> int:
    repository = FeedbackProjectionRepository(validate_private_feedback_path(Path(args.database)))
    try:
        cases = repository.list_cases(status=args.status, priority=args.priority, limit=args.limit)
        print(json.dumps([case.to_dict() for case in cases], ensure_ascii=False, sort_keys=True))
    finally:
        repository.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task17A local redacted feedback projection")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="read source SQLite read-only and build private projection")
    build.add_argument("--source-db", required=True)
    build.add_argument("--output-db", required=True)
    build.add_argument("--limit", type=int, default=100)
    build.set_defaults(handler=_build)
    listing = subparsers.add_parser("list", help="list redacted private cases")
    listing.add_argument("--database", required=True)
    listing.add_argument("--priority", choices=("P0", "P1", "P2", "P3"))
    listing.add_argument("--status")
    listing.add_argument("--limit", type=int, default=100)
    listing.set_defaults(handler=_list)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception as error:  # noqa: BLE001 - CLI must not print source text
        category = "private_path_or_contract" if isinstance(error, ValueError) else "runtime"
        print(f"error_type={type(error).__name__} category={category}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
