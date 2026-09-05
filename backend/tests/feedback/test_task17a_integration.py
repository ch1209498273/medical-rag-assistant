from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.feedback.models import ReviewDecision
from app.feedback.promotion import (
    EvidenceCheck,
    evaluate_promotion,
)
from app.feedback.repository import FeedbackProjectionRepository
from app.feedback.triage import deduplicate_cases, score_case


def _make_operational_db(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE chat_sessions(session_id TEXT PRIMARY KEY, created_at TEXT);
        CREATE TABLE messages(
            message_id TEXT PRIMARY KEY, session_id TEXT, role TEXT, content TEXT,
            status TEXT, reply_to_message_id TEXT, rewritten_question TEXT,
            citations_json TEXT, reason_code TEXT, reference_answer TEXT,
            audience_scope TEXT, workflow_summary_json TEXT, created_at TEXT
        );
        CREATE TABLE feedback(
            feedback_id TEXT PRIMARY KEY, session_id TEXT, message_id TEXT,
            helpful INTEGER, reason TEXT, created_at TEXT
        );
        """
    )
    connection.execute("INSERT INTO chat_sessions VALUES ('s1', '2026-09-03T10:00:00+08:00')")
    rows = [
        ("u1", "透析中低血压怎么处理？", "a1", "患者姓名：张三。请按制度处理。", "answered", None, True),
        ("u2", "透析中低血压 怎么处理", "a2", "请按制度处理。", "answered", None, False),
        ("u3", "当前没有的外部信息？", "a3", "当前资料没有覆盖。", "refused", "not_answered", False),
        ("u4", "高风险患者姓名：李四的急症？", "a4", "患者姓名：李四。", "answered", None, True),
        ("u5", "疑似标识 AB123456789012345678", "a5", "请人工核验。", "answered", None, False),
    ]
    for user_id, question, assistant_id, answer, status, reason, high_risk in rows:
        connection.execute(
            "INSERT INTO messages VALUES (?, 's1', 'user', ?, 'submitted', NULL, NULL, '[]', NULL, NULL, 'nurse', NULL, '2026-09-03T10:00:01+08:00')",
            (user_id, question),
        )
        connection.execute(
            "INSERT INTO messages VALUES (?, 's1', 'assistant', ?, ?, ?, NULL, ?, ?, NULL, 'nurse', ?, '2026-09-03T10:00:02+08:00')",
            (
                assistant_id,
                answer,
                status,
                user_id,
                json.dumps([{"chunk_id": "chunk-1"}]) if status == "answered" else "[]",
                reason,
                json.dumps(
                    {
                        "model_id": "deepseek-v4-flash",
                        "prompt_version": "c1",
                        "source_version": "2026-standard-manual-v1",
                        "high_risk": high_risk,
                    }
                ),
            ),
        )
        connection.execute(
            "INSERT INTO feedback VALUES (?, 's1', ?, ?, ?, '2026-09-03T10:01:00+08:00')",
            (f"f-{assistant_id}", assistant_id, 0 if status == "refused" else 1, reason),
        )
    connection.commit()
    connection.close()


def test_task17a_end_to_end_is_local_redacted_and_source_read_only(tmp_path: Path) -> None:
    source = tmp_path / "operational.sqlite3"
    _make_operational_db(source)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    projection_path = tmp_path / "data" / "private" / "feedback" / "feedback.db"
    projection = FeedbackProjectionRepository(projection_path)
    source_connection = sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)
    source_connection.row_factory = sqlite3.Row
    try:
        rows = source_connection.execute(
            "SELECT * FROM messages WHERE role='assistant' ORDER BY message_id"
        ).fetchall()
        for row in rows:
            from app.chat.models import ChatMessage, Feedback

            message = ChatMessage(
                message_id=row["message_id"],
                session_id=row["session_id"],
                role="assistant",
                content=row["content"],
                status=row["status"],
                rewritten_question=None,
                citations=tuple(json.loads(row["citations_json"])),
                reason_code=row["reason_code"],
                created_at=row["created_at"],
                reply_to_message_id=row["reply_to_message_id"],
                audience_scope="nurse",
                workflow_summary=json.loads(row["workflow_summary_json"]),
            )
            question = source_connection.execute(
                "SELECT content FROM messages WHERE message_id = ?", (row["reply_to_message_id"],)
            ).fetchone()[0]
            feedback_row = source_connection.execute(
                "SELECT message_id, helpful, reason, created_at FROM feedback WHERE message_id = ?",
                (message.message_id,),
            ).fetchone()
            feedback = Feedback(
                message_id=feedback_row["message_id"],
                helpful=bool(feedback_row["helpful"]),
                created_at=feedback_row["created_at"],
                reason=feedback_row["reason"],
            )
            case, event = projection.build_case_from_message(
                message,
                feedback,
                trace={**message.workflow_summary, "question": question},
                now=datetime(2026, 9, 3, 10, 2, tzinfo=timezone.utc),
            )
            projection.upsert_case(case)
            if event:
                projection.append_event(event)
            decision = score_case(
                case,
                high_risk=bool(message.workflow_summary.get("high_risk")),
                citation_mismatch=case.reason_code == "citation_mismatch",
            )
            projection.update_triage(case.case_id, decision.priority, decision.score, decision.reasons)
    finally:
        source_connection.close()

    cases = projection.list_cases(limit=20)
    assert len(cases) == 5
    assert any(case.redaction_status == "review_required" for case in cases)
    assert any(case.triage_priority == "P0" for case in cases)
    assert len(deduplicate_cases(cases)) == 4

    answerable = next(
        case
        for case in cases
        if case.redaction_status == "passed"
        and case.answer_status == "answered"
        and case.triage_priority != "P0"
    )
    answerable_decision = evaluate_promotion(
        answerable,
        evidence_check=EvidenceCheck(True, ("chunk-1",), True, True),
        corpus_check=None,
        review_decisions=(
            ReviewDecision(
                review_id="review-safe",
                case_id=answerable.case_id,
                reviewer_role="product",
                decision="approve",
                evidence_ok=True,
                points_ok=True,
                safety_ok=True,
                note_code="checked",
                review_version="v1",
                reviewed_at="2026-09-03T10:00:00+08:00",
            ),
        ),
        target_split="dev",
    )
    assert answerable_decision.status == "golden_v2_candidate"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert not any(path.name.endswith("raw_provider_response.json") for path in projection_path.parent.rglob("*"))
    projection.close()
