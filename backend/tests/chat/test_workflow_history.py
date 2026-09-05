from __future__ import annotations

import sqlite3

from app.storage.sqlite import SqliteDocumentRepository


def _summary(*, outcome: str = "answered") -> dict[str, object]:
    return {
        "workflow_version": "agent_workflow_v2a",
        "run_id": "run-12345678",
        "route": "verify",
        "outcome": outcome,
        "verifier_status": "passed" if outcome == "answered" else "failed",
        "http_calls": 4,
        "elapsed_ms": 820,
        "reason_code": None if outcome == "answered" else "GENERATION_UNAVAILABLE",
    }


def test_workflow_summary_round_trips_without_creating_a_second_message_table(tmp_path):
    database = tmp_path / "metadata.sqlite3"
    repository = SqliteDocumentRepository(database)
    try:
        session_id = repository.create_chat_session()
        user = repository.append_user_message(session_id, "请说明核验流程")
        assistant = repository.append_assistant_message(
            session_id,
            user.message_id,
            "已按资料核验。",
            "answered",
            (),
            None,
            workflow_summary=_summary(),
        )

        restored = repository.list_chat_messages(session_id)[-1]
        tables = {
            row[0]
            for row in repository._connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }

        assert assistant.workflow_summary == _summary()
        assert restored.workflow_summary == _summary()
        assert "workflow_summaries" not in tables
        assert repository.schema_version() == 8
    finally:
        repository.close()

    reopened = SqliteDocumentRepository(database)
    try:
        assert reopened.list_chat_messages(session_id)[-1].workflow_summary == _summary()
        assert reopened.schema_version() == 8
    finally:
        reopened.close()


def test_legacy_messages_keep_a_null_workflow_summary(tmp_path):
    repository = SqliteDocumentRepository(tmp_path / "metadata.sqlite3")
    try:
        session_id = repository.create_chat_session()
        user = repository.append_user_message(session_id, "旧问题")

        restored = repository.list_chat_messages(session_id)[0]

        assert restored.workflow_summary is None
        assert user.workflow_summary is None
    finally:
        repository.close()


def test_schema_v6_is_migrated_idempotently_to_the_nullable_summary_column(tmp_path):
    database = tmp_path / "legacy-v6.sqlite3"
    repository = SqliteDocumentRepository(database)
    repository.close()

    connection = sqlite3.connect(database)
    try:
        connection.execute("ALTER TABLE messages DROP COLUMN workflow_summary_json")
        connection.execute("UPDATE schema_version SET version = 6")
        connection.commit()
    finally:
        connection.close()

    migrated = SqliteDocumentRepository(database)
    try:
        columns = {
            row[1]
            for row in migrated._connection.execute("PRAGMA table_info(messages)").fetchall()
        }
        assert "workflow_summary_json" in columns
        assert migrated.schema_version() == 8
    finally:
        migrated.close()

    reopened = SqliteDocumentRepository(database)
    try:
        assert reopened.schema_version() == 8
    finally:
        reopened.close()


def test_corrupted_workflow_summary_degrades_history_to_a_safe_error(tmp_path):
    repository = SqliteDocumentRepository(tmp_path / "corrupt-summary.sqlite3")
    try:
        session_id = repository.create_chat_session()
        user = repository.append_user_message(session_id, "问题")
        repository.append_assistant_message(
            session_id,
            user.message_id,
            "答案",
            "answered",
            (),
            None,
        )
        repository._connection.execute(
            "UPDATE messages SET workflow_summary_json = ? WHERE role = 'assistant'",
            ("{not-json}",),
        )
        repository._connection.commit()

        restored = repository.list_chat_messages(session_id)[-1]

        assert restored.status == "error"
        assert restored.workflow_summary is None
        assert restored.content == "当前会话记录无法安全恢复。"
    finally:
        repository.close()
