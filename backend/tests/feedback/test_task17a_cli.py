from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

APP_ROOT = Path(__file__).parents[2].parent
PROJECTION_SCRIPT = APP_ROOT / "scripts" / "task17a_feedback_projection.py"
RETENTION_SCRIPT = APP_ROOT / "scripts" / "task17a_feedback_retention.py"


def _source_db(tmp_path: Path) -> Path:
    path = tmp_path / "source.sqlite3"
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
    connection.execute(
        "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "u1",
            "s1",
            "user",
            "患者姓名：张三的透析参数是什么？",
            "submitted",
            None,
            None,
            "[]",
            None,
            None,
            "nurse",
            None,
            "2026-09-03T10:00:01+08:00",
        ),
    )
    connection.execute(
        "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "a1",
            "s1",
            "assistant",
            "患者姓名：张三。请按制度处理。",
            "answered",
            "u1",
            None,
            json.dumps([{"chunk_id": "chunk-1", "excerpt": "secret"}]),
            None,
            None,
            "nurse",
            json.dumps({"model_id": "deepseek-v4-flash", "prompt_version": "c1"}),
            "2026-09-03T10:00:02+08:00",
        ),
    )
    connection.execute(
        "INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?)",
        ("f1", "s1", "a1", 0, "citation_mismatch", "2026-09-03T10:01:00+08:00"),
    )
    connection.commit()
    connection.close()
    return path


def _private_db(tmp_path: Path) -> Path:
    path = tmp_path / "data" / "private" / "feedback" / "feedback.db"
    path.parent.mkdir(parents=True)
    return path


def test_cli_help_and_path_rejection(tmp_path: Path) -> None:
    help_result = subprocess.run(
        [sys.executable, str(PROJECTION_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0
    rejected = subprocess.run(
        [
            sys.executable,
            str(PROJECTION_SCRIPT),
            "build",
            "--source-db",
            str(_source_db(tmp_path)),
            "--output-db",
            str(tmp_path / "unsafe.db"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert "private" in (rejected.stdout + rejected.stderr).lower()


def test_build_is_idempotent_and_list_is_redacted(tmp_path: Path) -> None:
    source = _source_db(tmp_path)
    database = _private_db(tmp_path)
    command = [
        sys.executable,
        str(PROJECTION_SCRIPT),
        "build",
        "--source-db",
        str(source),
        "--output-db",
        str(database),
    ]
    first = subprocess.run(command, capture_output=True, text=True, check=False)
    second = subprocess.run(command, capture_output=True, text=True, check=False)
    assert first.returncode == 0
    assert second.returncode == 0
    summary = json.loads(first.stdout)
    assert summary["network_calls"] == 0
    assert summary["provider_constructed"] is False
    assert summary["redaction_passed"] == 1

    listed = subprocess.run(
        [sys.executable, str(PROJECTION_SCRIPT), "list", "--database", str(database)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert listed.returncode == 0
    assert "张三" not in listed.stdout
    assert "secret" not in listed.stdout
    assert "[患者-1]" in listed.stdout


def test_purge_prints_counts_only(tmp_path: Path) -> None:
    source = _source_db(tmp_path)
    database = _private_db(tmp_path)
    built = subprocess.run(
        [sys.executable, str(PROJECTION_SCRIPT), "build", "--source-db", str(source), "--output-db", str(database)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0
    purged = subprocess.run(
        [sys.executable, str(RETENTION_SCRIPT), "purge", "--database", str(database), "--as-of", "2027-01-01"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert purged.returncode == 0
    payload = json.loads(purged.stdout)
    assert payload["cases_deleted"] == 1
    assert "张三" not in purged.stdout
    assert (database.parent / "retention-log.json").exists()
