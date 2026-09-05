from __future__ import annotations

import importlib.util
import json

import pytest
from app.storage.sqlite import SqliteDocumentRepository


def test_bad_case_state_machine_rejects_skip_to_closed() -> None:
    assert importlib.util.find_spec("app.evaluation.bad_cases") is not None
    from app.evaluation.bad_cases import apply_bad_case_transition

    with pytest.raises(ValueError, match="invalid transition"):
        apply_bad_case_transition("new", "closed")


def test_bad_case_export_has_no_question_or_answer_text(tmp_path) -> None:
    assert importlib.util.find_spec("scripts.export_bad_case_summary") is not None
    from scripts.export_bad_case_summary import build_bad_case_summary

    database = tmp_path / "private.sqlite3"
    repository = SqliteDocumentRepository(database)
    try:
        session_id = repository.create_chat_session()
        user = repository.append_user_message(session_id, "虚构原始问题")
        assistant = repository.append_assistant_message(
            session_id,
            user.message_id,
            "虚构答案正文",
            "answered",
            (),
            None,
        )
        repository.upsert_feedback(assistant.message_id, False, "missing_step")
    finally:
        repository.close()

    report = build_bad_case_summary(database)

    assert report["counts"]["USER_FEEDBACK_MISSING_STEP"] == 1
    encoded = json.dumps(report, ensure_ascii=False)
    assert "虚构原始问题" not in encoded
    assert "虚构答案正文" not in encoded
    assert str(database) not in encoded
    assert "DEEPSEEK_API_KEY" not in encoded
