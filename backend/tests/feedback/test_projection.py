from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from app.chat.models import ChatMessage, Feedback
from app.feedback.repository import FeedbackProjectionRepository


def test_build_case_from_message_keeps_redacted_content_and_safe_trace(tmp_path):
    repository = FeedbackProjectionRepository(
        tmp_path / "data" / "private" / "feedback" / "feedback.db"
    )
    repository.close()
    message = ChatMessage(
        message_id="message-123",
        session_id="session-456",
        role="assistant",
        content="患者姓名：张三。按制度处理。",
        status="answered",
        rewritten_question=None,
        citations=(
            {
                "chunk_id": "chunk-1",
                "file_name": "secret.docx",
                "excerpt": "不得保存到反馈投影",
            },
        ),
        reason_code=None,
        created_at="2026-09-03T10:00:00+08:00",
        reply_to_message_id="user-1",
        workflow_summary={"model_id": "deepseek-v4-flash", "reason_code": None},
    )
    feedback = Feedback(
        message_id=message.message_id,
        helpful=False,
        created_at="2026-09-03T10:01:00+08:00",
        reason="citation_mismatch",
    )

    case, event = FeedbackProjectionRepository.build_case_from_message(
        message,
        feedback,
        trace={
            "question": "患者姓名：张三的透析参数是什么？",
            "source_version": "2026-standard-manual-v1",
            "retrieval_profile": "baseline_v1+vector+c1",
            "model_id": "deepseek-v4-flash",
            "prompt_version": "c1",
            "latency_bucket": "lt_5s",
            "audience_scope": "nurse",
            "provider_response": "must never persist",
            "reasoning_content": "must never persist",
        },
        now=datetime(2026, 9, 3, 10, 2, tzinfo=timezone.utc),
    )

    assert case.message_id_hash == hashlib.sha256(b"message-123").hexdigest()
    assert "张三" not in case.question_redacted + case.answer_redacted
    assert case.citation_chunk_ids == ("chunk-1",)
    assert event is not None
    serialized = json.dumps(case.to_dict(), ensure_ascii=False)
    assert "provider_response" not in serialized
    assert "reasoning_content" not in serialized
    assert "secret.docx" not in serialized
