from __future__ import annotations

from app.chat.history import HistoryContextBuilder
from app.chat.models import ChatMessage


def _message(
    message_id: str,
    session_id: str,
    role: str,
    content: str,
    *,
    status: str,
    reply_to: str | None = None,
) -> ChatMessage:
    return ChatMessage(
        message_id=message_id,
        session_id=session_id,
        role=role,
        content=content,
        status=status,
        rewritten_question=None,
        citations=(),
        reason_code=None,
        created_at=message_id,
        reply_to_message_id=reply_to,
    )


def make_six_complete_turns(answer_chars: int = 500) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    for number in range(1, 7):
        user_id = f"u{number}"
        messages.append(_message(user_id, "s1", "user", f"问题{number}", status="submitted"))
        messages.append(
            _message(
                f"a{number}",
                "s1",
                "assistant",
                "答" * answer_chars,
                status="answered",
                reply_to=user_id,
            )
        )
    return messages


def test_history_keeps_newest_five_complete_turns_and_6000_chars():
    messages = make_six_complete_turns(answer_chars=500)

    context = HistoryContextBuilder(max_turns=5, max_chars=6000).build(messages)

    assert len(context.turns) == 5
    assert context.turns[0].user_question == "问题6"
    assert context.turns[-1].user_question == "问题2"


def test_history_does_not_split_a_question_answer_pair():
    messages = make_six_complete_turns(answer_chars=500)
    messages.extend(
        [
            _message("u-long", "s1", "user", "超长问题", status="submitted"),
            _message(
                "a-long",
                "s1",
                "assistant",
                "答" * 6000,
                status="answered",
                reply_to="u-long",
            ),
        ]
    )

    context = HistoryContextBuilder(max_turns=5, max_chars=6000).build(messages)

    assert all(turn.user_question != "超长问题" for turn in context.turns)


def test_history_ignores_submitted_and_error_messages():
    messages = [
        _message("u-incomplete", "s1", "user", "未完成问题", status="submitted"),
        _message(
            "a-error",
            "s1",
            "assistant",
            "暂时不可用",
            status="error",
            reply_to="u-incomplete",
        ),
        _message("u-complete", "s1", "user", "已完成问题", status="submitted"),
        _message(
            "a-complete",
            "s1",
            "assistant",
            "安全答复",
            status="refused",
            reply_to="u-complete",
        ),
    ]

    context = HistoryContextBuilder().build(messages)

    assert [turn.user_question for turn in context.turns] == ["已完成问题"]


def test_history_builder_rejects_budget_above_approved_cap():
    import pytest

    with pytest.raises(ValueError):
        HistoryContextBuilder(max_chars=6001)
