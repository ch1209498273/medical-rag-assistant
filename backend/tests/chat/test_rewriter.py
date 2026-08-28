from __future__ import annotations

import json

import pytest
from app.chat.models import ChatTurn, HistoryContext
from app.chat.rewriter import FollowUpRewriter, RewriteError
from app.errors import ProviderError
from app.settings import Settings


class FakeDeepSeek:
    def __init__(self, json_result=None, error=None):
        self.json_result = json_result
        self.error = error
        self.messages = None

    async def complete_json(self, messages):
        self.messages = messages
        if self.error is not None:
            raise self.error
        return self.json_result


def make_turn() -> ChatTurn:
    return ChatTurn(
        user_message_id="u1",
        user_question="透析中低血压怎么处理？",
        rewritten_question=None,
        assistant_message_id="a1",
        answer="应当按制度监测并处理。",
        status="answered",
        citations=(),
    )


@pytest.mark.asyncio
async def test_rewriter_returns_only_valid_standalone_question():
    deepseek = FakeDeepSeek(json_result={"standalone_question": "透析低血压如何预防？"})

    result = await FollowUpRewriter(deepseek).rewrite(
        "那怎么预防？", HistoryContext(turns=(make_turn(),))
    )

    assert result == "透析低血压如何预防？"
    assert "source_id" not in json.dumps(deepseek.messages, ensure_ascii=False)


@pytest.mark.asyncio
async def test_rewriter_rejects_extra_fields_and_does_not_fallback_to_original():
    deepseek = FakeDeepSeek(
        json_result={"standalone_question": "那怎么预防？", "answer": "猜测"}
    )

    with pytest.raises(RewriteError):
        await FollowUpRewriter(deepseek).rewrite(
            "那怎么预防？", HistoryContext(turns=(make_turn(),))
        )


@pytest.mark.asyncio
async def test_rewriter_rejects_provider_failure():
    deepseek = FakeDeepSeek(error=ProviderError("deepseek", "complete", None, True, "x"))

    with pytest.raises(RewriteError):
        await FollowUpRewriter(deepseek).rewrite(
            "那怎么预防？", HistoryContext(turns=(make_turn(),))
        )


@pytest.mark.asyncio
async def test_rewriter_serialized_prompt_stays_within_approved_history_budget():
    turns = tuple(
        ChatTurn(
            user_message_id=f"u{index}",
            user_question=f"问题{index}",
            rewritten_question="独立问题" * 40,
            assistant_message_id=f"a{index}",
            answer="答复" * 500,
            status="answered",
            citations=(),
        )
        for index in range(5)
    )
    deepseek = FakeDeepSeek(json_result={"standalone_question": "独立追问"})

    await FollowUpRewriter(deepseek).rewrite(
        "当前怎么处理？", HistoryContext(turns=turns)
    )

    assert len(deepseek.messages[1]["content"]) <= 6000


def test_settings_rejects_history_budget_above_approved_cap():
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        Settings(_env_file=None, chat_history_max_chars=6001)
