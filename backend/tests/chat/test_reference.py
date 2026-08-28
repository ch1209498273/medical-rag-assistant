from __future__ import annotations

import pytest
from app.chat.reference import ReferenceAnswerService
from app.errors import ProviderError


class FakeDeepSeek:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.messages = None
        self.calls = 0

    async def complete_json(self, messages):
        self.calls += 1
        self.messages = messages
        if self.error is not None:
            raise self.error
        return self.result


@pytest.mark.asyncio
async def test_reference_answer_uses_only_standalone_question():
    provider = FakeDeepSeek({"reference_answer": "这是一段未核验的通用学习参考。"})

    result = await ReferenceAnswerService(provider).generate("透析低血压一般有哪些原因？")

    assert result == "这是一段未核验的通用学习参考。"
    assert provider.calls == 1
    assert provider.messages[0]["role"] == "system"
    user_prompt = provider.messages[1]["content"]
    assert "透析低血压一般有哪些原因？" in user_prompt
    assert "source_id" not in user_prompt
    assert "内部制度原文" not in user_prompt
    assert "conversation-history" not in user_prompt


@pytest.mark.asyncio
async def test_reference_answer_rejects_extra_fields_and_unsafe_output():
    provider = FakeDeepSeek(
        {"reference_answer": "通用参考", "citations": [{"reference_id": "S1"}]}
    )
    assert await ReferenceAnswerService(provider).generate("问题") is None

    provider = FakeDeepSeek(
        {
            "reference_answer": "C:"
            + chr(92)
            + "private"
            + chr(92)
            + "secret.txt"
        }
    )
    assert await ReferenceAnswerService(provider).generate("问题") is None


@pytest.mark.asyncio
async def test_reference_answer_fails_closed_without_retry():
    provider = FakeDeepSeek(
        error=ProviderError("deepseek", "complete", None, True, "upstream")
    )

    assert await ReferenceAnswerService(provider).generate("问题") is None
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_reference_answer_rejects_empty_or_oversized_response():
    provider = FakeDeepSeek({"reference_answer": "   "})
    assert await ReferenceAnswerService(provider).generate("问题") is None

    provider = FakeDeepSeek({"reference_answer": "答案" * 2001})
    assert await ReferenceAnswerService(provider).generate("问题") is None
