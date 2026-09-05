"""Contract tests for DeepSeek streaming responses."""

from __future__ import annotations

from dataclasses import dataclass

import httpx
import pytest
from app.agents.budget import (
    BudgetConfigurationError,
    RequestBudget,
    WorkflowBudgetExceeded,
    request_scope,
    stage_scope,
)
from app.errors import ProviderError
from app.providers.deepseek import DeepSeekClient


@dataclass
class StreamTransport:
    chunks: list[bytes]

    def __post_init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def stream(self, method: str, url: str, **kwargs: object):
        self.calls.append({"method": method, "url": url, **kwargs})
        for chunk in self.chunks:
            yield chunk


@dataclass
class JsonTransport:
    payload: object
    status_code: int = 200

    def __post_init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def request(self, method: str, url: str, **kwargs: object):
        self.calls.append({"method": method, "url": url, **kwargs})
        return type(
            "Response",
            (),
            {"status_code": self.status_code, "json": lambda _: self.payload},
        )()


class ReadTimeoutThenSuccessTransport:
    def __init__(self, *, after_content: bool = False) -> None:
        self.calls = 0
        self.after_content = after_content

    async def stream(self, method: str, url: str, **kwargs: object):
        self.calls += 1
        if self.calls == 1:
            if self.after_content:
                yield 'data: {"choices":[{"delta":{"content":"依据"}}]}\n\n'.encode()
            raise httpx.ReadTimeout(
                "read timeout with Authorization: " + "Bearer " + "secret"
            )
        yield (
            'data: {"choices":[{"delta":{"content":"重试成功"}}]}\n\n'.encode()
        )
        yield b"data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_deepseek_stream_yields_only_content_across_arbitrary_chunks() -> None:
    wire = (
        b'data: {"choices":[{"delta":{"role":"assistant"}}]}\r\n\r\n'
        + 'data: {"choices":[{"delta":{"content":"依"}}]}\n\n'.encode()
        + 'data: {"choices":[{"delta":{"content":"据[S1]"}}]}\n\n'.encode()
        + b"data: [DONE]\n\n"
    )
    chunks = [wire[:3], wire[3:17], wire[17:29], wire[29:47], wire[47:]]
    transport = StreamTransport(chunks)
    client = DeepSeekClient("key", transport=transport)

    result = "".join([part async for part in client.stream_answer([])])

    assert result == "依据[S1]"
    assert transport.calls[0]["url"] == "https://api.deepseek.com/chat/completions"
    assert transport.calls[0]["json"]["stream"] is True  # type: ignore[index]
    assert "response_format" not in transport.calls[0]["json"]  # type: ignore[operator]


@pytest.mark.asyncio
async def test_deepseek_stream_can_request_json_object_response_format() -> None:
    transport = StreamTransport([b"data: [DONE]\n\n"])
    client = DeepSeekClient("key", transport=transport)

    assert [part async for part in client.stream_answer([], json_output=True)] == []

    request_json = transport.calls[0]["json"]
    assert isinstance(request_json, dict)
    assert request_json["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_deepseek_stream_forwards_per_call_answer_budget_and_thinking_mode() -> None:
    transport = StreamTransport([b"data: [DONE]\n\n"])
    client = DeepSeekClient(
        "key",
        transport=transport,
        max_tokens=2048,
        max_tokens_limit=32768,
        max_retries=0,
    )

    assert [
        part
        async for part in client.stream_answer(
            [], max_tokens=32768, thinking={"type": "disabled"}
        )
    ] == []

    request_json = transport.calls[0]["json"]
    assert isinstance(request_json, dict)
    assert request_json["max_tokens"] == 32768
    assert request_json["thinking"] == {"type": "disabled"}


@pytest.mark.asyncio
async def test_deepseek_stream_forwards_explicit_temperature_without_changing_default() -> None:
    transport = StreamTransport([b"data: [DONE]\n\n"])
    client = DeepSeekClient("key", transport=transport, max_retries=0)

    assert [part async for part in client.stream_answer([], temperature=0)] == []
    explicit_payload = transport.calls[0]["json"]
    assert isinstance(explicit_payload, dict)
    assert explicit_payload["temperature"] == 0.0

    transport_default = StreamTransport([b"data: [DONE]\n\n"])
    default_client = DeepSeekClient("key", transport=transport_default, max_retries=0)
    assert [part async for part in default_client.stream_answer([])] == []
    default_payload = transport_default.calls[0]["json"]
    assert isinstance(default_payload, dict)
    assert "temperature" not in default_payload


@pytest.mark.parametrize("value", [-0.1, 2.1, True])
def test_deepseek_rejects_invalid_stream_temperature_before_transport(value) -> None:
    transport = StreamTransport([b"data: [DONE]\n\n"])
    client = DeepSeekClient("key", transport=transport, max_retries=0)

    async def consume() -> None:
        async for _ in client.stream_answer([], temperature=value):
            pass

    with pytest.raises(ValueError, match="temperature"):
        import asyncio

        asyncio.run(consume())
    assert transport.calls == []


@pytest.mark.asyncio
async def test_deepseek_structured_default_stays_small_when_answer_limit_is_larger() -> None:
    transport = JsonTransport(
        {"choices": [{"message": {"content": '{"ok":true}'}}]}
    )
    client = DeepSeekClient(
        "key",
        transport=transport,
        max_tokens=2048,
        max_tokens_limit=32768,
        max_retries=0,
    )

    assert await client.complete_json([]) == {"ok": True}

    request_json = transport.calls[0]["json"]
    assert isinstance(request_json, dict)
    assert request_json["max_tokens"] == 2048
    assert "thinking" not in request_json


def test_deepseek_rejects_answer_budget_and_thinking_mode_before_transport() -> None:
    transport = JsonTransport(
        {"choices": [{"message": {"content": '{"ok":true}'}}]}
    )
    client = DeepSeekClient(
        "key", transport=transport, max_tokens=2048, max_tokens_limit=32768
    )

    with pytest.raises(ValueError, match="max_tokens"):
        client._payload([], stream=True, max_tokens=32769)
    with pytest.raises(ValueError, match="thinking"):
        client._payload([], stream=True, thinking={"type": "unknown"})
    with pytest.raises(ValueError, match="thinking"):
        client._payload([], stream=True, thinking={"type": 1})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="thinking"):
        client._payload([], stream=True, thinking="disabled")  # type: ignore[arg-type]

    assert transport.calls == []


@pytest.mark.asyncio
async def test_deepseek_stream_ignores_reasoning_content() -> None:
    transport = StreamTransport(
        [
            b'data: {"choices":[{"delta":{"reasoning_content":"private"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"visible"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]
    )
    client = DeepSeekClient("key", transport=transport, max_retries=0)

    assert "".join([part async for part in client.stream_answer([])]) == "visible"


@pytest.mark.asyncio
async def test_deepseek_complete_json_requests_json_object_response_format() -> None:
    transport = JsonTransport(
        {"choices": [{"message": {"content": '{"cases": []}'}}]}
    )
    client = DeepSeekClient("key", transport=transport)

    result = await client.complete_json([])

    assert result == {"cases": []}
    request_json = transport.calls[0]["json"]
    assert isinstance(request_json, dict)
    assert request_json["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_deepseek_complete_json_reports_only_numeric_usage() -> None:
    transport = JsonTransport(
        {
            "choices": [{"message": {"content": '{"ok":true}'}}],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "total_tokens": 20,
                "secret": "not forwarded",
            },
        }
    )
    observed: list[dict[str, object]] = []
    client = DeepSeekClient(
        "key", transport=transport, max_retries=0, usage_observer=observed.append
    )

    assert await client.complete_json([]) == {"ok": True}

    assert observed == [
        {
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "operation": "complete",
            "input_tokens": 12,
            "output_tokens": 8,
            "total_tokens": 20,
            "usage_complete": True,
        }
    ]


@pytest.mark.asyncio
async def test_candidate_provider_without_stage_reports_budget_configuration_error():
    transport = JsonTransport(
        {"choices": [{"message": {"content": '{"ok":true}'}}]}
    )
    client = DeepSeekClient("key", transport=transport, max_retries=0)

    async with request_scope(RequestBudget()):
        with pytest.raises(BudgetConfigurationError):
            await client.complete_json([])

    assert transport.calls == []


@pytest.mark.asyncio
async def test_candidate_provider_preserves_budget_exhaustion():
    transport = JsonTransport(
        {"choices": [{"message": {"content": '{"ok":true}'}}]}
    )
    client = DeepSeekClient("key", transport=transport, max_retries=0)
    budget = RequestBudget(max_calls=1)

    async with request_scope(budget):
        with stage_scope("routing"):
            assert await client.complete_json([]) == {"ok": True}
        with stage_scope("routing"), pytest.raises(WorkflowBudgetExceeded):
            await client.complete_json([])

    assert len(transport.calls) == 1


@pytest.mark.asyncio
async def test_deepseek_stream_reports_none_when_provider_omits_usage() -> None:
    transport = StreamTransport(
        [
            b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]
    )
    observed: list[dict[str, object]] = []
    client = DeepSeekClient(
        "key", transport=transport, max_retries=0, usage_observer=observed.append
    )

    assert "".join([part async for part in client.stream_answer([])]) == "ok"
    assert observed == [
        {
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "operation": "stream_answer",
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "usage_complete": False,
        }
    ]


@pytest.mark.asyncio
async def test_deepseek_complete_json_allows_bounded_semantic_judge_overrides() -> None:
    transport = JsonTransport(
        {"choices": [{"message": {"content": '{"point_verdicts":[]}'}}]}
    )
    client = DeepSeekClient("key", transport=transport, max_tokens=2048, max_retries=0)

    result = await client.complete_json(
        [],
        temperature=0.0,
        max_tokens=512,
        operation="semantic_judge",
    )

    assert result == {"point_verdicts": []}
    request_json = transport.calls[0]["json"]
    assert isinstance(request_json, dict)
    assert request_json["temperature"] == 0.0
    assert request_json["max_tokens"] == 512


@pytest.mark.asyncio
async def test_deepseek_complete_json_rejects_invalid_judge_override_before_transport() -> None:
    transport = JsonTransport(
        {"choices": [{"message": {"content": '{"point_verdicts":[]}'}}]}
    )
    client = DeepSeekClient("key", transport=transport, max_tokens=2048, max_retries=0)

    with pytest.raises(ValueError, match="temperature"):
        await client.complete_json([], temperature=-0.1)

    assert transport.calls == []


@pytest.mark.asyncio
async def test_deepseek_complete_omits_json_object_response_format() -> None:
    transport = JsonTransport(
        {"choices": [{"message": {"content": "ordinary answer"}}]}
    )
    client = DeepSeekClient("key", transport=transport)

    result = await client.complete([])

    assert result == "ordinary answer"
    request_json = transport.calls[0]["json"]
    assert isinstance(request_json, dict)
    assert "response_format" not in request_json


@pytest.mark.asyncio
async def test_formal_deepseek_client_keeps_default_output_and_retry_policy() -> None:
    transport = JsonTransport(
        {"choices": [{"message": {"content": "ordinary answer"}}]}
    )
    client = DeepSeekClient("key", transport=transport)

    assert await client.complete([]) == "ordinary answer"

    request = transport.calls[0]
    request_json = request["json"]
    assert isinstance(request_json, dict)
    assert request_json["max_tokens"] == 2048
    assert request_json["stream"] is False
    assert request["timeout"] == 30.0
    assert client.max_tokens == 2048
    assert client.timeout == 30.0
    assert client.max_retries == 3


@pytest.mark.asyncio
async def test_deepseek_stream_rejects_error_after_partial_content() -> None:
    transport = StreamTransport(
        [
            'data: {"choices":[{"delta":{"content":"依据"}}]}\n\n'.encode(),
            b'data: {"error":{"message":"upstream detail"}}\n\n',
        ]
    )
    client = DeepSeekClient("key", transport=transport)

    stream = client.stream_answer([])
    assert await anext(stream) == "依据"
    with pytest.raises(ProviderError) as raised:
        await anext(stream)

    assert raised.value.provider == "deepseek"
    assert raised.value.retryable is False
    assert raised.value.failure_kind == "schema"
    assert "upstream detail" not in str(raised.value)


@pytest.mark.asyncio
async def test_deepseek_complete_json_marks_invalid_structured_output_as_schema_failure() -> None:
    client = DeepSeekClient(
        "key",
        transport=JsonTransport(
            {"choices": [{"message": {"content": "not-json"}}]}
        ),
        max_retries=0,
    )

    with pytest.raises(ProviderError) as raised:
        await client.complete_json([])

    assert raised.value.failure_kind == "schema"


@pytest.mark.asyncio
async def test_deepseek_stream_requires_a_complete_done_event() -> None:
    transport = StreamTransport(
        ['data: {"choices":[{"delta":{"content":"半截"}}]}\n\n'.encode()]
    )
    client = DeepSeekClient("key", transport=transport)

    with pytest.raises(ProviderError, match="complete"):
        _ = "".join([part async for part in client.stream_answer([])])


@pytest.mark.asyncio
async def test_deepseek_read_timeout_retries_before_any_content_with_backoff() -> None:
    transport = ReadTimeoutThenSuccessTransport()
    delays: list[float] = []
    client = DeepSeekClient("key", transport=transport, sleep=delays.append)

    result = "".join([part async for part in client.stream_answer([])])

    assert result == "重试成功"
    assert transport.calls == 2
    assert delays == [0.5]


@pytest.mark.asyncio
async def test_deepseek_read_timeout_after_content_does_not_replay_stream() -> None:
    transport = ReadTimeoutThenSuccessTransport(after_content=True)
    client = DeepSeekClient("key", transport=transport)
    stream = client.stream_answer([])

    assert await anext(stream) == "依据"
    with pytest.raises(ProviderError) as raised:
        await anext(stream)

    assert raised.value.retryable is True
    assert transport.calls == 1


@pytest.mark.asyncio
async def test_deepseek_stream_hard_byte_limit_closes_and_fails_closed() -> None:
    transport = StreamTransport(
        [
            b'data: {"choices":[{"delta":{"content":"123456"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]
    )
    client = DeepSeekClient("key", transport=transport, max_stream_bytes=32)

    with pytest.raises(ProviderError, match="limit"):
        _ = "".join([part async for part in client.stream_answer([])])


@pytest.mark.asyncio
async def test_deepseek_stream_deadline_rejects_a_never_done_source() -> None:
    class NeverDone:
        async def stream(self, method: str, url: str, **kwargs: object):
            while True:
                await asyncio.sleep(0.01)
                yield b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'

    import asyncio

    client = DeepSeekClient(
        "key", transport=NeverDone(), stream_timeout_seconds=0.001, max_retries=0
    )

    with pytest.raises(ProviderError, match="interrupted"):
        _ = "".join([part async for part in client.stream_answer([])])


@pytest.mark.asyncio
async def test_deepseek_event_limit_counts_multiple_sse_events_in_one_raw_chunk():
    transport = StreamTransport(
        [
            (
                b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
                b'data: {"choices":[{"delta":{"content":"b"}}]}\n\n'
                b"data: [DONE]\n\n"
            )
        ]
    )
    client = DeepSeekClient("key", transport=transport, max_stream_events=2)

    with pytest.raises(ProviderError, match="event limit"):
        _ = "".join([part async for part in client.stream_answer([])])


@pytest.mark.asyncio
async def test_deepseek_retry_backoff_consumes_one_total_deadline():
    transport = ReadTimeoutThenSuccessTransport()

    async def slow_sleep(delay: float):
        await asyncio.sleep(0.01)

    import asyncio

    client = DeepSeekClient(
        "key",
        transport=transport,
        sleep=slow_sleep,
        stream_timeout_seconds=0.001,
        max_retries=1,
    )

    with pytest.raises(ProviderError, match="interrupted"):
        _ = "".join([part async for part in client.stream_answer([])])
    assert transport.calls == 1
