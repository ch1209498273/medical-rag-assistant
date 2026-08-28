"""Contract tests for DeepSeek streaming responses."""

from __future__ import annotations

from dataclasses import dataclass

import httpx
import pytest
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
    assert "upstream detail" not in str(raised.value)


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
