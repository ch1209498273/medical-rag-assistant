"""DeepSeek official OpenAI-compatible streaming adapter."""

from __future__ import annotations

import asyncio
import codecs
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import httpx

from app.errors import ProviderError
from app.providers.http import ProviderHttpClient

MAX_STREAM_BYTES_HARD = 8 * 1024 * 1024
MAX_STREAM_CHARS_HARD = 1_000_000
MAX_STREAM_EVENTS_HARD = 10_000
MAX_STREAM_TIMEOUT_HARD = 300.0


def _stream_interrupted(error: BaseException) -> ProviderError:
    return ProviderError(
        "deepseek",
        "stream_answer",
        None,
        True,
        "DeepSeek stream was interrupted",
    )


class DeepSeekClient(ProviderHttpClient):
    """Stream grounded answers without exposing provider response details."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        *,
        model: str = "deepseek-v4-flash",
        max_tokens: int = 2048,
        transport: Any = None,
        sleep: Any = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        max_stream_bytes: int = 1 * 1024 * 1024,
        max_stream_chars: int = 200_000,
        max_stream_events: int = 2_048,
        stream_timeout_seconds: float = 60.0,
    ) -> None:
        if not 1 <= max_stream_bytes <= MAX_STREAM_BYTES_HARD:
            raise ValueError("max_stream_bytes exceeds the hard safety bound")
        if not 1 <= max_stream_chars <= MAX_STREAM_CHARS_HARD:
            raise ValueError("max_stream_chars exceeds the hard safety bound")
        if not 1 <= max_stream_events <= MAX_STREAM_EVENTS_HARD:
            raise ValueError("max_stream_events exceeds the hard safety bound")
        if not 0 < stream_timeout_seconds <= MAX_STREAM_TIMEOUT_HARD:
            raise ValueError("stream_timeout_seconds exceeds the hard safety bound")
        super().__init__(
            "deepseek",
            api_key,
            base_url,
            transport=transport,
            sleep=sleep,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.model = model
        self.max_tokens = max_tokens
        self.max_stream_bytes = max_stream_bytes
        self.max_stream_chars = max_stream_chars
        self.max_stream_events = max_stream_events
        self.stream_timeout_seconds = stream_timeout_seconds

    async def stream_answer(
        self, messages: Sequence[Mapping[str, Any]], *, json_output: bool = False
    ) -> AsyncIterator[str]:
        """Yield only ``delta.content`` from complete SSE events.

        ``json_output`` requests the provider's JSON-object mode for callers
        that enforce a structured response after streaming.
        """

        payload = self._payload(messages, stream=True)
        if json_output:
            payload["response_format"] = {"type": "json_object"}
        yielded_content = False
        deadline = asyncio.get_running_loop().time() + self.stream_timeout_seconds
        for attempt in range(self.max_retries + 1):
            handle = None
            completed = False
            try:
                try:
                    # Stream consumption owns the retry budget from this point;
                    # a failed HTTP status is retried in the same loop.
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise TimeoutError
                    handle = await asyncio.wait_for(
                        self.open_stream(
                            "stream_answer", "/chat/completions", payload, retries=0
                        ),
                        timeout=remaining,
                    )
                except ProviderError as error:
                    if (
                        handle is None
                        and error.retryable
                        and not yielded_content
                        and attempt < self.max_retries
                    ):
                        remaining = deadline - asyncio.get_running_loop().time()
                        if remaining <= 0:
                            raise _stream_interrupted(error)
                        try:
                            await asyncio.wait_for(
                                self._wait_before_retry(attempt), timeout=remaining
                            )
                        except TimeoutError as timeout_error:
                            raise _stream_interrupted(timeout_error) from timeout_error
                        continue
                    raise

                decoder = codecs.getincrementaldecoder("utf-8")()
                buffer = ""
                done = False
                stream_bytes = 0
                stream_events = 0
                stream_chars = 0
                iterator = handle.iter_bytes().__aiter__()
                while True:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise TimeoutError
                    try:
                        raw = await asyncio.wait_for(iterator.__anext__(), remaining)
                    except StopAsyncIteration:
                        break
                    stream_bytes += len(raw)
                    if stream_bytes > self.max_stream_bytes:
                        raise ProviderError(
                            "deepseek",
                            "stream_answer",
                            200,
                            False,
                            "DeepSeek stream exceeded the byte limit",
                        )
                    if stream_events > self.max_stream_events:
                        raise ProviderError(
                            "deepseek",
                            "stream_answer",
                            200,
                            False,
                            "DeepSeek stream exceeded the event limit",
                        )
                    buffer += decoder.decode(raw)
                    complete_lines = buffer.splitlines(keepends=True)
                    if complete_lines and not complete_lines[-1].endswith(("\n", "\r")):
                        buffer = complete_lines.pop()
                    else:
                        buffer = ""
                    for line in complete_lines:
                        if line.lstrip().startswith("data:"):
                            stream_events += 1
                            if stream_events > self.max_stream_events:
                                raise ProviderError(
                                    "deepseek",
                                    "stream_answer",
                                    200,
                                    False,
                                    "DeepSeek stream exceeded the event limit",
                                )
                        content, line_done = _parse_sse_line(line)
                        if line_done:
                            done = True
                        if content is not None:
                            stream_chars += len(content)
                            if stream_chars > self.max_stream_chars:
                                raise ProviderError(
                                    "deepseek",
                                    "stream_answer",
                                    200,
                                    False,
                                    "DeepSeek stream exceeded the character limit",
                                )
                            yielded_content = True
                            yield content
                        if done:
                            break
                    if done:
                        break

                buffer += decoder.decode(b"", final=True)
                if buffer.strip() and not done:
                    if buffer.lstrip().startswith("data:"):
                        stream_events += 1
                        if stream_events > self.max_stream_events:
                            raise ProviderError(
                                "deepseek",
                                "stream_answer",
                                200,
                                False,
                                "DeepSeek stream exceeded the event limit",
                            )
                    content, line_done = _parse_sse_line(buffer)
                    done = line_done
                    if content is not None:
                        stream_chars += len(content)
                        if stream_chars > self.max_stream_chars:
                            raise ProviderError(
                                "deepseek",
                                "stream_answer",
                                200,
                                False,
                                "DeepSeek stream exceeded the character limit",
                            )
                        yielded_content = True
                        yield content
                if not done:
                    raise ProviderError(
                        "deepseek",
                        "stream_answer",
                        200,
                        True,
                        "DeepSeek stream ended before complete event",
                    )
                completed = True
                return
            except TimeoutError as error:
                raise ProviderError(
                    "deepseek",
                    "stream_answer",
                    None,
                    True,
                    "DeepSeek stream was interrupted",
                ) from error
            except ProviderError:
                raise
            except httpx.TimeoutException as error:
                if not yielded_content and attempt < self.max_retries:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise _stream_interrupted(error)
                    try:
                        await asyncio.wait_for(
                            self._wait_before_retry(attempt), timeout=remaining
                        )
                    except TimeoutError as timeout_error:
                        raise _stream_interrupted(timeout_error) from timeout_error
                    continue
                raise ProviderError(
                    "deepseek",
                    "stream_answer",
                    None,
                    True,
                    "DeepSeek stream was interrupted",
                ) from error
            except (ConnectionError, OSError) as error:
                if not yielded_content and attempt < self.max_retries:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise _stream_interrupted(error)
                    try:
                        await asyncio.wait_for(
                            self._wait_before_retry(attempt), timeout=remaining
                        )
                    except TimeoutError as timeout_error:
                        raise _stream_interrupted(timeout_error) from timeout_error
                    continue
                raise ProviderError(
                    "deepseek",
                    "stream_answer",
                    None,
                    True,
                    "DeepSeek stream was interrupted",
                ) from error
            except Exception as error:
                raise ProviderError(
                    "deepseek",
                    "stream_answer",
                    200,
                    False,
                    "DeepSeek stream response was invalid",
                ) from error
            finally:
                if handle is not None:
                    try:
                        await handle.close()
                    except Exception as error:
                        if completed:
                            raise ProviderError(
                                "deepseek",
                                "stream_answer",
                                None,
                                False,
                                "DeepSeek stream cleanup failed",
                            ) from error

    async def complete(self, messages: Sequence[Mapping[str, Any]]) -> str:
        """Obtain one non-streaming assistant message for evaluation fallback."""

        return await self._complete_content(self._payload(messages, stream=False))

    async def complete_json(self, messages: Sequence[Mapping[str, Any]]) -> Any:
        """Parse a structured non-streaming response without accepting prose."""

        payload = self._payload(messages, stream=False)
        payload["response_format"] = {"type": "json_object"}
        content = await self._complete_content(payload)
        try:
            return json.loads(content)
        except (TypeError, ValueError) as error:
            raise ProviderError(
                "deepseek",
                "complete",
                200,
                False,
                "DeepSeek response was not valid structured JSON",
            ) from error

    async def _complete_content(self, payload: Mapping[str, Any]) -> str:
        response = await self.request_json("complete", "/chat/completions", payload)
        content = _extract_content(response)
        if not isinstance(content, str) or not content.strip():
            raise ProviderError(
                "deepseek",
                "complete",
                200,
                False,
                "DeepSeek response did not contain explicit text",
            )
        return content.strip()

    def _payload(self, messages: Sequence[Mapping[str, Any]], *, stream: bool) -> dict[str, Any]:
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            raise TypeError("messages must be a sequence")
        normalised: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, Mapping):
                raise TypeError("each message must be a mapping")
            normalised.append(dict(message))
        return {
            "model": self.model,
            "messages": normalised,
            "stream": stream,
            "max_tokens": self.max_tokens,
        }


def _parse_sse_line(line: str) -> tuple[str | None, bool]:
    stripped = line.strip("\r\n")
    if not stripped or stripped.startswith(":"):
        return None, False
    if not stripped.startswith("data:"):
        return None, False
    data = stripped[5:].lstrip()
    if data == "[DONE]":
        return None, True
    try:
        event = json.loads(data)
    except (TypeError, ValueError) as error:
        raise ProviderError(
            "deepseek",
            "stream_answer",
            200,
            False,
            "DeepSeek stream contained invalid JSON",
        ) from error
    if not isinstance(event, Mapping):
        raise ProviderError(
            "deepseek",
            "stream_answer",
            200,
            False,
            "DeepSeek stream event was invalid",
        )
    if "error" in event:
        raise ProviderError(
            "deepseek",
            "stream_answer",
            200,
            False,
            "DeepSeek stream returned an upstream error",
        )
    choices = event.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderError(
            "deepseek",
            "stream_answer",
            200,
            False,
            "DeepSeek stream event has no choices",
        )
    first = choices[0]
    if not isinstance(first, Mapping):
        raise ProviderError(
            "deepseek",
            "stream_answer",
            200,
            False,
            "DeepSeek stream choice was invalid",
        )
    delta = first.get("delta")
    if not isinstance(delta, Mapping):
        raise ProviderError(
            "deepseek",
            "stream_answer",
            200,
            False,
            "DeepSeek stream delta was invalid",
        )
    content = delta.get("content")
    if content is None:
        return None, False
    if not isinstance(content, str):
        raise ProviderError(
            "deepseek",
            "stream_answer",
            200,
            False,
            "DeepSeek stream content was not text",
        )
    return content, False


def _extract_content(response: Any) -> Any:
    if not isinstance(response, Mapping):
        raise ProviderError("deepseek", "complete", 200, False, "DeepSeek response was invalid")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise ProviderError("deepseek", "complete", 200, False, "DeepSeek response was invalid")
    message = choices[0].get("message")
    if not isinstance(message, Mapping) or "content" not in message:
        raise ProviderError("deepseek", "complete", 200, False, "DeepSeek response was invalid")
    return message["content"]
