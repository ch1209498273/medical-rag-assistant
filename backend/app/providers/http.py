"""Small, dependency-injected HTTP policy shared by provider adapters.

The application owns the default ``httpx.AsyncClient``.  A caller-supplied
transport remains caller-owned and is never closed by these adapters; this is
important for tests and for applications sharing one connection pool.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self

import httpx

from app.errors import ProviderError

logger = logging.getLogger(__name__)

Sleep = Callable[[float], Awaitable[None] | None]


class _RequestFailure(Exception):
    def __init__(self, status_code: int | None, retryable: bool, message: str) -> None:
        self.status_code = status_code
        self.retryable = retryable
        self.message = message
        super().__init__(message)


@dataclass
class StreamHandle:
    """Normalised stream source plus its optional async-context cleanup."""

    source: Any
    context: Any = None

    async def iter_bytes(self) -> AsyncIterator[bytes]:
        source = self.source
        if isinstance(source, bytes):
            yield source
            return
        if isinstance(source, str):
            yield source.encode("utf-8")
            return
        aiter_bytes = getattr(source, "aiter_bytes", None)
        if callable(aiter_bytes):
            iterator = aiter_bytes()
            async for chunk in iterator:
                yield _as_bytes(chunk)
            return
        if hasattr(source, "__aiter__"):
            async for chunk in source:
                yield _as_bytes(chunk)
            return
        iter_bytes = getattr(source, "iter_bytes", None)
        if callable(iter_bytes):
            for chunk in iter_bytes():
                yield _as_bytes(chunk)
            return
        content = getattr(source, "content", None)
        if content is not None:
            yield _as_bytes(content)
            return
        raise TypeError("stream response has no byte iterator")

    async def close(self) -> None:
        if self.context is not None:
            await _maybe_await(self.context.__aexit__(None, None, None))
            self.context = None
            return
        close = getattr(self.source, "aclose", None)
        if callable(close):
            await _maybe_await(close())


class _NonClosingTransport(httpx.AsyncBaseTransport):
    """Let an adapter-owned client use, but never close, an external transport."""

    def __init__(self, delegate: httpx.AsyncBaseTransport) -> None:
        self._delegate = delegate

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._delegate.handle_async_request(request)

    async def aclose(self) -> None:
        # The caller owns the delegated transport and must close it explicitly.
        return None


def _as_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    raise TypeError("stream yielded a non-byte value")


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class ProviderHttpClient:
    """Retrying HTTP boundary with injected transport and clock."""

    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str,
        *,
        transport: Any = None,
        sleep: Sleep | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        max_response_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
            or timeout > 300
        ):
            raise ValueError("timeout must be between 0 and 300 seconds")
        if (
            isinstance(max_response_bytes, bool)
            or not isinstance(max_response_bytes, int)
            or max_response_bytes <= 0
            or max_response_bytes > 8 * 1024 * 1024
        ):
            raise ValueError("max_response_bytes exceeds the hard safety bound")
        self.provider = provider
        credential = api_key.strip()
        self.api_key = credential
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.max_response_bytes = max_response_bytes
        self._sleep = sleep or asyncio.sleep
        self._owns_transport = transport is None
        if transport is None:
            self._transport = httpx.AsyncClient(
                timeout=httpx.Timeout(timeout),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
        elif isinstance(transport, httpx.AsyncBaseTransport):
            self._transport = httpx.AsyncClient(
                transport=_NonClosingTransport(transport),
                timeout=httpx.Timeout(timeout),
            )
            self._owns_transport = True
        else:
            self._transport = transport

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _ensure_key(self, operation: str) -> None:
        if not self.api_key:
            raise ProviderError(
                self.provider,
                operation,
                None,
                False,
                "provider credentials are not configured",
            )

    async def aclose(self) -> None:
        """Close only a transport created by this adapter."""

        if not self._owns_transport:
            return
        close = getattr(self._transport, "aclose", None)
        if callable(close):
            await _maybe_await(close())
        else:
            close = getattr(self._transport, "close", None)
            if callable(close):
                await _maybe_await(close())

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def _wait_before_retry(self, attempt: int) -> None:
        delay = (0.5, 1.0, 2.0)[min(attempt, 2)]
        result = self._sleep(delay)
        if inspect.isawaitable(result):
            await result

    async def _call_request(self, method: str, url: str, payload: Mapping[str, Any]) -> Any:
        request = getattr(self._transport, "request", None)
        if request is None:
            request = getattr(self._transport, method.lower(), None)
        if request is None:
            raise _RequestFailure(None, False, "transport does not support HTTP requests")
        kwargs = {
            "headers": self._headers(),
            "json": dict(payload),
            "timeout": self.timeout,
        }
        try:
            try:
                result = request(method, url, **kwargs)
            except TypeError as error:
                # Minimal test transports often omit the optional timeout keyword.
                if "timeout" not in str(error):
                    raise
                kwargs.pop("timeout")
                result = request(method, url, **kwargs)
            return await _maybe_await(result)
        except _RequestFailure:
            raise
        except (httpx.TimeoutException, TimeoutError, ConnectionError, OSError) as error:
            raise _RequestFailure(
                None, True, "provider request timed out or was unavailable"
            ) from error
        except Exception as error:
            raise _RequestFailure(None, False, "provider transport failed") from error

    async def _request_once(self, operation: str, path: str, payload: Mapping[str, Any]) -> Any:
        try:
            response = await self._call_request("POST", self._url(path), payload)
        except _RequestFailure:
            raise
        except (httpx.TimeoutException, TimeoutError, ConnectionError, OSError) as error:
            raise _RequestFailure(None, True, "provider request timed out or was unavailable") from error

        status_code = int(getattr(response, "status_code", 200))
        if status_code >= 400:
            await _close_response(response)
            raise _RequestFailure(
                status_code,
                status_code == 429 or status_code >= 500,
                "provider returned an HTTP error",
            )
        try:
            data = await self._response_json(response)
        finally:
            await _close_response(response)
        return data

    async def _response_json(self, response: Any) -> Any:
        if isinstance(response, Mapping):
            return response
        content = getattr(response, "content", None)
        if isinstance(content, (bytes, bytearray)) and len(content) > self.max_response_bytes:
            raise _RequestFailure(200, False, "provider response exceeded the size limit")
        parser = getattr(response, "json", None)
        if not callable(parser):
            raise _RequestFailure(200, False, "provider response was not JSON")
        try:
            value = await _maybe_await(parser())
        except Exception as error:
            raise _RequestFailure(200, False, "provider response was not valid JSON") from error
        return value

    async def request_json(
        self,
        operation: str,
        path: str,
        payload: Mapping[str, Any],
    ) -> Any:
        """POST JSON with bounded retries and a safe public error."""

        self._ensure_key(operation)
        for attempt in range(self.max_retries + 1):
            try:
                return await self._request_once(operation, path, payload)
            except _RequestFailure as failure:
                if failure.retryable and attempt < self.max_retries:
                    logger.warning(
                        "provider request retry",
                        extra={
                            "provider": self.provider,
                            "operation": operation,
                            "status_code": failure.status_code,
                            "attempt": attempt + 1,
                        },
                    )
                    await self._wait_before_retry(attempt)
                    continue
                raise ProviderError(
                    self.provider,
                    operation,
                    failure.status_code,
                    failure.retryable,
                    failure.message,
                ) from None
            except ProviderError:
                raise
            except Exception:  # noqa: BLE001 - provider boundary must fail closed
                raise ProviderError(
                    self.provider,
                    operation,
                    None,
                    False,
                    "provider request failed",
                ) from None

    async def _open_stream_once(
        self,
        operation: str,
        path: str,
        payload: Mapping[str, Any],
    ) -> StreamHandle:
        stream = getattr(self._transport, "stream", None)
        if callable(stream):
            try:
                candidate = stream(
                    "POST",
                    self._url(path),
                    headers=self._headers(),
                    json=dict(payload),
                    timeout=self.timeout,
                )
            except TypeError as error:
                if "timeout" not in str(error):
                    raise
                candidate = stream(
                    "POST",
                    self._url(path),
                    headers=self._headers(),
                    json=dict(payload),
                )
            try:
                candidate = await _maybe_await(candidate)
                if hasattr(candidate, "__aenter__"):
                    context = candidate
                    response = await _maybe_await(candidate.__aenter__())
                else:
                    context = None
                    response = candidate
            except (httpx.TimeoutException, TimeoutError, ConnectionError, OSError) as error:
                raise _RequestFailure(None, True, "provider stream timed out or was unavailable") from error
        else:
            response = await self._call_request("POST", self._url(path), payload)
            context = None

        status_code = int(getattr(response, "status_code", 200))
        if status_code >= 400:
            await _close_response(response)
            if context is not None:
                await _maybe_await(context.__aexit__(None, None, None))
            raise _RequestFailure(
                status_code,
                status_code == 429 or status_code >= 500,
                "provider returned an HTTP error",
            )
        return StreamHandle(response, context)

    async def open_stream(
        self,
        operation: str,
        path: str,
        payload: Mapping[str, Any],
        *,
        retries: int | None = None,
    ) -> StreamHandle:
        """Open a streaming response, retrying only before it is consumed."""

        self._ensure_key(operation)
        retry_limit = self.max_retries if retries is None else max(0, retries)
        for attempt in range(retry_limit + 1):
            try:
                return await self._open_stream_once(operation, path, payload)
            except _RequestFailure as failure:
                if failure.retryable and attempt < retry_limit:
                    logger.warning(
                        "provider stream retry",
                        extra={
                            "provider": self.provider,
                            "operation": operation,
                            "status_code": failure.status_code,
                            "attempt": attempt + 1,
                        },
                    )
                    await self._wait_before_retry(attempt)
                    continue
                raise ProviderError(
                    self.provider,
                    operation,
                    failure.status_code,
                    failure.retryable,
                    failure.message,
                ) from None
            except ProviderError:
                raise
            except Exception:  # noqa: BLE001 - provider boundary must fail closed
                raise ProviderError(
                    self.provider,
                    operation,
                    None,
                    False,
                    "provider stream failed",
                ) from None


async def _close_response(response: Any) -> None:
    close = getattr(response, "aclose", None)
    if callable(close):
        await _maybe_await(close())
