"""Contract tests for the SiliconFlow adapters.

All responses in this module are fictional and are returned by an in-memory
transport.  The tests must never contact a cloud provider or spend tokens.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

import httpx
import pytest
from app.errors import ProviderError
from app.providers.siliconflow import RankedItem, SiliconFlowClient


@dataclass
class MockResponse:
    status_code: int
    payload: object

    def json(self) -> object:
        return self.payload


class MockTransport:
    """Small async transport with response queues and call recording."""

    def __init__(self, responses: list[MockResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []
        self.closed = False

    async def request(self, method: str, url: str, **kwargs: object) -> MockResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        if not self.responses:
            raise AssertionError("mock transport ran out of responses")
        return self.responses.pop(0)


class ExternalHttpxTransport(httpx.AsyncBaseTransport):
    """A caller-owned low-level transport used only for ownership regression."""

    def __init__(self) -> None:
        self.closed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]},
            request=request,
        )

    async def aclose(self) -> None:
        self.closed = True


class ExplodingTransport:
    async def request(self, method: str, url: str, **kwargs: object) -> object:
        raise ValueError(
            "Authorization: "
            + "Bearer "
            + "super-secret-key; context=full policy text"
        )


@pytest.mark.asyncio
async def test_embedding_preserves_input_order_and_payload() -> None:
    transport = MockTransport(
        [
            MockResponse(
                200,
                {
                    "data": [
                        {"index": 1, "embedding": [0.0, 1.0]},
                        {"index": 0, "embedding": [1.0, 0.0]},
                    ]
                },
            )
        ]
    )
    client = SiliconFlowClient("key", transport=transport)

    vectors = await client.embed(["甲", "乙"])

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert transport.calls[0]["url"] == "https://api.siliconflow.cn/v1/embeddings"
    assert transport.calls[0]["json"] == {
        "model": "BAAI/bge-m3",
        "input": ["甲", "乙"],
        "encoding_format": "float",
    }


@pytest.mark.asyncio
async def test_injected_httpx_transport_remains_caller_owned() -> None:
    transport = ExternalHttpxTransport()
    client = SiliconFlowClient("key", transport=transport)

    assert await client.embed(["甲"]) == [[1.0, 0.0]]
    await client.aclose()

    assert transport.closed is False
    await transport.aclose()


@pytest.mark.asyncio
async def test_embedding_batches_at_most_32_items_without_reordering() -> None:
    texts = [f"制度-{index}" for index in range(33)]
    transport = MockTransport(
        [
            MockResponse(
                200,
                {
                    "data": [
                        {"index": index, "embedding": [float(index)]}
                        for index in range(32)
                    ]
                },
            ),
            MockResponse(200, {"data": [{"index": 0, "embedding": [32.0]}]}),
        ]
    )
    client = SiliconFlowClient("key", transport=transport)

    vectors = await client.embed(texts)

    assert len(transport.calls) == 2
    assert len(transport.calls[0]["json"]["input"]) == 32  # type: ignore[index]
    assert len(transport.calls[1]["json"]["input"]) == 1  # type: ignore[index]
    assert [vector[0] for vector in vectors] == list(map(float, range(33)))


@pytest.mark.asyncio
async def test_rate_limit_becomes_retryable_provider_error_without_leaking_headers() -> (
    None
):
    transport = MockTransport([MockResponse(429, {}) for _ in range(4)])
    delays: list[float] = []
    client = SiliconFlowClient(
        "super-secret-key", transport=transport, sleep=delays.append
    )

    with pytest.raises(ProviderError) as raised:
        await client.embed(["甲"])

    error = raised.value
    assert error.retryable is True
    assert error.provider == "siliconflow"
    assert error.operation == "embed"
    assert error.status_code == 429
    assert "super-secret-key" not in str(error)
    assert delays == [0.5, 1.0, 2.0]
    assert len(transport.calls) == 4


@pytest.mark.asyncio
async def test_unauthorized_response_is_not_retried() -> None:
    transport = MockTransport([MockResponse(401, {})])
    client = SiliconFlowClient("key", transport=transport)

    with pytest.raises(ProviderError) as raised:
        await client.embed(["甲"])

    assert raised.value.retryable is False
    assert len(transport.calls) == 1


@pytest.mark.asyncio
async def test_unexpected_transport_exception_becomes_safe_provider_error() -> None:
    client = SiliconFlowClient("key", transport=ExplodingTransport())

    with pytest.raises(ProviderError) as raised:
        await client.embed(["甲"])

    error = raised.value
    assert error.provider == "siliconflow"
    assert error.operation == "embed"
    assert error.status_code is None
    assert error.retryable is False
    assert "super-secret-key" not in str(error)
    assert "full policy text" not in str(error)


@pytest.mark.asyncio
async def test_rerank_maps_indexes_and_rejects_duplicate_indexes() -> None:
    transport = MockTransport(
        [
            MockResponse(
                200,
                {
                    "results": [
                        {"index": 1, "relevance_score": 0.2},
                        {"index": 0, "relevance_score": 0.9},
                    ]
                },
            )
        ]
    )
    client = SiliconFlowClient("key", transport=transport)

    ranked = await client.rerank("问题", ["甲", "乙"], top_n=2)

    assert ranked == [
        RankedItem(index=1, document="乙", score=0.2),
        RankedItem(index=0, document="甲", score=0.9),
    ]

    duplicate_transport = MockTransport(
        [
            MockResponse(
                200,
                {
                    "results": [
                        {"index": 0, "relevance_score": 0.2},
                        {"index": 0, "relevance_score": 0.1},
                    ]
                },
            )
        ]
    )
    with pytest.raises(ProviderError, match="index"):
        await SiliconFlowClient("key", transport=duplicate_transport).rerank(
            "问题", ["甲", "乙"], top_n=2
        )


@pytest.mark.asyncio
async def test_ocr_sends_base64_multimodal_payload_and_accepts_explicit_text() -> None:
    transport = MockTransport(
        [MockResponse(200, {"choices": [{"message": {"content": "OCR制度文本"}}]})]
    )
    client = SiliconFlowClient("key", transport=transport)

    result = await client.ocr_page(b"fictional-png")

    assert result == "OCR制度文本"
    payload = transport.calls[0]["json"]
    image_part = payload["messages"][0]["content"][1]  # type: ignore[index]
    encoded = image_part["image_url"]["url"].split(",", 1)[1]  # type: ignore[index]
    assert base64.b64decode(encoded) == b"fictional-png"
    assert payload["model"] == "PaddlePaddle/PaddleOCR-VL-1.5"  # type: ignore[index]


@pytest.mark.asyncio
async def test_ocr_empty_or_ambiguous_content_fails_closed() -> None:
    empty = MockTransport(
        [MockResponse(200, {"choices": [{"message": {"content": "  "}}]})]
    )
    with pytest.raises(ProviderError, match="OCR"):
        await SiliconFlowClient("key", transport=empty).ocr_page(b"image")

    ambiguous = MockTransport(
        [
            MockResponse(
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "content": [{"type": "tool_call", "value": "guess"}]
                            }
                        }
                    ]
                },
            )
        ]
    )
    with pytest.raises(ProviderError, match="OCR"):
        await SiliconFlowClient("key", transport=ambiguous).ocr_page(b"image")
