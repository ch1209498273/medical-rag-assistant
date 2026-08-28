"""SiliconFlow Embedding, Reranker and multimodal OCR adapters."""

from __future__ import annotations

import base64
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.errors import ProviderError
from app.providers.http import ProviderHttpClient


@dataclass(frozen=True)
class RankedItem:
    """A reranked document with its original candidate index."""

    index: int
    document: str
    score: float


class SiliconFlowClient(ProviderHttpClient):
    """Cloud model adapter used by indexing and retrieval services."""

    MAX_EMBED_BATCH = 32

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.siliconflow.cn/v1",
        *,
        embedding_model: str = "BAAI/bge-m3",
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        ocr_model: str = "PaddlePaddle/PaddleOCR-VL-1.5",
        transport: Any = None,
        sleep: Any = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        super().__init__(
            "siliconflow",
            api_key,
            base_url,
            transport=transport,
            sleep=sleep,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.ocr_model = ocr_model

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts in bounded batches while restoring each batch's order."""

        values = list(texts)
        if not values:
            return []
        if any(not isinstance(text, str) for text in values):
            raise ValueError("embedding inputs must be strings")

        vectors: list[list[float]] = []
        for start in range(0, len(values), self.MAX_EMBED_BATCH):
            batch = values[start : start + self.MAX_EMBED_BATCH]
            payload = {
                "model": self.embedding_model,
                "input": batch,
                "encoding_format": "float",
            }
            response = await self.request_json("embed", "/embeddings", payload)
            vectors.extend(_parse_embeddings(response, len(batch)))
        return vectors

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int,
    ) -> list[RankedItem]:
        """Return strictly validated results mapped back to candidate text."""

        values = list(documents)
        if not isinstance(query, str) or any(
            not isinstance(item, str) for item in values
        ):
            raise ValueError("rerank query and documents must be strings")
        if not values:
            return []
        if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n <= 0:
            raise ValueError("top_n must be a positive integer")

        payload = {
            "model": self.reranker_model,
            "query": query,
            "documents": values,
            "top_n": min(top_n, len(values)),
            "return_documents": False,
        }
        response = await self.request_json("rerank", "/rerank", payload)
        return _parse_rerank(response, values, top_n)

    async def ocr_page(self, image_bytes: bytes) -> str:
        """Recognise one page only when the provider returns explicit text."""

        if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
            raise ValueError("image_bytes must be non-empty bytes")
        encoded = base64.b64encode(bytes(image_bytes)).decode("ascii")
        payload = {
            "model": self.ocr_model,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "请准确识别图片中的文字；无法确认的内容不要猜测。",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded}"},
                        },
                    ],
                }
            ],
        }
        response = await self.request_json("ocr_page", "/chat/completions", payload)
        content = _message_content(response)
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            pieces: list[str] = []
            for part in content:
                if not isinstance(part, Mapping) or part.get("type") != "text":
                    raise _invalid(
                        "ocr_page", "OCR response did not contain explicit text"
                    )
                text = part.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise _invalid(
                        "ocr_page", "OCR response did not contain explicit text"
                    )
                pieces.append(text)
            if pieces:
                return "".join(pieces).strip()
        raise _invalid("ocr_page", "OCR response did not contain explicit text")


def _invalid(operation: str, message: str) -> ProviderError:
    return ProviderError("siliconflow", operation, 200, False, message)


def _parse_embeddings(response: Any, expected: int) -> list[list[float]]:
    if not isinstance(response, Mapping):
        raise _invalid("embed", "embedding response has an invalid structure")
    data = response.get("data")
    if not isinstance(data, list) or len(data) != expected:
        raise _invalid("embed", "embedding response count did not match the request")

    by_index: dict[int, list[float]] = {}
    for item in data:
        if not isinstance(item, Mapping):
            raise _invalid("embed", "embedding response item is invalid")
        index = item.get("index")
        embedding = item.get("embedding")
        if not isinstance(index, int) or isinstance(index, bool):
            raise _invalid("embed", "embedding response index is invalid")
        if index < 0 or index >= expected or index in by_index:
            raise _invalid("embed", "embedding response indexes are invalid")
        if not isinstance(embedding, list) or not embedding:
            raise _invalid("embed", "embedding vector is invalid")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in embedding
        ):
            raise _invalid("embed", "embedding vector contains a non-finite value")
        by_index[index] = [float(value) for value in embedding]

    if set(by_index) != set(range(expected)):
        raise _invalid("embed", "embedding response indexes are incomplete")
    return [by_index[index] for index in range(expected)]


def _parse_rerank(response: Any, documents: list[str], top_n: int) -> list[RankedItem]:
    if not isinstance(response, Mapping):
        raise _invalid("rerank", "rerank response has an invalid structure")
    results = response.get("results")
    if not isinstance(results, list) or len(results) > min(top_n, len(documents)):
        raise _invalid("rerank", "rerank response count is invalid")

    ranked: list[RankedItem] = []
    seen: set[int] = set()
    for item in results:
        if not isinstance(item, Mapping):
            raise _invalid("rerank", "rerank result is invalid")
        index = item.get("index")
        score = item.get("relevance_score", item.get("score"))
        if not isinstance(index, int) or isinstance(index, bool):
            raise _invalid("rerank", "rerank result index is invalid")
        if index < 0 or index >= len(documents) or index in seen:
            raise _invalid("rerank", "rerank result index is invalid")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
        ):
            raise _invalid("rerank", "rerank score is invalid")
        seen.add(index)
        ranked.append(
            RankedItem(index=index, document=documents[index], score=float(score))
        )
    return ranked


def _message_content(response: Any) -> Any:
    if not isinstance(response, Mapping):
        raise _invalid("ocr_page", "OCR response has an invalid structure")
    choices = response.get("choices")
    if (
        not isinstance(choices, list)
        or not choices
        or not isinstance(choices[0], Mapping)
    ):
        raise _invalid("ocr_page", "OCR response has an invalid structure")
    message = choices[0].get("message")
    if not isinstance(message, Mapping) or "content" not in message:
        raise _invalid("ocr_page", "OCR response has an invalid structure")
    return message["content"]
