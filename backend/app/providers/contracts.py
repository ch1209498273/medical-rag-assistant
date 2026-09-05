"""Provider protocols shared by cloud adapters and the offline demo."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Protocol

from app.providers.siliconflow import RankedItem


class KnowledgeProvider(Protocol):
    """Embedding, reranking, and OCR operations used by ingestion and RAG."""

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int,
    ) -> list[RankedItem]: ...

    async def ocr_page(self, image_bytes: bytes) -> str: ...

    async def aclose(self) -> None: ...


class LanguageModelProvider(Protocol):
    """Structured answer and JSON completion operations used by chat."""

    def stream_answer(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        json_output: bool = False,
        max_tokens: int | None = None,
        thinking: Mapping[str, str] | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]: ...

    async def complete_json(self, messages: Sequence[Mapping[str, Any]]) -> Any: ...

    async def aclose(self) -> None: ...


__all__ = ["KnowledgeProvider", "LanguageModelProvider"]
