from __future__ import annotations

import pytest
from app.main import DocumentRuntime, RagRuntime


@pytest.mark.asyncio
async def test_document_runtime_closes_resources_in_order_and_only_once():
    calls: list[str] = []

    class Ingestion:
        async def aclose(self):
            calls.append("ingestion")

    class Provider:
        async def aclose(self):
            calls.append("provider")

    class VectorStore:
        def close(self):
            calls.append("vector")

    class Repository:
        def close(self):
            calls.append("repository")

    runtime = DocumentRuntime(Repository(), VectorStore(), Provider(), Ingestion())

    await runtime.close()
    await runtime.close()

    assert calls == ["ingestion", "provider", "vector", "repository"]
    assert runtime.closed is True


@pytest.mark.asyncio
async def test_document_runtime_continues_cleanup_after_close_error():
    calls: list[str] = []

    class Ingestion:
        async def aclose(self):
            calls.append("ingestion")
            raise RuntimeError("fictional ingestion close failure")

    class Provider:
        async def aclose(self):
            calls.append("provider")
            raise RuntimeError("fictional provider close failure")

    class VectorStore:
        def close(self):
            calls.append("vector")

    class Repository:
        def close(self):
            calls.append("repository")

    runtime = DocumentRuntime(Repository(), VectorStore(), Provider(), Ingestion())

    with pytest.raises(RuntimeError, match="ingestion close failure"):
        await runtime.close()

    assert calls == ["ingestion", "provider", "vector", "repository"]
    assert runtime.closed is True


@pytest.mark.asyncio
async def test_rag_runtime_closes_model_before_document_runtime_and_only_once():
    calls: list[str] = []

    class DeepSeek:
        async def aclose(self):
            calls.append("deepseek")
            raise RuntimeError("fictional deepseek close failure")

    class Documents:
        async def close(self):
            calls.append("documents")

    runtime = RagRuntime(
        Documents(),
        DeepSeek(),
        object(),
        object(),
        object(),
    )

    with pytest.raises(RuntimeError, match="deepseek close failure"):
        await runtime.close()
    await runtime.close()

    assert calls == ["deepseek", "documents"]
    assert runtime.closed is True
