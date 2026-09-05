from __future__ import annotations

from typing import ClassVar

import app.main as main_module
import pytest
from app.settings import Settings
from pydantic import ValidationError


class _DeepSeek:
    async def aclose(self) -> None:
        return None


class _VectorStore:
    def snapshot_active_chunks(self, active_versions):
        raise AssertionError("vector C1 runtime must not read lexical snapshots")


class _DocumentRuntime:
    repository = object()
    vector_store = _VectorStore()
    provider = object()

    async def close(self) -> None:
        return None


class _Retrieval:
    kwargs: ClassVar[dict[str, object]] = {}

    def __init__(self, **kwargs):
        type(self).kwargs = kwargs


class _LexicalIndex:
    instance: ClassVar[_LexicalIndex | None] = None
    provider: ClassVar[object | None] = None

    def __init__(self, snapshot_provider):
        type(self).instance = self
        type(self).provider = snapshot_provider


@pytest.mark.asyncio
async def test_default_vector_runtime_does_not_read_active_lexical_snapshot(
    monkeypatch,
):
    deepseek = _DeepSeek()
    monkeypatch.setattr(main_module, "DemoLanguageModel", lambda scenarios: deepseek)
    monkeypatch.setattr(main_module, "load_demo_scenarios", lambda path: ())
    monkeypatch.setattr(main_module, "RetrievalService", _Retrieval)

    settings = Settings(_env_file=None, app_runtime_mode="demo")

    assert settings.retrieval_strategy == "vector"
    runtime = await main_module.build_rag_runtime_async(settings, _DocumentRuntime())

    assert _Retrieval.kwargs["retrieval_strategy"] == "vector"
    assert _Retrieval.kwargs["lexical_index"] is None
    await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["hybrid", "hybrid_normalized"])
async def test_hybrid_runtime_injects_qdrant_snapshot_provider_lazily(
    monkeypatch, strategy
):
    deepseek = _DeepSeek()
    monkeypatch.setattr(main_module, "DemoLanguageModel", lambda scenarios: deepseek)
    monkeypatch.setattr(main_module, "load_demo_scenarios", lambda path: ())
    monkeypatch.setattr(main_module, "RetrievalService", _Retrieval)
    monkeypatch.setattr(main_module, "ActiveLexicalIndex", _LexicalIndex, raising=False)

    settings = Settings(
        _env_file=None,
        app_runtime_mode="demo",
        retrieval_strategy=strategy,
    )

    runtime = await main_module.build_rag_runtime_async(settings, _DocumentRuntime())

    assert _LexicalIndex.provider is _DocumentRuntime.vector_store
    assert _Retrieval.kwargs["retrieval_strategy"] == strategy
    assert _Retrieval.kwargs["lexical_index"] is _LexicalIndex.instance
    await runtime.close()


def test_retrieval_strategy_rejects_values_outside_experiment_allowlist():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, retrieval_strategy="lexical")
