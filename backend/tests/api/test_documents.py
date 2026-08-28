"""API contract tests for configured-directory document management."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from app.main import create_app
from app.settings import Settings
from fastapi.testclient import TestClient


@dataclass
class FakeDocumentApiService:
    scanned: int = 0
    reindexed: list[int] = field(default_factory=list)

    async def scan(self):
        self.scanned += 1
        return {
            "results": [
                {
                    "document_id": 1,
                    "file_name": "制度.docx",
                    "status": "active",
                    "version_id": "version-1",
                    "failure_reason": None,
                }
            ],
            "deactivated": [],
        }

    def list_documents(self):
        return [
            {
                "document_id": 1,
                "file_name": "制度.docx",
                "status": "active",
                "version_id": "version-1",
                "failure_reason": None,
            }
        ]

    async def reindex(self, document_id: int):
        self.reindexed.append(document_id)
        return {
            "document_id": document_id,
            "file_name": "制度.docx",
            "status": "active",
            "version_id": "version-2",
            "failure_reason": None,
        }


def _client(tmp_path: Path, service: FakeDocumentApiService) -> TestClient:
    settings = Settings(
        source_documents_dir=tmp_path,
        sqlite_path=tmp_path / "metadata.sqlite3",
        qdrant_path=tmp_path / "qdrant",
    )
    return TestClient(create_app(settings, document_service=service))


def _cloud_settings(tmp_path: Path) -> Settings:
    provider_settings = {
        "deepseek_api_key": "fixture",
        "siliconflow_api_key": "fixture",
    }
    return Settings(
        _env_file=None,
        source_documents_dir=tmp_path,
        private_data_dir=tmp_path / "private",
        sqlite_path=tmp_path / "metadata.sqlite3",
        qdrant_path=tmp_path / "qdrant",
        **provider_settings,
    )


def test_document_list_returns_status_metadata_without_private_path_or正文(tmp_path):
    service = FakeDocumentApiService()
    response = _client(tmp_path, service).get("/api/documents")

    assert response.status_code == 200
    payload = response.json()
    assert payload["documents"][0]["file_name"] == "制度.docx"
    assert str(tmp_path) not in response.text
    assert "text" not in payload["documents"][0]
    assert "正文" not in response.text


def test_document_api_replaces_untrusted_failure_reason_with_safe_code(tmp_path):
    service = FakeDocumentApiService()
    service.scan = lambda: {
        "results": [
            {
                "document_id": 1,
                "file_name": "制度.docx",
                "status": "failed",
                "version_id": "version-1",
                "failure_reason": f"permission denied: {tmp_path}\\企业正文",
            }
        ],
        "deactivated": [],
    }

    response = _client(tmp_path, service).post("/api/documents/scan")

    assert response.status_code == 200
    failure_reason = response.json()["results"][0]["failure_reason"]
    assert failure_reason == "INDEX_FAILED"
    assert str(tmp_path) not in response.text
    assert "企业正文" not in response.text


def test_document_api_fails_closed_for_tampered_metadata_and_timestamps(tmp_path):
    service = FakeDocumentApiService()
    service.list_documents = lambda: [
        {
            "document_id": "not-an-integer",
            "file_name": "C:" + chr(92) + "private" + chr(92) + "provider-stack.docx",
            "status": "provider-stack",
            "version_id": r"..\secret-version",
            "updated_at": "Traceback from provider",
            "failure_reason": "C:" + chr(92) + "private" + chr(92) + "raw-error",
        }
    ]

    response = _client(tmp_path, service).get("/api/documents")

    assert response.status_code == 200
    row = response.json()["documents"][0]
    assert row["status"] == "failed"
    assert row["failure_reason"] == "INDEX_FAILED"
    assert row["file_name"] == "未命名资料.docx"
    assert row["version_id"] is None
    assert row["updated_at"] is None
    assert "provider-stack" not in response.text
    assert "Traceback" not in response.text
    assert "private" not in response.text


def test_scan_uses_configured_directory_and_returns_index_status(tmp_path):
    service = FakeDocumentApiService()
    response = _client(tmp_path, service).post("/api/documents/scan")

    assert response.status_code == 200
    assert service.scanned == 1
    assert response.json()["results"][0]["status"] == "active"


def test_scan_preserves_skipped_status_for_unchanged_active_document(tmp_path):
    service = FakeDocumentApiService()
    service.scan = lambda: {
        "results": [
            {
                "document_id": 1,
                "file_name": "制度.docx",
                "status": "skipped",
                "version_id": "version-1",
                "failure_reason": None,
            }
        ],
        "deactivated": [],
    }

    response = _client(tmp_path, service).post("/api/documents/scan")

    assert response.status_code == 200
    assert response.json()["results"][0]["status"] == "skipped"
    assert response.json()["results"][0]["failure_reason"] is None


def test_reindex_uses_opaque_document_id_and_never_accepts_a_local_path(tmp_path):
    service = FakeDocumentApiService()
    client = _client(tmp_path, service)

    response = client.post("/api/documents/1/reindex")

    assert response.status_code == 200
    assert service.reindexed == [1]
    assert response.json()["version_id"] == "version-2"


def test_reindex_path_traversal_is_not_a_valid_document_identifier(tmp_path):
    service = FakeDocumentApiService()
    client = _client(tmp_path, service)

    response = client.post("/api/documents/..%2Fsecret/reindex")

    assert response.status_code in {404, 422}
    assert service.reindexed == []


def test_cloud_app_fails_closed_before_startup_without_exposing_key(tmp_path):
    canary = "configured-deepseek-test-key"
    settings_kwargs = {"deepseek_api_key": canary}
    settings = Settings(
        _env_file=None,
        source_documents_dir=tmp_path,
        sqlite_path=tmp_path / "metadata.sqlite3",
        qdrant_path=tmp_path / "qdrant",
        siliconflow_api_key="",
        **settings_kwargs,
    )

    with pytest.raises(
        RuntimeError, match="required cloud providers are not configured"
    ) as error, TestClient(create_app(settings)):
        pass

    assert canary not in str(error.value)
    assert "api_key" not in str(error.value).lower()


class NoNetworkTransport:
    async def request(self, method: str, url: str, **kwargs):
        raise AssertionError("composition smoke test must not call a provider")


def test_composed_runtime_closes_sqlite_qdrant_and_provider_resources(tmp_path):
    from app.main import build_document_runtime

    settings = _cloud_settings(tmp_path)
    runtime = build_document_runtime(settings, transport=NoNetworkTransport())

    with TestClient(create_app(settings, document_runtime=runtime)) as client:
        assert client.get("/api/documents").status_code == 200

    assert runtime.closed is True


def test_document_runtime_builder_closes_partial_resources_on_each_failure(
    tmp_path, monkeypatch
):
    import app.main as main_module

    closed: list[str] = []

    class Repository:
        def close(self):
            closed.append("sqlite")

    class VectorStore:
        def close(self):
            closed.append("qdrant")

    class Provider:
        def close(self):
            closed.append("siliconflow")

    class Service:
        pass

    monkeypatch.setattr(main_module, "SqliteDocumentRepository", lambda path: Repository())

    def fail_qdrant(path):
        raise RuntimeError("fictional qdrant constructor failure")

    monkeypatch.setattr(main_module, "QdrantLocalVectorStore", fail_qdrant)
    settings = _cloud_settings(tmp_path)

    with pytest.raises(RuntimeError, match="qdrant constructor"):
        main_module.build_document_runtime(settings)
    assert closed == ["sqlite"]

    closed.clear()
    monkeypatch.setattr(main_module, "QdrantLocalVectorStore", lambda path: VectorStore())
    monkeypatch.setattr(main_module, "SiliconFlowClient", lambda *args, **kwargs: Provider())
    monkeypatch.setattr(
        main_module,
        "IngestionService",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("fictional service failure")),
    )
    with pytest.raises(RuntimeError, match="service failure"):
        main_module.build_document_runtime(settings)
    assert closed == ["siliconflow", "qdrant", "sqlite"]


def test_rag_runtime_builder_closes_deepseek_when_later_assembly_fails(
    tmp_path, monkeypatch
):
    import app.main as main_module

    closed: list[str] = []

    class DeepSeek:
        def close(self):
            closed.append("deepseek")

    class Retrieval:
        def __init__(self, **kwargs):
            raise RuntimeError("fictional retrieval constructor failure")

    class DocumentRuntime:
        repository = object()
        vector_store = object()
        provider = object()

    document_runtime = DocumentRuntime()
    monkeypatch.setattr(main_module, "DeepSeekClient", lambda *args, **kwargs: DeepSeek())
    monkeypatch.setattr(main_module, "RetrievalService", Retrieval)
    settings = _cloud_settings(tmp_path)

    with pytest.raises(RuntimeError, match="retrieval constructor"):
        main_module.build_rag_runtime(settings, document_runtime)
    assert closed == ["deepseek"]


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_rag_runtime_feature_flag_controls_optional_claim_verifier(
    tmp_path, monkeypatch, enabled: bool
):
    import app.main as main_module
    from app.rag.deepseek_verifier import DeepSeekClaimVerifier

    class DeepSeek:
        async def aclose(self):
            return None

    class Retrieval:
        def __init__(self, **kwargs):
            pass

    class DocumentRuntime:
        repository = object()
        vector_store = object()
        provider = object()

        async def close(self):
            return None

    deepseek = DeepSeek()
    monkeypatch.setattr(main_module, "DeepSeekClient", lambda *args, **kwargs: deepseek)
    monkeypatch.setattr(main_module, "RetrievalService", Retrieval)
    settings_kwargs = {
        "deepseek_api_key": "fixture",
        "siliconflow_api_key": "fixture",
    }
    settings = Settings(
        _env_file=None,
        source_documents_dir=tmp_path,
        sqlite_path=tmp_path / "metadata.sqlite3",
        qdrant_path=tmp_path / "qdrant",
        stage_b_verification_enabled=enabled,
        **settings_kwargs,
    )

    runtime = await main_module.build_rag_runtime_async(settings, DocumentRuntime())

    if enabled:
        assert isinstance(runtime.answerer.verifier, DeepSeekClaimVerifier)
        assert runtime.answerer.verifier.deepseek is deepseek
    else:
        assert runtime.answerer.verifier is None
    await runtime.close()


def test_document_runtime_close_marks_each_resource_before_close_and_never_retries():
    import asyncio

    from app.main import DocumentRuntime

    calls: list[str] = []

    class Provider:
        async def aclose(self):
            calls.append("provider")

    class Vector:
        def close(self):
            calls.append("vector")

    class Repository:
        def close(self):
            calls.append("repository")
            raise RuntimeError("fictional close failure")

    runtime = DocumentRuntime(Repository(), Vector(), Provider(), object())
    with pytest.raises(RuntimeError, match="close failure"):
        asyncio.run(runtime.close())
    asyncio.run(runtime.close())
    assert calls == ["provider", "vector", "repository"]
    assert runtime.closed is True


def test_document_runtime_drains_ingestion_before_closing_provider_and_storage():
    import asyncio

    from app.main import DocumentRuntime

    calls: list[str] = []

    class Ingestion:
        async def aclose(self):
            calls.append("ingestion")

    class Provider:
        async def aclose(self):
            calls.append("provider")

    class Vector:
        def close(self):
            calls.append("vector")

    class Repository:
        def close(self):
            calls.append("repository")

    runtime = DocumentRuntime(Repository(), Vector(), Provider(), Ingestion())

    asyncio.run(runtime.close())

    assert calls == ["ingestion", "provider", "vector", "repository"]


def test_sync_runtime_builder_rejects_running_loop_instead_of_detached_cleanup(
    tmp_path, monkeypatch
):
    import asyncio

    import app.main as main_module

    closed: list[str] = []

    class Repository:
        def close(self):
            closed.append("repository")

    class Vector:
        def close(self):
            closed.append("vector")

    class Provider:
        async def aclose(self):
            closed.append("provider")

    monkeypatch.setattr(main_module, "SqliteDocumentRepository", lambda path: Repository())
    monkeypatch.setattr(main_module, "QdrantLocalVectorStore", lambda path: Vector())
    monkeypatch.setattr(main_module, "SiliconFlowClient", lambda *args, **kwargs: Provider())
    monkeypatch.setattr(
        main_module,
        "IngestionService",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("fictional service failure")),
    )
    settings = _cloud_settings(tmp_path)

    async def run():
        with pytest.raises(RuntimeError, match="running event loop"):
            main_module.build_document_runtime(settings)
        assert closed == []

    asyncio.run(run())


@pytest.mark.asyncio
async def test_async_runtime_builder_awaits_cleanup_before_returning_failure(
    tmp_path, monkeypatch
):
    import app.main as main_module

    closed: list[str] = []

    class AsyncProvider:
        async def aclose(self):
            await asyncio.sleep(0)
            closed.append("provider")

    class Repository:
        def close(self):
            closed.append("repository")

    class Vector:
        def close(self):
            closed.append("vector")

    monkeypatch.setattr(main_module, "SqliteDocumentRepository", lambda path: Repository())
    monkeypatch.setattr(main_module, "QdrantLocalVectorStore", lambda path: Vector())
    monkeypatch.setattr(main_module, "SiliconFlowClient", lambda *args, **kwargs: AsyncProvider())
    monkeypatch.setattr(
        main_module,
        "IngestionService",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("fictional async service failure")),
    )
    settings = _cloud_settings(tmp_path)

    with pytest.raises(RuntimeError, match="async service failure"):
        await main_module.build_document_runtime_async(settings)
    assert closed == ["provider", "vector", "repository"]
