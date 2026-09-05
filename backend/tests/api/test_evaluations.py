from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from app.evaluation.service import (
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationReport,
    EvaluationSet,
    calculate_report,
)
from app.main import create_app
from app.providers.minimax import EvaluationGeneration
from app.settings import Settings
from fastapi.testclient import TestClient


def valid_case(source_id: str = "S1") -> EvaluationCase:
    return EvaluationCase(
        question="问题",
        reference_points=["要点"],
        source_ids=[source_id],
        type="fact",
    )


@dataclass
class FakeEvaluationService:
    evaluation_set: EvaluationSet
    report: EvaluationReport

    async def generate(self, count: int = 30):
        return self.evaluation_set

    async def run(self, set_id: str):
        return self.report

    def get(self, set_id: str):
        return self.evaluation_set if set_id == self.evaluation_set.set_id else None


def test_evaluation_routes_assemble_safe_json(tmp_path: Path):
    evaluation_set = EvaluationSet(
        set_id="set-1",
        cases=(
            EvaluationCase(
                question="问题",
                reference_points=["要点"],
                source_ids=["S1"],
                type="fact",
            ),
        ),
        provider="minimax",
        model_id="MiniMax-M2.7",
    )
    report = EvaluationReport(
        total=1,
        citation_accuracy=1.0,
        answer_availability_rate=1.0,
        runtime_citation_visibility_rate=1.0,
        results=(
            EvaluationCaseResult(
                case_id="1",
                question="问题",
                type="fact",
                answered=True,
                citation_valid=True,
                citation_source_ids=("S1",),
                answer_failure_code="ANSWER_CITATION_INVALID",
                generation_available=True,
                source_visibility=True,
                answer_failure_detail="SOURCE_MARKER",
            ),
        ),
    )
    service = FakeEvaluationService(evaluation_set, report)
    settings = Settings(
        _env_file=None,
        source_documents_dir=tmp_path,
        sqlite_path=tmp_path / "metadata.sqlite3",
        qdrant_path=tmp_path / "qdrant",
    )
    client = TestClient(
        create_app(
            settings,
            document_service=object(),
            rag_service=object(),
            evaluation_service=service,
        )
    )

    generated = client.post("/api/evaluations/generate", json={"count": 1})
    fetched = client.get("/api/evaluations/set-1")
    run = client.post("/api/evaluations/set-1/run")

    assert generated.status_code == 200
    assert fetched.status_code == 200
    assert run.status_code == 200
    assert generated.json()["set_id"] == "set-1"
    assert run.json()["total"] == 1
    assert run.json()["answer_availability_rate"] == 1.0
    assert run.json()["runtime_citation_visibility_rate"] == 1.0
    safe_result = run.json()["results"][0]
    assert safe_result["generation_available"] is True
    assert safe_result["source_visibility"] is True
    assert "answer_failure_code" not in safe_result
    assert "answer_failure_detail" not in safe_result
    assert "answer" not in safe_result
    assert "provider_response" not in safe_result


def test_evaluation_report_exposes_industry_aggregates_without_private_ids(
    tmp_path: Path,
):
    evaluation_set = EvaluationSet(
        set_id="set-industry",
        cases=(valid_case("private-chunk"),),
        provider="evaluation",
        model_id="fictional",
    )
    report = calculate_report(
        [
            EvaluationCaseResult(
                case_id="1",
                question="问题",
                type="fact",
                answered=True,
                expected_source_ids=("private-chunk",),
                candidate_source_ids=("private-chunk",),
                evidence_source_ids=("private-chunk",),
                resolved_citation_source_ids=("private-chunk",),
            )
        ]
    )
    service = FakeEvaluationService(evaluation_set, report)
    settings = Settings(
        _env_file=None,
        source_documents_dir=tmp_path,
        sqlite_path=tmp_path / "metadata.sqlite3",
        qdrant_path=tmp_path / "qdrant",
    )

    with TestClient(
        create_app(
            settings,
            document_service=object(),
            rag_service=object(),
            evaluation_service=service,
        )
    ) as client:
        payload = client.post("/api/evaluations/set-industry/run").json()

    assert payload["industry_metrics"]["protocol_version"] == "task12a-v1"
    assert "numerator" in payload["industry_metrics"]["generation"]["answer_availability"]
    assert "private-chunk" not in json.dumps(payload, ensure_ascii=False)


def test_evaluation_report_allowlist_exposes_only_boolean_observability_fields(
    tmp_path: Path,
):
    evaluation_set = EvaluationSet(
        set_id="set-allowlist",
        cases=(valid_case(),),
        provider="evaluation",
        model_id="fictional",
    )

    @dataclass
    class UnsafeResult:
        case_id: str = "1"
        question: str = "fictional question"
        type: str = "fact"
        answered: bool = True
        refused: bool = False
        no_answer: bool = False
        reference_points_covered: bool = False
        citation_valid: bool = True
        citation_source_ids: tuple[str, ...] = ("S1",)
        generation_available: bool = True
        source_visibility: bool = True
        answer: str = "raw answer must not cross API"
        provider_response: str = "raw provider payload must not cross API"

    class UnsafeReport:
        def to_dict(self):
            return {
                "total": 1,
                "results": [UnsafeResult().__dict__],
            }

    report = UnsafeReport()
    service = FakeEvaluationService(evaluation_set, report)
    settings = Settings(
        _env_file=None,
        source_documents_dir=tmp_path,
        sqlite_path=tmp_path / "metadata.sqlite3",
        qdrant_path=tmp_path / "qdrant",
    )

    with TestClient(
        create_app(
            settings,
            document_service=object(),
            rag_service=object(),
            evaluation_service=service,
        )
    ) as client:
        payload = client.post("/api/evaluations/set-allowlist/run").json()

    safe_result = payload["results"][0]
    assert safe_result["generation_available"] is True
    assert safe_result["source_visibility"] is True
    assert "answer" not in safe_result
    assert "provider_response" not in safe_result


def test_default_lifespan_assembles_evaluation_service_without_network(
    tmp_path: Path, monkeypatch
):
    import app.main as main_module

    class FakeRepository:
        def active_version_ids(self):
            return {"version-1"}

    class FakeVectorStore:
        def snapshot_active_chunks(self, active_versions):
            assert active_versions == {"version-1"}
            return {"S1": {"text": "虚构制度片段"}}

    class FakeDocumentRuntime:
        repository = FakeRepository()
        vector_store = FakeVectorStore()
        service = object()

        async def close(self):
            return None

    class FakeRagService:
        async def stream(self, question):
            if False:
                yield question

    class FakeRagRuntime:
        document_runtime = FakeDocumentRuntime()
        deepseek = object()
        service = FakeRagService()

        async def close(self):
            return None

    class FakeGenerator:
        async def generate(self, context):
            return EvaluationGeneration(
                cases=(valid_case("S1"),),
                provider="minimax",
                model_id="MiniMax-M2.7",
            )

    async def build_documents(settings):
        return FakeDocumentRuntime()

    async def build_rag(settings, document_runtime):
        return FakeRagRuntime()

    monkeypatch.setattr(main_module, "build_document_runtime_async", build_documents)
    monkeypatch.setattr(main_module, "build_rag_runtime_async", build_rag)
    monkeypatch.setattr(
        main_module,
        "create_evaluation_generator",
        lambda **kwargs: FakeGenerator(),
        raising=False,
    )
    settings = Settings(
        _env_file=None,
        source_documents_dir=tmp_path,
        private_data_dir=tmp_path / "private",
        sqlite_path=tmp_path / "metadata.sqlite3",
        qdrant_path=tmp_path / "qdrant",
    )

    with TestClient(create_app(settings)) as client:
        response = client.post("/api/evaluations/generate", json={"count": 1})

    assert response.status_code == 200
    assert response.json()["cases"][0]["source_ids"] == ["S1"]


def test_production_evaluation_runtime_uses_only_deepseek_for_candidate_generation(
    tmp_path: Path, monkeypatch
):
    import app.main as main_module

    formal_deepseek = object()
    captured: dict[str, object] = {}

    class FakeRepository:
        def active_version_ids(self):
            return {"version-1"}

    class FakeVectorStore:
        def snapshot_active_chunks(self, active_versions):
            assert active_versions == {"version-1"}
            return {"S1": {"text": "虚构制度片段"}}

    class FakeDocumentRuntime:
        repository = FakeRepository()
        vector_store = FakeVectorStore()
        service = object()

        async def close(self):
            return None

    class FakeRagService:
        async def stream(self, question):
            if False:
                yield question

    class FakeRagRuntime:
        document_runtime = FakeDocumentRuntime()
        deepseek = formal_deepseek
        service = FakeRagService()

        async def close(self):
            return None

    class FakeGenerator:
        generators = ()

    async def build_documents(settings):
        return FakeDocumentRuntime()

    async def build_rag(settings, document_runtime):
        return FakeRagRuntime()

    def capture_evaluation_generator(**kwargs):
        captured.update(kwargs)
        return FakeGenerator()

    monkeypatch.setattr(main_module, "build_document_runtime_async", build_documents)
    monkeypatch.setattr(main_module, "build_rag_runtime_async", build_rag)
    monkeypatch.setattr(
        main_module, "create_evaluation_generator", capture_evaluation_generator
    )
    settings_kwargs = {
        "deepseek_api_key": "fixture",
        "minimax_api_key": "fixture",
    }
    settings = Settings(
        _env_file=None,
        source_documents_dir=tmp_path,
        private_data_dir=tmp_path / "private",
        sqlite_path=tmp_path / "metadata.sqlite3",
        qdrant_path=tmp_path / "qdrant",
        **settings_kwargs,
    )

    with TestClient(create_app(settings)):
        pass

    assert captured["deepseek_client"] is None
    assert "minimax_api_key" not in captured


def test_direct_evaluation_runtime_ignores_injected_chat_client(
    tmp_path: Path, monkeypatch
):
    import app.main as main_module

    captured: dict[str, object] = {}
    injected_chat_client = object()

    class FakeRepository:
        def active_version_ids(self):
            return {"version-1"}

    class FakeVectorStore:
        def snapshot_active_chunks(self, active_versions):
            assert active_versions == {"version-1"}
            return {"S1": {"text": "虚构制度片段"}}

    class FakeDocumentRuntime:
        repository = FakeRepository()
        vector_store = FakeVectorStore()

    class FakeRagService:
        async def stream(self, question):
            if False:
                yield question

    class FakeGenerator:
        generators = ()

    def capture_evaluation_generator(**kwargs):
        captured.update(kwargs)
        return FakeGenerator()

    monkeypatch.setattr(
        main_module, "create_evaluation_generator", capture_evaluation_generator
    )
    settings_kwargs = {"deepseek_api_key": "fixture"}
    settings = Settings(
        _env_file=None,
        source_documents_dir=tmp_path,
        private_data_dir=tmp_path / "private",
        sqlite_path=tmp_path / "metadata.sqlite3",
        qdrant_path=tmp_path / "qdrant",
        **settings_kwargs,
    )

    runtime = asyncio.run(
        main_module.build_evaluation_runtime_async(
            settings,
            FakeDocumentRuntime(),
            FakeRagService(),
            deepseek=injected_chat_client,
        )
    )

    assert captured["deepseek_client"] is None
    asyncio.run(runtime.close())
