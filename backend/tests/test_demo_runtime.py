from __future__ import annotations

import json
from pathlib import Path

import pytest
from app.main import build_document_runtime_async, build_rag_runtime_async
from app.settings import Settings


def forbidden_constructor(*args, **kwargs):
    raise AssertionError("cloud client must not be constructed")


def write_demo_questions(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scenarios": [
                    {
                        "scenario_id": "demo",
                        "type": "fact",
                        "question": "培训需要多少学时？",
                        "standalone_question": "培训需要多少学时？",
                        "answer": "至少8学时。",
                        "refused": False,
                        "evidence_terms": ["培训", "8学时"],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def demo_settings(tmp_path: Path) -> Settings:
    questions = write_demo_questions(tmp_path / "questions.json")
    return Settings(
        _env_file=None,
        app_runtime_mode="demo",
        demo_questions_path=questions,
        source_documents_dir=tmp_path / "documents",
        private_data_dir=tmp_path / "private",
        qdrant_path=tmp_path / "qdrant",
        sqlite_path=tmp_path / "app.sqlite3",
    )


def cloud_settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "_env_file": None,
        "app_runtime_mode": "cloud",
        "source_documents_dir": tmp_path / "documents",
        "private_data_dir": tmp_path / "private",
        "qdrant_path": tmp_path / "qdrant",
        "sqlite_path": tmp_path / "app.sqlite3",
    }
    values.update(overrides)
    return Settings(**values)


@pytest.mark.asyncio
async def test_demo_runtime_never_constructs_cloud_clients(monkeypatch, tmp_path):
    monkeypatch.setattr("app.main.SiliconFlowClient", forbidden_constructor)
    monkeypatch.setattr("app.main.DeepSeekClient", forbidden_constructor)
    settings = demo_settings(tmp_path)
    document_runtime = await build_document_runtime_async(settings)
    rag_runtime = await build_rag_runtime_async(settings, document_runtime)
    assert type(document_runtime.provider).__name__ == "DemoKnowledgeProvider"
    assert type(rag_runtime.deepseek).__name__ == "DemoLanguageModel"
    await rag_runtime.close()


@pytest.mark.asyncio
async def test_cloud_runtime_does_not_fallback_when_key_is_missing(tmp_path):
    settings = cloud_settings(tmp_path, deepseek_api_key="", siliconflow_api_key="")
    with pytest.raises(RuntimeError, match="required cloud providers are not configured"):
        await build_document_runtime_async(settings)
