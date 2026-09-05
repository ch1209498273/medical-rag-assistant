from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from app.domain.models import Chunk, SourceRef
from app.domain.ports import SearchHit
from app.ingestion.docling_adapter import StructuredDocument, StructuredElement
from app.ingestion.staging_chunker import chunk_with_docling
from app.ingestion.task15_models import (
    SourceFileFingerprint,
    StagingManifest,
    Task15RunStatus,
)
from app.ingestion.task15_staging import (
    build_hybrid_staging_retriever,
    build_staging_repository,
    build_staging_vector_store,
    load_staging_candidates,
    materialize_staging,
    publish_staging_metadata,
)


def _manifest(run_dir: Path) -> StagingManifest:
    return StagingManifest(
        run_id="RUN-STAGING",
        source_version="2026-standard-manual-v1",
        completed_volume_count=8,
        planned_volume_count=1,
        files=(
            SourceFileFingerprint(
                relative_path="volume-1.docx",
                size_bytes=10,
                modified_ns=20,
                sha256="a" * 64,
            ),
        ),
        parser_backend="docling",
        fallback_backend="legacy",
        chunker_config={
            "max_tokens": 512,
            "repeat_table_header": True,
            "merge_peers": True,
        },
        retrieval_config={
            "strategy": "hybrid",
            "fusion_profile": "balanced",
            "rrf_k": 60,
            "vector_weight": 1.0,
            "lexical_weight": 1.0,
        },
        output_root=str(run_dir),
        status=Task15RunStatus.CHUNKED,
    )


def _candidates():
    document = StructuredDocument(
        source=SourceFileFingerprint(
            relative_path="volume-1.docx",
            size_bytes=10,
            modified_ns=20,
            sha256="a" * 64,
        ),
        elements=(
            StructuredElement(
                text="身份核对记录。",
                heading_path=("第一册", "核对"),
                content_kind="text",
                provenance_digest="b" * 64,
            ),
        ),
        parser_backend="docling",
        tokenizer_backend="fake",
    )
    return chunk_with_docling(document)


def _write_chunk_manifest(run_dir: Path, candidates) -> None:
    candidate = candidates[0]
    relative = Path("chunks") / "docling" / "candidate.jsonl"
    path = run_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "text": candidate.text,
                "source": asdict(candidate.source),
                "record": asdict(candidate.record),
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "chunk-manifest.json").write_text(
        json.dumps(
            {
                "run_id": "RUN-STAGING",
                "source_version": "2026-standard-manual-v1",
                "status": "chunked",
                "active_backend": None,
                "files": [
                    {
                        "relative_path": "volume-1.docx",
                        "coverage": {"status": "pass"},
                        "docling": {"path": str(relative), "count": 1},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


@dataclass
class _FakeEmbedder:
    async def embed(self, _questions):
        return [[0.1] * 1024]


@dataclass
class _FakeReranker:
    async def rerank(self, _question, candidates):
        return [1.0 for _ in candidates]


class _FakeVectorStore:
    def search(self, _vector, active_versions, limit):
        assert active_versions == {"2026-standard-manual-v1"}
        return [
            SearchHit(
                version_id="2026-standard-manual-v1",
                chunk=Chunk(
                    chunk_id="vector-hit",
                    text="身份核对记录。",
                    source=SourceRef(file_name="volume-1.docx", heading_path=("第一册", "核对")),
                    content_hash="c" * 64,
                ),
                score=0.9,
            )
        ][:limit]


def test_staging_repository_and_vector_store_use_isolated_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "RUN-STAGING"
    run_dir.mkdir()
    manifest = _manifest(run_dir)

    repository = build_staging_repository(run_dir, manifest=manifest)
    candidates = _candidates()
    publish_staging_metadata(repository, candidates, version_id=manifest.source_version)
    vector_store = build_staging_vector_store(run_dir)

    assert (run_dir / "sqlite-staging").is_dir()
    assert (run_dir / "qdrant-staging").is_dir()
    assert repository.active_version_ids() == {"2026-standard-manual-v1"}
    assert repository.snapshot_active_chunks({"old-version"}) == []
    vector_store.close()
    repository.close()


def test_loader_reads_only_the_approved_passing_backend(tmp_path: Path) -> None:
    run_dir = tmp_path / "RUN-STAGING"
    run_dir.mkdir()
    manifest = _manifest(run_dir)
    candidates = _candidates()
    _write_chunk_manifest(run_dir, candidates)

    loaded = load_staging_candidates(
        run_dir,
        manifest=manifest,
        approved_backend="docling",
    )

    assert len(loaded) == 1
    assert loaded[0].record.parser_backend == "docling"
    assert loaded[0].record.source_version == manifest.source_version


def test_materialize_staging_records_backend_and_keeps_vectors_empty(tmp_path: Path) -> None:
    run_dir = tmp_path / "RUN-STAGING"
    run_dir.mkdir()
    manifest = _manifest(run_dir)
    candidates = _candidates()
    _write_chunk_manifest(run_dir, candidates)

    summary_path = materialize_staging(
        run_dir,
        manifest=manifest,
        approved_backend="docling",
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["approved_backend"] == "docling"
    assert summary["candidate_count"] == 1
    assert summary["sqlite_built"] is True
    assert summary["qdrant_built"] is True
    assert summary["vectors_indexed"] is False
    assert (run_dir / "sqlite-staging" / "staging.sqlite3").exists()
    assert (run_dir / "qdrant-staging").is_dir()


def test_hybrid_staging_retriever_has_fixed_rrf_configuration(tmp_path: Path) -> None:
    run_dir = tmp_path / "RUN-HYBRID"
    run_dir.mkdir()
    manifest = _manifest(run_dir)
    repository = build_staging_repository(run_dir, manifest=manifest)
    candidates = _candidates()
    publish_staging_metadata(repository, candidates, version_id=manifest.source_version)
    vector_store = _FakeVectorStore()

    retriever = build_hybrid_staging_retriever(
        vector_store=vector_store,
        chunks=candidates,
        repository=repository,
        embedder=_FakeEmbedder(),
        reranker=_FakeReranker(),
    )

    assert retriever.retrieval_strategy == "hybrid"
    assert retriever.rrf_k == 60
    assert retriever.fusion_profile == "balanced"
    assert retriever.vector_weight == 1.0
    assert retriever.lexical_weight == 1.0
    repository.close()
