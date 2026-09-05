"""Isolated SQLite/Qdrant staging stores and a fixed hybrid retriever."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from app.domain.models import Chunk, SourceRef
from app.domain.ports import SearchHit
from app.ingestion.staging_chunker import StagingChunkCandidate
from app.ingestion.task15_models import (
    StagingChunkRecord,
    StagingManifest,
    Task15RunStatus,
)
from app.rag.lexical import ActiveLexicalIndex
from app.rag.retrieval import RetrievalService
from app.storage.qdrant import QdrantLocalVectorStore


class Task15StagingRepository:
    """Small private repository for technical staging metadata and text.

    It deliberately has no business-approval methods.  The staging retriever
    opts into ``require_business_eligibility=False`` and can therefore be used
    for technical smoke checks without making draft material production-ready.
    """

    def __init__(self, database_path: Path, *, version_id: str) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(database_path)
        self._connection.row_factory = sqlite3.Row
        self._version_id = version_id
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS staging_versions (
                version_id TEXT PRIMARY KEY,
                source_version TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('active', 'retired'))
            );
            CREATE TABLE IF NOT EXISTS staging_chunks (
                version_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                text TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                source_sha256 TEXT NOT NULL,
                file_name TEXT NOT NULL,
                heading_path TEXT NOT NULL,
                page INTEGER,
                page_end INTEGER,
                paragraph_start INTEGER,
                paragraph_end INTEGER,
                table_id TEXT,
                parser_backend TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                PRIMARY KEY(version_id, chunk_id),
                FOREIGN KEY(version_id) REFERENCES staging_versions(version_id)
            );
            """
        )
        columns = {
            str(row[1])
            for row in self._connection.execute("PRAGMA table_info(staging_chunks)")
        }
        if "table_id" not in columns:
            self._connection.execute("ALTER TABLE staging_chunks ADD COLUMN table_id TEXT")
        self._connection.execute(
            "INSERT OR IGNORE INTO staging_versions(version_id, source_version, status) VALUES (?, ?, 'active')",
            (version_id, version_id),
        )
        self._connection.commit()

    def publish_chunks(
        self,
        candidates: Sequence[StagingChunkCandidate],
        *,
        version_id: str,
    ) -> None:
        if version_id != self._version_id:
            raise ValueError("staging version_id does not match repository")
        for candidate in candidates:
            record = candidate.record
            if record.source_version != version_id:
                raise ValueError("chunk source_version does not match staging version")
            self._connection.execute(
                """
                INSERT OR REPLACE INTO staging_chunks(
                    version_id, chunk_id, text, content_hash, source_sha256,
                    file_name, heading_path, page, page_end,
                    paragraph_start, paragraph_end, table_id, parser_backend, token_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version_id,
                    record.chunk_id,
                    candidate.text,
                    record.content_hash,
                    record.source_sha256,
                    candidate.source.file_name,
                    json.dumps(list(record.heading_path), ensure_ascii=False),
                    candidate.source.page,
                    candidate.source.page_end,
                    candidate.source.paragraph_start,
                    candidate.source.paragraph_end,
                    candidate.source.table_id,
                    record.parser_backend,
                    record.token_count,
                ),
            )
        self._connection.commit()

    def active_version_ids(self) -> set[str]:
        rows = self._connection.execute(
            "SELECT version_id FROM staging_versions WHERE status = 'active'"
        )
        return {str(row["version_id"]) for row in rows}

    def snapshot_active_chunks(
        self,
        active_versions: set[str],
        limit: int = 10_000,
    ) -> list[SearchHit]:
        if not active_versions or limit <= 0:
            return []
        if not active_versions.issubset(self.active_version_ids()):
            active_versions = active_versions & self.active_version_ids()
        if not active_versions:
            return []
        placeholders = ",".join("?" for _ in active_versions)
        rows = self._connection.execute(
            f"SELECT * FROM staging_chunks WHERE version_id IN ({placeholders}) ORDER BY version_id, chunk_id LIMIT ?",
            (*sorted(active_versions), limit),
        )
        hits: list[SearchHit] = []
        for row in rows:
            heading_path = tuple(json.loads(row["heading_path"]))
            source = SourceRef(
                file_name=str(row["file_name"]),
                heading_path=heading_path,
                page=row["page"],
                page_end=row["page_end"],
                paragraph_start=row["paragraph_start"],
                paragraph_end=row["paragraph_end"],
                table_id=row["table_id"],
            )
            hits.append(
                SearchHit(
                    version_id=str(row["version_id"]),
                    chunk=Chunk(
                        chunk_id=str(row["chunk_id"]),
                        text=str(row["text"]),
                        source=source,
                        content_hash=str(row["content_hash"]),
                    ),
                    score=0.0,
                )
            )
        return hits

    def close(self) -> None:
        self._connection.close()


class _InMemoryStagingRepository:
    def __init__(self, candidates: Sequence[StagingChunkCandidate]) -> None:
        self._hits = tuple(_candidate_to_hit(candidate) for candidate in candidates)

    def active_version_ids(self) -> set[str]:
        return {"2026-standard-manual-v1"}

    def snapshot_active_chunks(self, active_versions: set[str], limit: int = 10_000) -> list[SearchHit]:
        if "2026-standard-manual-v1" not in active_versions:
            return []
        return list(self._hits[:limit])


def load_staging_candidates(
    run_dir: Path,
    *,
    manifest: StagingManifest,
    approved_backend: str,
) -> tuple[StagingChunkCandidate, ...]:
    """Load one passing candidate backend from a private chunk manifest.

    The loader is deliberately strict: every source file must expose the
    selected backend, its coverage must be ``pass``, and every JSONL path must
    remain under this run directory.  This prevents an accidental partial
    staging build or path traversal from silently becoming the active input.
    """

    if approved_backend not in {"docling", "legacy"}:
        raise ValueError("approved_backend must be docling or legacy")
    root = Path(run_dir).resolve()
    if Path(manifest.output_root).resolve() != root:
        raise ValueError("staging run_dir does not match manifest output_root")
    manifest_path = root / "chunk-manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("chunk manifest is invalid") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("files"), list):
        raise TypeError("chunk manifest is invalid")
    if payload.get("run_id") != manifest.run_id:
        raise ValueError("chunk manifest run_id does not match manifest")
    if payload.get("source_version") != manifest.source_version:
        raise ValueError("chunk manifest source_version does not match manifest")
    if payload.get("status") != Task15RunStatus.CHUNKED.value:
        raise ValueError("chunk manifest is not ready for staging")
    source_files = {item.relative_path: item for item in manifest.files}
    candidates: list[StagingChunkCandidate] = []
    seen_ids: set[str] = set()
    for item in payload["files"]:
        if not isinstance(item, dict):
            raise TypeError("chunk manifest file entry is invalid")
        relative_source = item.get("relative_path")
        if not isinstance(relative_source, str) or relative_source not in source_files:
            raise ValueError("chunk manifest source file is not in manifest")
        coverage = item.get("coverage")
        if not isinstance(coverage, Mapping) or coverage.get("status") != "pass":
            raise ValueError("selected backend does not have passing coverage")
        details = item.get(approved_backend)
        if not isinstance(details, Mapping):
            raise TypeError("selected backend candidate is missing")
        relative_candidate = details.get("path")
        expected_count = details.get("count")
        if (
            not isinstance(relative_candidate, str)
            or not relative_candidate.strip()
            or isinstance(expected_count, bool)
            or not isinstance(expected_count, int)
            or expected_count <= 0
        ):
            raise ValueError("selected backend candidate metadata is invalid")
        candidate_path = (root / relative_candidate).resolve()
        try:
            candidate_path.relative_to(root)
        except ValueError as error:
            raise ValueError("candidate path escapes staging run") from error
        try:
            lines = candidate_path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise ValueError("private candidate file is unavailable") from error
        if len(lines) != expected_count:
            raise ValueError("candidate count does not match manifest")
        for line in lines:
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("private candidate record is invalid") from error
            candidate = _candidate_from_private_record(raw)
            record = candidate.record
            if record.parser_backend != approved_backend:
                raise ValueError("candidate parser backend does not match selection")
            if record.source_version != manifest.source_version:
                raise ValueError("candidate source version does not match manifest")
            if record.source_relative_path not in source_files:
                raise ValueError("candidate source is not in manifest")
            if record.chunk_id in seen_ids:
                raise ValueError("selected backend contains duplicate chunk_id")
            seen_ids.add(record.chunk_id)
            candidates.append(candidate)
    if len({item.get("relative_path") for item in payload["files"] if isinstance(item, dict)}) != len(
        source_files
    ):
        raise ValueError("chunk manifest does not cover every manifest source")
    if not candidates:
        raise ValueError("approved backend has no candidates")
    return tuple(candidates)


def materialize_staging(
    run_dir: Path,
    *,
    manifest: StagingManifest,
    approved_backend: str,
) -> Path:
    """Persist the selected candidate backend into isolated local stores.

    This stage writes metadata/text to the run-local SQLite database and
    creates an empty run-local Qdrant collection.  Vectors are intentionally
    absent until the separately authorized cloud Embedding stage.
    """

    candidates = load_staging_candidates(
        run_dir,
        manifest=manifest,
        approved_backend=approved_backend,
    )
    repository = build_staging_repository(run_dir, manifest=manifest)
    vector_store = build_staging_vector_store(run_dir)
    try:
        publish_staging_metadata(repository, candidates, version_id=manifest.source_version)
    finally:
        repository.close()
        vector_store.close()

    root = Path(run_dir).resolve()
    summary = {
        "run_id": manifest.run_id,
        "source_version": manifest.source_version,
        "approved_backend": approved_backend,
        "candidate_count": len(candidates),
        "sqlite_built": True,
        "qdrant_built": True,
        "vectors_indexed": False,
        "sqlite_path": "sqlite-staging/staging.sqlite3",
        "qdrant_path": "qdrant-staging",
        "version_consistency": "pass",
        "retrieval_config": dict(manifest.retrieval_config),
        "cloud_embedding_executed": False,
        "production_unchanged": True,
        "deletion_executed": False,
        "status": "staged",
    }
    summary_path = root / "staging-summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    staged_manifest = replace(manifest, status=Task15RunStatus.STAGED)
    (root / "manifest.json").write_text(
        json.dumps(staged_manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary_path


def build_staging_repository(
    run_dir: Path,
    *,
    manifest: StagingManifest,
) -> Task15StagingRepository:
    """Create an independent private SQLite staging repository."""

    return Task15StagingRepository(
        Path(run_dir) / "sqlite-staging" / "staging.sqlite3",
        version_id=manifest.source_version,
    )


def publish_staging_metadata(
    repository: Task15StagingRepository,
    chunks: Sequence[StagingChunkCandidate],
    *,
    version_id: str,
) -> None:
    """Persist private chunk text and metadata into the staging database."""

    repository.publish_chunks(chunks, version_id=version_id)


def build_staging_vector_store(
    run_dir: Path,
    *,
    dimensions: int = 1024,
) -> QdrantLocalVectorStore:
    """Create a separate local Qdrant collection for staging vectors."""

    if dimensions != QdrantLocalVectorStore.vector_dimension:
        raise ValueError("Task 15 staging requires 1024-dimensional vectors")
    return QdrantLocalVectorStore(Path(run_dir) / "qdrant-staging")


def build_hybrid_staging_retriever(
    *,
    vector_store,
    chunks: Sequence[StagingChunkCandidate],
    repository=None,
    embedder=None,
    reranker=None,
    retrieval_limit: int = 20,
    rerank_limit: int = 6,
    threshold: float = 0.35,
) -> RetrievalService:
    """Build the explicitly configured staging hybrid retriever."""

    resolved_repository = repository or _InMemoryStagingRepository(chunks)
    lexical_index = ActiveLexicalIndex(resolved_repository)
    return RetrievalService(
        repository=resolved_repository,
        vector_store=vector_store,
        embedder=embedder or _UnavailableProvider("embedding"),
        reranker=reranker or _UnavailableProvider("reranker"),
        retrieval_limit=retrieval_limit,
        rerank_limit=rerank_limit,
        relevance_threshold=threshold,
        retrieval_strategy="hybrid",
        lexical_index=lexical_index,
        rrf_k=60,
        fusion_profile="balanced",
        vector_weight=1.0,
        lexical_weight=1.0,
        require_business_eligibility=False,
    )


class _UnavailableProvider:
    def __init__(self, name: str) -> None:
        self._name = name

    async def embed(self, _questions):
        raise RuntimeError(f"staging {self._name} provider was not configured")

    async def rerank(self, _question, _candidates):
        raise RuntimeError(f"staging {self._name} provider was not configured")


def _candidate_to_hit(candidate: StagingChunkCandidate) -> SearchHit:
    return SearchHit(
        version_id=candidate.record.source_version,
        chunk=Chunk(
            chunk_id=candidate.record.chunk_id,
            text=candidate.text,
            source=candidate.source,
            content_hash=candidate.record.content_hash,
        ),
        score=0.0,
    )


def _candidate_from_private_record(raw: object) -> StagingChunkCandidate:
    if not isinstance(raw, dict):
        raise TypeError("private candidate record is invalid")
    text = raw.get("text")
    source_payload = raw.get("source")
    record_payload = raw.get("record")
    if (
        not isinstance(text, str)
        or not text.strip()
        or not isinstance(source_payload, Mapping)
        or not isinstance(record_payload, Mapping)
    ):
        raise ValueError("private candidate record is invalid")
    try:
        record = StagingChunkRecord(
            chunk_id=str(record_payload["chunk_id"]),
            source_version=str(record_payload["source_version"]),
            source_relative_path=str(record_payload["source_relative_path"]),
            source_sha256=str(record_payload["source_sha256"]),
            heading_path=tuple(record_payload["heading_path"]),
            content_kind=str(record_payload["content_kind"]),
            provenance_digest=str(record_payload["provenance_digest"]),
            content_hash=str(record_payload["content_hash"]),
            token_count=int(record_payload["token_count"]),
            parser_backend=str(record_payload["parser_backend"]),
            ordinal=int(record_payload["ordinal"]),
        )
        source = SourceRef(
            file_name=str(source_payload["file_name"]),
            heading_path=tuple(source_payload["heading_path"]),
            page=_optional_int(source_payload.get("page")),
            page_end=_optional_int(source_payload.get("page_end")),
            paragraph_start=_optional_int(source_payload.get("paragraph_start")),
            paragraph_end=_optional_int(source_payload.get("paragraph_end")),
            table_id=_optional_table_id(source_payload.get("table_id")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("private candidate record is invalid") from error
    normalised_text = text.strip()
    if record.content_hash != sha256(normalised_text.encode("utf-8")).hexdigest():
        raise ValueError("private candidate content hash is invalid")
    return StagingChunkCandidate(record=record, text=normalised_text, source=source)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("source location must be an integer or null")
    return value


def _optional_table_id(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.startswith("table-"):
        raise TypeError("source table_id must be a table-N value or null")
    suffix = value.removeprefix("table-")
    if not suffix.isdigit() or int(suffix) < 1:
        raise ValueError("source table_id must be a table-N value or null")
    return value


__all__ = [
    "Task15StagingRepository",
    "build_hybrid_staging_retriever",
    "build_staging_repository",
    "build_staging_vector_store",
    "load_staging_candidates",
    "materialize_staging",
    "publish_staging_metadata",
]
