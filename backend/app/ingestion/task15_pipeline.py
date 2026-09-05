"""Local, isolated Task 15 inventory/extraction/chunking pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from typing import Any

from app.ingestion.docling_adapter import (
    DoclingAdapter,
    DoclingUnavailableError,
    LocalTokenizerUnavailableError,
)
from app.ingestion.docx_audit import audit_docx_path
from app.ingestion.extractors import DocumentResourceLimits
from app.ingestion.staging_chunker import (
    StagingChunkCandidate,
    chunk_with_docling,
    chunk_with_legacy_fallback,
    compare_coverage,
)
from app.ingestion.task15_models import (
    DocxStructureAudit,
    SourceFileFingerprint,
    StagingManifest,
    Task15RunStatus,
)

SOURCE_VERSION = "2026-standard-manual-v1"
COMPLETED_VOLUME_COUNT = 8
PLANNED_VOLUME_COUNT = 1
STAGING_MAX_FILE_BYTES = 64 * 1024 * 1024
STAGING_MAX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
STAGING_MAX_DOCX_PARAGRAPHS = 50_000
STAGING_MAX_TOKENS = 512
STAGING_PRIMARY_TOKENIZER = "BAAI/bge-m3"
STAGING_TOKENIZER_FALLBACK = "cl100k_base"
STAGING_RETRIEVAL_CONFIG = {
    "strategy": "hybrid",
    "fusion_profile": "balanced",
    "rrf_k": 60,
    "vector_weight": 1.0,
    "lexical_weight": 1.0,
}


class Task15PipelineError(RuntimeError):
    """A source or staging invariant prevented safe continuation."""


def run_inventory(
    source_dir: Path,
    run_dir: Path,
    *,
    expected_file_count: int = COMPLETED_VOLUME_COUNT,
) -> StagingManifest:
    """Scan the source directory twice and write a stable private manifest."""

    source_root = Path(source_dir).resolve()
    output_root = Path(run_dir).resolve()
    if not source_root.is_dir():
        raise Task15PipelineError("source directory is unavailable")
    if expected_file_count <= 0:
        raise ValueError("expected_file_count must be positive")
    output_root.mkdir(parents=True, exist_ok=True)

    paths = sorted(
        path
        for path in source_root.rglob("*.docx")
        if path.is_file() and not path.is_symlink()
    )
    if len(paths) != expected_file_count:
        raise Task15PipelineError(
            f"expected {expected_file_count} DOCX files, found {len(paths)}"
        )

    first = {
        path: _source_fingerprint(path, relative_path=path.relative_to(source_root))
        for path in paths
    }
    second = {
        path: _source_fingerprint(path, relative_path=path.relative_to(source_root))
        for path in paths
    }
    if first != second:
        raise Task15PipelineError("source changed during scan")

    run_id = output_root.name or "task15-run"
    manifest = StagingManifest(
        run_id=run_id,
        source_version=SOURCE_VERSION,
        completed_volume_count=COMPLETED_VOLUME_COUNT,
        planned_volume_count=PLANNED_VOLUME_COUNT,
        files=tuple(first[path] for path in paths),
        parser_backend="docling",
        fallback_backend="legacy",
        chunker_config={
            "max_tokens": STAGING_MAX_TOKENS,
            "repeat_table_header": True,
            "merge_peers": True,
            "omit_header_on_overflow": False,
            "tokenizer_name": STAGING_PRIMARY_TOKENIZER,
            "tokenizer_fallback": STAGING_TOKENIZER_FALLBACK,
            "tokenizer_resolution": "local_only_explicit_fallback",
        },
        retrieval_config=STAGING_RETRIEVAL_CONFIG,
        output_root=str(output_root),
        status=Task15RunStatus.INVENTORIED,
    )
    _write_json(
        output_root / "inventory.json",
        {
            **manifest.to_dict(),
            "source_dir": str(source_root),
            "planned_volume": {"status": "planned", "included": False},
            "staging_limits": {
                "max_file_bytes": STAGING_MAX_FILE_BYTES,
                "max_docx_uncompressed_bytes": STAGING_MAX_UNCOMPRESSED_BYTES,
            },
        },
    )
    _persist_manifest(manifest)
    return manifest


def load_manifest(run_dir: Path) -> StagingManifest:
    """Load the last safe manifest so later CLI stages are resumable."""

    root = Path(run_dir).resolve()
    payload = _read_json(root / "manifest.json")
    files_payload = payload.get("files")
    if not isinstance(files_payload, list):
        raise Task15PipelineError("manifest files are missing")
    files: list[SourceFileFingerprint] = []
    for item in files_payload:
        if not isinstance(item, dict):
            raise Task15PipelineError("manifest file entry is invalid")
        files.append(
            SourceFileFingerprint(
                relative_path=str(item.get("relative_path", "")),
                size_bytes=int(item.get("size_bytes", 0)),
                modified_ns=int(item.get("modified_ns", 0)),
                sha256=str(item.get("sha256", "")),
            )
        )
    try:
        status = Task15RunStatus(str(payload.get("status", "")))
    except ValueError as error:
        raise Task15PipelineError("manifest status is invalid") from error
    return StagingManifest(
        run_id=str(payload.get("run_id", root.name)),
        source_version=str(payload.get("source_version", "")),
        completed_volume_count=int(payload.get("completed_volume_count", 0)),
        planned_volume_count=int(payload.get("planned_volume_count", 0)),
        files=tuple(files),
        parser_backend=str(payload.get("parser_backend", "docling")),
        fallback_backend=str(payload.get("fallback_backend", "legacy")),
        chunker_config=dict(payload.get("chunker_config", {})),
        retrieval_config=dict(payload.get("retrieval_config", {})),
        output_root=str(payload.get("output_root", root)),
        status=status,
    )


def run_extraction(manifest: StagingManifest) -> StagingManifest:
    """Audit all sources and write an aggregate extraction report."""

    _require_status(manifest, Task15RunStatus.INVENTORIED)
    inventory = _read_json(_run_path(manifest, "inventory.json"))
    source_root = _require_source_root(inventory)
    limits = DocumentResourceLimits(
        max_file_bytes=STAGING_MAX_FILE_BYTES,
        max_docx_paragraphs=STAGING_MAX_DOCX_PARAGRAPHS,
        max_docx_uncompressed_bytes=STAGING_MAX_UNCOMPRESSED_BYTES,
    )
    records: list[dict[str, object]] = []
    hard_failures: list[str] = []
    for fingerprint in manifest.files:
        path = _source_path(source_root, fingerprint)
        try:
            audit = audit_docx_path(path, limits=limits)
            audit = replace(audit, fingerprint=fingerprint)
            records.append({"relative_path": fingerprint.relative_path, "audit": _audit_to_dict(audit)})
        except Exception as error:  # noqa: BLE001 - report each source safely
            reason = _error_code(error)
            hard_failures.append(f"{fingerprint.relative_path}:{reason}")
            records.append(
                {
                    "relative_path": fingerprint.relative_path,
                    "status": "blocked",
                    "error_code": reason,
                }
            )
    status = Task15RunStatus.BLOCKED if hard_failures else Task15RunStatus.EXTRACTED
    next_manifest = replace(manifest, status=status)
    _write_json(
        _run_path(manifest, "extraction-report.json"),
        {
            "run_id": manifest.run_id,
            "source_version": manifest.source_version,
            "status": status.value,
            "files": records,
            "hard_failures": hard_failures,
        },
    )
    _persist_manifest(next_manifest)
    if hard_failures:
        raise Task15PipelineError("extraction blocked: " + "; ".join(hard_failures))
    return next_manifest


def run_chunking(manifest: StagingManifest) -> StagingManifest:
    """Produce Docling and legacy candidate chunks in private files."""

    if manifest.status not in {Task15RunStatus.EXTRACTED, Task15RunStatus.CHUNKED}:
        raise Task15PipelineError("chunking requires a successful extraction stage")
    inventory = _read_json(_run_path(manifest, "inventory.json"))
    extraction = _read_json(_run_path(manifest, "extraction-report.json"))
    source_root = _require_source_root(inventory)
    audit_by_path = {
        str(item["relative_path"]): _audit_from_dict(item["audit"], manifest)
        for item in extraction.get("files", [])
        if isinstance(item, dict) and isinstance(item.get("audit"), dict)
    }
    adapter = DoclingAdapter(tokenizer_name=STAGING_PRIMARY_TOKENIZER)
    private_chunk_root = _run_path(manifest, "chunks")
    private_chunk_root.mkdir(parents=True, exist_ok=True)
    file_reports: list[dict[str, object]] = []
    for fingerprint in manifest.files:
        source_path = _source_path(source_root, fingerprint)
        docling_candidates: tuple[StagingChunkCandidate, ...] = ()
        docling_error: str | None = None
        docling_primary_error: str | None = None
        docling_tokenizer_backend: str | None = None
        try:
            docling_document = adapter.convert(
                source_path,
                max_tokens=STAGING_MAX_TOKENS,
                repeat_table_header=True,
                merge_peers=True,
                omit_header_on_overflow=False,
            )
            docling_candidates = chunk_with_docling(
                replace(
                    docling_document,
                    source=replace(
                        docling_document.source,
                        relative_path=fingerprint.relative_path,
                    ),
                ),
                max_tokens=STAGING_MAX_TOKENS,
            )
            docling_tokenizer_backend = docling_document.tokenizer_backend
        except LocalTokenizerUnavailableError as error:
            # BAAI/bge-m3 is the preferred tokenizer when already cached.  A
            # local cl100k_base tokenizer is an explicit, deterministic
            # fallback for token budgeting only; it is not the embedding model.
            docling_primary_error = _error_code(error)
            fallback_adapter = DoclingAdapter(tokenizer_name=STAGING_TOKENIZER_FALLBACK)
            try:
                docling_document = fallback_adapter.convert(
                    source_path,
                    max_tokens=STAGING_MAX_TOKENS,
                    repeat_table_header=True,
                    merge_peers=True,
                    omit_header_on_overflow=False,
                )
                docling_candidates = chunk_with_docling(
                    replace(
                        docling_document,
                        source=replace(
                            docling_document.source,
                            relative_path=fingerprint.relative_path,
                        ),
                    ),
                    max_tokens=STAGING_MAX_TOKENS,
                )
                docling_tokenizer_backend = docling_document.tokenizer_backend
            except (DoclingUnavailableError, LocalTokenizerUnavailableError, ValueError) as fallback_error:
                docling_error = _error_code(fallback_error)
        except (DoclingUnavailableError, ValueError) as error:
            docling_error = _error_code(error)

        try:
            fallback_candidates = chunk_with_legacy_fallback(
                source_path,
                max_tokens=STAGING_MAX_TOKENS,
                max_docx_paragraphs=STAGING_MAX_DOCX_PARAGRAPHS,
            )
            fallback_candidates = _rebase_candidates(
                fallback_candidates,
                relative_path=fingerprint.relative_path,
            )
            fallback_error = None
        except Exception as error:  # noqa: BLE001 - preserve per-file failure
            fallback_candidates = ()
            fallback_error = _error_code(error)

        audit = audit_by_path.get(fingerprint.relative_path)
        if audit is None:
            audit = DocxStructureAudit(
                fingerprint=fingerprint,
                paragraph_count=0,
                paragraph_chars=0,
                heading_count=0,
                textbox_count=0,
                textbox_chars=0,
                table_count=0,
                nonempty_cell_count=0,
                table_chars=0,
                extracted_chars=0,
                coverage_status="blocked",
                blocking_reasons=("missing_extraction_audit",),
            )
        coverage = compare_coverage(
            audit,
            docling_chunks=docling_candidates,
            fallback_chunks=fallback_candidates,
        )
        docling_details = _write_candidate_file(
            private_chunk_root,
            fingerprint,
            "docling",
            docling_candidates,
            error_code=docling_error,
        )
        docling_details["primary_tokenizer"] = STAGING_PRIMARY_TOKENIZER
        docling_details["tokenizer_backend"] = docling_tokenizer_backend
        docling_details["primary_tokenizer_error"] = docling_primary_error
        docling_details["used_explicit_tokenizer_fallback"] = docling_primary_error is not None
        file_reports.append(
            {
                "relative_path": fingerprint.relative_path,
                "docling": docling_details,
                "legacy": _write_candidate_file(
                    private_chunk_root,
                    fingerprint,
                    "legacy",
                    fallback_candidates,
                    error_code=fallback_error,
                ),
                "coverage": coverage,
                "active_backend": None,
            }
        )
    next_manifest = replace(manifest, status=Task15RunStatus.CHUNKED)
    _write_json(
        _run_path(manifest, "chunk-manifest.json"),
        {
            "run_id": manifest.run_id,
            "source_version": manifest.source_version,
            "status": next_manifest.status.value,
            "files": file_reports,
            "active_backend": None,
        },
    )
    _persist_manifest(next_manifest)
    return next_manifest


def run_report(manifest: StagingManifest) -> Path:
    """Create a safe aggregate quality report for the current run."""

    if manifest.status not in {Task15RunStatus.CHUNKED, Task15RunStatus.STAGED}:
        raise Task15PipelineError("quality report requires chunking")
    extraction = _read_json(_run_path(manifest, "extraction-report.json"))
    chunks = _read_json(_run_path(manifest, "chunk-manifest.json"))
    file_reports = chunks.get("files", [])
    extraction_by_path = {
        str(item.get("relative_path")): item
        for item in extraction.get("files", [])
        if isinstance(item, dict)
    }
    blocking: list[str] = []
    warnings: list[str] = []
    for item in file_reports:
        if not isinstance(item, dict):
            blocking.append("invalid_chunk_manifest_record")
            continue
        coverage = item.get("coverage", {})
        if isinstance(coverage, dict):
            for reason in coverage.get("blocking_reasons", []):
                blocking.append(f"{item.get('relative_path')}:{reason}")
            extraction_item = extraction_by_path.get(str(item.get("relative_path")))
            extraction_audit = (
                extraction_item.get("audit")
                if isinstance(extraction_item, dict)
                else None
            )
            if (
                isinstance(extraction_audit, dict)
                and extraction_audit.get("coverage_status") == "blocked"
            ):
                if coverage.get("status") == "pass":
                    warnings.append(
                        f"{item.get('relative_path')}:"
                        "extraction_audit_gap_recovered_by_candidate"
                    )
                else:
                    blocking.append(
                        f"{item.get('relative_path')}:extraction_audit_not_recovered"
                    )
        docling_details = item.get("docling", {})
        if isinstance(docling_details, dict):
            if docling_details.get("error_code") == "docling_unavailable":
                warnings.append(f"{item.get('relative_path')}:docling_unavailable")
            if docling_details.get("used_explicit_tokenizer_fallback") is True:
                warnings.append(f"{item.get('relative_path')}:primary_tokenizer_fallback")
    extraction_summary = _extraction_summary(manifest, extraction)
    chunk_summary = {
        backend: _candidate_summary(manifest, file_reports, backend)
        for backend in ("docling", "legacy")
    }
    staging_root = _run_path(manifest, "sqlite-staging")
    qdrant_root = _run_path(manifest, "qdrant-staging")
    staging_summary = _read_optional_staging_summary(manifest)
    cloud_embedding_executed = bool(
        staging_summary is not None
        and staging_summary.get("cloud_embedding_executed") is True
        and staging_summary.get("vectors_indexed") is True
    )
    staging_info: dict[str, object] = {
        "sqlite_built": staging_root.is_dir(),
        "qdrant_built": qdrant_root.is_dir(),
        "active_backend": None,
        "version_consistency": "not_built",
        "retrieval_config": dict(manifest.retrieval_config),
    }
    active_backend: str | None = None
    if staging_summary is not None:
        active_backend = str(staging_summary["approved_backend"])
        staging_info.update(
            {
                "sqlite_built": staging_summary["sqlite_built"],
                "qdrant_built": staging_summary["qdrant_built"],
                "active_backend": active_backend,
                "version_consistency": staging_summary["version_consistency"],
                "retrieval_config": staging_summary["retrieval_config"],
                "vectors_indexed": staging_summary["vectors_indexed"],
            }
        )
    report_path = _run_path(manifest, "quality-report.json")
    _write_json(
        report_path,
        {
            "run_id": manifest.run_id,
            "source_version": manifest.source_version,
            "status": "blocked" if blocking else "pass",
            "file_count": len(manifest.files),
            "planned_volume": {"count": manifest.planned_volume_count, "included": False},
            "extraction_status": extraction.get("status"),
            "extraction_summary": extraction_summary,
            "chunk_summary": chunk_summary,
            "docling_chunk_count": chunk_summary["docling"]["count"],
            "legacy_chunk_count": chunk_summary["legacy"]["count"],
            "staging": staging_info,
            "blocking_reasons": sorted(set(blocking)),
            "warnings": sorted(set(warnings)),
            "production_unchanged": True,
            "active_backend": active_backend,
            "rollback_path": str(Path(manifest.output_root)),
            "cloud_embedding_executed": cloud_embedding_executed,
            "deletion_executed": False,
        },
    )
    return report_path


def _read_optional_staging_summary(manifest: StagingManifest) -> dict[str, object] | None:
    path = _run_path(manifest, "staging-summary.json")
    if not path.exists():
        return None
    try:
        payload = _read_json(path)
    except Task15PipelineError as error:
        raise Task15PipelineError("staging summary is invalid") from error
    required = {
        "run_id",
        "source_version",
        "approved_backend",
        "candidate_count",
        "sqlite_built",
        "qdrant_built",
        "vectors_indexed",
        "version_consistency",
        "retrieval_config",
    }
    if set(payload) < required:
        raise Task15PipelineError("staging summary fields are missing")
    if payload["run_id"] != manifest.run_id or payload["source_version"] != manifest.source_version:
        raise Task15PipelineError("staging summary identity does not match manifest")
    if payload["approved_backend"] not in {"docling", "legacy"}:
        raise Task15PipelineError("staging summary backend is invalid")
    if payload["version_consistency"] != "pass":
        raise Task15PipelineError("staging summary version consistency is invalid")
    if payload["retrieval_config"] != dict(manifest.retrieval_config):
        raise Task15PipelineError("staging summary retrieval config does not match manifest")
    for field in ("sqlite_built", "qdrant_built", "vectors_indexed"):
        if not isinstance(payload[field], bool):
            raise Task15PipelineError(f"staging summary {field} is invalid")
    return payload


def _extraction_summary(
    manifest: StagingManifest,
    extraction: dict[str, Any],
) -> dict[str, object]:
    totals = {
        "paragraph_count": 0,
        "paragraph_chars": 0,
        "heading_count": 0,
        "textbox_count": 0,
        "textbox_chars": 0,
        "table_count": 0,
        "nonempty_cell_count": 0,
        "table_chars": 0,
        "extracted_chars": 0,
    }
    coverage_status_counts: dict[str, int] = {}
    audited_count = 0
    blocked_count = 0
    for item in extraction.get("files", []):
        if not isinstance(item, dict):
            blocked_count += 1
            continue
        audit = item.get("audit")
        if not isinstance(audit, dict):
            blocked_count += 1
            continue
        audited_count += 1
        status = str(audit.get("coverage_status", "unknown"))
        coverage_status_counts[status] = coverage_status_counts.get(status, 0) + 1
        for field in totals:
            value = audit.get(field, 0)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                totals[field] += value
    blocked_count += max(0, len(manifest.files) - audited_count - blocked_count)
    return {
        "file_count": len(manifest.files),
        "audited_file_count": audited_count,
        "blocked_file_count": blocked_count,
        **totals,
        "coverage_status_counts": coverage_status_counts,
    }


def _candidate_summary(
    manifest: StagingManifest,
    file_reports: object,
    backend: str,
) -> dict[str, object]:
    summary = {
        "count": 0,
        "files_with_candidates": 0,
        "max_token_count": 0,
        "empty_chunk_count": 0,
        "token_over_limit_count": 0,
        "duplicate_chunk_id_count": 0,
        "duplicate_content_hash_count": 0,
        "source_retraceable_count": 0,
        "source_retraceable_rate": "0/0",
        "invalid_record_count": 0,
    }
    if not isinstance(file_reports, list):
        return summary
    chunk_ids: set[str] = set()
    content_hashes: set[str] = set()
    retraceable = 0
    for item in file_reports:
        if not isinstance(item, dict):
            continue
        details = item.get(backend)
        if not isinstance(details, dict):
            continue
        count = details.get("count")
        if not isinstance(count, int) or count < 0:
            summary["invalid_record_count"] += 1
            continue
        candidate_path = details.get("path")
        if count > 0:
            summary["files_with_candidates"] += 1
        if not isinstance(candidate_path, str):
            if count:
                summary["invalid_record_count"] += 1
            continue
        path = Path(candidate_path)
        if not path.is_absolute():
            path = Path(manifest.output_root) / path
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            summary["invalid_record_count"] += 1
            continue
        if len(lines) != count:
            summary["invalid_record_count"] += 1
        for line in lines:
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                summary["invalid_record_count"] += 1
                continue
            if not isinstance(raw, dict):
                summary["invalid_record_count"] += 1
                continue
            text = raw.get("text")
            record = raw.get("record")
            if not isinstance(text, str) or not text.strip():
                summary["empty_chunk_count"] += 1
            if not isinstance(record, dict):
                summary["invalid_record_count"] += 1
                continue
            summary["count"] += 1
            token_count = record.get("token_count")
            if isinstance(token_count, int) and not isinstance(token_count, bool):
                summary["max_token_count"] = max(summary["max_token_count"], token_count)
                if token_count > STAGING_MAX_TOKENS:
                    summary["token_over_limit_count"] += 1
            chunk_id = record.get("chunk_id")
            if isinstance(chunk_id, str):
                if chunk_id in chunk_ids:
                    summary["duplicate_chunk_id_count"] += 1
                chunk_ids.add(chunk_id)
            content_hash = record.get("content_hash")
            if isinstance(content_hash, str):
                if content_hash in content_hashes:
                    summary["duplicate_content_hash_count"] += 1
                content_hashes.add(content_hash)
            source = raw.get("source")
            if (
                isinstance(source, dict)
                and isinstance(source.get("file_name"), str)
                and isinstance(source.get("heading_path"), list)
                and source.get("heading_path")
            ):
                retraceable += 1
    summary["source_retraceable_count"] = retraceable
    summary["source_retraceable_rate"] = f"{retraceable}/{summary['count']}"
    return summary


def _require_status(manifest: StagingManifest, status: Task15RunStatus) -> None:
    if manifest.status is not status:
        raise Task15PipelineError(
            f"stage requires {status.value}, current status is {manifest.status.value}"
        )


def _run_path(manifest: StagingManifest, name: str) -> Path:
    return Path(manifest.output_root) / name


def _source_path(source_root: Path, fingerprint: SourceFileFingerprint) -> Path:
    candidate = (source_root / Path(fingerprint.relative_path)).resolve()
    if source_root not in candidate.parents:
        raise Task15PipelineError("source path escapes source directory")
    if not candidate.is_file():
        raise Task15PipelineError(f"source file is unavailable: {fingerprint.relative_path}")
    return candidate


def _source_fingerprint(
    path: Path,
    *,
    relative_path: Path | None = None,
) -> SourceFileFingerprint:
    try:
        size = path.stat().st_size
        data = path.read_bytes()
        modified_ns = path.stat().st_mtime_ns
    except OSError as error:
        raise Task15PipelineError("source file cannot be read") from error
    if size > STAGING_MAX_FILE_BYTES or len(data) > STAGING_MAX_FILE_BYTES:
        raise Task15PipelineError("document file size limit exceeded")
    return SourceFileFingerprint(
        relative_path=(relative_path or Path(path.name)).as_posix(),
        size_bytes=len(data),
        modified_ns=modified_ns,
        sha256=sha256(data).hexdigest(),
    )


def _require_source_root(inventory: dict[str, object]) -> Path:
    source_dir = inventory.get("source_dir")
    if not isinstance(source_dir, str) or not source_dir.strip():
        raise Task15PipelineError("inventory source_dir is missing")
    path = Path(source_dir).resolve()
    if not path.is_dir():
        raise Task15PipelineError("inventory source_dir is unavailable")
    return path


def _audit_to_dict(audit: DocxStructureAudit) -> dict[str, object]:
    return {
        "fingerprint": {
            "relative_path": audit.fingerprint.relative_path,
            "size_bytes": audit.fingerprint.size_bytes,
            "modified_ns": audit.fingerprint.modified_ns,
            "sha256": audit.fingerprint.sha256,
        },
        "paragraph_count": audit.paragraph_count,
        "paragraph_chars": audit.paragraph_chars,
        "heading_count": audit.heading_count,
        "textbox_count": audit.textbox_count,
        "textbox_chars": audit.textbox_chars,
        "table_count": audit.table_count,
        "nonempty_cell_count": audit.nonempty_cell_count,
        "table_chars": audit.table_chars,
        "extracted_chars": audit.extracted_chars,
        "coverage_status": audit.coverage_status,
        "blocking_reasons": list(audit.blocking_reasons),
    }


def _audit_from_dict(payload: dict[str, object], manifest: StagingManifest) -> DocxStructureAudit:
    raw_fp = payload.get("fingerprint")
    if not isinstance(raw_fp, dict):
        raise Task15PipelineError("extraction audit fingerprint is missing")
    fp = SourceFileFingerprint(
        relative_path=str(raw_fp.get("relative_path", "")),
        size_bytes=int(raw_fp.get("size_bytes", 0)),
        modified_ns=int(raw_fp.get("modified_ns", 0)),
        sha256=str(raw_fp.get("sha256", "")),
    )
    matched = next((item for item in manifest.files if item.relative_path == fp.relative_path), fp)
    return DocxStructureAudit(
        fingerprint=matched,
        paragraph_count=int(payload.get("paragraph_count", 0)),
        paragraph_chars=int(payload.get("paragraph_chars", 0)),
        heading_count=int(payload.get("heading_count", 0)),
        textbox_count=int(payload.get("textbox_count", 0)),
        textbox_chars=int(payload.get("textbox_chars", 0)),
        table_count=int(payload.get("table_count", 0)),
        nonempty_cell_count=int(payload.get("nonempty_cell_count", 0)),
        table_chars=int(payload.get("table_chars", 0)),
        extracted_chars=int(payload.get("extracted_chars", 0)),
        coverage_status=str(payload.get("coverage_status", "blocked")),
        blocking_reasons=tuple(str(item) for item in payload.get("blocking_reasons", [])),
    )


def _rebase_candidates(
    candidates: tuple[StagingChunkCandidate, ...],
    *,
    relative_path: str,
) -> tuple[StagingChunkCandidate, ...]:
    return tuple(
        replace(
            candidate,
            record=replace(candidate.record, source_relative_path=relative_path),
            source=replace(candidate.source, file_name=relative_path),
        )
        for candidate in candidates
    )


def _write_candidate_file(
    root: Path,
    fingerprint: SourceFileFingerprint,
    backend: str,
    candidates: tuple[StagingChunkCandidate, ...],
    *,
    error_code: str | None,
) -> dict[str, object]:
    backend_dir = root / backend
    backend_dir.mkdir(parents=True, exist_ok=True)
    file_path = backend_dir / f"{fingerprint.sha256}.jsonl"
    with file_path.open("w", encoding="utf-8", newline="\n") as handle:
        for candidate in candidates:
            handle.write(json.dumps(_private_candidate_dict(candidate), ensure_ascii=False))
            handle.write("\n")
    result: dict[str, object] = {
        "count": len(candidates),
        "path": str(file_path.relative_to(root.parent)),
        "error_code": error_code,
    }
    return result


def _private_candidate_dict(candidate: StagingChunkCandidate) -> dict[str, object]:
    return {
        "text": candidate.text,
        "source": {
            "file_name": candidate.source.file_name,
            "heading_path": list(candidate.source.heading_path),
            "page": candidate.source.page,
            "page_end": candidate.source.page_end,
            "paragraph_start": candidate.source.paragraph_start,
            "paragraph_end": candidate.source.paragraph_end,
            "table_id": candidate.source.table_id,
        },
        "record": asdict(candidate.record),
    }


def _error_code(error: BaseException) -> str:
    text = str(error).casefold()
    if "docling is unavailable" in text or "local tokenizer" in text:
        return "docling_unavailable"
    if "no chunks" in text:
        return "docling_no_chunks"
    if "size limit" in text:
        return "resource_limit_exceeded"
    if "invalid" in text or "malformed" in text:
        return "invalid_document"
    return error.__class__.__name__.casefold()


def _persist_manifest(manifest: StagingManifest) -> None:
    _write_json(Path(manifest.output_root) / "manifest.json", manifest.to_dict())


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Task15PipelineError(f"private stage file is invalid: {path.name}") from error
    if not isinstance(value, dict):
        raise Task15PipelineError(f"private stage file must be an object: {path.name}")
    return value


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


__all__ = [
    "Task15PipelineError",
    "load_manifest",
    "run_chunking",
    "run_extraction",
    "run_inventory",
    "run_report",
]
