from __future__ import annotations

import pytest
from app.ingestion.task15_models import (
    DocxStructureAudit,
    SourceFileFingerprint,
    StagingChunkRecord,
    StagingManifest,
    Task15RunStatus,
)


def _fingerprint(name: str = "vol-a.docx") -> SourceFileFingerprint:
    return SourceFileFingerprint(
        relative_path=name,
        size_bytes=123,
        modified_ns=456,
        sha256="a" * 64,
    )


def _manifest(**overrides: object) -> StagingManifest:
    values: dict[str, object] = {
        "run_id": "RUN-001",
        "source_version": "2026-standard-manual-v1",
        "completed_volume_count": 8,
        "planned_volume_count": 1,
        "files": (_fingerprint(),),
        "parser_backend": "docling",
        "fallback_backend": "legacy",
        "chunker_config": {
            "max_tokens": 512,
            "repeat_table_header": True,
            "merge_peers": True,
        },
        "retrieval_config": {
            "strategy": "hybrid",
            "fusion_profile": "balanced",
            "rrf_k": 60,
            "vector_weight": 1.0,
            "lexical_weight": 1.0,
        },
        "output_root": "data/private/knowledge-refresh/task15/RUN-001",
        "status": Task15RunStatus.INVENTORIED,
    }
    values.update(overrides)
    return StagingManifest(**values)


def test_manifest_keeps_stable_version_and_hybrid_configuration() -> None:
    manifest = _manifest()

    payload = manifest.to_dict()

    assert payload["source_version"] == "2026-standard-manual-v1"
    assert payload["status"] == "inventoried"
    assert payload["retrieval_config"] == {
        "strategy": "hybrid",
        "fusion_profile": "balanced",
        "rrf_k": 60,
        "vector_weight": 1.0,
        "lexical_weight": 1.0,
    }


def test_manifest_serialization_does_not_contain_raw_text_or_credentials() -> None:
    manifest = _manifest()

    serialized = repr(manifest.to_dict())

    assert "正文" not in serialized
    assert "api_key" not in serialized.casefold()
    assert "question" not in serialized.casefold()
    assert "answer" not in serialized.casefold()


def test_invalid_fingerprint_is_rejected() -> None:
    with pytest.raises(ValueError, match="relative_path"):
        SourceFileFingerprint(
            relative_path="../outside.docx",
            size_bytes=1,
            modified_ns=1,
            sha256="a" * 64,
        )

    with pytest.raises(ValueError, match="sha256"):
        SourceFileFingerprint(
            relative_path="inside.docx",
            size_bytes=1,
            modified_ns=1,
            sha256="not-a-digest",
        )


def test_manifest_rejects_non_hybrid_or_unbalanced_retrieval() -> None:
    with pytest.raises(ValueError, match="hybrid"):
        _manifest(
            retrieval_config={
                "strategy": "vector",
                "fusion_profile": "balanced",
                "rrf_k": 60,
                "vector_weight": 1.0,
                "lexical_weight": 1.0,
            }
        )

    with pytest.raises(ValueError, match="balanced"):
        _manifest(
            retrieval_config={
                "strategy": "hybrid",
                "fusion_profile": "vector_dominant",
                "rrf_k": 60,
                "vector_weight": 3.0,
                "lexical_weight": 1.0,
            }
        )


def test_audit_and_chunk_records_reject_negative_or_empty_values() -> None:
    with pytest.raises(ValueError, match="paragraph_count"):
        DocxStructureAudit(
            fingerprint=_fingerprint(),
            paragraph_count=-1,
            paragraph_chars=0,
            heading_count=0,
            textbox_count=0,
            textbox_chars=0,
            table_count=0,
            nonempty_cell_count=0,
            table_chars=0,
            extracted_chars=0,
            coverage_status="pass",
            blocking_reasons=(),
        )

    with pytest.raises(ValueError, match="token_count"):
        StagingChunkRecord(
            chunk_id="chunk-1",
            source_version="2026-standard-manual-v1",
            source_relative_path="vol-a.docx",
            source_sha256="a" * 64,
            heading_path=("卷 A",),
            content_kind="text",
            provenance_digest="b" * 64,
            content_hash="c" * 64,
            token_count=0,
            parser_backend="docling",
            ordinal=0,
        )


def test_models_are_immutable() -> None:
    manifest = _manifest()

    with pytest.raises(AttributeError):
        manifest.run_id = "changed"  # type: ignore[misc]
