"""Private, deterministic records used by the Task 15 refresh pipeline.

The models deliberately contain metadata and digests only.  They are safe to
serialize into an aggregate manifest; document text belongs in the private
staging directory and is never part of these records.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_VERSION = "2026-standard-manual-v1"


class Task15RunStatus(StrEnum):
    """Lifecycle status for one isolated refresh run."""

    INVENTORIED = "inventoried"
    EXTRACTED = "extracted"
    CHUNKED = "chunked"
    STAGED = "staged"
    BLOCKED = "blocked"


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _require_nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _require_digest(value: object, field: str) -> str:
    text = _require_text(value, field).lower()
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{field} must be a SHA-256 digest")
    return text


def _require_relative_path(value: object) -> str:
    path = _require_text(value, "relative_path").replace("\\", "/")
    if path.startswith("/") or re.match(r"^[A-Za-z]:/", path):
        raise ValueError("relative_path must be relative")
    parts = tuple(part for part in path.split("/") if part)
    if not parts or any(part in {".", ".."} for part in parts):
        raise ValueError("relative_path must stay inside the source directory")
    return "/".join(parts)


def _json_value(value: object) -> object:
    """Convert nested immutable values into JSON-safe values."""

    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class SourceFileFingerprint:
    """Stable identity for one source file, without its content."""

    relative_path: str
    size_bytes: int
    modified_ns: int
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "relative_path", _require_relative_path(self.relative_path))
        object.__setattr__(self, "size_bytes", _require_nonnegative_int(self.size_bytes, "size_bytes"))
        object.__setattr__(self, "modified_ns", _require_nonnegative_int(self.modified_ns, "modified_ns"))
        object.__setattr__(self, "sha256", _require_digest(self.sha256, "sha256"))


@dataclass(frozen=True, slots=True)
class DocxStructureAudit:
    """Aggregate structure and coverage facts for one DOCX source."""

    fingerprint: SourceFileFingerprint
    paragraph_count: int
    paragraph_chars: int
    heading_count: int
    textbox_count: int
    textbox_chars: int
    table_count: int
    nonempty_cell_count: int
    table_chars: int
    extracted_chars: int
    coverage_status: str
    blocking_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.fingerprint, SourceFileFingerprint):
            raise TypeError("fingerprint must be a SourceFileFingerprint")
        for field in (
            "paragraph_count",
            "paragraph_chars",
            "heading_count",
            "textbox_count",
            "textbox_chars",
            "table_count",
            "nonempty_cell_count",
            "table_chars",
            "extracted_chars",
        ):
            object.__setattr__(self, field, _require_nonnegative_int(getattr(self, field), field))
        status = _require_text(self.coverage_status, "coverage_status").casefold()
        if status not in {"pass", "blocked", "unknown"}:
            raise ValueError("coverage_status must be pass, blocked, or unknown")
        reasons = tuple(_require_text(item, "blocking_reasons") for item in self.blocking_reasons)
        if status == "blocked" and not reasons:
            raise ValueError("blocked coverage requires blocking_reasons")
        object.__setattr__(self, "coverage_status", status)
        object.__setattr__(self, "blocking_reasons", reasons)


@dataclass(frozen=True, slots=True)
class StagingChunkRecord:
    """Private chunk metadata that can be mapped to the project's ``Chunk``."""

    chunk_id: str
    source_version: str
    source_relative_path: str
    source_sha256: str
    heading_path: tuple[str, ...]
    content_kind: str
    provenance_digest: str
    content_hash: str
    token_count: int
    parser_backend: str
    ordinal: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "chunk_id", _require_text(self.chunk_id, "chunk_id"))
        object.__setattr__(self, "source_version", _require_text(self.source_version, "source_version"))
        object.__setattr__(self, "source_relative_path", _require_relative_path(self.source_relative_path))
        if self.source_version != _VERSION:
            raise ValueError(f"source_version must be {_VERSION}")
        object.__setattr__(self, "source_sha256", _require_digest(self.source_sha256, "source_sha256"))
        heading_path = tuple(_require_text(item, "heading_path") for item in self.heading_path)
        if not heading_path:
            raise ValueError("heading_path must not be empty")
        object.__setattr__(self, "heading_path", heading_path)
        object.__setattr__(self, "content_kind", _require_text(self.content_kind, "content_kind"))
        object.__setattr__(self, "provenance_digest", _require_digest(self.provenance_digest, "provenance_digest"))
        object.__setattr__(self, "content_hash", _require_digest(self.content_hash, "content_hash"))
        token_count = self.token_count
        if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count <= 0:
            raise ValueError("token_count must be a positive integer")
        object.__setattr__(self, "token_count", token_count)
        object.__setattr__(self, "parser_backend", _require_text(self.parser_backend, "parser_backend"))
        ordinal = self.ordinal
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise ValueError("ordinal must be a non-negative integer")
        object.__setattr__(self, "ordinal", ordinal)


@dataclass(frozen=True, slots=True)
class StagingManifest:
    """Run-level manifest with explicit chunking and hybrid retrieval config."""

    run_id: str
    source_version: str
    completed_volume_count: int
    planned_volume_count: int
    files: tuple[SourceFileFingerprint, ...]
    parser_backend: str
    fallback_backend: str
    chunker_config: Mapping[str, object]
    retrieval_config: Mapping[str, object]
    output_root: str
    status: Task15RunStatus

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _require_text(self.run_id, "run_id"))
        object.__setattr__(self, "source_version", _require_text(self.source_version, "source_version"))
        if self.source_version != _VERSION:
            raise ValueError(f"source_version must be {_VERSION}")
        for field in ("completed_volume_count", "planned_volume_count"):
            object.__setattr__(self, field, _require_nonnegative_int(getattr(self, field), field))
        files = tuple(self.files)
        if any(not isinstance(item, SourceFileFingerprint) for item in files):
            raise ValueError("files must contain SourceFileFingerprint values")
        if len({item.relative_path for item in files}) != len(files):
            raise ValueError("files must not contain duplicate relative_path values")
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "parser_backend", _require_text(self.parser_backend, "parser_backend"))
        object.__setattr__(self, "fallback_backend", _require_text(self.fallback_backend, "fallback_backend"))
        chunker = dict(self.chunker_config)
        max_tokens = chunker.get("max_tokens")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("chunker_config.max_tokens must be a positive integer")
        for field in ("repeat_table_header", "merge_peers"):
            if not isinstance(chunker.get(field), bool):
                raise TypeError(f"chunker_config.{field} must be a boolean")
        object.__setattr__(self, "chunker_config", MappingProxyType(chunker))
        retrieval = dict(self.retrieval_config)
        if retrieval.get("strategy") != "hybrid":
            raise ValueError("retrieval_config.strategy must be hybrid")
        if retrieval.get("fusion_profile") != "balanced":
            raise ValueError("retrieval_config.fusion_profile must be balanced")
        if retrieval.get("rrf_k") != 60:
            raise ValueError("retrieval_config.rrf_k must be 60")
        if retrieval.get("vector_weight") != 1.0 or retrieval.get("lexical_weight") != 1.0:
            raise ValueError("retrieval_config weights must be balanced at 1.0")
        object.__setattr__(self, "retrieval_config", MappingProxyType(retrieval))
        object.__setattr__(self, "output_root", _require_text(self.output_root, "output_root"))
        status = self.status
        if not isinstance(status, Task15RunStatus):
            try:
                status = Task15RunStatus(status)
            except (TypeError, ValueError) as error:
                raise ValueError("status is invalid") from error
        object.__setattr__(self, "status", status)

    def to_dict(self) -> dict[str, object]:
        """Return an aggregate, JSON-safe representation without raw content."""

        return {
            "run_id": self.run_id,
            "source_version": self.source_version,
            "completed_volume_count": self.completed_volume_count,
            "planned_volume_count": self.planned_volume_count,
            "files": [
                {
                    "relative_path": item.relative_path,
                    "size_bytes": item.size_bytes,
                    "modified_ns": item.modified_ns,
                    "sha256": item.sha256,
                }
                for item in self.files
            ],
            "parser_backend": self.parser_backend,
            "fallback_backend": self.fallback_backend,
            "chunker_config": _json_value(self.chunker_config),
            "retrieval_config": _json_value(self.retrieval_config),
            "output_root": self.output_root,
            "status": self.status.value,
        }


__all__ = [
    "DocxStructureAudit",
    "SourceFileFingerprint",
    "StagingChunkRecord",
    "StagingManifest",
    "Task15RunStatus",
]
