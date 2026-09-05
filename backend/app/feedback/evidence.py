"""Fail-closed local evidence resolution for feedback review."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.chat.safety import is_safe_answer_text, is_safe_file_name

EvidenceStatus = Literal["available", "unavailable", "version_mismatch", "unsafe"]


@dataclass(frozen=True)
class EvidenceReference:
    chunk_id: str
    source_version: str
    file_name: str
    heading_path: tuple[str, ...]
    page: int | None
    page_end: int | None
    paragraph_start: int | None
    paragraph_end: int | None
    excerpt: str
    table_id: str | None = None


@dataclass(frozen=True)
class EvidenceBundle:
    status: EvidenceStatus
    references: tuple[EvidenceReference, ...] = ()
    reason_code: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason_code": self.reason_code,
            "references": [
                {
                    "chunk_id": item.chunk_id,
                    "source_version": item.source_version,
                    "file_name": item.file_name,
                    "heading_path": list(item.heading_path),
                    "page": item.page,
                    "page_end": item.page_end,
                    "paragraph_start": item.paragraph_start,
                    "paragraph_end": item.paragraph_end,
                    "table_id": item.table_id,
                    "excerpt": item.excerpt,
                }
                for item in self.references
            ],
        }


class LocalEvidenceResolver:
    """Resolve only exact citation IDs from a local approved snapshot."""

    def __init__(self, source_snapshot) -> None:
        self.source_snapshot = source_snapshot

    def resolve(
        self, *, source_version: str | None, citation_chunk_ids: tuple[str, ...]
    ) -> EvidenceBundle:
        if not isinstance(source_version, str) or not source_version.strip():
            return EvidenceBundle("unavailable", reason_code="source_version_missing")
        if not citation_chunk_ids:
            return EvidenceBundle("unavailable", reason_code="citation_ids_missing")
        if any(
            not isinstance(chunk_id, str)
            or not chunk_id.strip()
            or len(chunk_id) > 160
            or any(char.isspace() for char in chunk_id)
            for chunk_id in citation_chunk_ids
        ):
            return EvidenceBundle("unsafe", reason_code="citation_ids_unsafe")

        hits = self.source_snapshot.snapshot_active_chunks({source_version}, limit=10_000)
        by_id = {hit.chunk.chunk_id: hit for hit in hits if hit.version_id == source_version}
        if not all(chunk_id in by_id for chunk_id in citation_chunk_ids):
            snapshot_all = getattr(self.source_snapshot, "snapshot_all_chunks", None)
            if callable(snapshot_all):
                all_hits = snapshot_all(limit=10_000)
                if any(
                    hit.chunk.chunk_id in citation_chunk_ids
                    and hit.version_id != source_version
                    for hit in all_hits
                ):
                    return EvidenceBundle("version_mismatch", reason_code="citation_version_mismatch")
            return EvidenceBundle("unavailable", reason_code="citation_chunk_missing")

        references: list[EvidenceReference] = []
        for chunk_id in citation_chunk_ids:
            hit = by_id[chunk_id]
            source = hit.chunk.source
            if (
                not is_safe_file_name(source.file_name)
                or not source.heading_path
                or any(not is_safe_answer_text(item) for item in source.heading_path)
                or not hit.chunk.text.strip()
                or not is_safe_answer_text(hit.chunk.text)
            ):
                return EvidenceBundle("unsafe", reason_code="citation_metadata_unsafe")
            references.append(
                EvidenceReference(
                    chunk_id=chunk_id,
                    source_version=source_version,
                    file_name=source.file_name,
                    heading_path=tuple(source.heading_path),
                    page=source.page,
                    page_end=source.page_end,
                    paragraph_start=source.paragraph_start,
                    paragraph_end=source.paragraph_end,
                    table_id=source.table_id,
                    excerpt=hit.chunk.text[:300],
                )
            )
        return EvidenceBundle("available", tuple(references))


__all__ = ["EvidenceBundle", "EvidenceReference", "EvidenceStatus", "LocalEvidenceResolver"]
