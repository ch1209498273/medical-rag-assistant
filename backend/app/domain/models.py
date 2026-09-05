"""Immutable domain values used by document ingestion."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceRef:
    """A human-locatable position in an imported policy document."""

    file_name: str
    heading_path: tuple[str, ...] = ()
    page: int | None = None
    page_end: int | None = None
    paragraph_start: int | None = None
    paragraph_end: int | None = None
    table_id: str | None = None


@dataclass(frozen=True)
class ExtractedBlock:
    """Text extracted from one original document position."""

    text: str
    source: SourceRef


@dataclass(frozen=True)
class ExtractedDocument:
    """Text blocks extracted locally from one supported source file."""

    file_name: str
    blocks: tuple[ExtractedBlock, ...]
    needs_ocr: bool = False


@dataclass(frozen=True)
class Chunk:
    """A stable, source-carrying unit that can later be embedded."""

    chunk_id: str
    text: str
    source: SourceRef
    content_hash: str
