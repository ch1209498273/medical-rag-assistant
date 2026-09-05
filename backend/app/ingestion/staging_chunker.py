"""Book-structure-aware staging chunking with a deterministic fallback."""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from docx import Document

from app.domain.models import Chunk, ExtractedBlock, ExtractedDocument, SourceRef
from app.ingestion.chunker import chunk_document
from app.ingestion.docling_adapter import (
    DoclingUnavailableError,
    StructuredDocument,
    StructuredElement,
    extract_with_fallback,
)
from app.ingestion.extractors import DocumentResourceLimits
from app.ingestion.task15_models import (
    DocxStructureAudit,
    SourceFileFingerprint,
    StagingChunkRecord,
)

_BOUNDARIES = re.compile(r"(?<=[。；\n])")
_MAX_TOKENS = 16_384


@dataclass(frozen=True, slots=True)
class StagingChunkCandidate:
    """Private text plus its safe metadata record and source reference."""

    record: StagingChunkRecord
    text: str
    source: SourceRef


def chunk_with_docling(
    document: StructuredDocument,
    *,
    max_tokens: int = 512,
) -> tuple[StagingChunkCandidate, ...]:
    """Adapt Docling's already token-aware chunks into project records."""

    _validate_max_tokens(max_tokens)
    return _chunk_structured_document(document, max_tokens=max_tokens)


def chunk_with_legacy_fallback(
    path: Path,
    *,
    max_tokens: int = 512,
    max_docx_paragraphs: int = 10_000,
) -> tuple[StagingChunkCandidate, ...]:
    """Use the existing deterministic chunker and add explicit DOCX tables."""

    _validate_max_tokens(max_tokens)
    if (
        isinstance(max_docx_paragraphs, bool)
        or not isinstance(max_docx_paragraphs, int)
        or max_docx_paragraphs <= 0
    ):
        raise ValueError("max_docx_paragraphs must be a positive integer")
    source_path = Path(path)
    try:
        document = extract_with_fallback(
            source_path,
            limits=DocumentResourceLimits(
                max_file_bytes=64 * 1024 * 1024,
                max_docx_paragraphs=max_docx_paragraphs,
                max_docx_uncompressed_bytes=64 * 1024 * 1024,
            ),
        )
    except DoclingUnavailableError:
        document = StructuredDocument(
            source=_fingerprint(source_path),
            elements=(),
            parser_backend="legacy",
            tokenizer_backend="none",
        )

    elements = list(document.elements)
    if source_path.suffix.casefold() == ".docx":
        elements.extend(_extract_table_elements(source_path, document.source))
    if not elements:
        raise DoclingUnavailableError("legacy extraction produced no elements")

    # The existing project chunker remains the baseline: its sentence-boundary
    # and overlap behavior is preserved before the staging token cap is applied.
    extracted = ExtractedDocument(
        file_name=source_path.name,
        blocks=tuple(
            ExtractedBlock(
                text=element.text,
                source=SourceRef(
                    file_name=source_path.name,
                    heading_path=element.heading_path,
                    page=element.page,
                    paragraph_start=element.paragraph_start,
                    paragraph_end=element.paragraph_end,
                    table_id=element.table_id,
                ),
            )
            for element in elements
            if element.text.strip()
        ),
    )
    baseline_chunks = chunk_document(extracted, target_chars=800, overlap_chars=120)
    return _candidates_from_chunks(
        source=document.source,
        parser_backend="legacy",
        chunks=baseline_chunks,
        max_tokens=max_tokens,
    )


def compare_coverage(
    audit: DocxStructureAudit,
    *,
    docling_chunks: tuple[StagingChunkCandidate, ...],
    fallback_chunks: tuple[StagingChunkCandidate, ...],
) -> dict[str, object]:
    """Compare candidate coverage using counts only, never returning text."""

    expected_chars = audit.paragraph_chars + audit.table_chars + audit.textbox_chars
    if expected_chars <= 0:
        expected_chars = audit.extracted_chars
    docling_chars = sum(len(candidate.text) for candidate in docling_chunks)
    fallback_chars = sum(len(candidate.text) for candidate in fallback_chunks)
    reasons: list[str] = []
    if not docling_chunks:
        reasons.append("docling_no_chunks")
    if expected_chars and docling_chars < expected_chars:
        reasons.append("docling_coverage_below_audit")
    if fallback_chunks and expected_chars and fallback_chars < expected_chars:
        reasons.append("fallback_coverage_below_audit")
    if audit.coverage_status == "blocked" and expected_chars and docling_chars < expected_chars:
        reasons.extend(reason for reason in audit.blocking_reasons if reason not in reasons)
    return {
        "status": "blocked" if reasons else "pass",
        "expected_chars": expected_chars,
        "docling_chars": docling_chars,
        "fallback_chars": fallback_chars,
        "docling_chunk_count": len(docling_chunks),
        "fallback_chunk_count": len(fallback_chunks),
        "available_backends": [
            backend
            for backend, candidates in (
                ("docling", docling_chunks),
                ("legacy", fallback_chunks),
            )
            if candidates
        ],
        "blocking_reasons": reasons,
    }


def validate_chunk_coverage(report: dict[str, object]) -> None:
    """Raise before active staging when the coverage report is blocked."""

    if report.get("status") != "pass":
        reasons = report.get("blocking_reasons") or ("unknown_coverage_failure",)
        raise ValueError(f"chunk coverage is blocked: {', '.join(map(str, reasons))}")


def choose_active_candidate(
    report: dict[str, object],
    *,
    approved_backend: str | None = None,
) -> str:
    """Require an explicit backend choice; never silently promote a candidate."""

    validate_chunk_coverage(report)
    available = tuple(str(item) for item in report.get("available_backends", ()))
    if not available:
        raise ValueError("no chunk backend is available")
    if approved_backend is None:
        raise ValueError("active backend requires explicit approval")
    if approved_backend not in available:
        raise ValueError("approved backend is unavailable")
    return approved_backend


def _chunk_structured_document(
    document: StructuredDocument,
    *,
    max_tokens: int,
) -> tuple[StagingChunkCandidate, ...]:
    candidates: list[StagingChunkCandidate] = []
    ordinal = 0
    for element in document.elements:
        for text in _split_element_text(element, max_tokens=max_tokens):
            source = SourceRef(
                file_name=document.source.relative_path,
                heading_path=element.heading_path or ("正文",),
                page=element.page,
                paragraph_start=element.paragraph_start,
                paragraph_end=element.paragraph_end,
                table_id=element.table_id,
            )
            candidates.append(
                _candidate(
                    source_file=document.source,
                    source=source,
                    text=text,
                    content_kind=element.content_kind,
                    provenance_digest=element.provenance_digest,
                    parser_backend=document.parser_backend,
                    ordinal=ordinal,
                    max_tokens=max_tokens,
                    token_count=element.token_count
                    if text == element.text.strip()
                    else None,
                )
            )
            ordinal += 1
    return tuple(candidates)


def _candidates_from_chunks(
    *,
    source: SourceFileFingerprint,
    parser_backend: str,
    chunks: list[Chunk],
    max_tokens: int,
) -> tuple[StagingChunkCandidate, ...]:
    candidates: list[StagingChunkCandidate] = []
    ordinal = 0
    for chunk in chunks:
        for text in _split_text(chunk.text, max_tokens=max_tokens):
            source_ref = chunk.source
            kind = "table" if source_ref.heading_path and source_ref.heading_path[0] == "表格" else "text"
            candidates.append(
                _candidate(
                    source_file=source,
                    source=source_ref,
                    text=text,
                    content_kind=kind,
                    provenance_digest=_digest(
                        f"{source_ref.file_name}:{source_ref.heading_path}:"
                        f"{source_ref.paragraph_start}:{source_ref.paragraph_end}"
                    ),
                    parser_backend=parser_backend,
                    ordinal=ordinal,
                    max_tokens=max_tokens,
                )
            )
            ordinal += 1
    return tuple(candidates)


def _candidate(
    *,
    source_file: SourceFileFingerprint,
    source: SourceRef,
    text: str,
    content_kind: str,
    provenance_digest: str,
    parser_backend: str,
    ordinal: int,
    max_tokens: int,
    token_count: int | None = None,
) -> StagingChunkCandidate:
    clean = text.strip()
    if not clean:
        raise ValueError("staging chunks must not be empty")
    measured_tokens = (
        token_count
        if isinstance(token_count, int)
        and not isinstance(token_count, bool)
        and token_count > 0
        else _estimate_tokens(clean)
    )
    if measured_tokens > max_tokens:
        raise ValueError("staging chunk exceeds token budget")
    content_hash = _digest(clean)
    chunk_id = _digest(
        f"2026-standard-manual-v1\n{source_file.sha256}\n{provenance_digest}\n"
        f"{ordinal}\n{content_hash}"
    )
    record = StagingChunkRecord(
        chunk_id=chunk_id,
        source_version="2026-standard-manual-v1",
        source_relative_path=source_file.relative_path,
        source_sha256=source_file.sha256,
        heading_path=source.heading_path or ("正文",),
        content_kind=content_kind,
        provenance_digest=provenance_digest,
        content_hash=content_hash,
        token_count=measured_tokens,
        parser_backend=parser_backend,
        ordinal=ordinal,
    )
    return StagingChunkCandidate(record=record, text=clean, source=source)


def _split_element_text(element: StructuredElement, *, max_tokens: int) -> tuple[str, ...]:
    text = element.text.strip()
    measured_tokens = element.token_count or _estimate_tokens(text)
    if measured_tokens <= max_tokens:
        return (text,)
    if element.content_kind.casefold() == "table":
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        header = lines[0] if lines else ""
        return _split_table_lines(lines, header=header, max_tokens=max_tokens)
    return _split_text(text, max_tokens=max_tokens)


def _split_table_lines(lines: list[str], *, header: str, max_tokens: int) -> tuple[str, ...]:
    if not lines:
        return ()
    if not header:
        return _split_text("\n".join(lines), max_tokens=max_tokens)
    if _estimate_tokens(header) > max_tokens:
        # An oversized header cannot be repeated without violating the hard
        # cap.  Keep it visible in deterministic pieces and let the report
        # expose the overflow for human review.
        return tuple(
            header[index : index + max_tokens]
            for index in range(0, len(header), max_tokens)
        )

    chunks: list[str] = [header]
    available = max_tokens - len(header) - 1
    if available <= 0:
        # Preserve the header as its own chunk; rows are kept separately and
        # remain locatable even though the header cannot fit beside them.
        chunks.extend(
            line[index : index + max_tokens]
            for line in lines[1:]
            for index in range(0, len(line), max_tokens)
        )
        return tuple(chunk for chunk in chunks if chunk.strip())

    for line in lines[1:]:
        for index in range(0, len(line), available):
            piece = line[index : index + available]
            chunks.append(f"{header}\n{piece}")
    return tuple(chunk for chunk in chunks if chunk.strip())


def _split_text(text: str, *, max_tokens: int) -> tuple[str, ...]:
    pieces: list[str] = []
    current = ""
    for sentence in filter(None, _BOUNDARIES.split(text)):
        if _estimate_tokens(sentence) > max_tokens:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(
                sentence[index : index + max_tokens]
                for index in range(0, len(sentence), max_tokens)
            )
        elif current and _estimate_tokens(f"{current}{sentence}") > max_tokens:
            pieces.append(current)
            current = sentence
        else:
            current += sentence
    if current:
        pieces.append(current)
    return tuple(piece.strip() for piece in pieces if piece.strip())


def _extract_table_elements(
    path: Path,
    source: SourceFileFingerprint,
) -> list[StructuredElement]:
    try:
        document = Document(path)
    except Exception as error:
        raise DoclingUnavailableError("legacy table extraction failed") from error
    elements: list[StructuredElement] = []
    for index, table in enumerate(document.tables, start=1):
        rows: list[str] = []
        for row in table.rows:
            cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            if any(cells):
                rows.append(" | ".join(cells))
        if not rows:
            continue
        elements.append(
            StructuredElement(
                text="\n".join(rows),
                heading_path=("表格", f"表{index}"),
                content_kind="table",
                provenance_digest=_digest(f"{source.sha256}:table:{index}"),
                table_id=f"table-{index}",
            )
        )
    return elements


def _fingerprint(path: Path) -> SourceFileFingerprint:
    data = path.read_bytes()
    try:
        modified_ns = path.stat().st_mtime_ns
    except OSError:
        modified_ns = 0
    return SourceFileFingerprint(
        relative_path=path.name,
        size_bytes=len(data),
        modified_ns=modified_ns,
        sha256=sha256(data).hexdigest(),
    )


def _estimate_tokens(text: str) -> int:
    # A deterministic upper-bound estimator is used for the fallback and for
    # validating fake Docling output.  The real HybridChunker enforces the
    # model tokenizer's limit before this adapter sees its chunks.
    return max(1, len(text))


def _validate_max_tokens(max_tokens: int) -> None:
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= _MAX_TOKENS:
        raise ValueError(f"max_tokens must be between 1 and {_MAX_TOKENS}")


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "StagingChunkCandidate",
    "choose_active_candidate",
    "chunk_with_docling",
    "chunk_with_legacy_fallback",
    "compare_coverage",
    "validate_chunk_coverage",
]
