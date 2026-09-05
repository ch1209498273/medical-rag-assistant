"""Deterministic, source-preserving policy text chunking."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from hashlib import sha256

from app.domain.models import Chunk, ExtractedBlock, ExtractedDocument, SourceRef

_SENTENCE_BOUNDARIES = re.compile(r"(?<=[。；\n])")


def chunk_document(
    document: ExtractedDocument,
    target_chars: int = 800,
    overlap_chars: int = 120,
) -> list[Chunk]:
    """Create stable chunks while keeping headings and source ranges locatable."""

    if target_chars <= 0:
        raise ValueError("target_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= target_chars:
        raise ValueError("overlap_chars must be between zero and target_chars")

    chunks: list[Chunk] = []
    current_text = ""
    current_source: SourceRef | None = None
    current_has_new_text = False
    last_piece_source: SourceRef | None = None
    pending_overlap = ""
    pending_overlap_source: SourceRef | None = None

    def flush() -> None:
        nonlocal current_has_new_text, pending_overlap, pending_overlap_source
        if not current_has_new_text or current_source is None:
            return
        chunks.append(_make_chunk(document.file_name, current_text, current_source))
        pending_overlap = current_text[-overlap_chars:] if overlap_chars else ""
        pending_overlap_source = last_piece_source if pending_overlap else None
        current_has_new_text = False

    def reset_current() -> None:
        nonlocal current_text, current_source, current_has_new_text
        current_text = ""
        current_source = None
        current_has_new_text = False

    previous_heading: tuple[str, ...] | None = None
    for block in document.blocks:
        if previous_heading is not None and block.source.heading_path != previous_heading:
            flush()
            reset_current()
            pending_overlap = ""
            pending_overlap_source = None
        previous_heading = block.source.heading_path

        for piece in _pieces_for_block(block, target_chars):
            remaining_text = piece.text
            while remaining_text:
                if not current_text and pending_overlap_source is not None:
                    if pending_overlap_source.heading_path == piece.source.heading_path:
                        current_text = pending_overlap
                        current_source = pending_overlap_source
                    pending_overlap = ""
                    pending_overlap_source = None

                if current_has_new_text and len(current_text) + len(remaining_text) > target_chars:
                    flush()
                    reset_current()
                    continue

                available_chars = target_chars - len(current_text)
                text_to_add = remaining_text[:available_chars]
                current_text += text_to_add
                current_source = (
                    piece.source
                    if current_source is None
                    else _merge_source_refs(current_source, piece.source)
                )
                current_has_new_text = True
                last_piece_source = piece.source
                remaining_text = remaining_text[len(text_to_add) :]

    flush()
    return chunks


def _pieces_for_block(block: ExtractedBlock, target_chars: int) -> list[ExtractedBlock]:
    if len(block.text) <= target_chars:
        return [block]
    return [
        ExtractedBlock(text=text, source=block.source)
        for text in _split_at_sentence_boundaries(block.text, target_chars)
    ]


def _split_at_sentence_boundaries(text: str, target_chars: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for sentence in filter(None, _SENTENCE_BOUNDARIES.split(text)):
        if len(sentence) > target_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(
                sentence[index : index + target_chars]
                for index in range(0, len(sentence), target_chars)
            )
        elif current and len(current) + len(sentence) > target_chars:
            pieces.append(current)
            current = sentence
        else:
            current += sentence
    if current:
        pieces.append(current)
    return pieces


def _merge_source_refs(first: SourceRef, last: SourceRef) -> SourceRef:
    if first.file_name != last.file_name or first.heading_path != last.heading_path:
        raise ValueError("chunks may only merge one document heading at a time")
    return SourceRef(
        file_name=first.file_name,
        heading_path=first.heading_path,
        page=_minimum(first.page, last.page),
        page_end=_maximum(first.page_end or first.page, last.page_end or last.page),
        paragraph_start=_minimum(first.paragraph_start, last.paragraph_start),
        paragraph_end=_maximum(
            first.paragraph_end or first.paragraph_start,
            last.paragraph_end or last.paragraph_start,
        ),
        table_id=first.table_id if first.table_id == last.table_id else None,
    )


def _minimum(first: int | None, last: int | None) -> int | None:
    return min(value for value in (first, last) if value is not None) if any(
        value is not None for value in (first, last)
    ) else None


def _maximum(first: int | None, last: int | None) -> int | None:
    return max(value for value in (first, last) if value is not None) if any(
        value is not None for value in (first, last)
    ) else None


def _make_chunk(file_name: str, text: str, source: SourceRef) -> Chunk:
    source_key = json.dumps(asdict(source), ensure_ascii=False, sort_keys=True)
    chunk_key = f"{file_name}\n{source_key}\n{text}".encode()
    return Chunk(
        chunk_id=sha256(chunk_key).hexdigest(),
        text=text,
        source=source,
        content_hash=sha256(text.encode("utf-8")).hexdigest(),
    )
