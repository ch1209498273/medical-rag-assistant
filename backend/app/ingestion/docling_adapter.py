"""Optional Docling adapter for the isolated Task 15 staging pipeline.

Docling is intentionally optional.  Importing this module must not download a
model or change the production ingestion path.  Callers can inject converter,
chunker and tokenizer fakes for deterministic tests.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

from app.domain.models import ExtractedDocument
from app.ingestion.extractors import DocumentResourceLimits, extract_document
from app.ingestion.task15_models import SourceFileFingerprint


class DoclingUnavailableError(RuntimeError):
    """Docling was unavailable or could not produce a usable document."""


class LocalTokenizerUnavailableError(RuntimeError):
    """A tokenizer was not available locally without a network fetch."""


@dataclass(frozen=True, slots=True)
class StructuredElement:
    """A normalized text/table element emitted by Docling or the fallback."""

    text: str
    heading_path: tuple[str, ...]
    content_kind: str
    provenance_digest: str
    page: int | None = None
    paragraph_start: int | None = None
    paragraph_end: int | None = None
    table_id: str | None = None
    token_count: int | None = None


@dataclass(frozen=True, slots=True)
class StructuredDocument:
    """Parser-neutral structured document used by the staging chunker."""

    source: SourceFileFingerprint
    elements: tuple[StructuredElement, ...]
    parser_backend: str
    tokenizer_backend: str


class DoclingAdapter:
    """Convert a DOCX through Docling without implicit network access."""

    def __init__(
        self,
        *,
        tokenizer_name: str,
        tokenizer_path: Path | None = None,
        tokenizer: object | None = None,
        converter_factory: Callable[[], object] | None = None,
        chunker_factory: Callable[..., object] | None = None,
    ) -> None:
        if not isinstance(tokenizer_name, str) or not tokenizer_name.strip():
            raise ValueError("tokenizer_name must be a non-empty string")
        self._tokenizer_name = tokenizer_name.strip()
        self._tokenizer_path = tokenizer_path
        self._tokenizer = tokenizer
        self._converter_factory = converter_factory
        self._chunker_factory = chunker_factory

    def convert(
        self,
        path: Path,
        *,
        max_tokens: int = 512,
        repeat_table_header: bool = True,
        merge_peers: bool = True,
        omit_header_on_overflow: bool = False,
    ) -> StructuredDocument:
        """Return normalized hierarchy elements from one DOCX."""

        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        for name, value in (
            ("repeat_table_header", repeat_table_header),
            ("merge_peers", merge_peers),
            ("omit_header_on_overflow", omit_header_on_overflow),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean")
        source_path = Path(path)
        if not source_path.is_file():
            raise DoclingUnavailableError("source DOCX is unavailable")

        tokenizer = self._tokenizer or resolve_local_tokenizer(
            tokenizer_name=self._tokenizer_name,
            tokenizer_path=self._tokenizer_path,
        )
        try:
            converter = self._build_converter()
            converted = _call_converter(converter, source_path)
            docling_document = getattr(converted, "document", converted)
            chunker = self._build_chunker(
                tokenizer=tokenizer,
                max_tokens=max_tokens,
                repeat_table_header=repeat_table_header,
                merge_peers=merge_peers,
                omit_header_on_overflow=omit_header_on_overflow,
            )
            raw_chunks = chunker.chunk(docling_document)
            count_tokens = getattr(
                getattr(chunker, "tokenizer", None), "count_tokens", None
            )
            elements = tuple(
                _normalise_chunk(
                    raw_chunk,
                    ordinal=index,
                    count_tokens=count_tokens if callable(count_tokens) else None,
                )
                for index, raw_chunk in enumerate(raw_chunks)
            )
            elements = _enrich_docx_paragraph_locations(source_path, elements)
        except DoclingUnavailableError:
            raise
        except Exception as error:
            raise DoclingUnavailableError("Docling conversion failed") from error
        if not elements:
            raise DoclingUnavailableError("Docling produced no chunks")
        return StructuredDocument(
            source=_fingerprint(source_path),
            elements=elements,
            parser_backend="docling",
            tokenizer_backend=self._tokenizer_name,
        )

    def _build_converter(self) -> object:
        if self._converter_factory is not None:
            return self._converter_factory()
        try:
            from docling.document_converter import DocumentConverter
        except Exception as error:
            raise DoclingUnavailableError("Docling is unavailable") from error
        return DocumentConverter()

    def _build_chunker(self, **options: object) -> object:
        if self._chunker_factory is not None:
            return self._chunker_factory(**options)
        try:
            from docling.chunking import HybridChunker
        except Exception as error:
            raise DoclingUnavailableError("Docling HybridChunker is unavailable") from error
        accepted = inspect.signature(HybridChunker).parameters
        kwargs = {name: value for name, value in options.items() if name in accepted}
        if "tokenizer" in kwargs:
            max_tokens = options.get("max_tokens")
            if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
                raise DoclingUnavailableError("Docling tokenizer budget is invalid")
            try:
                kwargs["tokenizer"] = _as_docling_tokenizer(
                    kwargs["tokenizer"],
                    max_tokens=max_tokens,
                    tokenizer_name=self._tokenizer_name,
                )
            except DoclingUnavailableError:
                raise
            except Exception as error:
                raise DoclingUnavailableError(
                    "Docling tokenizer configuration failed"
                ) from error
        try:
            return HybridChunker(**kwargs)
        except Exception as error:
            raise DoclingUnavailableError("Docling HybridChunker configuration failed") from error


def resolve_local_tokenizer(
    *, tokenizer_name: str, tokenizer_path: Path | None = None
) -> object:
    """Resolve a tokenizer using local files only; never trigger a download."""

    if not isinstance(tokenizer_name, str) or not tokenizer_name.strip():
        raise ValueError("tokenizer_name must be a non-empty string")
    if tokenizer_path is not None and not Path(tokenizer_path).exists():
        raise LocalTokenizerUnavailableError("local tokenizer path does not exist")

    source: str | Path = Path(tokenizer_path) if tokenizer_path is not None else tokenizer_name
    if tokenizer_name == "cl100k_base" and tokenizer_path is None:
        try:
            import tiktoken

            return tiktoken.get_encoding("cl100k_base")
        except Exception as error:
            raise LocalTokenizerUnavailableError("local tokenizer is unavailable") from error

    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            source,
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as error:
        raise LocalTokenizerUnavailableError("local tokenizer is unavailable") from error


def _as_docling_tokenizer(
    tokenizer: object,
    *,
    max_tokens: int,
    tokenizer_name: str,
) -> object:
    """Adapt a local tokenizer object to Docling's versioned BaseTokenizer API.

    Docling 2.x reads the chunk budget from ``BaseTokenizer.get_max_tokens``;
    passing ``max_tokens`` directly to ``HybridChunker`` is ignored by newer
    Pydantic-based releases.  Keep the adapter local so the rest of the app
    never depends on Docling's tokenizer classes.
    """

    try:
        from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
    except Exception as error:
        raise DoclingUnavailableError("Docling tokenizer API is unavailable") from error
    if isinstance(tokenizer, BaseTokenizer):
        if tokenizer.get_max_tokens() != max_tokens:
            raise DoclingUnavailableError(
                "provided Docling tokenizer has a different token budget"
            )
        return tokenizer
    try:
        if tokenizer_name == "cl100k_base":
            from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

            return OpenAITokenizer(tokenizer=tokenizer, max_tokens=max_tokens)
        from docling_core.transforms.chunker.tokenizer.huggingface import (
            HuggingFaceTokenizer,
        )

        return HuggingFaceTokenizer(tokenizer=tokenizer, max_tokens=max_tokens)
    except Exception as error:
        raise DoclingUnavailableError("Docling tokenizer wrapper is unavailable") from error


def extract_with_fallback(
    path: Path,
    *,
    limits: DocumentResourceLimits | None = None,
) -> StructuredDocument:
    """Map the existing conservative extractor into the adapter contract."""

    source_path = Path(path)
    try:
        data = source_path.read_bytes()
        extracted: ExtractedDocument = extract_document(
            source_path,
            data=data,
            limits=limits
            or DocumentResourceLimits(
                max_file_bytes=64 * 1024 * 1024,
                max_docx_uncompressed_bytes=64 * 1024 * 1024,
            ),
            heading_mode="conservative",
        )
    except Exception as error:
        raise DoclingUnavailableError("legacy extraction failed") from error
    elements = tuple(
        StructuredElement(
            text=block.text,
            heading_path=block.source.heading_path or ("正文",),
            content_kind="text",
            provenance_digest=_digest(
                f"{block.source.file_name}:{block.source.paragraph_start}:"
                f"{block.source.paragraph_end}:{block.source.heading_path}"
            ),
            page=block.source.page,
            paragraph_start=block.source.paragraph_start,
            paragraph_end=block.source.paragraph_end,
        )
        for block in extracted.blocks
        if block.text.strip()
    )
    if not elements:
        raise DoclingUnavailableError("legacy extraction produced no elements")
    return StructuredDocument(
        source=_fingerprint(source_path),
        elements=elements,
        parser_backend="legacy",
        tokenizer_backend="none",
    )


def _call_converter(converter: object, path: Path) -> object:
    convert = getattr(converter, "convert", None)
    if not callable(convert):
        raise DoclingUnavailableError("Docling converter has no convert method")
    try:
        return convert(str(path))
    except TypeError:
        return convert(source=str(path))


def _normalise_chunk(
    raw_chunk: object,
    *,
    ordinal: int,
    count_tokens: Callable[[str], int] | None = None,
) -> StructuredElement:
    text = getattr(raw_chunk, "text", raw_chunk if isinstance(raw_chunk, str) else None)
    if not isinstance(text, str) or not text.strip():
        raise DoclingUnavailableError(f"Docling chunk {ordinal} is empty")
    metadata = getattr(raw_chunk, "meta", None)
    heading_path = _heading_path(raw_chunk, metadata)
    content_kind = _content_kind(raw_chunk, metadata)
    page = _int_attr(raw_chunk, metadata, "page")
    paragraph_start = _int_attr(raw_chunk, metadata, "paragraph_start")
    paragraph_end = _int_attr(raw_chunk, metadata, "paragraph_end")
    raw_table_id = _text_attr(raw_chunk, metadata, "table_id")
    table_id = raw_table_id
    if table_id is None:
        table_id = _table_id_from_doc_items(metadata)
    # Keep the compatibility identity independent of derived locators.  A
    # metadata repair must not invalidate already-built vector identities.
    provenance = _text_attr(raw_chunk, metadata, "provenance") or (
        f"{ordinal}:{heading_path}:None:None:None:{raw_table_id}"
    )
    return StructuredElement(
        text=text.strip(),
        heading_path=heading_path,
        content_kind=content_kind,
        provenance_digest=_digest(provenance),
        page=page,
        paragraph_start=paragraph_start,
        paragraph_end=paragraph_end,
        table_id=table_id,
        token_count=_safe_token_count(text, count_tokens),
    )


def _safe_token_count(
    text: str,
    count_tokens: Callable[[str], int] | None,
) -> int | None:
    if count_tokens is None:
        return None
    try:
        value = count_tokens(text)
    except Exception:  # noqa: BLE001 - optional tokenizer implementations vary
        return None
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _content_kind(raw: object, metadata: object) -> str:
    explicit = _text_attr(raw, metadata, "content_kind")
    if explicit:
        return explicit
    doc_items = getattr(metadata, "doc_items", None)
    if isinstance(doc_items, Iterable):
        for item in doc_items:
            label = getattr(item, "label", None)
            value = getattr(label, "value", label)
            if isinstance(value, str) and value.strip():
                normalized = value.casefold().replace("_", " ")
                if "table" in normalized:
                    return "table"
                if "heading" in normalized or "title" in normalized:
                    return "heading"
                if "list" in normalized:
                    return "list"
    return "text"


def _heading_path(raw: object, metadata: object) -> tuple[str, ...]:
    value = getattr(raw, "heading_path", None)
    if value is None and metadata is not None:
        value = getattr(metadata, "heading_path", None)
    if value is None and metadata is not None:
        value = getattr(metadata, "headings", None)
    if isinstance(value, str):
        value = (value,)
    if isinstance(value, Iterable):
        items = tuple(str(item).strip() for item in value if str(item).strip())
        if items:
            return items
    return ("正文",)


def _text_attr(raw: object, metadata: object, field: str) -> str | None:
    value = getattr(raw, field, None)
    if value is None and metadata is not None:
        value = getattr(metadata, field, None)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _table_id_from_doc_items(metadata: object) -> str | None:
    """Derive a stable table locator from Docling item self references."""

    items = getattr(metadata, "doc_items", None)
    if not isinstance(items, Iterable):
        return None
    for item in items:
        self_ref = getattr(item, "self_ref", None)
        if not isinstance(self_ref, str):
            continue
        match = re.fullmatch(r"#/tables/(\d+)", self_ref.strip())
        if match is not None:
            # Docling self references are zero-based; expose a human-friendly
            # one-based table number in citations.
            return f"table-{int(match.group(1)) + 1}"
    return None


def _enrich_docx_paragraph_locations(
    path: Path, elements: tuple[StructuredElement, ...]
) -> tuple[StructuredElement, ...]:
    """Backfill real DOCX paragraph ranges without inventing page numbers."""

    if path.suffix.casefold() != ".docx" or not elements:
        return elements
    try:
        extracted = extract_document(
            path,
            limits=DocumentResourceLimits(
                max_file_bytes=64 * 1024 * 1024,
                max_docx_paragraphs=100_000,
                max_docx_uncompressed_bytes=256 * 1024 * 1024,
            ),
            heading_mode="conservative",
        )
    except Exception:  # noqa: BLE001 - enrichment must not disable Docling
        return elements
    blocks = tuple(
        block
        for block in extracted.blocks
        if block.text.strip()
        and block.source.paragraph_start is not None
        and block.source.paragraph_end is not None
    )
    if not blocks:
        return elements
    normalised_blocks = tuple(_normalise_locator_text(block.text) for block in blocks)
    joined = "".join(normalised_blocks)
    offsets: list[int] = []
    offset = 0
    for value in normalised_blocks:
        offsets.append(offset)
        offset += len(value)
    cursor = 0
    enriched: list[StructuredElement] = []
    for element in elements:
        if (
            element.content_kind.casefold() == "table"
            or element.paragraph_start is not None
            or element.paragraph_end is not None
        ):
            enriched.append(element)
            continue
        needle = _normalise_locator_text(element.text)
        if not needle:
            enriched.append(element)
            continue
        position = joined.find(needle, cursor)
        if position < 0:
            enriched.append(element)
            continue
        start_index = _offset_index(offsets, position)
        end_index = _offset_index(offsets, position + len(needle) - 1)
        start = blocks[start_index].source.paragraph_start
        end = blocks[end_index].source.paragraph_end
        enriched.append(replace(element, paragraph_start=start, paragraph_end=end))
        cursor = position + len(needle)
    return tuple(enriched)


def _normalise_locator_text(value: str) -> str:
    import unicodedata

    normalised = unicodedata.normalize("NFKC", value).casefold()
    return "".join(
        character
        for character in normalised
        if not character.isspace()
        and unicodedata.category(character)[0] not in {"P", "S"}
    )


def _offset_index(offsets: list[int], position: int) -> int:
    for index in range(len(offsets) - 1, -1, -1):
        if offsets[index] <= position:
            return index
    return 0


def _int_attr(raw: object, metadata: object, field: str) -> int | None:
    value = getattr(raw, field, None)
    if value is None and metadata is not None:
        value = getattr(metadata, field, None)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


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


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "DoclingAdapter",
    "DoclingUnavailableError",
    "LocalTokenizerUnavailableError",
    "StructuredDocument",
    "StructuredElement",
    "extract_with_fallback",
    "resolve_local_tokenizer",
]
