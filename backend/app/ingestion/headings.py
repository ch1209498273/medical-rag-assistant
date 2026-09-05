"""Conservative, explicit heading signals for local document extraction."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Literal
from xml.etree import ElementTree

HeadingMode = Literal["legacy", "conservative"]
HeadingPath = tuple[str, ...]
FALLBACK_HEADING_PATH: HeadingPath = ("正文",)
MAX_HEADING_CHARS = 200

_WORD_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_HEADING_STYLE = re.compile(r"^heading ?([1-9])$", re.IGNORECASE)


@dataclass(frozen=True)
class PdfHeadingResolution:
    """Page-level heading paths and safe aggregate diagnostics."""

    paths_by_page: tuple[HeadingPath, ...]
    valid_entry_count: int
    invalid_entry_count: int
    ambiguous_page_count: int


@dataclass(frozen=True)
class DocxXmlParagraph:
    """One non-empty XML paragraph recovered in deterministic order."""

    element: ElementTree.Element
    text: str
    ordinal: int
    heading_level: int | None


def normalise_heading_title(value: object, *, max_chars: int = MAX_HEADING_CHARS) -> str | None:
    """Return a bounded, whitespace-normalised title or None."""

    if not isinstance(value, str):
        return None
    if isinstance(max_chars, bool) or max_chars <= 0:
        raise ValueError("max_chars must be positive")
    normalised = unicodedata.normalize("NFKC", value)
    normalised = " ".join(normalised.split())
    if not normalised or len(normalised) > max_chars:
        return None
    if any(unicodedata.category(char).startswith("C") for char in normalised):
        return None
    return normalised


def update_heading_path(
    current: Sequence[str],
    level: int,
    title: object,
) -> HeadingPath | None:
    """Apply one explicit heading to a known path without inventing parents."""

    if isinstance(level, bool) or not isinstance(level, int) or not 1 <= level <= 9:
        return None
    normalised = normalise_heading_title(title)
    if normalised is None:
        return None
    known = tuple(item for item in current if isinstance(item, str) and item)
    effective_level = min(level, len(known) + 1)
    return (*known[: effective_level - 1], normalised)


def resolve_pdf_toc(toc: object, *, page_count: int) -> PdfHeadingResolution:
    """Map explicit PDF outline entries to pages, failing closed on ambiguity."""

    if isinstance(page_count, bool) or not isinstance(page_count, int) or page_count < 0:
        raise ValueError("page_count must be a non-negative integer")
    paths: list[HeadingPath] = [FALLBACK_HEADING_PATH] * page_count
    valid_entries: list[tuple[int, int, str, int]] = []
    invalid_entry_count = 0
    if isinstance(toc, Sequence) and not isinstance(toc, (str, bytes, bytearray)):
        for ordinal, entry in enumerate(toc):
            if not isinstance(entry, Sequence) or isinstance(entry, (str, bytes, bytearray)):
                invalid_entry_count += 1
                continue
            if len(entry) < 3:
                invalid_entry_count += 1
                continue
            level, title, page = entry[0], entry[1], entry[2]
            normalised_title = normalise_heading_title(title)
            if (
                isinstance(level, bool)
                or not isinstance(level, int)
                or not 1 <= level <= 9
                or isinstance(page, bool)
                or not isinstance(page, int)
                or not 1 <= page <= page_count
                or normalised_title is None
            ):
                invalid_entry_count += 1
                continue
            valid_entries.append((ordinal, level, normalised_title, page))

    ordered_entries = sorted(valid_entries, key=lambda item: (item[3], item[0]))
    entries_by_page: dict[int, list[HeadingPath]] = {}
    current: HeadingPath = ()
    paths_by_entry: list[tuple[int, HeadingPath]] = []
    for _ordinal, level, title, page in ordered_entries:
        updated = update_heading_path(current, level, title)
        if updated is None:
            invalid_entry_count += 1
            continue
        current = updated
        paths_by_entry.append((page, current))
        entries_by_page.setdefault(page, []).append(current)

    ambiguous_page_count = sum(
        1 for page_entries in entries_by_page.values() if len(page_entries) > 1
    )
    current_path: HeadingPath = ()
    entry_index = 0
    for page in range(1, page_count + 1):
        while entry_index < len(paths_by_entry) and paths_by_entry[entry_index][0] <= page:
            current_path = paths_by_entry[entry_index][1]
            entry_index += 1
        page_entries = entries_by_page.get(page, ())
        if len(page_entries) > 1:
            paths[page - 1] = FALLBACK_HEADING_PATH
        elif current_path:
            paths[page - 1] = current_path
    return PdfHeadingResolution(
        paths_by_page=tuple(paths),
        valid_entry_count=len(paths_by_entry),
        invalid_entry_count=invalid_entry_count,
        ambiguous_page_count=ambiguous_page_count,
    )


def docx_heading_level(paragraph: ElementTree.Element) -> int | None:
    """Return an explicit DOCX heading level, never a visual-style guess."""

    paragraph_properties = next(
        (
            child
            for child in paragraph
            if _local_name(child.tag) == "pPr"
        ),
        None,
    )
    if paragraph_properties is None:
        return None

    outline = next(
        (
            child
            for child in paragraph_properties
            if _local_name(child.tag) == "outlineLvl"
        ),
        None,
    )
    if outline is not None:
        raw_value = _attribute(outline, "val")
        try:
            value = int(raw_value) if raw_value is not None else -1
        except (TypeError, ValueError):
            value = -1
        if 0 <= value <= 8:
            return value + 1

    style = next(
        (
            child
            for child in paragraph_properties
            if _local_name(child.tag) == "pStyle"
        ),
        None,
    )
    if style is None:
        return None
    match = _HEADING_STYLE.fullmatch(_attribute(style, "val") or "")
    return int(match.group(1)) if match else None


def iter_docx_xml_paragraphs(root: ElementTree.Element) -> Iterator[DocxXmlParagraph]:
    """Yield non-empty paragraphs while preserving textbox traversal semantics."""

    ordinal = 0

    def walk(element: ElementTree.Element) -> Iterator[DocxXmlParagraph]:
        nonlocal ordinal
        local_name = _local_name(element.tag)
        if local_name in {"Fallback", "del"}:
            return
        if local_name == "p":
            text = _paragraph_own_text(element)
            if text:
                ordinal += 1
                yield DocxXmlParagraph(
                    element=element,
                    text=text,
                    ordinal=ordinal,
                    heading_level=docx_heading_level(element),
                )
        for child in element:
            yield from walk(child)

    yield from walk(root)


def _paragraph_own_text(paragraph: ElementTree.Element) -> str:
    text_parts: list[str] = []

    def collect(element: ElementTree.Element, *, root: bool = False) -> None:
        local_name = _local_name(element.tag)
        if local_name in {"Fallback", "del"}:
            return
        if not root and local_name == "p":
            return
        if local_name == "t" and element.text:
            text_parts.append(element.text)
        for child in element:
            collect(child)

    collect(paragraph, root=True)
    return "".join(text_parts).strip()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _attribute(element: ElementTree.Element, name: str) -> str | None:
    return element.get(f"{{{_WORD_NAMESPACE}}}{name}") or element.get(name)
