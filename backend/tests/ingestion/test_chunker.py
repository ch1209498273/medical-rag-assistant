from hashlib import sha256
from itertools import pairwise

from app.domain.models import ExtractedBlock, ExtractedDocument, SourceRef
from app.ingestion.chunker import chunk_document


def _block(text: str, paragraph: int, heading: tuple[str, ...] = ("虚构制度",)) -> ExtractedBlock:
    return ExtractedBlock(
        text=text,
        source=SourceRef(
            file_name="制度.docx",
            heading_path=heading,
            paragraph_start=paragraph,
            paragraph_end=paragraph,
        ),
    )


def test_chunker_preserves_source_range_and_plain_text_overlap() -> None:
    """Breaks if overlap or the cited paragraph range is lost at a chunk break."""

    document = ExtractedDocument(
        file_name="制度.docx",
        blocks=(_block("甲" * 60, 2), _block("乙" * 60, 3)),
    )

    chunks = chunk_document(document, target_chars=80, overlap_chars=20)

    assert len(chunks) == 2
    assert chunks[0].source.file_name == "制度.docx"
    assert chunks[0].source.paragraph_start == 2
    assert chunks[0].source.paragraph_end == 2
    assert chunks[0].text[-20:] in chunks[1].text
    assert chunks[1].source.paragraph_start == 2
    assert chunks[1].source.paragraph_end == 3


def test_chunker_starts_a_new_chunk_when_heading_changes() -> None:
    """Breaks if sections with distinct headings are merged into one citation."""

    document = ExtractedDocument(
        file_name="制度.docx",
        blocks=(
            _block("第一章的虚构条款。", 2, ("第一章",)),
            _block("第二章的虚构条款。", 3, ("第二章",)),
        ),
    )

    chunks = chunk_document(document)

    assert [chunk.source.heading_path for chunk in chunks] == [("第一章",), ("第二章",)]
    assert [chunk.source.paragraph_start for chunk in chunks] == [2, 3]


def test_chunker_splits_long_blocks_at_chinese_sentence_boundaries() -> None:
    """Breaks if a long clause is split through a sentence despite a boundary."""

    document = ExtractedDocument(
        file_name="制度.docx",
        blocks=(_block("甲" * 45 + "。" + "乙" * 45 + "；" + "丙" * 45 + "。", 7),),
    )

    chunks = chunk_document(document, target_chars=55, overlap_chars=0)

    assert [chunk.text for chunk in chunks] == ["甲" * 45 + "。", "乙" * 45 + "；", "丙" * 45 + "。"]
    assert all(chunk.source.paragraph_start == 7 for chunk in chunks)


def test_chunker_keeps_hard_split_chunks_within_target_after_overlap() -> None:
    """Breaks if overlap makes a no-boundary chunk exceed the configured size."""

    document = ExtractedDocument(
        file_name="制度.docx",
        blocks=(_block("".join(chr(0x4E00 + index) for index in range(200)), 11),),
    )

    chunks = chunk_document(document, target_chars=100, overlap_chars=20)

    assert all(len(chunk.text) <= 100 for chunk in chunks)
    assert all(
        following.text.startswith(previous.text[-20:])
        for previous, following in pairwise(chunks)
    )
    assert all(chunk.source.file_name == "制度.docx" for chunk in chunks)
    assert all(chunk.source.paragraph_start == 11 for chunk in chunks)
    assert all(chunk.source.paragraph_end == 11 for chunk in chunks)


def test_chunk_ids_and_content_hashes_are_deterministic() -> None:
    """Breaks if stable document input receives unstable vector-store identifiers."""

    document = ExtractedDocument(
        file_name="制度.docx",
        blocks=(_block("可重复切分的虚构条款。", 2),),
    )

    first = chunk_document(document)
    second = chunk_document(document)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert first[0].content_hash == sha256(first[0].text.encode("utf-8")).hexdigest()
