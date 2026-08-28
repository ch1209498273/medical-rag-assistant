"""Integration tests for the local Qdrant vector-store boundary."""

from __future__ import annotations

import pytest
from app.domain.models import Chunk, SourceRef
from app.storage.qdrant import QdrantLocalVectorStore


def make_chunk(text: str, *, chunk_id: str = "chunk-1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        content_hash=f"hash-{chunk_id}",
        source=SourceRef(
            file_name="制度.docx",
            heading_path=("第二章", "值班"),
            paragraph_start=12,
            paragraph_end=14,
        ),
    )


def vector(first: float) -> list[float]:
    return [first] + [0.0] * 1023


@pytest.fixture
def vector_store(tmp_path):
    instance = QdrantLocalVectorStore(tmp_path / "qdrant")
    try:
        yield instance
    finally:
        instance.close()


def test_vector_search_only_returns_active_versions(vector_store):
    """A missing active-version filter could surface a superseded policy."""
    vector_store.upsert("v1", [make_chunk("旧制度", chunk_id="old")], [vector(1.0)])
    vector_store.upsert("v2", [make_chunk("新制度", chunk_id="new")], [vector(1.0)])

    hits = vector_store.search(vector(1.0), {"v2"}, limit=5)

    assert [hit.chunk.text for hit in hits] == ["新制度"]
    assert hits[0].version_id == "v2"


def test_vector_search_with_no_active_versions_is_safely_empty(vector_store):
    """An empty allowlist must never turn into an unfiltered vector query."""
    vector_store.upsert("v1", [make_chunk("旧制度")], [vector(1.0)])

    assert vector_store.search(vector(1.0), set(), limit=5) == []


def test_vector_payload_preserves_locatable_source_fields(vector_store):
    """A payload regression would prevent an answer citation from being checked."""
    chunk = make_chunk("值班人员交接班。")
    vector_store.upsert("v1", [chunk], [vector(1.0)])

    hit = vector_store.search(vector(1.0), {"v1"}, limit=1)[0]

    assert hit.chunk.chunk_id == "chunk-1"
    assert hit.chunk.source.file_name == "制度.docx"
    assert hit.chunk.source.heading_path == ("第二章", "值班")
    assert hit.chunk.source.paragraph_start == 12
    assert hit.chunk.source.paragraph_end == 14


def test_local_qdrant_data_persists_across_restart(tmp_path):
    """A process restart must retain vectors already written to the local store."""
    data_path = tmp_path / "qdrant"
    first = QdrantLocalVectorStore(data_path)
    first.upsert("v1", [make_chunk("可检索制度")], [vector(1.0)])
    first.close()

    restarted = QdrantLocalVectorStore(data_path)
    try:
        hits = restarted.search(vector(1.0), {"v1"}, limit=1)
        assert [hit.chunk.text for hit in hits] == ["可检索制度"]
    finally:
        restarted.close()


def test_rejects_vectors_with_wrong_dimension(vector_store):
    """Wrong embedding dimensions must be rejected before corrupting the collection."""
    with pytest.raises(ValueError, match="1024"):
        vector_store.upsert("v1", [make_chunk("制度")], [[1.0, 0.0]])


def test_rejects_duplicate_chunk_ids_before_writing_any_points(vector_store):
    """A duplicate id must fail rather than silently dropping one citation-bearing chunk."""
    duplicate_chunks = [
        make_chunk("第一段", chunk_id="duplicate"),
        make_chunk("第二段", chunk_id="duplicate"),
    ]

    with pytest.raises(ValueError, match="duplicate chunk_id"):
        vector_store.upsert("v1", duplicate_chunks, [vector(1.0), vector(0.5)])

    assert vector_store.search(vector(1.0), {"v1"}, limit=5) == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("text", 123),
        ("file_name", None),
        ("heading_path", [None]),
        ("page", 1.5),
        ("page_end", "2"),
        ("paragraph_start", True),
    ],
)
def test_malformed_payload_types_fail_closed_at_qdrant_boundary(field, value):
    payload = {
        "version_id": "v1",
        "chunk_id": "chunk-1",
        "content_hash": "hash-1",
        "text": "虚构可定位片段",
        "file_name": "制度.docx",
        "heading_path": ["正文"],
        "page": None,
        "page_end": None,
        "paragraph_start": 1,
        "paragraph_end": 1,
    }
    payload[field] = value

    with pytest.raises(ValueError, match="payload"):
        QdrantLocalVectorStore._search_hit(payload, 0.9)


def test_creates_the_required_cosine_collection(vector_store):
    """A wrong collection configuration would make cloud embeddings incompatible."""
    collection = vector_store.collection_info()

    assert vector_store.collection_name == "policy_chunks_v1"
    assert collection.config.params.vectors.size == 1024
    assert collection.config.params.vectors.distance.value == "Cosine"


def test_collection_initialization_failure_preserves_root_error_when_close_also_fails(
    tmp_path, monkeypatch
):
    import app.storage.qdrant as qdrant_module

    class DoubleFailureClient:
        def collection_exists(self, name):
            raise ValueError("collection-init-original")

        def close(self):
            raise RuntimeError("qdrant-close-secondary")

    monkeypatch.setattr(
        qdrant_module, "QdrantClient", lambda **kwargs: DoubleFailureClient()
    )

    with pytest.raises(ValueError, match="collection-init-original"):
        QdrantLocalVectorStore(tmp_path / "qdrant")
