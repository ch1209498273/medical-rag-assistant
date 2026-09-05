from __future__ import annotations

from app.domain.models import Chunk, SourceRef
from app.domain.ports import SearchHit
from app.feedback.evidence import LocalEvidenceResolver


class FakeSource:
    def __init__(self, hits):
        self.hits = hits
        self.calls = 0

    def snapshot_active_chunks(self, active_versions, limit=10000):
        self.calls += 1
        return [hit for hit in self.hits if hit.version_id in active_versions][:limit]

    def snapshot_all_chunks(self, limit=10000):
        self.calls += 1
        return self.hits[:limit]


def _hit(version_id: str, chunk_id: str, text: str = "低血压时按制度处理") -> SearchHit:
    return SearchHit(
        version_id=version_id,
        chunk=Chunk(
            chunk_id=chunk_id,
            text=text,
            source=SourceRef(
                file_name="制度.docx",
                heading_path=("透析管理", "低血压处理"),
                paragraph_start=3,
                paragraph_end=4,
            ),
            content_hash="a" * 64,
        ),
        score=0.0,
    )


def test_evidence_resolver_requires_exact_version_and_chunk_ids():
    source = FakeSource([_hit("2026-standard-manual-v1", "c-1")])
    resolver = LocalEvidenceResolver(source)

    bundle = resolver.resolve(
        source_version="2026-standard-manual-v1", citation_chunk_ids=("c-1",)
    )
    mismatch = resolver.resolve(source_version="old-version", citation_chunk_ids=("c-1",))

    assert bundle.status == "available"
    assert bundle.references[0].chunk_id == "c-1"
    assert mismatch.status == "version_mismatch"


def test_evidence_resolver_fails_closed_for_missing_or_unsafe_chunks():
    source = FakeSource([_hit("2026-standard-manual-v1", "c-1", "")])
    resolver = LocalEvidenceResolver(source)

    missing = resolver.resolve(
        source_version="2026-standard-manual-v1", citation_chunk_ids=("unknown",)
    )
    empty = resolver.resolve(source_version="2026-standard-manual-v1", citation_chunk_ids=())
    unsafe = resolver.resolve(
        source_version="2026-standard-manual-v1", citation_chunk_ids=("c-1",)
    )

    assert missing.status == "unavailable"
    assert empty.status == "unavailable"
    assert unsafe.status == "unsafe"
    assert source.calls == 3
