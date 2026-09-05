from __future__ import annotations

import json

from app.domain.models import Chunk, SourceRef
from app.domain.ports import SearchHit
from app.rag.retrieval import (
    deduplicate_hits_with_diagnostics,
    merge_ranked_hits_with_diagnostics,
)


def _hit(
    chunk_id: str,
    text: str,
    *,
    content_hash: str,
    heading_path: tuple[str, ...] = ("制度",),
) -> SearchHit:
    return SearchHit(
        version_id="v1",
        chunk=Chunk(
            chunk_id=chunk_id,
            text=text,
            source=SourceRef(file_name="synthetic.pdf", heading_path=heading_path),
            content_hash=content_hash,
        ),
        score=0.5,
    )


def test_rrf_diagnostics_mark_shared_and_branch_only_candidates() -> None:
    merged, fusion_facts, drops = merge_ranked_hits_with_diagnostics(
        [
            _hit("shared-v", "共同候选", content_hash="same"),
            _hit("vector-only", "向量独有", content_hash="vector"),
        ],
        [
            _hit("lexical-only", "词法独有", content_hash="lexical"),
            _hit("shared-l", "共同候选另一份", content_hash="same"),
        ],
        limit=20,
    )

    memberships = {fact.candidate_id: fact.membership for fact in fusion_facts}
    assert memberships == {
        "same": "shared",
        "vector": "vector_only",
        "lexical": "lexical_only",
    }
    assert {hit.chunk.content_hash for hit in merged} == {
        "same",
        "vector",
        "lexical",
    }
    assert drops == ()


def test_post_fusion_dedup_reports_exact_near_and_heading_cap_drops() -> None:
    hits = [
        _hit("keep", "第一条完整制度说明", content_hash="keep"),
        _hit("exact", "第二条内容", content_hash="keep"),
        _hit("near", "第一条完整制度说明。", content_hash="near"),
        _hit("heading-2", "同标题下的第二条独立说明", content_hash="heading-2"),
        _hit("heading-3", "同标题下的第三条夜间交接流程", content_hash="heading-3"),
    ]

    retained, drops = deduplicate_hits_with_diagnostics(hits, post_fusion_dedup=True)

    assert [hit.chunk.chunk_id for hit in retained] == ["keep", "heading-2"]
    assert [drop.reason for drop in drops] == [
        "exact_duplicate",
        "near_duplicate",
        "heading_cap",
    ]

    audit_retained, audit_drops = deduplicate_hits_with_diagnostics(
        hits, post_fusion_dedup=False
    )
    assert [hit.chunk.chunk_id for hit in audit_retained] == [
        "keep",
        "exact",
        "near",
        "heading-2",
        "heading-3",
    ]
    assert audit_drops == ()


def test_diagnostic_facts_never_serialize_chunk_text_or_source_file() -> None:
    _, fusion_facts, _ = merge_ranked_hits_with_diagnostics(
        [_hit("vector", "不能出现在 trace 的正文", content_hash="vector")],
        [],
        limit=20,
    )

    serialized = json.dumps(
        [fact.to_dict() for fact in fusion_facts], ensure_ascii=False
    )
    assert "不能出现在 trace 的正文" not in serialized
    assert "synthetic.pdf" not in serialized
    assert "text" not in serialized
