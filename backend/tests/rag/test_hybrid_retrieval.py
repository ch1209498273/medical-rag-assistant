from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from app.domain.models import Chunk, SourceRef
from app.domain.ports import SearchHit
from app.providers.siliconflow import RankedItem
from app.rag.retrieval import RetrievalService, merge_ranked_hits


def make_hit(
    chunk_id: str,
    text: str,
    *,
    version_id: str = "v1",
    content_hash: str | None = None,
) -> SearchHit:
    return SearchHit(
        version_id=version_id,
        chunk=Chunk(
            chunk_id=chunk_id,
            text=text,
            source=SourceRef(
                file_name="虚构制度.docx", heading_path=(f"章节-{chunk_id}",)
            ),
            content_hash=content_hash or f"hash-{chunk_id}",
        ),
        score=0.5,
    )


@dataclass
class FakeRepository:
    active_versions: set[str] = field(default_factory=lambda: {"v1"})

    def active_version_ids(self) -> set[str]:
        return set(self.active_versions)


@dataclass
class FakeVectorStore:
    hits: list[SearchHit]
    calls: list[tuple[object, set[str], int]] = field(default_factory=list)

    def search(self, query_vector, active_versions, limit):
        self.calls.append((query_vector, set(active_versions), limit))
        return self.hits


@dataclass
class FakeLexicalIndex:
    hits: list[SearchHit]
    calls: list[tuple[tuple[str, ...], set[str], int]] = field(
        default_factory=list
    )

    def search(self, queries, active_versions, limit=20):
        self.calls.append((tuple(queries), set(active_versions), limit))
        return self.hits


@dataclass
class FakeEmbedder:
    calls: list[list[str]] = field(default_factory=list)

    async def embed(self, texts):
        self.calls.append(list(texts))
        return [[0.0] * 4 for _ in texts]


@dataclass
class FakeReranker:
    score: float = 1.0

    async def rerank(self, query, documents, top_n):
        return [
            RankedItem(index=index, document=document, score=self.score)
            for index, document in enumerate(documents[:top_n])
        ]


def make_service(
    vector_hits: list[SearchHit],
    lexical_hits: list[SearchHit] | None = None,
    *,
    strategy: str = "vector",
    active_versions: set[str] | None = None,
    embedder: FakeEmbedder | None = None,
    reranker: FakeReranker | None = None,
) -> tuple[RetrievalService, FakeVectorStore, FakeLexicalIndex | None, FakeEmbedder]:
    vector_store = FakeVectorStore(vector_hits)
    lexical_index = (
        None if lexical_hits is None else FakeLexicalIndex(lexical_hits)
    )
    active_versions = active_versions or {"v1"}
    embedder = embedder or FakeEmbedder()
    service = RetrievalService(
        repository=FakeRepository(active_versions),
        vector_store=vector_store,
        embedder=embedder,
        reranker=reranker or FakeReranker(),
        retrieval_limit=20,
        rerank_limit=6,
        relevance_threshold=0.35,
        retrieval_strategy=strategy,  # type: ignore[arg-type]
        lexical_index=lexical_index,
    )
    return service, vector_store, lexical_index, embedder


@pytest.mark.asyncio
async def test_default_vector_strategy_only_calls_vector_store() -> None:
    vector = make_hit("vector", "向量候选内容足够独特")
    service, vector_store, lexical_index, embedder = make_service([vector])

    candidates = await service.retrieve_candidates("原始问题")

    assert [hit.chunk.chunk_id for hit in candidates.hits] == ["vector"]
    assert len(vector_store.calls) == 1
    assert lexical_index is None
    assert embedder.calls == [["原始问题"]]


@pytest.mark.asyncio
async def test_hybrid_merges_two_branches_once_and_filters_inactive_hits() -> None:
    vector = make_hit("vector", "向量分支独有内容")
    common_vector = make_hit("common", "共同命中内容向量")
    stale_vector = make_hit("stale-vector", "过期向量内容", version_id="old")
    lexical = make_hit("lexical", "词法分支独有内容")
    common_lexical = make_hit("common", "共同命中内容词法")
    stale_lexical = make_hit("stale-lexical", "过期词法内容", version_id="old")
    service, vector_store, lexical_index, embedder = make_service(
        [common_vector, vector, stale_vector],
        [lexical, common_lexical, stale_lexical],
        strategy="hybrid",
    )

    candidates = await service.retrieve_candidates("原始问题")

    assert lexical_index is not None
    assert lexical_index.calls == [(("原始问题",), {"v1"}, 20)]
    assert len(vector_store.calls) == 1
    assert embedder.calls == [["原始问题"]]
    assert {hit.version_id for hit in candidates.hits} == {"v1"}
    assert [hit.chunk.chunk_id for hit in candidates.hits] == [
        "common",
        "lexical",
        "vector",
    ]


@pytest.mark.asyncio
async def test_hybrid_normalized_passes_original_and_local_query_variants() -> None:
    lexical_hit = make_hit("lexical", "规范化词法候选")
    service, _, lexical_index, embedder = make_service(
        [make_hit("vector", "向量候选")],
        [lexical_hit],
        strategy="hybrid_normalized",
    )

    await service.retrieve_candidates("  血透　HD  ")

    assert lexical_index is not None
    queries, active_versions, limit = lexical_index.calls[0]
    assert queries[0] == "血透　HD"
    assert "血透 HD" in queries
    assert "血液透析 血液透析" in queries
    assert active_versions == {"v1"}
    assert limit == 20
    assert embedder.calls == [["血透　HD"]]


def test_merge_ranked_hits_uses_rrf_and_content_hash_deduplication() -> None:
    common_vector = make_hit("common-v", "共同候选-向量", content_hash="same")
    vector_only = make_hit("vector-only", "向量独有候选")
    lexical_only = make_hit("lexical-only", "词法独有候选")
    common_lexical = make_hit("common-l", "共同候选-词法", content_hash="same")

    merged = merge_ranked_hits(
        [common_vector, vector_only],
        [lexical_only, common_lexical],
        limit=20,
    )

    assert [hit.chunk.chunk_id for hit in merged] == [
        "common-v",
        "lexical-only",
        "vector-only",
    ]
    assert merged[0].chunk.content_hash == "same"
    assert merged[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert len({hit.chunk.content_hash for hit in merged}) == len(merged)


def test_merge_ranked_hits_consumes_one_rank_per_identity_in_each_branch() -> None:
    duplicate_first = make_hit(
        "duplicate-first", "重复候选第一次出现", content_hash="same"
    )
    duplicate_second = make_hit(
        "duplicate-second", "重复候选第二次出现", content_hash="same"
    )
    independent_vector = make_hit(
        "independent-vector", "独立候选向量命中", content_hash="independent"
    )
    independent_lexical = make_hit(
        "independent-lexical", "独立候选词法命中", content_hash="independent"
    )

    merged = merge_ranked_hits(
        [duplicate_first, duplicate_second, independent_vector],
        [independent_lexical],
        limit=20,
    )

    assert [hit.chunk.chunk_id for hit in merged] == [
        "independent-vector",
        "duplicate-first",
    ]
    assert merged[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert merged[1].score == pytest.approx(1 / 61)


def test_merge_ranked_hits_uses_stable_key_for_equal_scores_and_caps_twenty() -> None:
    vector_hits = [make_hit(f"v-{index:02d}", f"向量候选-{index}-独特内容") for index in range(25)]

    merged = merge_ranked_hits([], vector_hits, limit=100)

    assert len(merged) == 20
    assert [hit.chunk.chunk_id for hit in merged[:3]] == ["v-00", "v-01", "v-02"]


@pytest.mark.asyncio
async def test_empty_lexical_branch_falls_back_to_vector_without_relaxing_gate() -> None:
    service, _, lexical_index, _ = make_service(
        [make_hit("vector", "向量候选内容")],
        [],
        strategy="hybrid",
        reranker=FakeReranker(score=0.349),
    )

    result = await service.retrieve("问题")

    assert lexical_index is not None
    assert result.status == "refused"
    assert result.reason_code == "INSUFFICIENT_EVIDENCE"


def test_hybrid_strategy_requires_a_lexical_index() -> None:
    with pytest.raises(ValueError, match="lexical"):
        make_service([make_hit("vector", "向量候选")], strategy="hybrid")


@pytest.mark.parametrize("rrf_k", [0, -1, True, 1.5])
def test_invalid_rrf_k_is_rejected(rrf_k: object) -> None:
    with pytest.raises(ValueError, match="rrf_k"):
        RetrievalService(
            repository=FakeRepository(),
            vector_store=FakeVectorStore([]),
            embedder=FakeEmbedder(),
            reranker=FakeReranker(),
            rrf_k=rrf_k,  # type: ignore[arg-type]
        )
