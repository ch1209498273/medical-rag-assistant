from __future__ import annotations

import pytest

from app.domain.models import Chunk, SourceRef
from app.domain.ports import EligibilityResult, SearchHit
from app.providers.siliconflow import RankedItem
from app.rag.retrieval import RetrievalService


class FakeRepository:
    def active_version_ids(self) -> set[str]:
        return {"v1"}


class EligibleRepository(FakeRepository):
    def eligible_version_ids(self, audience_scope, as_of):
        return EligibilityResult(frozenset({"v1"}), None)


class FakeVectorStore:
    def __init__(self, hits: list[SearchHit], *, ignore_limit: bool = False) -> None:
        self.hits = hits
        self.ignore_limit = ignore_limit
        self.requested_limit: int | None = None
        self.active_versions: set[str] | None = None

    def search(self, query_vector, active_versions, limit):
        self.requested_limit = limit
        self.active_versions = set(active_versions)
        return self.hits if self.ignore_limit else self.hits[:limit]


class FakeCloud:
    def __init__(self, rerank_scores: list[float] | None = None) -> None:
        self.rerank_scores = rerank_scores
        self.top_n: int | None = None
        self.document_count: int | None = None

    async def embed(self, texts):
        return [[0.0] * 1024 for _ in texts]

    async def rerank(self, query, documents, top_n):
        self.top_n = top_n
        self.document_count = len(documents)
        scores = self.rerank_scores or [1.0 - index / 100 for index in range(len(documents))]
        return [
            RankedItem(index=index, document=documents[index], score=scores[index])
            for index in range(min(top_n, len(documents)))
        ]


def make_hit(index: int, text: str, heading=("制度",), version_id: str = "v1") -> SearchHit:
    chunk = Chunk(
        chunk_id=f"c{index}",
        text=text,
        source=SourceRef(file_name="虚构制度.docx", heading_path=heading),
        content_hash=f"hash-{index}",
    )
    return SearchHit(version_id=version_id, chunk=chunk, score=1.0 - index / 100)


def make_retrieval(hits, scores=None):
    vector_store = FakeVectorStore(hits)
    cloud = FakeCloud(scores)
    service = RetrievalService(
        repository=FakeRepository(),
        vector_store=vector_store,
        embedder=cloud,
        reranker=cloud,
        retrieval_limit=20,
        rerank_limit=6,
        relevance_threshold=0.35,
    )
    return service, vector_store, cloud


def make_eligible_retrieval(hits):
    vector_store = FakeVectorStore(hits)
    cloud = FakeCloud()
    service = RetrievalService(
        repository=EligibleRepository(),
        vector_store=vector_store,
        embedder=cloud,
        reranker=cloud,
        retrieval_limit=20,
        rerank_limit=6,
        relevance_threshold=0.35,
    )
    return service, vector_store, cloud


@pytest.mark.asyncio
async def test_retrieval_recalls_20_then_deduplicates_and_reranks_to_6():
    service, vector_store, cloud = make_retrieval(
        [
            make_hit(index, chr(0x4E00 + index) * 12, (f"章节{index}",))
            for index in range(20)
        ]
    )
    result = await service.retrieve("请假需要履行什么流程？")
    assert vector_store.requested_limit == 20
    assert cloud.top_n == 6
    assert len(result.evidence) == 6
    assert [item.reference_id for item in result.evidence] == [
        "S1", "S2", "S3", "S4", "S5", "S6"
    ]


@pytest.mark.asyncio
async def test_retrieval_uses_business_eligible_allowlist_when_repository_provides_one():
    service, vector_store, _ = make_eligible_retrieval([make_hit(0, "正式制度")])

    await service.retrieve("制度流程")

    assert vector_store.active_versions == {"v1"}


@pytest.mark.asyncio
async def test_same_heading_keeps_at_most_two_and_drops_high_overlap():
    heading = ("培训制度", "签到")
    hits = [
        make_hit(0, "培训签到后由负责人确认。", heading),
        make_hit(1, "培训签到后由负责人确认。", heading),
        make_hit(2, "迟到人员需要登记。", heading),
        make_hit(3, "缺席人员需要补训。", heading),
    ]
    service, _, _ = make_retrieval(hits)
    result = await service.retrieve("培训")
    same_heading = [
        item for item in result.evidence
        if item.hit.chunk.source.heading_path == ("培训制度", "签到")
    ]
    assert [item.hit.chunk.text for item in same_heading] == [
        "培训签到后由负责人确认。",
        "迟到人员需要登记。",
    ]
    assert len({item.hit.chunk.content_hash for item in result.evidence}) == len(result.evidence)


@pytest.mark.asyncio
async def test_top_rerank_score_below_point_35_returns_refusal():
    service, _, _ = make_retrieval([make_hit(0, "无关制度")], [0.349])
    result = await service.retrieve("年终奖多少？")
    assert result.status == "refused"
    assert result.reason_code == "INSUFFICIENT_EVIDENCE"


@pytest.mark.asyncio
async def test_retrieval_never_exposes_more_than_six_evidence_items():
    vector_store = FakeVectorStore(
        [make_hit(index, f"第{index}条内容", (f"章节{index}",)) for index in range(10)]
    )
    cloud = FakeCloud()
    service = RetrievalService(
        repository=FakeRepository(),
        vector_store=vector_store,
        embedder=cloud,
        reranker=cloud,
        retrieval_limit=20,
        rerank_limit=20,
        relevance_threshold=0.35,
    )
    result = await service.retrieve("培训流程？")
    assert len(result.evidence) == 6
    assert [item.reference_id for item in result.evidence] == [
        "S1", "S2", "S3", "S4", "S5", "S6"
    ]


@pytest.mark.asyncio
async def test_retrieval_drops_hits_outside_active_version_allowlist():
    service, _, _ = make_retrieval(
        [
            make_hit(0, "当前制度", version_id="v1"),
            make_hit(1, "旧版制度", version_id="old"),
        ]
    )
    result = await service.retrieve("制度？")
    assert [item.hit.version_id for item in result.evidence] == ["v1"]


@pytest.mark.asyncio
async def test_high_overlap_body_is_deduplicated_across_headings():
    base = "培训签到完成后由负责人确认并登记，异常情况需要及时报告。"
    service, _, _ = make_retrieval(
        [
            make_hit(0, base, ("培训", "签到")),
            make_hit(1, base + " ", ("培训", "记录")),
        ]
    )
    result = await service.retrieve("培训签到")
    assert [item.hit.chunk.text for item in result.evidence] == [base]


@pytest.mark.asyncio
async def test_numeric_high_overlap_body_is_deduplicated_without_exceptions():
    first = "第1条培训签到完成后由负责人确认并登记，异常情况需要及时报告。"
    second = "第2条培训签到完成后由负责人确认并登记，异常情况需要及时报告。"
    service, _, _ = make_retrieval(
        [
            make_hit(0, first, ("培训", "签到")),
            make_hit(1, second, ("培训", "记录")),
        ]
    )
    result = await service.retrieve("培训签到")
    assert [item.hit.chunk.text for item in result.evidence] == [first]


@pytest.mark.asyncio
async def test_retrieval_hard_caps_provider_overflow_at_twenty_before_rerank():
    hits = [
        make_hit(index, chr(0x4E00 + index) * 12, (f"章节{index}",))
        for index in range(25)
    ]
    vector_store = FakeVectorStore(hits, ignore_limit=True)
    cloud = FakeCloud()
    service = RetrievalService(
        repository=FakeRepository(),
        vector_store=vector_store,
        embedder=cloud,
        reranker=cloud,
        retrieval_limit=20,
        rerank_limit=6,
        relevance_threshold=0.35,
    )
    result = await service.retrieve("培训流程？")
    assert vector_store.requested_limit == 20
    assert cloud.document_count == 20
    assert len(result.evidence) == 6


@pytest.mark.asyncio
async def test_highest_rerank_score_controls_gate_and_order():
    service, _, _ = make_retrieval(
        [make_hit(0, "较弱条款", ("章节一",)), make_hit(1, "强相关条款", ("章节二",))],
        [0.20, 0.90],
    )
    result = await service.retrieve("流程？")
    assert result.status == "ready"
    assert result.evidence[0].rerank_score == 0.90
    assert result.evidence[0].hit.chunk.text == "强相关条款"
