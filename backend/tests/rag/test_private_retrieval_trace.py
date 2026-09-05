from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.domain.models import Chunk, SourceRef
from app.domain.ports import SearchHit
from app.rag.answering import RagService
from app.rag.models import CandidateSet, ChatEvent, Evidence, RetrievalResult


def make_hit(chunk_id: str, text: str) -> SearchHit:
    chunk = Chunk(
        chunk_id=chunk_id,
        text=text,
        source=SourceRef(file_name="制度.pdf", heading_path=("第一章",)),
        content_hash=f"hash-{chunk_id}",
    )
    return SearchHit(version_id="version-1", chunk=chunk, score=0.8)


@dataclass
class FakeRetrieval:
    first: SearchHit
    second: SearchHit

    async def retrieve_candidates(self, question: str) -> CandidateSet:
        return CandidateSet(question=question, hits=(self.first, self.second))

    async def rerank(self, question: str, candidates: CandidateSet) -> RetrievalResult:
        return RetrievalResult(
            status="ready",
            evidence=(
                Evidence(reference_id="S1", hit=self.first, rerank_score=0.9),
                Evidence(reference_id="S2", hit=self.second, rerank_score=0.8),
            ),
        )


class FakeAnswerer:
    async def stream(self, question: str, retrieval: RetrievalResult):
        yield ChatEvent(
            type="final",
            data={"refused": False, "citations": [{"reference_id": "S1"}]},
        )


@pytest.mark.asyncio
async def test_rag_stream_writes_private_trace_without_leaking_ids_to_events() -> None:
    first = make_hit("chunk-a", "先提交申请")
    second = make_hit("chunk-b", "再等待审批")
    service = RagService(
        retrieval=FakeRetrieval(first, second), answerer=FakeAnswerer()
    )
    trace: dict[str, object] = {}

    events = [
        event
        async for event in service.stream("如何申请？", private_trace=trace)
    ]

    assert trace["candidate_source_ids"] == ["chunk-a", "chunk-b"]
    assert trace["evidence_source_ids"] == ["chunk-a", "chunk-b"]
    assert trace["evidence_reference_map"] == {"S1": "chunk-a", "S2": "chunk-b"}
    assert trace["candidate_count"] == 2
    assert trace["evidence_count"] == 2
    assert trace["status"] == "ready"
    assert all("candidate_source_ids" not in event.data for event in events)
    assert all("evidence_source_ids" not in event.data for event in events)


@pytest.mark.asyncio
async def test_rag_stream_without_trace_keeps_existing_event_shape() -> None:
    first = make_hit("chunk-a", "先提交申请")
    second = make_hit("chunk-b", "再等待审批")
    service = RagService(
        retrieval=FakeRetrieval(first, second), answerer=FakeAnswerer()
    )

    events = [event async for event in service.stream("如何申请？")]

    final = next(event for event in events if event.type == "final")
    diagnostics = final.data["retrieval_diagnostics"]
    assert diagnostics["candidate_count"] == 2
    assert "candidate_source_ids" not in diagnostics
    assert "evidence_source_ids" not in diagnostics
