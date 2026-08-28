"""Independent Task 6 acceptance tests at the public HTTP/SSE boundary.

These tests deliberately assemble the real retrieval, answer-validation, RAG,
and FastAPI layers while replacing only cloud/storage boundaries with fakes.
All chunks are fictional and are created in memory; no network or credential
is needed to run the suite.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pytest
from app.domain.models import Chunk, SourceRef
from app.domain.ports import SearchHit
from app.errors import ProviderError
from app.main import create_app
from app.providers.siliconflow import RankedItem
from app.rag.answering import AnswerService, RagService
from app.rag.models import ChatEvent
from app.rag.retrieval import RetrievalService
from app.settings import Settings
from fastapi.testclient import TestClient


class FictionalRepository:
    def __init__(self, active_version_ids: Iterable[str] = ("v-fictional",)) -> None:
        self._active_version_ids = set(active_version_ids)

    def active_version_ids(self) -> set[str]:
        return set(self._active_version_ids)


class FictionalVectorStore:
    def __init__(self, hits: Iterable[SearchHit]) -> None:
        self.hits = list(hits)
        self.search_calls = 0

    def search(self, query_vector, active_versions, limit):
        self.search_calls += 1
        return self.hits[:limit]


class FictionalCloudProvider:
    """Fake SiliconFlow boundary with observable calls and failure injection."""

    def __init__(
        self,
        *,
        rerank_score: float = 0.9,
        embed_error: BaseException | None = None,
        rerank_error: BaseException | None = None,
    ) -> None:
        self.rerank_score = rerank_score
        self.embed_error = embed_error
        self.rerank_error = rerank_error
        self.embed_calls = 0
        self.rerank_calls = 0

    async def embed(self, texts):
        self.embed_calls += 1
        if self.embed_error is not None:
            raise self.embed_error
        return [[0.0] * 4 for _ in texts]

    async def rerank(self, query, documents, top_n):
        self.rerank_calls += 1
        if self.rerank_error is not None:
            raise self.rerank_error
        return [
            RankedItem(index=index, document=document, score=self.rerank_score)
            for index, document in enumerate(documents[:top_n])
        ]


class FictionalDeepSeek:
    """Fake DeepSeek stream boundary; raw chunks are never trusted by tests."""

    def __init__(
        self,
        raw_chunks: Iterable[str] = (
            '{"refused":false,"answer":"先申请，再',
            '审批。"}',
        ),
        *,
        error: BaseException | None = None,
        error_after_chunks: int | None = None,
    ) -> None:
        self.raw_chunks = tuple(raw_chunks)
        self.error = error
        self.error_after_chunks = error_after_chunks
        self.calls = 0

    async def stream_answer(self, messages, *, json_output=False):
        self.calls += 1
        if self.error is not None:
            for index, chunk in enumerate(self.raw_chunks, start=1):
                yield chunk
                if (
                    self.error_after_chunks is not None
                    and index >= self.error_after_chunks
                ):
                    raise self.error
            raise self.error
        for chunk in self.raw_chunks:
            yield chunk


def fictional_hit(
    *,
    version_id: str = "v-fictional",
    file_name: str = "fictional-leave-policy.docx",
) -> SearchHit:
    return SearchHit(
        version_id=version_id,
        chunk=Chunk(
            chunk_id="fictional-chunk-1",
            text="员工请假应先提交申请，再由负责人审批。",
            source=SourceRef(
                file_name=file_name,
                heading_path=("虚构请假制度", "申请流程"),
                paragraph_start=2,
                paragraph_end=2,
            ),
            content_hash="fictional-content-hash-1",
        ),
        score=0.95,
    )


def assemble_rag(
    *,
    tmp_path: Path,
    active_version_ids: Iterable[str] = ("v-fictional",),
    hits: Iterable[SearchHit] = (),
    cloud: FictionalCloudProvider | None = None,
    deepseek: FictionalDeepSeek | None = None,
):
    cloud = cloud or FictionalCloudProvider()
    deepseek = deepseek or FictionalDeepSeek()
    repository = FictionalRepository(active_version_ids)
    vector_store = FictionalVectorStore(hits)
    retrieval = RetrievalService(
        repository=repository,
        vector_store=vector_store,
        embedder=cloud,
        reranker=cloud,
        retrieval_limit=20,
        rerank_limit=6,
        relevance_threshold=0.35,
    )
    rag = RagService(
        retrieval=retrieval,
        answerer=AnswerService(deepseek=deepseek),
    )
    settings = Settings(
        _env_file=None,
        source_documents_dir=tmp_path,
        sqlite_path=tmp_path / "fictional-metadata.sqlite3",
        qdrant_path=tmp_path / "fictional-qdrant",
    )
    return create_app(settings, document_service=object(), rag_service=rag), cloud, deepseek, vector_store


def parse_sse(body: str) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for frame in body.strip().split("\n\n"):
        lines = frame.splitlines()
        frames.append(
            {
                "event": lines[0].removeprefix("event: "),
                "data": json.loads(lines[1].removeprefix("data: ")),
            }
        )
    return frames


def status_stages(events: list[dict[str, object]]) -> list[object]:
    return [
        event["data"]["stage"]
        for event in events
        if event["event"] == "status"
    ]


def post_question(app, question: str) -> tuple[object, list[dict[str, object]]]:
    with TestClient(app) as client:
        response = client.post("/api/chat/stream", json={"question": question})
    return response, parse_sse(response.text)


def test_public_stream_emits_backend_built_citation_after_validation(tmp_path):
    app, cloud, deepseek, _ = assemble_rag(
        tmp_path=tmp_path,
        hits=(fictional_hit(),),
    )

    response, events = post_question(app, "虚构制度中的请假申请流程是什么？")

    assert response.status_code == 200
    assert [event["event"] for event in events] == [
        "status",
        "status",
        "status",
        "status",
        "answer_delta",
        "final",
    ]
    assert status_stages(events) == [
        "retrieving",
        "reranking",
        "generating",
        "validating",
    ]
    validating_index = next(
        index
        for index, event in enumerate(events)
        if event["event"] == "status"
        and event["data"]["stage"] == "validating"
    )
    delta_index = next(
        index for index, event in enumerate(events) if event["event"] == "answer_delta"
    )
    assert delta_index > validating_index
    assert events[4]["data"] == {
        "text": "先申请，再审批。",
        "source_ids": ["S1"],
    }
    assert events[5]["data"] == {
        "refused": False,
        "citations": [
            {
                "reference_id": "S1",
                "file_name": "fictional-leave-policy.docx",
                "heading_path": ["虚构请假制度", "申请流程"],
                "page": None,
                "page_end": None,
                "paragraph_start": 2,
                "paragraph_end": 2,
                "excerpt": "员工请假应先提交申请，再由负责人审批。",
            }
        ],
    }
    assert cloud.embed_calls == 1
    assert cloud.rerank_calls == 1
    assert deepseek.calls == 1


def test_low_relevance_refuses_at_http_boundary_without_deepseek_call(tmp_path):
    cloud = FictionalCloudProvider(rerank_score=0.349)
    deepseek = FictionalDeepSeek(raw_chunks=("must-not-be-read",))
    app, _, _, _ = assemble_rag(
        tmp_path=tmp_path,
        hits=(fictional_hit(),),
        cloud=cloud,
        deepseek=deepseek,
    )

    response, events = post_question(app, "完全无关的问题")

    assert response.status_code == 200
    assert [event["event"] for event in events] == ["status", "status", "final"]
    assert status_stages(events) == ["retrieving", "reranking"]
    assert events[-1]["data"] == {
        "refused": True,
        "reason_code": "INSUFFICIENT_EVIDENCE",
        "citations": [],
    }
    assert cloud.embed_calls == 1
    assert cloud.rerank_calls == 1
    assert deepseek.calls == 0


def test_empty_knowledge_base_refuses_without_vector_search_or_deepseek(tmp_path):
    cloud = FictionalCloudProvider()
    deepseek = FictionalDeepSeek()
    app, _, _, vector_store = assemble_rag(
        tmp_path=tmp_path,
        active_version_ids=(),
        hits=(fictional_hit(),),
        cloud=cloud,
        deepseek=deepseek,
    )

    response, events = post_question(app, "虚构知识库为空时应如何回答？")

    assert response.status_code == 200
    assert events[-1]["data"]["reason_code"] == "INSUFFICIENT_EVIDENCE"
    assert events[-1]["data"]["refused"] is True
    assert cloud.embed_calls == 1
    assert cloud.rerank_calls == 0
    assert vector_store.search_calls == 0
    assert deepseek.calls == 0


@pytest.mark.parametrize(
    ("operation", "failure", "expected_stage"),
    [
        (
            "embed",
            TimeoutError("fictional SiliconFlow timeout: secret-buffer"),
            "retrieving",
        ),
        (
            "embed",
            RuntimeError("fictional SiliconFlow exception: private-upstream-detail"),
            "retrieving",
        ),
        (
            "rerank",
            TimeoutError("fictional rerank timeout: rerank-secret-buffer"),
            "reranking",
        ),
        (
            "rerank",
            ProviderError(
                "siliconflow",
                "rerank",
                503,
                True,
                "fictional private-rerank-detail",
            ),
            "reranking",
        ),
    ],
)
def test_provider_failures_become_safe_sse_errors(
    tmp_path, operation, failure, expected_stage
):
    cloud = FictionalCloudProvider(
        embed_error=failure if operation == "embed" else None,
        rerank_error=failure if operation == "rerank" else None,
    )
    app, _, deepseek, _ = assemble_rag(
        tmp_path=tmp_path,
        hits=(fictional_hit(),),
        cloud=cloud,
    )

    response, events = post_question(app, "检索供应商失败时不应泄露错误")

    assert response.status_code == 200
    expected_events = ["status", "error"]
    if operation == "rerank":
        expected_events = ["status", "status", "error"]
    assert [event["event"] for event in events] == expected_events
    expected_stages = ["retrieving"]
    if operation == "rerank":
        expected_stages.append("reranking")
    assert status_stages(events) == expected_stages
    assert events[-1]["data"] == {
        "stage": expected_stage,
        "reason_code": "RETRIEVAL_UNAVAILABLE",
    }
    assert "secret-buffer" not in response.text
    assert "private-upstream-detail" not in response.text
    assert "rerank-secret-buffer" not in response.text
    assert "private-rerank-detail" not in response.text
    assert cloud.embed_calls == 1
    assert cloud.rerank_calls == (0 if operation == "embed" else 1)
    assert deepseek.calls == 0


def test_deepseek_failure_discards_buffer_and_exposes_only_exact_candidate_label(tmp_path):
    deepseek = FictionalDeepSeek(
        raw_chunks=("partial answer that must never be emitted",),
        error=TimeoutError("fictional DeepSeek timeout: upstream-secret"),
        error_after_chunks=1,
    )
    app, cloud, _, _ = assemble_rag(
        tmp_path=tmp_path,
        hits=(fictional_hit(),),
        deepseek=deepseek,
    )

    response, events = post_question(app, "生成供应商失败时怎么办？")

    assert response.status_code == 200
    assert [event["event"] for event in events] == [
        "status",
        "status",
        "status",
        "error",
    ]
    assert status_stages(events) == ["retrieving", "reranking", "generating"]
    assert all(event["event"] != "answer_delta" for event in events)
    assert events[-1]["data"] == {
        "stage": "generating",
        "reason_code": "GENERATION_UNAVAILABLE",
        "candidate_sources": [
            {
                "label": "可能相关的原文，尚未生成答案",
                "reference_id": "S1",
                "file_name": "fictional-leave-policy.docx",
                "heading_path": ["虚构请假制度", "申请流程"],
            }
        ],
    }
    assert "partial answer that must never be emitted" not in response.text
    assert "upstream-secret" not in response.text
    assert cloud.embed_calls == 1
    assert cloud.rerank_calls == 1
    assert deepseek.calls == 1


@pytest.mark.parametrize(
    "raw_answer",
    [
        '{"refused":false,"answer":"有文字但未知来源","extra":"拒绝"}',
        '{"refused":false,"answer":""}',
        '{"refused":false,"answer":null}',
        "not-json",
    ],
)
def test_invalid_answer_contract_is_filtered_as_one_answer(tmp_path, raw_answer):
    deepseek = FictionalDeepSeek(raw_chunks=(raw_answer,))
    app, cloud, deepseek, _ = assemble_rag(
        tmp_path=tmp_path,
        hits=(fictional_hit(),),
        deepseek=deepseek,
    )

    response, events = post_question(app, "引用不完整或伪造时整条拒答")

    assert response.status_code == 200
    assert status_stages(events) == [
        "retrieving",
        "reranking",
        "generating",
        "validating",
    ]
    assert all(event["event"] != "answer_delta" for event in events)
    assert events[-1]["event"] == "final"
    assert events[-1]["data"] == {
        "refused": True,
        "reason_code": "ANSWER_NOT_VERIFIABLE",
        "citations": [],
    }
    assert "有文字但未知来源" not in response.text
    assert "拒绝" not in response.text
    assert cloud.embed_calls == 1
    assert cloud.rerank_calls == 1
    assert deepseek.calls == 1


def test_model_supplied_citations_metadata_is_rejected_as_unverifiable(tmp_path):
    raw_answer = (
        '{"refused":false,"answer":"后端应生成引用",'
        '"citations":[{"reference_id":"S9",'
        '"file_name":"forged-model-citation.docx","page":999}]}'
    )
    deepseek = FictionalDeepSeek(raw_chunks=(raw_answer,))
    app, cloud, deepseek, _ = assemble_rag(
        tmp_path=tmp_path,
        hits=(fictional_hit(),),
        deepseek=deepseek,
    )

    response, events = post_question(app, "模型自带 citation 元数据不得生效")

    assert response.status_code == 200
    assert status_stages(events) == [
        "retrieving",
        "reranking",
        "generating",
        "validating",
    ]
    assert all(event["event"] != "answer_delta" for event in events)
    assert events[-1]["data"] == {
        "refused": True,
        "reason_code": "ANSWER_NOT_VERIFIABLE",
        "citations": [],
    }
    assert "forged-model-citation.docx" not in response.text
    assert cloud.embed_calls == 1
    assert cloud.rerank_calls == 1
    assert deepseek.calls == 1


def test_public_sse_rejects_illegal_event_without_echoing_payload(tmp_path):
    class UntrustedRagService:
        async def stream(self, question):
            yield ChatEvent(
                type="not-a-contract-event",
                data={"secret": "fictional-payload-must-not-echo"},
            )

    settings = Settings(
        _env_file=None,
        source_documents_dir=tmp_path,
        sqlite_path=tmp_path / "fictional-metadata.sqlite3",
        qdrant_path=tmp_path / "fictional-qdrant",
    )
    app = create_app(
        settings,
        document_service=object(),
        rag_service=UntrustedRagService(),
    )

    response, events = post_question(app, "非法事件不得回显")

    assert response.status_code == 200
    assert events == [
        {
            "event": "error",
            "data": {"stage": "chat", "reason_code": "INVALID_EVENT"},
        }
    ]
    assert "not-a-contract-event" not in response.text
    assert "fictional-payload-must-not-echo" not in response.text
