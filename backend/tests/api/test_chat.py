from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.chat import _safe_event
from app.chat.history import HistoryContextBuilder
from app.chat.orchestrator import ChatOrchestrator
from app.main import create_app
from app.rag.models import ChatEvent
from app.settings import Settings
from app.storage.sqlite import SqliteDocumentRepository


def test_rag_runtime_closes_deepseek_before_document_resources():
    import asyncio

    from app.main import RagRuntime

    closed: list[str] = []

    class Resource:
        async def aclose(self):
            closed.append("deepseek")

    class DocumentRuntime:
        async def close(self):
            closed.extend(("siliconflow", "qdrant", "sqlite"))

    runtime = RagRuntime(
        document_runtime=DocumentRuntime(),
        deepseek=Resource(),
        retrieval=object(),
        answerer=object(),
        service=object(),
    )
    asyncio.run(runtime.close())
    asyncio.run(runtime.close())
    assert closed == ["deepseek", "siliconflow", "qdrant", "sqlite"]


def test_rag_runtime_does_not_retry_a_failed_close():
    import asyncio

    from app.main import RagRuntime

    calls: list[str] = []

    class DeepSeek:
        async def aclose(self):
            calls.append("deepseek")

    class DocumentRuntime:
        async def close(self):
            calls.append("documents")
            raise RuntimeError("close failed")

    runtime = RagRuntime(
        document_runtime=DocumentRuntime(),
        deepseek=DeepSeek(),
        retrieval=object(),
        answerer=object(),
        service=object(),
    )
    with pytest.raises(RuntimeError):
        asyncio.run(runtime.close())
    asyncio.run(runtime.close())
    assert calls == ["deepseek", "documents"]


def test_lifespan_closes_document_runtime_when_rag_assembly_fails(tmp_path, monkeypatch):
    import app.main as main_module

    closed: list[str] = []

    class DocumentRuntime:
        service = object()

        async def close(self):
            closed.append("documents")

    runtime = DocumentRuntime()
    async def build_document_runtime(settings):
        return runtime

    monkeypatch.setattr(main_module, "build_document_runtime_async", build_document_runtime)

    async def fail_build_rag(settings, document_runtime):
        raise RuntimeError("rag assembly failed")

    monkeypatch.setattr(main_module, "build_rag_runtime_async", fail_build_rag)
    settings = Settings(
        _env_file=None,
        source_documents_dir=tmp_path,
        sqlite_path=tmp_path / "metadata.sqlite3",
        qdrant_path=tmp_path / "qdrant",
    )
    with pytest.raises(RuntimeError, match="rag assembly failed"), TestClient(
        create_app(settings)
    ):
        pass
    assert closed == ["documents"]


class FakeRagService:
    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, question):
        self.calls += 1
        for stage in ("retrieving", "reranking", "generating", "validating"):
            yield ChatEvent(type="status", data={"stage": stage})
        yield ChatEvent(
            type="answer_delta", data={"text": "先申请。", "source_ids": ["S1"]}
        )
        yield ChatEvent(type="final", data={"refused": False, "citations": []})


class AudienceAwareRagService:
    def __init__(self) -> None:
        self.audience_scope = None

    async def stream(self, question, *, audience_scope):
        self.audience_scope = audience_scope
        yield ChatEvent(type="status", data={"stage": "retrieving"})
        yield ChatEvent(
            type="final",
            data={
                "refused": True,
                "citations": [],
                "reason_code": "EVIDENCE_SCOPE_UNCLEAR",
            },
        )


def parse_sse(text: str) -> list[dict[str, object]]:
    parsed = []
    for frame in text.strip().split("\n\n"):
        lines = frame.splitlines()
        parsed.append(
            {
                "event": lines[0].removeprefix("event: "),
                "data": json.loads(lines[1].removeprefix("data: ")),
            }
        )
    return parsed


def make_chat_client(tmp_path, rag=None):
    rag = rag or FakeRagService()
    settings = Settings(
        _env_file=None,
        source_documents_dir=tmp_path,
        sqlite_path=tmp_path / "metadata.sqlite3",
        qdrant_path=tmp_path / "qdrant",
    )
    client = TestClient(
        create_app(settings, document_service=object(), rag_service=rag)
    )
    return client, rag


def test_chat_stream_preserves_safe_event_order(tmp_path):
    client, _ = make_chat_client(tmp_path)
    response = client.post("/api/chat/stream", json={"question": "请假流程？"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert [event["event"] for event in events] == [
        "status", "status", "status", "status", "answer_delta", "final"
    ]
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_chat_stream_passes_audience_scope_as_a_data_hint_not_identity(tmp_path):
    rag = AudienceAwareRagService()
    client, _ = make_chat_client(tmp_path, rag)

    response = client.post(
        "/api/chat/stream",
        json={"question": "护士培训流程？", "audience_scope": "nurse"},
    )

    assert response.status_code == 200
    assert rag.audience_scope == "nurse"


def test_chat_stream_accepts_pharmacist_audience_scope(tmp_path):
    rag = AudienceAwareRagService()
    client, _ = make_chat_client(tmp_path, rag)

    response = client.post(
        "/api/chat/stream",
        json={"question": "药师培训流程？", "audience_scope": "pharmacist"},
    )

    assert response.status_code == 200
    assert rag.audience_scope == "pharmacist"


def test_chat_rejects_empty_or_oversized_question_without_model_call(tmp_path):
    client, rag = make_chat_client(tmp_path)
    assert client.post("/api/chat/stream", json={"question": ""}).status_code == 422
    assert client.post(
        "/api/chat/stream", json={"question": "问" * 1001}
    ).status_code == 422
    assert rag.calls == 0


def test_chat_rejects_unexpected_runtime_event_type_without_echoing_it(tmp_path):
    class UnexpectedRagService:
        async def stream(self, question):
            yield ChatEvent(type="unexpected", data={"secret": "do-not-echo"})

    client, _ = make_chat_client(tmp_path, UnexpectedRagService())
    response = client.post("/api/chat/stream", json={"question": "问题"})
    assert response.status_code == 200
    events = parse_sse(response.text)
    assert [event["event"] for event in events] == ["error"]
    assert events[0]["data"]["reason_code"] == "INVALID_EVENT"
    assert "unexpected" not in response.text
    assert "do-not-echo" not in response.text


def test_chat_allows_multiline_answer_delta_without_invalidating_the_stream(tmp_path):
    class MultilineRagService:
        async def stream(self, question):
            yield ChatEvent(type="status", data={"stage": "generating"})
            yield ChatEvent(type="answer_delta", data={"text": "第一行\n第二行\r\n第三行\t。"})
            yield ChatEvent(type="final", data={"refused": True, "citations": []})

    client, _ = make_chat_client(tmp_path, MultilineRagService())

    response = client.post("/api/chat/stream", json={"question": "问题"})
    events = parse_sse(response.text)

    assert [event["event"] for event in events] == ["status", "answer_delta", "final"]
    assert events[1]["data"]["text"] == "第一行\n第二行\r\n第三行\t。"


def test_legacy_sse_rejects_unknown_reason_code_without_echoing_it(tmp_path):
    class UnsafeRagService:
        async def stream(self, question):
            yield ChatEvent(
                type="error",
                data={"stage": "retrieving", "reason_code": "secret-provider-detail"},
            )

    client, _ = make_chat_client(tmp_path, UnsafeRagService())

    response = client.post("/api/chat/stream", json={"question": "问题"})
    events = parse_sse(response.text)

    assert events == [{"event": "error", "data": {"stage": "chat", "reason_code": "INVALID_EVENT"}}]
    assert "secret-provider-detail" not in response.text


def test_legacy_sse_rejects_unsafe_candidate_source_metadata(tmp_path):
    class UnsafeRagService:
        async def stream(self, question):
            yield ChatEvent(
                type="error",
                data={
                    "stage": "generating",
                    "reason_code": "GENERATION_UNAVAILABLE",
                    "candidate_sources": [
                        {
                            "label": "safe label",
                            "reference_id": "S1",
                            "file_name": "C:" + chr(92) + "secret" + chr(92) + "policy.docx",
                            "heading_path": ["heading"],
                        }
                    ],
                },
            )

    client, _ = make_chat_client(tmp_path, UnsafeRagService())

    response = client.post("/api/chat/stream", json={"question": "问题"})
    events = parse_sse(response.text)

    assert events == [{"event": "error", "data": {"stage": "chat", "reason_code": "INVALID_EVENT"}}]
    assert "policy.docx" not in response.text


def test_rewriting_event_rejects_unsafe_session_metadata():
    event = ChatEvent(
        type="status",
        data={"stage": "rewriting", "session_id": "../../secret"},
    )

    assert _safe_event(event, require_terminal_metadata=True) is None


def test_stateful_sse_rejects_incomplete_citation_shape():
    event = ChatEvent(
        type="final",
        data={
            "session_id": "s1",
            "message_id": "a1",
            "refused": False,
            "citations": [{"reference_id": "S1", "file_name": "制度.docx"}],
        },
    )

    assert _safe_event(event, require_terminal_metadata=True) is None


def test_chat_session_list_returns_safe_metadata_only(tmp_path):
    repository = SqliteDocumentRepository(tmp_path / "metadata.sqlite3")
    first = repository.create_chat_session()
    second = repository.create_chat_session()
    repository.append_user_message(first, "不应出现在列表预览")
    repository.append_user_message(second, "第二会话问题")
    settings = Settings(
        _env_file=None,
        source_documents_dir=tmp_path,
        sqlite_path=tmp_path / "metadata.sqlite3",
        qdrant_path=tmp_path / "qdrant",
    )
    try:
        holder = SimpleNamespace(repository=repository)
        with TestClient(
            create_app(settings, document_service=object(), chat_orchestrator=holder)
        ) as client:
            response = client.get("/api/chat/sessions?limit=20")

        assert response.status_code == 200
        item = response.json()["sessions"][0]
        assert set(item) == {
            "session_id",
            "title",
            "created_at",
            "last_activity_at",
            "message_count",
        }
        assert item["session_id"] == second
        assert item["title"] == "第二会话问题"
        assert item["message_count"] == 1
        assert '"content"' not in response.text
    finally:
        repository.close()


def test_final_reference_answer_is_kept_separate_on_answered_messages():
    event = ChatEvent(
        type="final",
        data={
            "refused": False,
            "citations": [
                {
                    "reference_id": "S1",
                    "file_name": "制度.docx",
                    "heading_path": ["正文"],
                    "page": None,
                    "page_end": None,
                    "paragraph_start": 1,
                    "paragraph_end": 1,
                    "excerpt": "资料摘录",
                }
            ],
            "reference_answer": "单独标注的通用参考",
        },
    )

    safe = _safe_event(event, require_terminal_metadata=False)

    assert safe is not None
    assert safe.data["refused"] is False
    assert safe.data["reference_answer"] == "单独标注的通用参考"


def test_workflow_summary_is_public_only_as_fixed_safe_facts():
    event = ChatEvent(
        type="final",
        data={
            "refused": True,
            "citations": [],
            "reason_code": "INSUFFICIENT_EVIDENCE",
            "workflow_summary": {
                "workflow_version": "agent_workflow_v2a",
                "run_id": "run-12345678",
                "route": "direct",
                "outcome": "refused",
                "verifier_status": "not_run",
                "http_calls": 1,
                "elapsed_ms": 40,
                "reason_code": "INSUFFICIENT_EVIDENCE",
            },
        },
    )

    safe = _safe_event(event, require_terminal_metadata=False)

    assert safe is not None
    assert safe.data["workflow_summary"]["verifier_status"] == "not_run"
    assert "run_id" in safe.data["workflow_summary"]

    unsafe = ChatEvent(
        type="final",
        data={
            "refused": True,
            "citations": [],
            "reason_code": "INSUFFICIENT_EVIDENCE",
            "workflow_summary": {
                "workflow_version": "agent_workflow_v2a",
                "run_id": "run-12345678",
                "route": "direct",
                "outcome": "refused",
                "verifier_status": "not_run",
                "http_calls": 1,
                "elapsed_ms": 40,
                "reason_code": "provider secret",
            },
        },
    )
    assert _safe_event(unsafe, require_terminal_metadata=False) is None


class StatefulRagService:
    async def stream(self, question):
        yield ChatEvent(type="status", data={"stage": "retrieving"})
        yield ChatEvent(
            type="final",
            data={"refused": True, "reason_code": "INSUFFICIENT_EVIDENCE", "citations": []},
        )


class SummaryRagService:
    async def stream(self, question):
        yield ChatEvent(type="status", data={"stage": "routing"})
        yield ChatEvent(type="answer_delta", data={"text": "依据资料回答。", "source_ids": ["S1"]})
        yield ChatEvent(
            type="final",
            data={
                "refused": False,
                "citations": [
                    {
                        "reference_id": "S1",
                        "file_name": "制度.docx",
                        "heading_path": ["正文"],
                        "page": None,
                        "page_end": None,
                        "paragraph_start": 1,
                        "paragraph_end": 1,
                        "excerpt": "公开样例资料。",
                    }
                ],
                "workflow_summary": {
                    "workflow_version": "agent_workflow_v2a",
                    "run_id": "run-12345678",
                    "route": "direct",
                    "outcome": "answered",
                    "verifier_status": "not_run",
                    "http_calls": 1,
                    "elapsed_ms": 120,
                    "reason_code": None,
                },
            },
        )


class NeverCalledRewriter:
    async def rewrite(self, question, history):
        raise AssertionError("first question must not be rewritten")


def test_workflow_summary_is_streamed_and_restored_from_history(tmp_path):
    repository = SqliteDocumentRepository(tmp_path / "metadata.sqlite3")
    orchestrator = ChatOrchestrator(
        repository,
        NeverCalledRewriter(),
        SummaryRagService(),
        HistoryContextBuilder(),
    )
    settings = Settings(
        _env_file=None,
        source_documents_dir=tmp_path,
        sqlite_path=tmp_path / "metadata.sqlite3",
        qdrant_path=tmp_path / "qdrant",
    )
    try:
        with TestClient(
            create_app(settings, document_service=object(), chat_orchestrator=orchestrator)
        ) as client:
            response = client.post("/api/chat/stream", json={"question": "问题"})
            events = parse_sse(response.text)
            session_id = events[0]["data"]["session_id"]
            final = events[-1]

            assert final["data"]["workflow_summary"]["route"] == "direct"
            restored = client.get(f"/api/chat/sessions/{session_id}").json()
            assert restored["messages"][-1]["workflow_summary"]["verifier_status"] == "not_run"
    finally:
        repository.close()


def test_stateful_partial_answer_hides_internal_reference_question(tmp_path):
    class PartialRagService:
        async def stream(self, question):
            yield ChatEvent(
                type="answer_delta",
                data={"text": "资料可确认应先申请。", "source_ids": ["S1"]},
            )
            yield ChatEvent(
                type="final",
                data={
                    "refused": False,
                    "citations": [
                        {
                            "reference_id": "S1",
                            "file_name": "制度.docx",
                            "heading_path": ["正文"],
                            "page": None,
                            "page_end": None,
                            "paragraph_start": 1,
                            "paragraph_end": 1,
                            "excerpt": "应先申请。",
                        }
                    ],
                    "reference_question": "审批通常需要多长时间？",
                },
            )

    class ReferenceGenerator:
        async def generate(self, question):
            assert question == "审批通常需要多长时间？"
            return "审批时长需向负责人确认。"

    repository = SqliteDocumentRepository(tmp_path / "metadata.sqlite3")
    orchestrator = ChatOrchestrator(
        repository,
        NeverCalledRewriter(),
        PartialRagService(),
        HistoryContextBuilder(),
        reference_generator=ReferenceGenerator(),
    )
    settings = Settings(
        _env_file=None,
        source_documents_dir=tmp_path,
        sqlite_path=tmp_path / "metadata.sqlite3",
        qdrant_path=tmp_path / "qdrant",
    )
    try:
        with TestClient(
            create_app(settings, document_service=object(), chat_orchestrator=orchestrator)
        ) as client:
            response = client.post("/api/chat/stream", json={"question": "请假流程和时长？"})
            events = parse_sse(response.text)
            session_id = events[0]["data"]["session_id"]
            final = events[-1]["data"]

            assert "reference_question" not in response.text
            assert final["refused"] is False
            assert final["reference_answer"] == "审批时长需向负责人确认。"
            assert final["citations"][0]["reference_id"] == "S1"

            restored = client.get(f"/api/chat/sessions/{session_id}").json()
            assistant = restored["messages"][-1]
            assert assistant["content"] == "资料可确认应先申请。"
            assert assistant["reference_answer"] == "审批时长需向负责人确认。"
    finally:
        repository.close()


def test_stateful_chat_persists_session_and_feedback(tmp_path):
    repository = SqliteDocumentRepository(tmp_path / "metadata.sqlite3")
    orchestrator = ChatOrchestrator(
        repository,
        NeverCalledRewriter(),
        StatefulRagService(),
        HistoryContextBuilder(),
    )
    settings = Settings(
        _env_file=None,
        source_documents_dir=tmp_path,
        sqlite_path=tmp_path / "metadata.sqlite3",
        qdrant_path=tmp_path / "qdrant",
    )
    try:
        with TestClient(
            create_app(settings, document_service=object(), chat_orchestrator=orchestrator)
        ) as client:
            response = client.post("/api/chat/stream", json={"question": "问题"})
            events = parse_sse(response.text)
            session_id = events[0]["data"]["session_id"]
            assistant_id = events[-1]["data"]["message_id"]
            assert events[0]["data"]["stage"] == "accepted"
            assert events[-1]["event"] == "final"

            restored = client.get(f"/api/chat/sessions/{session_id}")
            assert restored.status_code == 200
            assert len(restored.json()["messages"]) == 2

            feedback = client.put(
                f"/api/chat/messages/{assistant_id}/feedback", json={"helpful": True}
            )
            assert feedback.status_code == 200
            assert feedback.json()["helpful"] is True
            assert client.put(
                f"/api/chat/messages/{assistant_id}/feedback", json={"helpful": False}
            ).json()["helpful"] is False
            assert client.put(
                "/api/chat/messages/not-user/feedback", json={"helpful": True}
            ).status_code == 404
    finally:
        repository.close()


def test_feedback_reason_is_returned_and_helpful_feedback_cannot_carry_one(tmp_path):
    repository = SqliteDocumentRepository(tmp_path / "metadata.sqlite3")
    orchestrator = ChatOrchestrator(
        repository,
        NeverCalledRewriter(),
        StatefulRagService(),
        HistoryContextBuilder(),
    )
    settings = Settings(
        _env_file=None,
        source_documents_dir=tmp_path,
        sqlite_path=tmp_path / "metadata.sqlite3",
        qdrant_path=tmp_path / "qdrant",
    )
    try:
        with TestClient(
            create_app(settings, document_service=object(), chat_orchestrator=orchestrator)
        ) as client:
            response = client.post("/api/chat/stream", json={"question": "问题"})
            assistant_id = parse_sse(response.text)[-1]["data"]["message_id"]

            saved = client.put(
                f"/api/chat/messages/{assistant_id}/feedback",
                json={"helpful": False, "reason": "missing_step"},
            )
            assert saved.status_code == 200
            assert saved.json()["reason"] == "missing_step"

            invalid = client.put(
                f"/api/chat/messages/{assistant_id}/feedback",
                json={"helpful": True, "reason": "too_slow"},
            )
            assert invalid.status_code in {400, 422}
    finally:
        repository.close()


def test_stateful_chat_persists_audience_scope_on_user_message(tmp_path):
    repository = SqliteDocumentRepository(tmp_path / "metadata.sqlite3")
    orchestrator = ChatOrchestrator(
        repository,
        NeverCalledRewriter(),
        StatefulRagService(),
        HistoryContextBuilder(),
    )
    settings = Settings(
        _env_file=None,
        source_documents_dir=tmp_path,
        sqlite_path=tmp_path / "metadata.sqlite3",
        qdrant_path=tmp_path / "qdrant",
    )
    try:
        with TestClient(
            create_app(settings, document_service=object(), chat_orchestrator=orchestrator)
        ) as client:
            response = client.post(
                "/api/chat/stream",
                json={"question": "护士培训流程？", "audience_scope": "nurse"},
            )
            session_id = parse_sse(response.text)[0]["data"]["session_id"]
            restored = client.get(f"/api/chat/sessions/{session_id}").json()

        assert restored["messages"][0]["audience_scope"] == "nurse"
    finally:
        repository.close()


def test_stateful_invalid_delta_persists_error_and_emits_terminal_ids(tmp_path):
    class UnsafeStatefulRag:
        async def stream(self, question):
            yield ChatEvent(type="status", data={"stage": "generating"})
            yield ChatEvent(
                type="answer_delta",
                data={"text": "C:" + chr(92) + "private" + chr(92) + "provider.log", "source_ids": ["S1"]},
            )

    repository = SqliteDocumentRepository(tmp_path / "metadata.sqlite3")
    orchestrator = ChatOrchestrator(
        repository,
        NeverCalledRewriter(),
        UnsafeStatefulRag(),
        HistoryContextBuilder(),
    )
    settings = Settings(
        _env_file=None,
        source_documents_dir=tmp_path,
        sqlite_path=tmp_path / "metadata.sqlite3",
        qdrant_path=tmp_path / "qdrant",
    )
    try:
        with TestClient(
            create_app(settings, document_service=object(), chat_orchestrator=orchestrator)
        ) as client:
            response = client.post("/api/chat/stream", json={"question": "问题"})
            events = parse_sse(response.text)
            assert events[-1]["event"] == "error"
            assert events[-1]["data"]["reason_code"] == "INVALID_EVENT"
            assert events[-1]["data"]["session_id"] == events[0]["data"]["session_id"]
            assert events[-1]["data"]["message_id"]
            rows = repository.list_chat_messages(events[0]["data"]["session_id"])
            assert rows[-1].status == "error"
            assert rows[-1].reason_code == "INVALID_EVENT"
            assert "private" not in response.text
    finally:
        repository.close()
