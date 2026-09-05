from __future__ import annotations

import pytest
from app.chat.history import HistoryContextBuilder
from app.chat.models import ChatMessage
from app.chat.orchestrator import ChatOrchestrator
from app.chat.rewriter import RewriteError
from app.rag.models import ChatEvent
from app.storage.sqlite import SqliteDocumentRepository


class FakeRepository:
    def __init__(self, with_history: bool = False):
        self.created_session_id = "new-session"
        self.messages: list[ChatMessage] = []
        self.last_user = None
        self.last_assistant = None
        if with_history:
            self.messages = [
                ChatMessage(
                    "old-user",
                    "s1",
                    "user",
                    "透析中低血压怎么处理？",
                    "submitted",
                    None,
                    (),
                    None,
                    "1",
                ),
                ChatMessage(
                    "old-assistant",
                    "s1",
                    "assistant",
                    "按制度处理。",
                    "answered",
                    None,
                    (),
                    None,
                    "2",
                    "old-user",
                ),
            ]

    def create_chat_session(self):
        self.messages = []
        return self.created_session_id

    def chat_session_exists(self, session_id):
        return session_id == "s1" or session_id == self.created_session_id

    def append_user_message(self, session_id, content):
        self.last_user = ChatMessage(
            "new-user",
            session_id,
            "user",
            content,
            "submitted",
            None,
            (),
            None,
            "3",
        )
        self.messages.append(self.last_user)
        return self.last_user

    def set_rewritten_question(self, message_id, rewritten_question):
        self.last_user = ChatMessage(
            self.last_user.message_id,
            self.last_user.session_id,
            "user",
            self.last_user.content,
            self.last_user.status,
            rewritten_question,
            self.last_user.citations,
            self.last_user.reason_code,
            self.last_user.created_at,
            self.last_user.reply_to_message_id,
        )
        self.messages[-1] = self.last_user

    def append_assistant_message(
        self,
        session_id,
        reply_to_message_id,
        content,
        status,
        citations,
        reason_code,
        reference_answer=None,
    ):
        self.last_assistant = ChatMessage(
            "new-assistant",
            session_id,
            "assistant",
            content,
            status,
            None,
            tuple(citations),
            reason_code,
            "4",
            reply_to_message_id,
            reference_answer,
        )
        self.messages.append(self.last_assistant)
        return self.last_assistant

    def list_chat_messages(self, session_id):
        return list(self.messages)


class FakeRewriter:
    def __init__(self, error: bool = False):
        self.calls = 0
        self.error = error

    async def rewrite(self, question, history):
        self.calls += 1
        if self.error:
            raise RewriteError()
        return "血液透析过程中低血压如何预防？"


class FakeRag:
    def __init__(self):
        self.questions = []

    async def stream(self, question):
        self.questions.append(question)
        yield ChatEvent(type="status", data={"stage": "retrieving"})
        yield ChatEvent(type="answer_delta", data={"text": "请按制度。"})
        yield ChatEvent(type="final", data={"refused": False, "citations": []})


class FakeReferenceGenerator:
    def __init__(self, answer="通用参考答案", error=None):
        self.answer = answer
        self.error = error
        self.questions = []

    async def generate(self, question):
        self.questions.append(question)
        if self.error is not None:
            raise self.error
        return self.answer


def make_orchestrator(with_history=False, rewrite_error=False):
    repository = FakeRepository(with_history=with_history)
    rewriter = FakeRewriter(error=rewrite_error)
    rag = FakeRag()
    return (
        ChatOrchestrator(
            repository,
            rewriter,
            rag,
            HistoryContextBuilder(),
        ),
        repository,
        rewriter,
        rag,
    )


@pytest.mark.asyncio
async def test_first_question_creates_session_without_rewrite():
    orchestrator, repository, rewriter, rag = make_orchestrator()

    events = [event async for event in orchestrator.stream("请假流程？")]

    assert events[0].data["stage"] == "accepted"
    assert events[0].data["session_id"] == repository.created_session_id
    assert rewriter.calls == 0
    assert rag.questions == ["请假流程？"]


@pytest.mark.asyncio
async def test_follow_up_rewrites_before_rag_and_persists_original_and_rewritten():
    orchestrator, repository, _rewriter, rag = make_orchestrator(with_history=True)

    events = [event async for event in orchestrator.stream("那怎么预防？", "s1")]

    assert [event.data["stage"] for event in events if event.type == "status"][:2] == [
        "accepted",
        "rewriting",
    ]
    assert rag.questions == ["血液透析过程中低血压如何预防？"]
    assert repository.last_user.rewritten_question == "血液透析过程中低血压如何预防？"


@pytest.mark.asyncio
async def test_rewrite_failure_never_calls_rag_and_persists_safe_error():
    orchestrator, repository, _rewriter, rag = make_orchestrator(
        with_history=True, rewrite_error=True
    )

    events = [event async for event in orchestrator.stream("那怎么办？", "s1")]

    assert rag.questions == []
    assert events[-1].type == "error"
    assert events[-1].data["reason_code"] == "FOLLOW_UP_UNAVAILABLE"
    assert repository.last_assistant.status == "error"


@pytest.mark.asyncio
async def test_sqlite_repository_restores_complete_stateful_turn(tmp_path):
    repository = SqliteDocumentRepository(tmp_path / "metadata.sqlite3")

    class RAG:
        async def stream(self, question):
            yield ChatEvent(type="answer_delta", data={"text": "安全提示"})
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
                            "excerpt": "公开样例制度摘录",
                        }
                    ],
                },
            )

    class Rewriter:
        async def rewrite(self, question, history):
            return "独立问题"

    try:
        orchestrator = ChatOrchestrator(
            repository, Rewriter(), RAG(), HistoryContextBuilder()
        )
        events = [event async for event in orchestrator.stream("首问")]
        session_id = events[0].data["session_id"]
        messages = repository.list_chat_messages(session_id)
        assert [message.role for message in messages] == ["user", "assistant"]
        assert messages[1].status == "answered"
        assert messages[1].citations[0]["reference_id"] == "S1"
    finally:
        repository.close()


@pytest.mark.asyncio
async def test_invalid_answer_delta_is_persisted_as_terminal_error_with_ids():
    class UnsafeRag:
        async def stream(self, question):
            yield ChatEvent(type="status", data={"stage": "generating"})
            yield ChatEvent(
                type="answer_delta",
                data={
                    "text": "Traceback: C:" + chr(92) + "private" + chr(92) + "provider.log",
                    "source_ids": ["S1"],
                },
            )

    repository = FakeRepository()
    orchestrator = ChatOrchestrator(repository, FakeRewriter(), UnsafeRag())

    events = [event async for event in orchestrator.stream("问题")]

    assert events[-1].type == "error"
    assert events[-1].data["reason_code"] == "INVALID_EVENT"
    assert events[-1].data["session_id"] == repository.created_session_id
    assert events[-1].data["message_id"] == repository.last_assistant.message_id
    assert repository.last_assistant.status == "error"
    assert repository.last_assistant.content == "当前服务暂时不可用，请稍后重试。"


@pytest.mark.asyncio
async def test_incomplete_final_citation_is_persisted_as_terminal_error_with_ids():
    class IncompleteCitationRag:
        async def stream(self, question):
            yield ChatEvent(type="final", data={
                "refused": False,
                "citations": [{"reference_id": "S1", "file_name": "制度.docx"}],
            })

    repository = FakeRepository()
    orchestrator = ChatOrchestrator(repository, FakeRewriter(), IncompleteCitationRag())

    events = [event async for event in orchestrator.stream("问题")]

    assert events[-1].type == "error"
    assert events[-1].data["reason_code"] == "INVALID_EVENT"
    assert repository.last_assistant.status == "error"


@pytest.mark.asyncio
async def test_reference_answer_is_added_only_for_grounding_refusals():
    class RefusedRag:
        async def stream(self, question):
            yield ChatEvent(
                type="final",
                data={
                    "refused": True,
                    "reason_code": "INSUFFICIENT_EVIDENCE",
                    "citations": [],
                },
            )

    repository = FakeRepository()
    reference = FakeReferenceGenerator()
    orchestrator = ChatOrchestrator(
        repository,
        FakeRewriter(),
        RefusedRag(),
        reference_generator=reference,
    )

    events = [event async for event in orchestrator.stream("问题")]

    assert reference.questions == ["问题"]
    assert events[-1].type == "final"
    assert events[-1].data["refused"] is True
    assert events[-1].data["citations"] == []
    assert events[-1].data["reference_answer"] == "通用参考答案"
    assert repository.last_assistant.status == "refused"
    assert repository.last_assistant.reference_answer == "通用参考答案"


@pytest.mark.asyncio
async def test_reference_answer_is_not_generated_for_model_refusal():
    class RefusedRag:
        async def stream(self, question):
            yield ChatEvent(
                type="final",
                data={
                    "refused": True,
                    "reason_code": "MODEL_REFUSED",
                    "citations": [],
                },
            )

    repository = FakeRepository()
    reference = FakeReferenceGenerator()
    orchestrator = ChatOrchestrator(
        repository,
        FakeRewriter(),
        RefusedRag(),
        reference_generator=reference,
    )

    events = [event async for event in orchestrator.stream("问题")]

    assert reference.questions == []
    assert events[-1].type == "final"
    assert "reference_answer" not in events[-1].data
    assert repository.last_assistant.reference_answer is None


@pytest.mark.asyncio
async def test_partial_grounded_answer_gets_a_separate_unverified_reference():
    class PartialAnswerRag:
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

    repository = FakeRepository()
    reference = FakeReferenceGenerator()
    orchestrator = ChatOrchestrator(
        repository,
        FakeRewriter(),
        PartialAnswerRag(),
        reference_generator=reference,
    )

    events = [event async for event in orchestrator.stream("请假流程和审批时长？")]

    assert reference.questions == ["审批通常需要多长时间？"]
    assert events[-1].data["refused"] is False
    assert events[-1].data["reference_answer"] == "通用参考答案"
    assert repository.last_assistant.content == "资料可确认应先申请。"
    assert repository.last_assistant.reference_answer == "通用参考答案"


@pytest.mark.asyncio
async def test_candidate_error_summary_is_validated_then_persisted_as_one_terminal():
    class CandidateRag:
        requires_request_scope = True

        async def stream(self, question):
            yield ChatEvent(type="status", data={"stage": "routing"})
            yield ChatEvent(
                type="error",
                data={
                    "stage": "routing",
                    "reason_code": "ROUTER_UNAVAILABLE",
                    "workflow_summary": {
                        "workflow_version": "agent_workflow_v2a",
                        "run_id": "0123456789abcdef",
                        "route": None,
                        "outcome": "error",
                        "verifier_status": "not_run",
                        "http_calls": 1,
                        "elapsed_ms": 2,
                        "reason_code": "ROUTER_UNAVAILABLE",
                    },
                    "reference_allowed": False,
                },
            )

    repository = FakeRepository()
    orchestrator = ChatOrchestrator(repository, FakeRewriter(), CandidateRag())

    events = [event async for event in orchestrator.stream("问题")]

    assert [event.type for event in events] == ["status", "status", "error"]
    assert events[-1].data["reason_code"] == "ROUTER_UNAVAILABLE"
    assert repository.last_assistant.status == "error"


@pytest.mark.asyncio
async def test_candidate_timeout_is_safe_terminal_and_does_not_fake_answer():
    class TimeoutRag:
        requires_request_scope = True

        async def stream(self, question):
            raise TimeoutError
            yield  # pragma: no cover - keeps this function an async generator

    repository = FakeRepository()
    orchestrator = ChatOrchestrator(repository, FakeRewriter(), TimeoutRag())

    events = [event async for event in orchestrator.stream("问题")]

    assert events[-1].type == "error"
    assert events[-1].data["reason_code"] == "WORKFLOW_TIMEOUT"
    assert repository.last_assistant.status == "error"


@pytest.mark.asyncio
async def test_candidate_verification_failure_cannot_trigger_reference_generation():
    class CandidateRag:
        requires_request_scope = True

        async def stream(self, question):
            yield ChatEvent(
                type="final",
                data={
                    "refused": True,
                    "citations": [],
                    "reason_code": "ANSWER_NOT_VERIFIABLE",
                    "workflow_summary": {
                        "workflow_version": "agent_workflow_v2a",
                        "run_id": "0123456789abcdef",
                        "route": "verify",
                        "outcome": "refused",
                        "verifier_status": "failed",
                        "http_calls": 5,
                        "elapsed_ms": 2,
                        "reason_code": "ANSWER_NOT_VERIFIABLE",
                    },
                    "reference_allowed": False,
                },
            )

    repository = FakeRepository()
    reference = FakeReferenceGenerator()
    orchestrator = ChatOrchestrator(
        repository, FakeRewriter(), CandidateRag(), reference_generator=reference
    )

    events = [event async for event in orchestrator.stream("复杂问题")]

    assert events[-1].type == "final"
    assert events[-1].data["refused"] is True
    assert "reference_answer" not in events[-1].data
    assert reference.questions == []
