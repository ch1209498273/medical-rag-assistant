"""FastAPI application assembly for the local-only backend."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agents import (
    AgentWorkflow,
    DeepSeekRouter,
    DeterministicRouter,
    DeterministicVerifier,
)
from app.agents.contracts import AgentVariant, validate_agent_variant
from app.api.admin_feedback import create_feedback_review_router
from app.api.chat import create_chat_router
from app.api.documents import create_documents_router
from app.api.evaluations import create_evaluations_router
from app.api.health import router as health_router
from app.chat.history import HistoryContextBuilder
from app.chat.orchestrator import ChatOrchestrator
from app.chat.reference import ReferenceAnswerService
from app.chat.rewriter import FollowUpRewriter
from app.evaluation.service import EvaluationService
from app.feedback.evidence import LocalEvidenceResolver
from app.feedback.repository import FeedbackProjectionRepository
from app.feedback.review_service import FeedbackReviewService
from app.ingestion.service import IngestionService
from app.providers.contracts import KnowledgeProvider, LanguageModelProvider
from app.providers.deepseek import DeepSeekClient
from app.providers.demo import (
    DemoKnowledgeProvider,
    DemoLanguageModel,
    load_demo_scenarios,
)
from app.providers.minimax import create_evaluation_generator
from app.providers.siliconflow import SiliconFlowClient
from app.rag.answering import AnswerService, RagService
from app.rag.deepseek_verifier import DeepSeekClaimVerifier
from app.rag.lexical import ActiveLexicalIndex
from app.rag.retrieval import RetrievalService
from app.settings import Settings, get_settings, require_cloud_keys
from app.storage.qdrant import QdrantLocalVectorStore
from app.storage.sqlite import SqliteDocumentRepository

LOGGER_NAME = "hemodialysis.backend"
APP_VERSION = "0.1.0"


def build_agent_services(
    variant: AgentVariant,
    *,
    direct_service: Any,
    verify_service: Any,
) -> tuple[Any, Any]:
    """Select already-built candidate services for one explicit variant."""

    selected = validate_agent_variant(variant)
    if selected == "router_direct_fallback":
        return direct_service, direct_service
    return direct_service, verify_service


class _SafeLogFilter(logging.Filter):
    """Keep the baseline application logger to approved operational fields."""

    _allowed_fields = frozenset(
        {"stage", "duration_ms", "status_code", "token_usage", "request_id"}
    )

    def filter(self, record: logging.LogRecord) -> bool:
        # The baseline logger is intentionally quiet unless a caller uses the
        # structured ``extra`` fields listed above.  This prevents accidental
        # request/response, environment, authorization, or path logging.
        record.msg = "operational event"
        record.args = ()
        for key in tuple(record.__dict__):
            if key.startswith("_") or key in _STANDARD_LOG_FIELDS:
                continue
            if key not in self._allowed_fields:
                record.__dict__.pop(key, None)
        return True


_STANDARD_LOG_FIELDS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
    }
)


@dataclass
class DocumentRuntime:
    """Resources owned by one application lifespan."""

    repository: SqliteDocumentRepository
    vector_store: QdrantLocalVectorStore
    provider: KnowledgeProvider
    service: IngestionService
    closed: bool = False
    _provider_close_started: bool = False
    _vector_close_started: bool = False
    _repository_close_started: bool = False
    _ingestion_close_started: bool = False

    async def close(self) -> None:
        if self.closed:
            return
        ingestion_error: BaseException | None = None
        provider_error: BaseException | None = None
        vector_error: BaseException | None = None
        repository_error: BaseException | None = None
        if not self._ingestion_close_started:
            self._ingestion_close_started = True
            close_ingestion = getattr(self.service, "aclose", None)
            if callable(close_ingestion):
                try:
                    await close_ingestion()
                except BaseException as error:  # noqa: BLE001 - teardown continues
                    ingestion_error = error
        if not self._provider_close_started:
            self._provider_close_started = True
            try:
                await self.provider.aclose()
            except BaseException as error:  # noqa: BLE001 - teardown continues
                provider_error = error
        if not self._vector_close_started:
            self._vector_close_started = True
            try:
                self.vector_store.close()
            except BaseException as error:  # noqa: BLE001 - teardown continues
                vector_error = error
        if not self._repository_close_started:
            self._repository_close_started = True
            try:
                self.repository.close()
            except BaseException as error:  # noqa: BLE001 - terminal cleanup is recorded
                repository_error = error
        self.closed = True
        if ingestion_error is not None:
            raise ingestion_error
        if provider_error is not None:
            raise provider_error
        if vector_error is not None:
            raise vector_error
        if repository_error is not None:
            raise repository_error


@dataclass
class RagRuntime:
    """Resources and services owned by the answer lifespan."""

    document_runtime: DocumentRuntime
    deepseek: LanguageModelProvider
    retrieval: RetrievalService
    answerer: AnswerService
    service: RagService
    rewriter: FollowUpRewriter | None = None
    history_builder: HistoryContextBuilder | None = None
    orchestrator: ChatOrchestrator | None = None
    workflow: AgentWorkflow | None = None
    closed: bool = False
    _deepseek_close_started: bool = False
    _document_close_started: bool = False

    async def close(self) -> None:
        """Close DeepSeek first, then the shared ingestion resources once."""

        if self.closed:
            return
        deepseek_error: BaseException | None = None
        document_error: BaseException | None = None
        if not self._deepseek_close_started:
            self._deepseek_close_started = True
            try:
                await self.deepseek.aclose()
            except BaseException as error:  # noqa: BLE001 - teardown must continue
                deepseek_error = error
        if not self._document_close_started:
            self._document_close_started = True
            try:
                await self.document_runtime.close()
            except BaseException as error:  # noqa: BLE001 - teardown is terminal
                document_error = error
        self.closed = True
        if deepseek_error is not None:
            raise deepseek_error
        if document_error is not None:
            raise document_error


@dataclass
class EvaluationRuntime:
    """Evaluation service plus only the provider clients it created."""

    service: EvaluationService
    owned_clients: tuple[Any, ...] = ()
    closed: bool = False
    _close_started: bool = False

    async def close(self) -> None:
        if self.closed or self._close_started:
            return
        self._close_started = True
        first_error: BaseException | None = None
        for client in self.owned_clients:
            try:
                await _dispose_async(client)
            except BaseException as error:  # noqa: BLE001 - teardown continues
                if first_error is None:
                    first_error = error
        self.closed = True
        if first_error is not None:
            raise first_error


async def build_document_runtime_async(
    settings: Settings,
    *,
    transport: Any = None,
    qdrant_create_if_missing: bool = True,
) -> DocumentRuntime:
    """Build document resources with awaitable, explicit ownership cleanup."""

    if not isinstance(qdrant_create_if_missing, bool):
        raise TypeError("qdrant_create_if_missing must be a boolean")

    if settings.app_runtime_mode == "cloud":
        require_cloud_keys(settings)

    acquired: list[Any] = []
    try:
        repository = SqliteDocumentRepository(settings.sqlite_path)
        acquired.append(repository)
        if qdrant_create_if_missing:
            vector_store = QdrantLocalVectorStore(settings.qdrant_path)
        else:
            vector_store = QdrantLocalVectorStore(
                settings.qdrant_path, create_if_missing=False
            )
        acquired.append(vector_store)
        if settings.app_runtime_mode == "demo":
            provider: KnowledgeProvider = DemoKnowledgeProvider()
        else:
            provider = SiliconFlowClient(
                settings.siliconflow_api_key,
                settings.siliconflow_base_url,
                embedding_model=settings.embedding_model,
                reranker_model=settings.reranker_model,
                ocr_model=settings.ocr_model,
                transport=transport,
                max_retries=(
                    0
                    if getattr(settings, "answer_workflow", "baseline_v1")
                    == "agent_workflow_v2a"
                    else 3
                ),
            )
        acquired.append(provider)
        service = IngestionService(
            repository=repository,
            vector_store=vector_store,
            embedder=provider,
            source_documents_dir=settings.source_documents_dir,
            ocr_client=provider,
            max_file_bytes=settings.ingestion_max_file_bytes,
            max_pdf_pages=settings.ingestion_max_pdf_pages,
            max_pdf_pixels=settings.ingestion_max_pdf_pixels,
            max_pdf_render_bytes=settings.ingestion_max_pdf_render_bytes,
            max_docx_paragraphs=settings.ingestion_max_docx_paragraphs,
            max_docx_uncompressed_bytes=settings.ingestion_max_docx_uncompressed_bytes,
            max_docx_zip_entries=settings.ingestion_max_docx_zip_entries,
            max_text_chars=settings.ingestion_max_text_chars,
            max_file_seconds=settings.ingestion_file_timeout_seconds,
            max_scan_files=settings.ingestion_scan_max_files,
        )
        return DocumentRuntime(repository, vector_store, provider, service)
    except BaseException:
        for resource in reversed(acquired):
            try:
                await _dispose_async(resource)
            except BaseException:  # noqa: BLE001 - cleanup must continue
                logging.getLogger(LOGGER_NAME).warning(
                    "runtime cleanup failed", extra={"stage": "runtime_cleanup"}
                )
        raise


def build_document_runtime(
    settings: Settings,
    *,
    transport: Any = None,
) -> DocumentRuntime:
    """Synchronous compatibility wrapper for callers outside an event loop."""

    _reject_running_loop()
    return asyncio.run(build_document_runtime_async(settings, transport=transport))


def build_rag_runtime(
    settings: Settings,
    document_runtime: DocumentRuntime,
    *,
    transport: Any = None,
) -> RagRuntime:
    """Synchronous compatibility wrapper for callers outside an event loop."""

    _reject_running_loop()
    return asyncio.run(
        build_rag_runtime_async(settings, document_runtime, transport=transport)
    )


async def build_rag_runtime_async(
    settings: Settings,
    document_runtime: DocumentRuntime,
    *,
    transport: Any = None,
    agent_variant: AgentVariant | None = None,
) -> RagRuntime:
    """Attach answer services with awaitable, explicit ownership cleanup."""

    candidate_variant: AgentVariant = "router_conditional_verify_fallback"
    if agent_variant is not None:
        candidate_variant = validate_agent_variant(agent_variant)
    if settings.app_runtime_mode == "cloud":
        require_cloud_keys(settings)

    acquired: list[Any] = []
    try:
        if settings.app_runtime_mode == "demo":
            deepseek: LanguageModelProvider = DemoLanguageModel(
                load_demo_scenarios(settings.demo_questions_path)
            )
        else:
            deepseek = DeepSeekClient(
                settings.deepseek_api_key,
                settings.deepseek_base_url,
                model=settings.deepseek_model,
                max_tokens=settings.deepseek_structured_max_tokens,
                max_tokens_limit=settings.deepseek_answer_max_tokens,
                transport=transport,
                max_retries=(
                    0
                    if getattr(settings, "answer_workflow", "baseline_v1")
                    == "agent_workflow_v2a"
                    else 3
                ),
                max_stream_bytes=settings.deepseek_stream_max_bytes,
                max_stream_chars=settings.deepseek_stream_max_chars,
                max_stream_events=settings.deepseek_stream_max_events,
                stream_timeout_seconds=settings.deepseek_stream_timeout_seconds,
            )
        acquired.append(deepseek)
        lexical_index = (
            ActiveLexicalIndex(document_runtime.vector_store)
            if settings.retrieval_strategy in {"hybrid", "hybrid_normalized"}
            else None
        )
        retrieval = RetrievalService(
            repository=document_runtime.repository,
            vector_store=document_runtime.vector_store,
            embedder=document_runtime.provider,
            reranker=document_runtime.provider,
            retrieval_limit=settings.retrieval_limit,
            rerank_limit=settings.rerank_limit,
            relevance_threshold=settings.relevance_threshold,
            retrieval_strategy=settings.retrieval_strategy,
            lexical_index=lexical_index,
            fusion_profile=settings.retrieval_fusion_profile,
            vector_weight=settings.vector_rrf_weight,
            lexical_weight=settings.lexical_rrf_weight,
            diagnostic_trace=settings.retrieval_diagnostic_trace,
            # Demo assets are fictional and intentionally have no operator
            # metadata-management step.  Keep the exception explicit and
            # runtime-scoped; cloud/real-document mode remains fail-closed.
            require_business_eligibility=settings.app_runtime_mode != "demo",
        )
        candidate_workflow = (
            getattr(settings, "answer_workflow", "baseline_v1")
            == "agent_workflow_v2a"
        )
        answer_kwargs = {
            "deepseek": deepseek,
            "max_answer_bytes": settings.deepseek_stream_max_bytes,
            "max_answer_chars": settings.deepseek_stream_max_chars,
            "max_answer_events": settings.deepseek_stream_max_events,
            "answer_timeout_seconds": settings.deepseek_stream_timeout_seconds,
            "prompt_profile": settings.answer_prompt_profile,
            "answer_max_tokens": settings.deepseek_answer_max_tokens,
            "answer_thinking_mode": settings.deepseek_answer_thinking_mode,
        }
        verifier = (
            DeepSeekClaimVerifier(deepseek)
            if settings.stage_b_verification_enabled
            else None
        )
        answerer = AnswerService(verifier=verifier, **answer_kwargs)
        service: Any = RagService(
            retrieval=retrieval,
            answerer=answerer,
            diagnostic_trace=settings.retrieval_diagnostic_trace,
        )
        workflow: AgentWorkflow | None = None
        if candidate_workflow:
            direct_answerer = AnswerService(verifier=None, **answer_kwargs)
            if settings.app_runtime_mode == "demo":
                candidate_router = DeterministicRouter()
                candidate_verifier = DeterministicVerifier()
            else:
                candidate_router = DeepSeekRouter(deepseek)
                candidate_verifier = DeepSeekClaimVerifier(deepseek)
            verify_answerer = AnswerService(
                verifier=candidate_verifier, **answer_kwargs
            )
            direct_service = RagService(
                retrieval=retrieval,
                answerer=direct_answerer,
                diagnostic_trace=settings.retrieval_diagnostic_trace,
            )
            verify_service = RagService(
                retrieval=retrieval,
                answerer=verify_answerer,
                diagnostic_trace=settings.retrieval_diagnostic_trace,
            )
            selected_direct_service, selected_verify_service = build_agent_services(
                candidate_variant,
                direct_service=direct_service,
                verify_service=verify_service,
            )
            workflow = AgentWorkflow(
                router=candidate_router,
                direct_service=selected_direct_service,
                verify_service=selected_verify_service,
                baseline_service=service,
                variant=candidate_variant,
                repository=(
                    document_runtime.repository
                    if settings.app_runtime_mode != "demo"
                    else None
                ),
            )
            service = workflow
        history_builder = HistoryContextBuilder(
            max_turns=settings.chat_history_max_turns,
            max_chars=settings.chat_history_max_chars,
        )
        rewriter = FollowUpRewriter(deepseek)
        reference_generator = ReferenceAnswerService(deepseek)
        orchestrator = ChatOrchestrator(
            document_runtime.repository,
            rewriter,
            service,
            history_builder,
            reference_generator,
        )
        return RagRuntime(
            document_runtime,
            deepseek,
            retrieval,
            answerer,
            service,
            rewriter,
            history_builder,
            orchestrator,
            workflow,
        )
    except BaseException:
        for resource in reversed(acquired):
            try:
                await _dispose_async(resource)
            except BaseException:  # noqa: BLE001 - cleanup must continue
                logging.getLogger(LOGGER_NAME).warning(
                    "runtime cleanup failed", extra={"stage": "runtime_cleanup"}
                )
        raise


def build_evaluation_runtime(
    settings: Settings,
    document_runtime: DocumentRuntime,
    rag_service: Any,
    *,
    deepseek: DeepSeekClient | None = None,
    transport: Any = None,
) -> EvaluationRuntime:
    """Synchronous compatibility wrapper for evaluation runtime assembly."""

    _reject_running_loop()
    return asyncio.run(
        build_evaluation_runtime_async(
            settings,
            document_runtime,
            rag_service,
            deepseek=deepseek,
            transport=transport,
        )
    )


async def build_evaluation_runtime_async(
    settings: Settings,
    document_runtime: DocumentRuntime,
    rag_service: Any,
    *,
    deepseek: DeepSeekClient | None = None,
    transport: Any = None,
) -> EvaluationRuntime:
    """Assemble private evaluation resources from the active indexed snapshot."""

    if settings.app_runtime_mode == "demo":
        raise RuntimeError("evaluation runtime is unavailable in demo mode")
    if rag_service is None:
        raise ValueError("RAG service is required for evaluation")
    ds_key = settings.deepseek_api_key
    generator = create_evaluation_generator(
        deepseek_api_key=ds_key,
        deepseek_base_url=settings.deepseek_base_url,
        deepseek_model=settings.deepseek_model,
        # Candidate generation must use its dedicated Responses client, never
        # a caller-injected formal-answer Chat Completions client.
        deepseek_client=None,
        transport=transport,
    )
    def active_source_snapshot() -> Any:
        active_versions = document_runtime.repository.active_version_ids()
        snapshot = getattr(document_runtime.vector_store, "snapshot_active_chunks", None)
        if not callable(snapshot):
            raise RuntimeError(  # noqa: TRY004 - missing runtime seam is operational
                "active evaluation source snapshot is unavailable"
            )
        return snapshot(active_versions)

    service = EvaluationService(
        generator=generator,
        source_provider=active_source_snapshot,
        rag_service=rag_service,
        storage_dir=Path(settings.private_data_dir) / "evaluations",
        diagnostic_trace=settings.retrieval_diagnostic_trace,
    )
    owned_clients: list[Any] = []
    generators = getattr(generator, "generators", (generator,))
    seen: set[int] = set()
    for item in generators:
        client = getattr(item, "client", None)
        if client is None or client is deepseek or id(client) in seen:
            continue
        if callable(getattr(client, "aclose", None)):
            seen.add(id(client))
            owned_clients.append(client)
    return EvaluationRuntime(service=service, owned_clients=tuple(owned_clients))


async def _dispose_async(resource: Any) -> None:
    close_async = getattr(resource, "aclose", None)
    if callable(close_async):
        await close_async()
        return
    close = getattr(resource, "close", None)
    if callable(close):
        result = close()
        if hasattr(result, "__await__"):
            await result


def _reject_running_loop() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError("synchronous runtime builder cannot run inside a running event loop")


def configure_logging() -> logging.Logger:
    """Configure a minimal application logger with a safe field baseline."""

    logger = logging.getLogger(LOGGER_NAME)
    logger.addFilter(_SafeLogFilter())
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def create_app(
    settings: Settings | None = None,
    document_service: Any | None = None,
    document_runtime: DocumentRuntime | None = None,
    rag_service: Any | None = None,
    chat_orchestrator: ChatOrchestrator | None = None,
    evaluation_service: Any | None = None,
    feedback_review_service: Any | None = None,
    evidence_resolver: Any | None = None,
) -> FastAPI:
    """Create the API app; local resources are opened only during lifespan."""

    selected = settings or Settings()
    configure_logging()
    owned_feedback_repository: FeedbackProjectionRepository | None = None
    if selected.feedback_review_ui_enabled and feedback_review_service is None:
        owned_feedback_repository = FeedbackProjectionRepository(
            Path(selected.private_data_dir) / "feedback" / "feedback.db"
        )
        feedback_review_service = FeedbackReviewService(owned_feedback_repository)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        runtime = document_runtime
        rag_runtime: RagRuntime | None = None
        evaluation_runtime: EvaluationRuntime | None = None
        try:
            if runtime is None and document_service is None:
                runtime = await build_document_runtime_async(selected)
                application.state.document_runtime = runtime
                application.state.document_service = runtime.service
            elif runtime is not None and document_service is None:
                application.state.document_service = runtime.service
            if rag_service is None and runtime is not None:
                rag_runtime = await build_rag_runtime_async(selected, runtime)
                application.state.rag_runtime = rag_runtime
                application.state.rag_service = rag_runtime.service
                application.state.chat_orchestrator = getattr(
                    rag_runtime, "orchestrator", None
                )
                application.state.chat_repository = getattr(
                    runtime, "repository", None
                )
                if (
                    selected.feedback_review_ui_enabled
                    and application.state.feedback_evidence_resolver is None
                ):
                    application.state.feedback_evidence_resolver = LocalEvidenceResolver(
                        runtime.vector_store
                    )
            if (
                evaluation_service is None
                and runtime is not None
                and selected.app_runtime_mode == "cloud"
            ):
                selected_rag_service = (
                    rag_runtime.service if rag_runtime is not None else rag_service
                )
                if selected_rag_service is not None:
                    evaluation_runtime = await build_evaluation_runtime_async(
                        selected,
                        runtime,
                        selected_rag_service,
                    )
                    application.state.evaluation_runtime = evaluation_runtime
                    application.state.evaluation_service = evaluation_runtime.service
            yield
        finally:
            try:
                if evaluation_runtime is not None:
                    await evaluation_runtime.close()
            finally:
                if rag_runtime is not None:
                    await rag_runtime.close()
                elif runtime is not None:
                    await runtime.close()
                if owned_feedback_repository is not None:
                    owned_feedback_repository.close()

    app = FastAPI(
        title="Medical Knowledge Q&A Assistant",
        version=APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    app.state.document_service = document_service
    app.state.document_runtime = document_runtime
    app.state.rag_service = rag_service
    app.state.chat_orchestrator = chat_orchestrator
    app.state.chat_repository = (
        getattr(chat_orchestrator, "repository", None)
        if chat_orchestrator is not None
        else getattr(document_runtime, "repository", None)
    )
    app.state.evaluation_service = evaluation_service
    app.state.evaluation_runtime = None
    app.state.feedback_review_ui_enabled = selected.feedback_review_ui_enabled
    app.state.feedback_review_service = feedback_review_service
    app.state.feedback_evidence_resolver = evidence_resolver
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(selected.cors_origins()),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "OPTIONS"],
        allow_headers=["Content-Type", "Accept"],
    )
    app.include_router(health_router)
    app.include_router(create_documents_router())
    app.include_router(create_chat_router())
    app.include_router(create_evaluations_router())
    app.include_router(create_feedback_review_router())
    # ``get_settings`` is used by the reusable health router, while each app
    # instance receives its own immutable selection for deterministic tests and
    # to avoid stale cached environment values between local processes.
    app.dependency_overrides[get_settings] = lambda: selected
    return app


app = create_app()


def run() -> None:
    """Run the local development server when invoked by a future start script."""

    import uvicorn

    settings = get_settings()
    uvicorn.run(app, host=settings.host, port=settings.port)


__all__ = [
    "DocumentRuntime",
    "EvaluationRuntime",
    "RagRuntime",
    "app",
    "build_document_runtime",
    "build_document_runtime_async",
    "build_evaluation_runtime",
    "build_evaluation_runtime_async",
    "build_rag_runtime",
    "build_rag_runtime_async",
    "create_app",
    "require_cloud_keys",
    "run",
]
