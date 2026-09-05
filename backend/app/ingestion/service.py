"""Safe, incremental orchestration for local policy document indexing."""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import os
import stat
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import fitz

from app.domain.models import ExtractedBlock, ExtractedDocument, SourceRef
from app.domain.ports import (
    DocumentBusinessMetadataInput,
    DocumentRepository,
    DocumentVersion,
)
from app.errors import ProviderError
from app.ingestion.chunker import chunk_document
from app.ingestion.extractors import DocumentResourceLimits, extract_document

IndexStatus = Literal["active", "skipped", "failed", "needs_ocr", "in_progress"]
SUPPORTED_SUFFIXES = frozenset({".pdf", ".docx"})
LOGGER = logging.getLogger(__name__)


class DocumentLimitError(ValueError):
    """A document exceeded a finite ingestion resource budget."""


class SourceChangedError(ValueError):
    """The validated source changed before or during bounded reading."""


@dataclass(frozen=True)
class IndexResult:
    """Public, non-sensitive result for one document indexing attempt."""

    file_name: str
    status: IndexStatus
    version_id: str | None = None
    document_id: int | None = None
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class _UpsertState:
    """Lifecycle state shared by an upsert worker and its caller."""

    write_started: bool = False
    write_succeeded: bool = False
    cleanup_task: asyncio.Task[Any] | None = None


@dataclass(frozen=True)
class ScanResult:
    """Outcome of one configured-directory snapshot."""

    results: list[IndexResult]
    deactivated: list[str]
    snapshot_complete: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "results": [result.to_dict() for result in self.results],
            "deactivated": list(self.deactivated),
            "snapshot_complete": self.snapshot_complete,
        }


class IngestionService:
    """Coordinate extraction, cloud embedding, vector upsert, and publication.

    The service owns the source-directory boundary.  Callers may ask it to
    index a path, but the path must resolve to a regular, first-level PDF/DOCX
    below the configured directory.  No user-supplied path is ever passed to a
    repository or provider after this check.
    """

    def __init__(
        self,
        repository: DocumentRepository,
        vector_store: Any,
        embedder: Any,
        source_documents_dir: Path,
        *,
        ocr_client: Any = None,
        target_chars: int = 800,
        overlap_chars: int = 120,
        max_file_bytes: int = 8 * 1024 * 1024,
        max_pdf_pages: int = 200,
        max_pdf_pixels: int = 20_000_000,
        max_pdf_render_bytes: int = 4 * 1024 * 1024,
        max_docx_paragraphs: int = 10_000,
        max_docx_uncompressed_bytes: int = 32 * 1024 * 1024,
        max_docx_zip_entries: int = 2_000,
        max_text_chars: int = 500_000,
        max_file_seconds: float = 30.0,
        max_scan_files: int = 100,
        upsert_cleanup_timeout_seconds: float = 5.0,
    ) -> None:
        self.repository = repository
        self.vector_store = vector_store
        self.embedder = embedder
        self.source_documents_dir = Path(source_documents_dir).resolve()
        self.ocr_client = ocr_client
        self.target_chars = target_chars
        self.overlap_chars = overlap_chars
        self.resource_limits = DocumentResourceLimits(
            max_file_bytes=max_file_bytes,
            max_pdf_pages=max_pdf_pages,
            max_pdf_pixels=max_pdf_pixels,
            max_pdf_render_bytes=max_pdf_render_bytes,
            max_docx_paragraphs=max_docx_paragraphs,
            max_docx_uncompressed_bytes=max_docx_uncompressed_bytes,
            max_docx_zip_entries=max_docx_zip_entries,
            max_text_chars=max_text_chars,
        )
        if not 0 < max_file_seconds <= 1800.0:
            raise ValueError("max_file_seconds exceeds the hard resource bound")
        if not 1 <= max_scan_files <= 1000:
            raise ValueError("max_scan_files exceeds the hard resource bound")
        if not 0 < upsert_cleanup_timeout_seconds <= 30.0:
            raise ValueError("upsert cleanup timeout exceeds the hard resource bound")
        self.max_file_seconds = max_file_seconds
        self.max_scan_files = max_scan_files
        self.upsert_cleanup_timeout_seconds = upsert_cleanup_timeout_seconds
        self._managed_operations: set[asyncio.Future[Any]] = set()
        self._managed_upserts: set[asyncio.Task[Any]] = set()
        self._managed_cleanups: set[asyncio.Task[Any]] = set()
        self._closing = False

    async def drain(self) -> None:
        """Await every upsert worker and its cleanup before storage teardown."""

        while self._managed_operations or self._managed_upserts or self._managed_cleanups:
            managed = self._managed_operations | self._managed_upserts | self._managed_cleanups
            results = await asyncio.gather(*tuple(managed), return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    _log_managed_cleanup_failure(result)

    async def aclose(self) -> None:
        """Stop new writes and drain all managed upsert/cleanup work."""

        self._closing = True
        drain_task = asyncio.create_task(self.drain())
        try:
            await asyncio.shield(drain_task)
        except asyncio.CancelledError:
            # A caller cancellation must not let runtime storage close while a
            # worker can still issue its compensating delete.
            await asyncio.shield(drain_task)
            raise

    def _begin_operation(self) -> asyncio.Future[Any]:
        operation = asyncio.get_running_loop().create_future()
        self._managed_operations.add(operation)
        return operation

    def _finish_operation(self, operation: asyncio.Future[Any]) -> None:
        if not operation.done():
            operation.set_result(None)
        self._managed_operations.discard(operation)

    async def index(self, path: Path, *, force: bool = False) -> IndexResult:
        if self._closing:
            raise RuntimeError("ingestion service is closing")
        operation = self._begin_operation()
        try:
            return await self._index_impl(path, force=force)
        finally:
            self._finish_operation(operation)

    async def _index_impl(self, path: Path, *, force: bool = False) -> IndexResult:
        """Index one safe source file and publish only after all writes succeed."""

        safe_path = self._validate_source_path(path)
        file_name = safe_path.name
        version = None
        try:
            deadline = asyncio.get_running_loop().time() + self.max_file_seconds
            async with asyncio.timeout_at(deadline):
                source_bytes = await asyncio.to_thread(
                    self._read_verified_source, safe_path
                )
                digest = sha256(source_bytes).hexdigest()
                version = self.repository.begin_index(file_name, digest, force=force)
                if version.action == "skip":
                    return self._result(version, "skipped")
                if version.action == "in_progress":
                    return self._result(version, "in_progress")
                return await self._process_version(
                    version, safe_path, source_bytes, deadline=deadline
                )
        except TimeoutError:
            if version is not None:
                self._fail_version(version.version_id, "DOCUMENT_TIMEOUT")
                return self._result(version, "failed", failure_reason="DOCUMENT_TIMEOUT")
            return IndexResult(
                file_name=file_name, status="failed", failure_reason="DOCUMENT_TIMEOUT"
            )
        except asyncio.CancelledError:
            if version is not None:
                self._fail_version(version.version_id, "DOCUMENT_CANCELLED")
            raise
        except Exception as error:  # noqa: BLE001 - source boundary fails closed
            return IndexResult(
                file_name=file_name,
                status="failed",
                failure_reason=_safe_failure_reason(error),
            )

    async def _process_version(
        self,
        version: DocumentVersion,
        safe_path: Path,
        source_bytes: bytes,
        *,
        deadline: float,
    ) -> IndexResult:
        upsert_state: _UpsertState | None = None
        try:
            if version.action == "needs_ocr":
                if self.ocr_client is None:
                    return self._result(version, "needs_ocr")
                version = self.repository.begin_ocr_indexing(version.version_id)
                document = await self._extract_with_ocr(safe_path, source_bytes)
            else:
                document = await asyncio.to_thread(
                    extract_document,
                    safe_path,
                    data=source_bytes,
                    limits=self.resource_limits,
                )
            if document.needs_ocr:
                self.repository.mark_needs_ocr(version.version_id)
                if self.ocr_client is None:
                    return self._result(version, "needs_ocr")
                version = self.repository.begin_ocr_indexing(version.version_id)
                document = await self._extract_with_ocr(safe_path, source_bytes)

            chunks = await asyncio.to_thread(
                chunk_document,
                document,
                target_chars=self.target_chars,
                overlap_chars=self.overlap_chars,
            )
            if not chunks:
                raise ValueError("document contains no extractable text")

            vectors = await self._embed([chunk.text for chunk in chunks])
            if len(vectors) != len(chunks):
                raise ValueError("embedding response count did not match chunks")
            upsert_state = await self._bounded_upsert(
                version.version_id, chunks, vectors, deadline=deadline
            )
            if self._closing:
                self._schedule_cleanup(version.version_id, upsert_state)
                raise RuntimeError("ingestion service is closing")
            self.repository.activate(version.version_id)
            return self._result(version, "active")
        except TimeoutError:
            self._fail_version(version.version_id, "DOCUMENT_TIMEOUT")
            return self._result(version, "failed", failure_reason="DOCUMENT_TIMEOUT")
        except asyncio.CancelledError:
            if upsert_state is not None:
                self._schedule_cleanup(version.version_id, upsert_state)
            raise
        except Exception as error:  # noqa: BLE001 - one file must fail safely
            reason = _safe_failure_reason(error)
            if upsert_state is not None:
                self._schedule_cleanup(version.version_id, upsert_state)
            self._fail_version(version.version_id, reason)
            return self._result(version, "failed", failure_reason=reason)

    async def scan(self) -> ScanResult:
        if self._closing:
            raise RuntimeError("ingestion service is closing")
        operation = self._begin_operation()
        try:
            return await self._scan_impl()
        finally:
            self._finish_operation(operation)

    async def _scan_impl(self) -> ScanResult:
        """Index a complete first-level snapshot and then deactivate missing files.

        Enumeration failures and unsafe entries make the snapshot incomplete;
        in that case the service reports the per-entry errors but deliberately
        skips the repository's bulk deactivation operation.
        """

        root = self.source_documents_dir
        if not root.is_dir():
            raise NotADirectoryError("configured source directory is not readable")

        # An enumeration error is intentionally allowed to propagate.  No
        # snapshot means no safe basis for mass deactivation.
        supported_entries: list[Path] = []
        for entry in root.iterdir():
            if entry.suffix.casefold() not in SUPPORTED_SUFFIXES:
                continue
            supported_entries.append(entry)
            if len(supported_entries) > self.max_scan_files:
                return ScanResult(
                    results=[
                        IndexResult(
                            file_name=item.name,
                            status="failed",
                            failure_reason="SCAN_LIMIT_EXCEEDED",
                        )
                        for item in supported_entries
                    ],
                    deactivated=[],
                    snapshot_complete=False,
                )

        results: list[IndexResult] = []
        present_file_names: set[str] = set()
        snapshot_complete = True
        for entry in supported_entries:
            if self._is_unsafe_path(entry):
                snapshot_complete = False
                results.append(
                    IndexResult(
                        file_name=entry.name,
                        status="failed",
                        failure_reason="source entry is not a safe regular file",
                    )
                )
                continue
            try:
                if not entry.is_file():
                    snapshot_complete = False
                    results.append(
                        IndexResult(
                            file_name=entry.name,
                            status="failed",
                            failure_reason="source entry is not a regular file",
                        )
                    )
                    continue
            except OSError:
                snapshot_complete = False
                results.append(
                    IndexResult(
                        file_name=entry.name,
                        status="failed",
                        failure_reason="source entry could not be inspected",
                    )
                )
                continue

            present_file_names.add(entry.name)
            try:
                results.append(await self.index(entry))
            except Exception as error:  # noqa: BLE001 - continue other files
                results.append(
                    IndexResult(
                        file_name=entry.name,
                        status="failed",
                        failure_reason=_safe_failure_reason(error),
                    )
                )

        deactivated = []
        if snapshot_complete:
            if self._closing:
                raise RuntimeError("ingestion service is closing")
            deactivated = self.repository.deactivate_missing(present_file_names)
        return ScanResult(
            results=results,
            deactivated=deactivated,
            snapshot_complete=snapshot_complete,
        )

    def list_documents(self) -> list[Any]:
        """Return repository metadata without source text or filesystem paths."""

        return self.repository.list_documents()

    def update_document_business_metadata(
        self, document_id: int, metadata: DocumentBusinessMetadataInput
    ) -> Any:
        """Persist controlled business metadata without reading document text."""

        if self._closing:
            raise RuntimeError("ingestion service is closing")
        return self.repository.upsert_document_business_metadata(document_id, metadata)

    async def reindex(self, document_id: int) -> IndexResult:
        """Force a fresh version using only the repository's safe file name."""

        if self._closing:
            raise RuntimeError("ingestion service is closing")
        operation = self._begin_operation()
        try:
            return await self._reindex_impl(document_id)
        finally:
            self._finish_operation(operation)

    async def _reindex_impl(self, document_id: int) -> IndexResult:
        document = self.repository.get_document(document_id)
        if document is None:
            raise KeyError(f"Unknown document id: {document_id}")
        path = self._validate_source_path(self.source_documents_dir / document.file_name)
        return await self.index(path, force=True)

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        result = self.embedder.embed(texts)
        if inspect.isawaitable(result):
            result = await result
        return [list(vector) for vector in result]

    async def _extract_with_ocr(self, path: Path, data: bytes) -> ExtractedDocument:
        blocks: list[ExtractedBlock] = []
        page_count = await asyncio.to_thread(_pdf_page_count, data)
        if page_count > self.resource_limits.max_pdf_pages:
            raise DocumentLimitError("PDF page limit exceeded")
        total_characters = 0
        for page_number in range(1, page_count + 1):
            image_bytes = await asyncio.to_thread(
                _render_ocr_page,
                data,
                page_number,
                self.resource_limits.max_pdf_pixels,
                self.resource_limits.max_pdf_render_bytes,
            )
            method = self.ocr_client.ocr_page
            if inspect.iscoroutinefunction(method):
                text = await method(image_bytes)
            else:
                text = await asyncio.to_thread(method, image_bytes)
            if inspect.isawaitable(text):
                text = await text
            if not isinstance(text, str) or not text.strip():
                raise ProviderError(
                    "siliconflow",
                    "ocr_page",
                    200,
                    False,
                    "OCR response did not contain explicit text",
                )
            blocks.append(
                ExtractedBlock(
                    text=text.strip(),
                    source=SourceRef(
                        file_name=path.name,
                        heading_path=("正文",),
                        page=page_number,
                        page_end=page_number,
                    ),
                )
            )
            total_characters += len(text.strip())
            if total_characters > self.resource_limits.max_text_chars:
                raise DocumentLimitError("document text character limit exceeded")
        return ExtractedDocument(file_name=path.name, blocks=tuple(blocks))

    async def _bounded_upsert(
        self, version_id: str, chunks: list[Any], vectors: list[list[float]], *, deadline: float
    ) -> _UpsertState:
        if self._closing:
            raise RuntimeError("ingestion service is closing")
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError
        timed_out = asyncio.Event()
        state = _UpsertState()
        worker = asyncio.create_task(
            self._run_upsert_worker(version_id, chunks, vectors, timed_out, state)
        )
        self._managed_upserts.add(worker)
        worker.add_done_callback(self._upsert_task_done)
        try:
            await asyncio.wait_for(asyncio.shield(worker), timeout=remaining)
        except TimeoutError:
            timed_out.set()
            if worker.done() and state.write_succeeded:
                self._schedule_cleanup(version_id, state)
            raise
        except asyncio.CancelledError:
            timed_out.set()
            if worker.done() and state.write_succeeded:
                self._schedule_cleanup(version_id, state)
            raise
        return state

    async def _run_upsert_worker(
        self,
        version_id: str,
        chunks: list[Any],
        vectors: list[list[float]],
        timed_out: asyncio.Event,
        state: _UpsertState,
    ) -> None:
        write_error: BaseException | None = None
        try:
            state.write_started = True
            await asyncio.to_thread(self.vector_store.upsert, version_id, chunks, vectors)
            state.write_succeeded = True
        except BaseException as error:
            write_error = error
            raise
        finally:
            if state.write_started and (write_error is not None or timed_out.is_set()):
                cleanup_task = self._schedule_cleanup(version_id, state)
                try:
                    await asyncio.shield(cleanup_task)
                except asyncio.CancelledError:
                    await asyncio.shield(cleanup_task)
                    raise

    async def _cleanup_upsert(self, version_id: str) -> None:
        cleanup = getattr(self.vector_store, "delete_version", None)
        if not callable(cleanup):
            return
        cleanup_task = asyncio.create_task(
            self._invoke_delete_version(cleanup, version_id)
        )
        self._track_cleanup_task(cleanup_task)
        try:
            await asyncio.wait_for(
                asyncio.shield(cleanup_task),
                timeout=self.upsert_cleanup_timeout_seconds,
            )
        except TimeoutError:
            # A running thread cannot be force-killed safely.  Wait for its
            # managed cleanup task before vector teardown; runtime.close()
            # drains this task so it can never close the client while
            # delete_version is live.
            _log_managed_cleanup_failure(TimeoutError("upsert cleanup deadline exceeded"))
        except BaseException as error:  # noqa: BLE001 - do not replace root error
            _log_managed_cleanup_failure(error)

    def _schedule_cleanup(self, version_id: str, state: _UpsertState) -> asyncio.Task[Any]:
        if state.cleanup_task is None:
            cleanup_task = asyncio.create_task(self._cleanup_upsert(version_id))
            state.cleanup_task = cleanup_task
            self._track_cleanup_task(cleanup_task)
        return state.cleanup_task

    def _track_cleanup_task(self, task: asyncio.Task[Any]) -> None:
        self._managed_cleanups.add(task)
        task.add_done_callback(self._cleanup_task_done)

    def _upsert_task_done(self, task: asyncio.Task[Any]) -> None:
        self._managed_upserts.discard(task)
        self._log_task_failure(task)

    def _cleanup_task_done(self, task: asyncio.Task[Any]) -> None:
        self._managed_cleanups.discard(task)
        self._log_task_failure(task)

    @staticmethod
    def _log_task_failure(task: asyncio.Task[Any]) -> None:
        if task.cancelled():
            return
        try:
            error = task.exception()
        except BaseException as caught:  # noqa: BLE001 - safe marker only
            _log_managed_cleanup_failure(caught)
        else:
            if error is not None:
                _log_managed_cleanup_failure(error)

    @staticmethod
    async def _invoke_delete_version(cleanup: Any, version_id: str) -> None:
        if inspect.iscoroutinefunction(cleanup):
            await cleanup(version_id)
            return
        result = await asyncio.to_thread(cleanup, version_id)
        if inspect.isawaitable(result):
            await result

    def _fail_version(self, version_id: str, reason: str) -> None:
        try:
            self.repository.fail(version_id, reason)
        except BaseException as error:  # noqa: BLE001 - failure state is best effort
            _log_managed_cleanup_failure(error)

    def _validate_source_path(self, path: Path) -> Path:
        candidate = Path(path)
        candidate = candidate if candidate.is_absolute() else self.source_documents_dir / candidate
        resolved = Path(os.path.abspath(candidate))
        root = self.source_documents_dir
        if resolved.parent != root:
            raise ValueError("path must be a first-level file in the configured source directory")
        if resolved.suffix.casefold() not in SUPPORTED_SUFFIXES:
            raise ValueError("only PDF and DOCX files are supported")
        if self._is_unsafe_path(resolved):
            raise ValueError("source path is not a safe regular file")
        if not resolved.is_file():
            raise FileNotFoundError("source document does not exist")
        return resolved

    def _read_verified_source(self, path: Path) -> bytes:
        before = os.lstat(path)
        self._ensure_safe_stat(before)
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(str(path), flags)
        try:
            opened = os.fstat(descriptor)
            self._ensure_safe_stat(opened)
            self._ensure_same_file(before, opened)
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(descriptor, min(1024 * 1024, self.resource_limits.max_file_bytes + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > self.resource_limits.max_file_bytes:
                    raise DocumentLimitError("document file size limit exceeded")
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        current = os.lstat(path)
        self._ensure_safe_stat(after)
        self._ensure_safe_stat(current)
        self._ensure_same_file(before, after)
        self._ensure_same_file(before, current)
        return b"".join(chunks)

    @staticmethod
    def _ensure_same_file(first: os.stat_result, second: os.stat_result) -> None:
        if (
            first.st_dev != second.st_dev
            or first.st_ino != second.st_ino
            or first.st_size != second.st_size
            or first.st_mtime_ns != second.st_mtime_ns
        ):
            raise SourceChangedError("source changed during bounded read")

    @staticmethod
    def _ensure_safe_stat(metadata: os.stat_result) -> None:
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError("source is not a single-link regular file")

    @classmethod
    def _is_unsafe_path(cls, path: Path) -> bool:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        if callable(is_junction) and is_junction():
            return True
        try:
            metadata = os.lstat(path)
        except OSError:
            return True
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        return (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or bool(getattr(metadata, "st_file_attributes", 0) & reparse_flag)
        )

    def _result(
        self,
        version: DocumentVersion,
        status: IndexStatus,
        *,
        failure_reason: str | None = None,
    ) -> IndexResult:
        summary = self.repository.list_documents()
        document_id = next(
            (
                document.document_id
                for document in summary
                if document.file_name == version.file_name
            ),
            None,
        )
        return IndexResult(
            file_name=version.file_name,
            status=status,
            version_id=version.version_id,
            document_id=document_id,
            failure_reason=failure_reason or version.failure_reason,
        )


def _safe_failure_reason(error: Exception) -> str:
    if isinstance(error, DocumentLimitError):
        message = str(error)
        if "file size" in message:
            return "DOCUMENT_TOO_LARGE"
        return "DOCUMENT_LIMIT_EXCEEDED"
    if isinstance(error, SourceChangedError):
        return "SOURCE_CHANGED"
    if isinstance(error, TimeoutError):
        return "DOCUMENT_TIMEOUT"
    if isinstance(error, ProviderError):
        return "PROVIDER_UNAVAILABLE"
    if isinstance(error, FileNotFoundError):
        return "SOURCE_FILE_MISSING"
    if isinstance(error, OSError):
        return "SOURCE_FILE_UNREADABLE"
    if isinstance(error, ValueError):
        if "limit exceeded" in str(error):
            return "DOCUMENT_LIMIT_EXCEEDED"
        return "DOCUMENT_NOT_INDEXABLE"
    return "INDEX_FAILED"


def _log_managed_cleanup_failure(error: BaseException) -> None:
    """Log only a fixed operational marker for managed cleanup failures."""

    LOGGER.warning("managed ingestion cleanup failed", extra={"stage": "ingestion_cleanup"})


def _pdf_page_count(data: bytes) -> int:
    with fitz.open(stream=data, filetype="pdf") as pdf:
        return pdf.page_count


def _render_ocr_page(
    data: bytes, page_number: int, maximum_pixels: int, maximum_bytes: int
) -> bytes:
    scale = 1.5
    with fitz.open(stream=data, filetype="pdf") as pdf:
        page = pdf.load_page(page_number - 1)
        width = math.ceil(page.rect.width * scale)
        height = math.ceil(page.rect.height * scale)
        if width <= 0 or height <= 0 or width * height > maximum_pixels:
            raise ValueError("PDF page pixel limit exceeded")
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        rendered_width = int(getattr(pixmap, "width", width))
        rendered_height = int(getattr(pixmap, "height", height))
        if rendered_width <= 0 or rendered_height <= 0:
            raise ValueError("PDF page pixel limit exceeded")
        if rendered_width * rendered_height > maximum_pixels:
            raise ValueError("PDF page pixel limit exceeded")
        image_bytes = pixmap.tobytes("png")
    if len(image_bytes) > maximum_bytes:
        raise ValueError("PDF render byte limit exceeded")
    return image_bytes
