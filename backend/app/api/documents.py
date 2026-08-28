"""Configured-directory document management endpoints."""

from __future__ import annotations

import inspect
from dataclasses import asdict, is_dataclass
from typing import Any

from app.chat.safety import (
    SAFE_DOCUMENT_STATUSES,
    SAFE_FAILURE_REASONS,
    is_safe_file_name,
    is_safe_id,
    is_safe_timestamp,
)
from fastapi import APIRouter, HTTPException, Request


def create_documents_router() -> APIRouter:
    """Create routes whose service dependency is supplied by application state."""

    router = APIRouter(prefix="/api/documents", tags=["documents"])

    @router.get("")
    async def list_documents(request: Request) -> dict[str, list[dict[str, object]]]:
        service = _service(request)
        documents = service.list_documents()
        if inspect.isawaitable(documents):
            documents = await documents
        return {"documents": [_safe_metadata(document) for document in documents]}

    @router.post("/scan")
    async def scan_documents(request: Request) -> dict[str, object]:
        service = _service(request)
        try:
            result = service.scan()
            if inspect.isawaitable(result):
                result = await result
            return _serialise_scan(result)
        except (NotADirectoryError, PermissionError, OSError) as error:
            raise HTTPException(
                status_code=503,
                detail="configured source directory could not be scanned",
            ) from error

    @router.post("/{document_id}/reindex")
    async def reindex_document(document_id: int, request: Request) -> dict[str, object]:
        service = _service(request)
        try:
            result = service.reindex(document_id)
            if inspect.isawaitable(result):
                result = await result
            return _safe_metadata(result)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="document not found") from error
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail="document is not present") from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail="document path is not allowed") from error
        except OSError as error:
            raise HTTPException(status_code=503, detail="document could not be read") from error

    return router


def _service(request: Request) -> Any:
    service = getattr(request.app.state, "document_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="document indexing service is not configured",
        )
    return service


def _serialise_scan(result: Any) -> dict[str, object]:
    if hasattr(result, "to_dict"):
        value = result.to_dict()
    elif is_dataclass(result):
        value = asdict(result)
    elif isinstance(result, dict):
        value = dict(result)
    else:
        raise HTTPException(status_code=500, detail="invalid scan result")
    raw_results = value.get("results", [])
    if not isinstance(raw_results, list):
        raw_results = []
    deactivated = value.get("deactivated", [])
    if not isinstance(deactivated, list):
        deactivated = []
    safe_deactivated = [
        item for item in deactivated if is_safe_file_name(item)
    ]
    snapshot_complete = value.get("snapshot_complete", True)
    if not isinstance(snapshot_complete, bool):
        snapshot_complete = False
    return {
        "results": [_safe_metadata(item) for item in raw_results],
        "deactivated": safe_deactivated,
        "snapshot_complete": snapshot_complete,
    }


def _safe_metadata(value: Any) -> dict[str, object]:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    elif is_dataclass(value):
        value = asdict(value)
    elif isinstance(value, dict):
        value = dict(value)
    else:
        raise HTTPException(status_code=500, detail="invalid document metadata")
    document_id = value.get("document_id")
    file_name = value.get("file_name")
    if isinstance(document_id, bool) or not isinstance(document_id, int) or document_id < 1:
        return _invalid_metadata()
    if not is_safe_file_name(file_name):
        return _invalid_metadata(document_id=document_id)

    version_id = value.get("version_id")
    if version_id is not None and not is_safe_id(version_id):
        version_id = None
    status = value.get("status")
    if not isinstance(status, str) or status not in SAFE_DOCUMENT_STATUSES:
        status = "failed"
    reason = value.get("failure_reason")
    if reason is not None and (
        not isinstance(reason, str) or reason not in SAFE_FAILURE_REASONS
    ):
        reason = "INDEX_FAILED"
    updated_at = value.get("updated_at")
    if updated_at is not None and not is_safe_timestamp(updated_at):
        updated_at = None
    return {
        "document_id": document_id,
        "file_name": file_name,
        "version_id": version_id,
        "status": status,
        "failure_reason": reason,
        "updated_at": updated_at,
    }


def _invalid_metadata(*, document_id: int = 0) -> dict[str, object]:
    return {
        "document_id": document_id,
        "file_name": "未命名资料.docx",
        "version_id": None,
        "status": "failed",
        "failure_reason": "INDEX_FAILED",
        "updated_at": None,
    }
