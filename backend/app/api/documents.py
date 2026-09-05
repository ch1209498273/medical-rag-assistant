"""Configured-directory document management endpoints."""

from __future__ import annotations

import inspect
import re
from dataclasses import asdict, is_dataclass
from typing import Any, Literal

from app.chat.safety import (
    SAFE_DOCUMENT_STATUSES,
    SAFE_FAILURE_REASONS,
    is_safe_answer_text,
    is_safe_file_name,
    is_safe_id,
    is_safe_timestamp,
)
from app.domain.ports import DocumentBusinessMetadataInput
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, StrictStr


class DocumentBusinessMetadataRequest(BaseModel):
    """Controlled local-manager fields; source text is never accepted here."""

    model_config = ConfigDict(extra="forbid", strict=True)

    content_type: Literal["policy", "training", "procedure", "other"]
    applicable_scope: Literal[
        "unspecified", "all_staff", "nurse", "doctor", "pharmacist", "administrator"
    ]
    effective_from: StrictStr | None = None
    review_due_at: StrictStr | None = None
    business_status: Literal["draft", "approved", "superseded", "retired", "unknown"]
    owner_role: StrictStr | None = None
    supersedes_version_id: StrictStr | None = None

    def to_domain(self) -> DocumentBusinessMetadataInput:
        return DocumentBusinessMetadataInput(
            content_type=self.content_type,
            applicable_scope=self.applicable_scope,
            effective_from=self.effective_from,
            review_due_at=self.review_due_at,
            business_status=self.business_status,
            owner_role=self.owner_role,
            supersedes_version_id=self.supersedes_version_id,
        )


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

    @router.put("/{document_id}/business-metadata")
    async def update_business_metadata(
        document_id: int,
        payload: DocumentBusinessMetadataRequest,
        request: Request,
    ) -> dict[str, object]:
        service = _service(request)
        try:
            result = service.update_document_business_metadata(
                document_id, payload.to_domain()
            )
            if inspect.isawaitable(result):
                result = await result
            return _safe_business_metadata(result)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="document not found") from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail="document metadata is invalid") from error

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
        "business_metadata": _safe_business_metadata(value.get("business_metadata")),
    }


def _safe_business_metadata(value: Any) -> dict[str, object] | None:
    if value is None:
        return None
    if is_dataclass(value):
        value = asdict(value)
    elif isinstance(value, dict):
        value = dict(value)
    else:
        return None
    content_type = value.get("content_type")
    scope = value.get("applicable_scope")
    status = value.get("business_status")
    if content_type not in {"policy", "training", "procedure", "other"}:
        return None
    if scope not in {"unspecified", "all_staff", "nurse", "doctor", "pharmacist", "administrator"}:
        return None
    if status not in {"draft", "approved", "superseded", "retired", "unknown"}:
        return None
    version_id = value.get("version_id")
    if not is_safe_id(version_id):
        return None
    dates: dict[str, str | None] = {}
    for key in ("effective_from", "review_due_at"):
        item = value.get(key)
        if item is not None and (
            not isinstance(item, str) or not _is_iso_date(item)
        ):
            return None
        dates[key] = item
    owner_role = value.get("owner_role")
    if owner_role is not None and (
        not isinstance(owner_role, str)
        or len(owner_role) > 80
        or not owner_role.strip()
        or not is_safe_answer_text(owner_role)
    ):
        return None
    supersedes_version_id = value.get("supersedes_version_id")
    if supersedes_version_id is not None and not is_safe_id(supersedes_version_id):
        return None
    updated_at = value.get("updated_at")
    if not is_safe_timestamp(updated_at):
        return None
    return {
        "version_id": version_id,
        "content_type": content_type,
        "applicable_scope": scope,
        "effective_from": dates["effective_from"],
        "review_due_at": dates["review_due_at"],
        "business_status": status,
        "owner_role": owner_role,
        "supersedes_version_id": supersedes_version_id,
        "updated_at": updated_at,
    }


def _is_iso_date(value: str) -> bool:
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value))


def _invalid_metadata(*, document_id: int = 0) -> dict[str, object]:
    return {
        "document_id": document_id,
        "file_name": "未命名资料.docx",
        "version_id": None,
        "status": "failed",
        "failure_reason": "INDEX_FAILED",
        "updated_at": None,
    }
