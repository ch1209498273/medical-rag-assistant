"""Private evaluation-set generation and regression endpoints."""

from __future__ import annotations

import inspect
from dataclasses import asdict, is_dataclass
from typing import Any

from app.errors import ProviderError
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field


class EvaluationGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = Field(default=30, ge=1, le=1000)


def create_evaluations_router() -> APIRouter:
    router = APIRouter(prefix="/api/evaluations", tags=["evaluations"])

    @router.post("/generate")
    async def generate_evaluation(
        payload: EvaluationGenerateRequest, request: Request
    ) -> dict[str, Any]:
        service = _service(request)
        try:
            result = service.generate(count=payload.count)
            if inspect.isawaitable(result):
                result = await result
            return _safe_set(result)
        except ProviderError as error:
            raise HTTPException(
                status_code=503, detail="EVALUATION_GENERATION_UNAVAILABLE"
            ) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail="evaluation set is invalid") from error

    @router.post("/{set_id}/run")
    async def run_evaluation(set_id: str, request: Request) -> dict[str, Any]:
        service = _service(request)
        try:
            result = service.run(set_id)
            if inspect.isawaitable(result):
                result = await result
            return _safe_report(result)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="evaluation set not found") from error
        except ProviderError as error:
            raise HTTPException(status_code=503, detail="EVALUATION_RUN_UNAVAILABLE") from error
        except ValueError as error:
            raise HTTPException(status_code=404, detail="evaluation set not found") from error

    @router.get("/{set_id}")
    async def get_evaluation(set_id: str, request: Request) -> dict[str, Any]:
        service = _service(request)
        try:
            result = service.get(set_id)
            if inspect.isawaitable(result):
                result = await result
        except ValueError as error:
            raise HTTPException(status_code=404, detail="evaluation set not found") from error
        if result is None:
            raise HTTPException(status_code=404, detail="evaluation set not found")
        return _safe_set(result)

    return router


def _service(request: Request) -> Any:
    service = getattr(request.app.state, "evaluation_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="evaluation service is not configured")
    return service


def _safe_set(value: Any) -> dict[str, Any]:
    payload = _as_dict(value)
    allowed = {
        "set_id",
        "cases",
        "provider",
        "model_id",
        "created_at",
        "checksum",
        "manual_review_required",
        "manual_review_minimum",
    }
    result = {key: payload[key] for key in allowed if key in payload}
    result["cases"] = [_safe_case(item) for item in payload.get("cases", [])]
    # UI must visibly prompt a first-pass manual check of at least ten cases.
    result.setdefault("manual_review_required", True)
    result.setdefault("manual_review_minimum", 10)
    result["review_notice"] = "需人工抽查至少 10 条"
    return result


def _safe_case(value: Any) -> dict[str, Any]:
    payload = _as_dict(value)
    allowed = {"question", "reference_points", "source_ids", "type"}
    return {key: payload.get(key) for key in allowed}


def _safe_report(value: Any) -> dict[str, Any]:
    payload = _as_dict(value)
    allowed = {
        "total",
        "citation_accuracy",
        "set_id",
        "answered_total",
        "grounded_accuracy",
        "no_answer_refusal_rate",
        "answer_availability_rate",
        "runtime_citation_visibility_rate",
        "first_status_latency_ms",
        "answer_duration_ms",
        "results",
        "created_at",
        "manual_review_required",
        "manual_review_minimum",
    }
    result = {key: payload[key] for key in allowed if key in payload}
    result["results"] = [_safe_result(item) for item in payload.get("results", [])]
    result["review_notice"] = "需人工抽查至少 10 条"
    return result


def _safe_result(value: Any) -> dict[str, Any]:
    payload = _as_dict(value)
    allowed = {
        "case_id",
        "question",
        "type",
        "answered",
        "refused",
        "no_answer",
        "reference_points_covered",
        "citation_valid",
        "citation_source_ids",
        "first_status_latency_ms",
        "answer_duration_ms",
        "reason_code",
        "generation_available",
        "source_visibility",
    }
    result = {key: payload[key] for key in allowed if key in payload}
    for key in ("generation_available", "source_visibility"):
        if key in result and not isinstance(result[key], bool):
            result.pop(key)
    return result


def _as_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    elif is_dataclass(value):
        value = asdict(value)
    elif isinstance(value, dict):
        value = dict(value)
    else:
        raise HTTPException(status_code=500, detail="invalid evaluation result")
    return value


__all__ = ["EvaluationGenerateRequest", "create_evaluations_router"]
