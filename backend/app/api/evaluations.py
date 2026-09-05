"""Private evaluation-set generation and regression endpoints."""

from __future__ import annotations

import inspect
import math
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
        "industry_metrics",
    }
    result = {key: payload[key] for key in allowed if key in payload}
    result["results"] = [_safe_result(item) for item in payload.get("results", [])]
    safe_metrics = _safe_industry_metrics(payload.get("industry_metrics"))
    if safe_metrics is None:
        result.pop("industry_metrics", None)
    else:
        result["industry_metrics"] = safe_metrics
    result["review_notice"] = "需人工抽查至少 10 条"
    return result


_RATE_FIELDS = frozenset(
    {"numerator", "denominator", "value", "ci95_low", "ci95_high"}
)


def _safe_industry_metrics(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    payload = _as_dict(value)
    if set(payload) != {"protocol_version", "retrieval", "generation", "safety"}:
        return None
    version = payload.get("protocol_version")
    if not isinstance(version, str) or version != "task12a-v1":
        return None
    retrieval = _safe_metric_group(
        payload.get("retrieval"),
        rate_fields=("recall_at_20", "recall_at_6", "precision_at_6", "negative_evidence_rate"),
        scalar_fields=("mrr_at_20", "mrr_denominator", "ndcg_at_6", "ndcg_denominator"),
    )
    generation = _safe_metric_group(
        payload.get("generation"),
        rate_fields=(
            "answer_availability",
            "grounded_answer_success",
            "answerable_over_refusal",
            "citation_precision",
            "citation_recall",
        ),
        scalar_fields=(),
    )
    safety = _safe_metric_group(
        payload.get("safety"),
        rate_fields=(
            "unsafe_answer_rate",
            "runtime_citation_visibility",
            "no_answer_refusal",
        ),
        scalar_fields=(),
    )
    if retrieval is None or generation is None or safety is None:
        return None
    return {
        "protocol_version": version,
        "retrieval": retrieval,
        "generation": generation,
        "safety": safety,
    }


def _safe_metric_group(
    value: Any,
    *,
    rate_fields: tuple[str, ...],
    scalar_fields: tuple[str, ...],
) -> dict[str, Any] | None:
    if value is None:
        return None
    payload = _as_dict(value)
    if set(payload) != set(rate_fields) | set(scalar_fields):
        return None
    result: dict[str, Any] = {}
    for field in rate_fields:
        safe_rate = _safe_rate(payload.get(field))
        if safe_rate is None:
            return None
        result[field] = safe_rate
    for field in scalar_fields:
        scalar = payload.get(field)
        if field.endswith("_denominator"):
            if isinstance(scalar, bool) or not isinstance(scalar, int) or scalar < 0:
                return None
            result[field] = scalar
            continue
        if scalar is not None:
            if isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
                return None
            scalar = float(scalar)
            if not math.isfinite(scalar) or not 0.0 <= scalar <= 1.0:
                return None
        result[field] = scalar
    return result


def _safe_rate(value: Any) -> dict[str, Any] | None:
    payload = _as_dict(value)
    if set(payload) != _RATE_FIELDS:
        return None
    numerator = payload.get("numerator")
    denominator = payload.get("denominator")
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or numerator < 0
        or denominator < 0
        or numerator > denominator
    ):
        return None
    values: dict[str, float] = {}
    for field in ("value", "ci95_low", "ci95_high"):
        raw = payload.get(field)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return None
        number = float(raw)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            return None
        values[field] = number
    if values["ci95_low"] > values["ci95_high"]:
        return None
    return {
        "numerator": numerator,
        "denominator": denominator,
        **values,
    }


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
