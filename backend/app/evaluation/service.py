"""Private, offline-first generation and regression measurement for RAG.

The service deliberately knows only the formal ``RagService`` stream contract
and a provider-supplied source snapshot.  It never imports the evaluation
fallback into formal answer code and persists generated material under an
operator-configured private directory.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import re
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.providers.minimax import EvaluationCase, EvaluationGeneration

_SET_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$")
_NO_ANSWER_TYPES = frozenset(
    {"no_answer", "no-answer", "no answer", "unanswerable", "refusal", "无答案", "无依据"}
)
_MODEL_REFUSAL_REASON = "MODEL_REFUSED"
# Backend-owned runtime citation ordinals, intentionally distinct from frozen source IDs.
_PUBLIC_CITATION_IDS = frozenset(f"S{index}" for index in range(1, 7))
EVALUATION_TYPES = (
    "fact",
    "procedure",
    "condition",
    "prohibition",
    "synthesis",
    "no_answer",
)
_TYPE_ALIASES = {
    "fact": "fact",
    "事实": "fact",
    "事实查询": "fact",
    "procedure": "procedure",
    "流程": "procedure",
    "流程步骤": "procedure",
    "condition": "condition",
    "适用条件": "condition",
    "applicability": "condition",
    "prohibition": "prohibition",
    "禁止": "prohibition",
    "禁止事项": "prohibition",
    "synthesis": "synthesis",
    "跨段归纳": "synthesis",
    "cross_section": "synthesis",
    "no_answer": "no_answer",
    "no-answer": "no_answer",
    "无答案": "no_answer",
    "无依据": "no_answer",
}
_DEFAULT_STORAGE_DIR = Path("../data/private/evaluations")


@dataclass(frozen=True)
class EvaluationSet:
    """An immutable generated set with private-storage metadata."""

    set_id: str
    cases: tuple[EvaluationCase, ...]
    provider: str
    model_id: str
    created_at: str = ""
    checksum: str | None = None
    manual_review_required: bool = True
    manual_review_minimum: int = 10

    def __post_init__(self) -> None:
        if not _SET_ID.fullmatch(self.set_id):
            raise ValueError("evaluation set id is invalid")
        object.__setattr__(self, "cases", tuple(self.cases))
        if not self.provider.strip() or not self.model_id.strip():
            raise ValueError("evaluation set provider metadata is required")
        if (
            isinstance(self.manual_review_minimum, bool)
            or not isinstance(self.manual_review_minimum, int)
            or self.manual_review_minimum < 10
        ):
            raise ValueError("manual review minimum must be at least 10")
        if not self.created_at:
            object.__setattr__(self, "created_at", _now())

    def to_dict(self, *, include_checksum: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "set_id": self.set_id,
            "cases": [_case_to_dict(case) for case in self.cases],
            "provider": self.provider,
            "model_id": self.model_id,
            "created_at": self.created_at,
            "manual_review_required": self.manual_review_required,
            "manual_review_minimum": self.manual_review_minimum,
        }
        if include_checksum and self.checksum is not None:
            payload["checksum"] = self.checksum
        return payload

    @property
    def id(self) -> str:
        return self.set_id


@dataclass(frozen=True)
class EvaluationCaseResult:
    """Safe, per-question regression facts; no raw model response is stored."""

    case_id: str = ""
    question: str = ""
    type: str = ""
    answered: bool = False
    refused: bool = False
    no_answer: bool = False
    reference_points_covered: bool = False
    citation_valid: bool = False
    citation_source_ids: tuple[str, ...] = ()
    first_status_latency_ms: float | None = None
    answer_duration_ms: float | None = None
    reason_code: str | None = None
    generation_available: bool = False
    source_visibility: bool = False
    candidate_count: int | None = None
    evidence_count: int | None = None
    highest_rerank_score: float | None = None
    retrieval_status: str | None = None
    retrieval_reason_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["citation_source_ids"] = list(self.citation_source_ids)
        return value

    @property
    def reference_covered(self) -> bool:
        return self.reference_points_covered

    @property
    def citation_hit(self) -> bool:
        return self.citation_valid

    @property
    def status_latency_ms(self) -> float | None:
        return self.first_status_latency_ms

    @property
    def total_duration_ms(self) -> float | None:
        return self.answer_duration_ms


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregated regression metrics plus safe per-question outcomes."""

    total: int = 0
    citation_accuracy: float = 0.0
    set_id: str | None = None
    answered_total: int = 0
    grounded_accuracy: float = 0.0
    no_answer_refusal_rate: float = 0.0
    answer_availability_rate: float = 0.0
    runtime_citation_visibility_rate: float = 0.0
    first_status_latency_ms: float | None = None
    answer_duration_ms: float | None = None
    results: tuple[EvaluationCaseResult, ...] = ()
    created_at: str = ""
    manual_review_required: bool = True
    manual_review_minimum: int = 10

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", tuple(self.results))
        if not self.created_at:
            object.__setattr__(self, "created_at", _now())

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "citation_accuracy": self.citation_accuracy,
            "set_id": self.set_id,
            "answered_total": self.answered_total,
            "grounded_accuracy": self.grounded_accuracy,
            "no_answer_refusal_rate": self.no_answer_refusal_rate,
            "answer_availability_rate": self.answer_availability_rate,
            "runtime_citation_visibility_rate": self.runtime_citation_visibility_rate,
            "first_status_latency_ms": self.first_status_latency_ms,
            "answer_duration_ms": self.answer_duration_ms,
            "results": [item.to_dict() for item in self.results],
            "created_at": self.created_at,
            "manual_review_required": self.manual_review_required,
            "manual_review_minimum": self.manual_review_minimum,
        }

    @property
    def answer_accuracy(self) -> float:
        """Alias used by regression dashboards for grounded conclusion rate."""

        return self.grounded_accuracy

    @property
    def refusal_rate(self) -> float:
        return self.no_answer_refusal_rate

    @property
    def first_stage_latency_ms(self) -> float | None:
        return self.first_status_latency_ms

    @property
    def total_answer_duration_ms(self) -> float | None:
        return self.answer_duration_ms

    @property
    def case_results(self) -> tuple[EvaluationCaseResult, ...]:
        return self.results


class EvaluationService:
    """Generate, privately persist, and run a grounded RAG evaluation set."""

    def __init__(
        self,
        generator: Any,
        source_provider: Any = None,
        *,
        context_provider: Any = None,
        sources: Any = None,
        source_ids: Any = None,
        known_source_ids: Any = None,
        source_chunks: Any = None,
        chunks: Any = None,
        source_catalog: Any = None,
        rag_service: Any = None,
        storage_dir: Path | str | None = None,
        private_data_dir: Path | str | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.generator = generator
        self.source_provider = source_provider
        if self.source_provider is None:
            self.source_provider = context_provider
        if self.source_provider is None:
            self.source_provider = sources
        if self.source_provider is None:
            self.source_provider = source_ids
        if self.source_provider is None:
            self.source_provider = known_source_ids
        if self.source_provider is None:
            self.source_provider = source_chunks
        if self.source_provider is None:
            self.source_provider = chunks
        if self.source_provider is None:
            self.source_provider = source_catalog
        self.rag_service = rag_service
        selected_dir = storage_dir if storage_dir is not None else private_data_dir
        self.storage_dir = Path(selected_dir or _DEFAULT_STORAGE_DIR)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._clock = clock or time.perf_counter

    async def generate(self, count: int = 30) -> EvaluationSet:
        """Generate at most ``count`` valid, source-grounded questions."""

        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 1000:
            raise ValueError("count must be between 1 and 1000")
        context, known_sources = await self._generation_context(count)
        raw_generation = await self.generator.generate(context)
        generation = _normalise_generation(raw_generation, self.generator)
        valid_cases = _filter_cases(generation.cases, known_sources, count)
        if not valid_cases:
            raise ValueError("evaluation generation produced no source-grounded cases")
        if count == 30:
            type_counts = {evaluation_type: 0 for evaluation_type in EVALUATION_TYPES}
            for case in valid_cases:
                if case.type in type_counts:
                    type_counts[case.type] += 1
            if len(valid_cases) != 30 or any(
                type_counts[evaluation_type] != 5
                for evaluation_type in EVALUATION_TYPES
            ):
                raise ValueError("evaluation generation did not satisfy exact type quotas")
        elif count >= len(EVALUATION_TYPES):
            covered_types = {case.type for case in valid_cases}
            if any(required_type not in covered_types for required_type in EVALUATION_TYPES):
                raise ValueError("evaluation generation did not cover all required types")
        result = EvaluationSet(
            set_id=uuid4().hex,
            cases=tuple(valid_cases),
            provider=generation.provider,
            model_id=generation.model_id,
        )
        checksum = _checksum(result.to_dict(include_checksum=False))
        result = EvaluationSet(
            set_id=result.set_id,
            cases=result.cases,
            provider=result.provider,
            model_id=result.model_id,
            created_at=result.created_at,
            checksum=checksum,
            manual_review_required=result.manual_review_required,
            manual_review_minimum=result.manual_review_minimum,
        )
        self._write_json(self._set_path(result.set_id), result.to_dict())
        return result

    async def run(self, set_id: str) -> EvaluationReport:
        """Run the formal RAG stream and persist safe regression outcomes."""

        evaluation_set = self.get(set_id)
        if evaluation_set is None:
            raise KeyError("evaluation set not found")
        if self.rag_service is None:
            raise RuntimeError("RAG service is not configured")
        results: list[EvaluationCaseResult] = []
        for index, case in enumerate(evaluation_set.cases, start=1):
            results.append(await self._run_case(str(index), case))
        report = calculate_report(results)
        report = EvaluationReport(
            total=report.total,
            citation_accuracy=report.citation_accuracy,
            set_id=evaluation_set.set_id,
            answered_total=report.answered_total,
            grounded_accuracy=report.grounded_accuracy,
            no_answer_refusal_rate=report.no_answer_refusal_rate,
            answer_availability_rate=report.answer_availability_rate,
            runtime_citation_visibility_rate=report.runtime_citation_visibility_rate,
            first_status_latency_ms=report.first_status_latency_ms,
            answer_duration_ms=report.answer_duration_ms,
            results=report.results,
            manual_review_required=evaluation_set.manual_review_required,
            manual_review_minimum=evaluation_set.manual_review_minimum,
        )
        self._write_json(self._report_path(evaluation_set.set_id), report.to_dict())
        return report

    def get(self, set_id: str) -> EvaluationSet | None:
        """Read and integrity-check a private evaluation set by opaque ID."""

        path = self._set_path(set_id)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            evaluation_set = _set_from_dict(payload)
            expected = _checksum(evaluation_set.to_dict(include_checksum=False))
            if evaluation_set.checksum != expected:
                raise ValueError("evaluation set integrity check failed")
            return evaluation_set
        except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError):
            # Do not expose local paths or parsing details through the API.
            raise ValueError("evaluation set could not be loaded") from None

    get_evaluation_set = get

    async def _generation_context(self, count: int) -> tuple[dict[str, Any], set[str]]:
        source_data: Any = []
        if self.source_provider is not None:
            if callable(self.source_provider):
                try:
                    signature = inspect.signature(self.source_provider)
                except (TypeError, ValueError):
                    signature = None
                required = (
                    signature is not None
                    and any(
                        parameter.default is inspect.Parameter.empty
                        and parameter.kind
                        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                        for parameter in signature.parameters.values()
                    )
                )
                source_data = self.source_provider(count) if required else self.source_provider()
            else:
                source_data = self.source_provider
            if not isinstance(source_data, (Mapping, Iterable, str, bytes)):
                for method_name in ("list_sources", "get_sources", "list_chunks", "get_chunks"):
                    method = getattr(source_data, method_name, None)
                    if callable(method):
                        source_data = method()
                        break
                if not isinstance(source_data, (Mapping, Iterable, str, bytes)) and hasattr(
                    source_data, "text"
                ):
                    source_data = [source_data]
            if inspect.isawaitable(source_data):
                source_data = await source_data
            if hasattr(source_data, "__aiter__"):
                source_data = [item async for item in source_data]
        sources = _normalise_sources(source_data)
        sampled = _stratified_sample(sources, count)
        known = {item["source_id"] for item in sources}
        # Preserve aliases (chunk_id/reference_id) so callers can use stable
        # IDs from Qdrant while tests and generated prompts use S1-style IDs.
        for item in sources:
            known.update(item.get("aliases", ()))
        return {
            "count": count,
            "question_types": list(EVALUATION_TYPES),
            "sources": sampled,
        }, known

    async def _run_case(self, case_id: str, case: EvaluationCase) -> EvaluationCaseResult:
        started = self._clock()
        first_status: float | None = None
        answer_text_parts: list[str] = []
        refused = False
        citations: list[str] = []
        reason_code: str | None = None
        retrieval_diagnostics: dict[str, object] | None = None
        try:
            stream = self.rag_service.stream(case.question)
            if inspect.isawaitable(stream):
                stream = await stream
            async for event in _iterate_events(stream):
                event_type, data = _event_data(event)
                now = self._clock()
                if event_type == "status" and first_status is None:
                    first_status = _milliseconds(now - started)
                elif event_type == "answer_delta":
                    text = data.get("text")
                    if isinstance(text, str):
                        answer_text_parts.append(text)
                    ids = data.get("source_ids")
                    if isinstance(ids, (list, tuple)):
                        citations.extend(item for item in ids if isinstance(item, str))
                elif event_type == "final":
                    refused = bool(data.get("refused", False))
                    final_text = data.get("text", data.get("answer"))
                    if isinstance(final_text, str):
                        answer_text_parts.append(final_text)
                    raw_citations = data.get("citations")
                    if isinstance(raw_citations, list):
                        citations.extend(
                            item.get("reference_id")
                            for item in raw_citations
                            if isinstance(item, Mapping) and isinstance(item.get("reference_id"), str)
                        )
                        citations.extend(item for item in raw_citations if isinstance(item, str))
                    raw_reason = data.get("reason_code")
                    if isinstance(raw_reason, str):
                        reason_code = raw_reason
                    retrieval_diagnostics = _safe_retrieval_diagnostics(
                        data.get("retrieval_diagnostics")
                    )
                elif event_type == "error":
                    raw_reason = data.get("reason_code")
                    reason_code = raw_reason if isinstance(raw_reason, str) else "RAG_UNAVAILABLE"
                    retrieval_diagnostics = _safe_retrieval_diagnostics(
                        data.get("retrieval_diagnostics")
                    )
            ended = self._clock()
            answer_text = "".join(answer_text_parts)
            citation_ids = tuple(dict.fromkeys(citations))
            answered = bool(answer_text.strip()) and not refused
            no_answer = _is_no_answer_type(case.type)
            valid_citation = bool(set(citation_ids) & set(case.source_ids)) if answered else refused and no_answer
            covered = _covers_reference_points(answer_text, case.reference_points) if answered else False
            if not answered and not reason_code and not refused:
                reason_code = "RAG_UNAVAILABLE"
            generation_available = _generation_available(
                answered=answered,
                refused=refused,
                reason_code=reason_code,
            )
            source_visibility = answered and bool(set(citation_ids) & _PUBLIC_CITATION_IDS)
            return EvaluationCaseResult(
                case_id=case_id,
                question=case.question,
                type=case.type,
                answered=answered,
                refused=refused,
                no_answer=no_answer,
                reference_points_covered=covered,
                citation_valid=valid_citation,
                citation_source_ids=citation_ids,
                first_status_latency_ms=first_status,
                answer_duration_ms=_milliseconds(ended - started),
                reason_code=reason_code,
                generation_available=generation_available,
                source_visibility=source_visibility,
                **_diagnostic_fields(retrieval_diagnostics),
            )
        except Exception:  # noqa: BLE001 - evaluation must isolate one case
            return EvaluationCaseResult(
                case_id=case_id,
                question=case.question,
                type=case.type,
                no_answer=_is_no_answer_type(case.type),
                first_status_latency_ms=first_status,
                answer_duration_ms=_milliseconds(self._clock() - started),
                reason_code="RAG_UNAVAILABLE",
                **_diagnostic_fields(retrieval_diagnostics),
            )

    def _set_path(self, set_id: str) -> Path:
        if not isinstance(set_id, str) or not _SET_ID.fullmatch(set_id):
            raise ValueError("evaluation set id is invalid")
        return self.storage_dir / f"{set_id}.json"

    def _report_path(self, set_id: str) -> Path:
        if not isinstance(set_id, str) or not _SET_ID.fullmatch(set_id):
            raise ValueError("evaluation set id is invalid")
        return self.storage_dir / f"{set_id}.report.json"

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def calculate_report(results: Sequence[Any]) -> EvaluationReport:
    """Calculate aggregate acceptance metrics from safe case facts."""

    normalised = tuple(_coerce_result(item, index) for index, item in enumerate(results, start=1))
    total = len(normalised)
    answered = tuple(item for item in normalised if item.answered)
    no_answer = tuple(item for item in normalised if item.no_answer)
    citation_accuracy = _ratio(sum(item.citation_valid for item in normalised), total)
    grounded_accuracy = _ratio(
        sum(item.reference_points_covered for item in answered), len(answered)
    )
    answer_availability_rate = _ratio(len(answered), total)
    runtime_citation_visibility_rate = _ratio(
        sum(item.source_visibility for item in answered), len(answered)
    )
    refusal_rate = _ratio(sum(item.refused for item in no_answer), len(no_answer))
    first_latencies = [item.first_status_latency_ms for item in normalised if item.first_status_latency_ms is not None]
    durations = [item.answer_duration_ms for item in normalised if item.answer_duration_ms is not None]
    return EvaluationReport(
        total=total,
        citation_accuracy=citation_accuracy,
        answered_total=len(answered),
        grounded_accuracy=grounded_accuracy,
        no_answer_refusal_rate=refusal_rate,
        answer_availability_rate=answer_availability_rate,
        runtime_citation_visibility_rate=runtime_citation_visibility_rate,
        first_status_latency_ms=_average(first_latencies),
        answer_duration_ms=_average(durations),
        results=normalised,
    )


def _normalise_generation(result: Any, generator: Any) -> EvaluationGeneration:
    if isinstance(result, EvaluationGeneration):
        return result
    if isinstance(result, (list, tuple)):
        provider = getattr(generator, "provider", "evaluation")
        model_id = getattr(generator, "model_id", "unknown")
        return EvaluationGeneration(tuple(result), str(provider), str(model_id))
    raise ValueError("evaluation generation response is invalid")


def _filter_cases(
    cases: Iterable[EvaluationCase], known_sources: set[str], count: int
) -> list[EvaluationCase]:
    valid: list[EvaluationCase] = []
    seen_questions: set[str] = set()
    for case in cases:
        if not isinstance(case, EvaluationCase):
            continue
        if not case.question.strip() or not case.reference_points or not case.source_ids:
            continue
        if any(source_id not in known_sources for source_id in case.source_ids):
            continue
        key = " ".join(case.question.split()).casefold()
        if key in seen_questions:
            continue
        seen_questions.add(key)
        case_type = _TYPE_ALIASES.get(case.type.strip().casefold())
        if case_type is None:
            continue
        reference_points = [point.strip() for point in case.reference_points if point.strip()]
        if not reference_points:
            continue
        valid.append(
            EvaluationCase(
                question=case.question.strip(),
                reference_points=reference_points,
                source_ids=list(dict.fromkeys(case.source_ids)),
                type=case_type,
            )
        )
        if len(valid) >= count:
            break
    return valid


def _normalise_sources(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        items: Iterable[Any] = [
            dict(item, source_id=key)
            if isinstance(item, Mapping)
            else {"source_id": key, "chunk": item}
            for key, item in value.items()
        ]
    elif isinstance(value, (str, bytes)) or value is None:
        items = []
    else:
        items = value if isinstance(value, Iterable) else []
    normalised: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        normalised_item = _source_item(item, index)
        if normalised_item is not None:
            normalised.append(normalised_item)
    return normalised


def _source_item(item: Any, index: int) -> dict[str, Any] | None:
    if isinstance(item, str):
        source_id = item.strip()
        if not source_id:
            return None
        return {
            "source_id": source_id,
            "aliases": (source_id,),
            "text": source_id,
            "file_name": "",
            "heading_path": [],
        }
    if isinstance(item, Mapping):
        source_object = item.get("chunk")
        source_id = item.get("source_id") or item.get("reference_id") or item.get("chunk_id") or f"S{index}"
        text = item.get("text") or item.get("content") or ""
        file_name = item.get("file_name", "")
        heading = item.get("heading_path", ())
        aliases = {
            alias
            for alias in (item.get("reference_id"), item.get("chunk_id"))
            if isinstance(alias, str) and alias.strip()
        }
        if source_object is not None and not text:
            chunk = getattr(source_object, "chunk", source_object)
            text = getattr(chunk, "text", "")
            source = getattr(chunk, "source", None)
            file_name = getattr(source, "file_name", file_name)
            heading = getattr(source, "heading_path", heading)
            chunk_id = getattr(chunk, "chunk_id", None)
            if isinstance(chunk_id, str) and chunk_id.strip():
                aliases.add(chunk_id)
    else:
        chunk = getattr(item, "chunk", item)
        source_id = getattr(chunk, "chunk_id", None) or getattr(item, "reference_id", None) or f"S{index}"
        text = getattr(chunk, "text", "")
        source = getattr(chunk, "source", None)
        file_name = getattr(source, "file_name", "")
        heading = getattr(source, "heading_path", ())
        aliases = {source_id}
        reference_id = getattr(item, "reference_id", None)
        if isinstance(reference_id, str):
            aliases.add(reference_id)
    if not isinstance(source_id, str) or not source_id.strip() or not isinstance(text, str) or not text.strip():
        return None
    if not isinstance(heading, (list, tuple)):
        heading = ()
    return {
        "source_id": source_id.strip(),
        "aliases": tuple(sorted(aliases)),
        "text": text.strip(),
        "file_name": str(file_name) if file_name is not None else "",
        "heading_path": [str(value) for value in heading if str(value).strip()],
    }


def _stratified_sample(sources: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
    for item in sources:
        groups[(item["file_name"], tuple(item["heading_path"]))].append(item)
    selected: list[dict[str, Any]] = []
    # Round-robin groups retains document/title diversity without random state
    # leaking into reproducibility or consuming private data unexpectedly.
    buckets = list(groups.values())
    while buckets and len(selected) < max(count * 2, count):
        next_buckets: list[list[dict[str, Any]]] = []
        for bucket in buckets:
            if bucket and len(selected) < max(count * 2, count):
                selected.append(bucket.pop(0))
            if bucket:
                next_buckets.append(bucket)
        buckets = next_buckets
    return selected


def _case_to_dict(case: EvaluationCase) -> dict[str, Any]:
    return {
        "question": case.question,
        "reference_points": list(case.reference_points),
        "source_ids": list(case.source_ids),
        "type": case.type,
    }


def _set_from_dict(value: Any) -> EvaluationSet:
    if not isinstance(value, Mapping):
        raise TypeError("evaluation set is not an object")
    cases = value.get("cases")
    if not isinstance(cases, list):
        raise TypeError("evaluation set cases are invalid")
    parsed_cases: list[EvaluationCase] = []
    for item in cases:
        if not isinstance(item, Mapping):
            raise TypeError("evaluation case is invalid")
        required = {"question", "reference_points", "source_ids", "type"}
        if set(item) != required:
            raise ValueError("evaluation case fields are invalid")
        if (
            not isinstance(item["question"], str)
            or not isinstance(item["reference_points"], list)
            or not isinstance(item["source_ids"], list)
            or not isinstance(item["type"], str)
            or not item["question"].strip()
            or not item["reference_points"]
            or any(not isinstance(point, str) or not point.strip() for point in item["reference_points"])
            or not item["source_ids"]
            or any(not isinstance(source_id, str) or not source_id.strip() for source_id in item["source_ids"])
            or not item["type"].strip()
        ):
            raise TypeError("evaluation case values are invalid")
        parsed_cases.append(
            EvaluationCase(
                question=item["question"],
                reference_points=item["reference_points"],
                source_ids=item["source_ids"],
                type=item["type"],
            )
        )
    return EvaluationSet(
        set_id=str(value.get("set_id", "")),
        cases=tuple(parsed_cases),
        provider=str(value.get("provider", "")),
        model_id=str(value.get("model_id", "")),
        created_at=str(value.get("created_at", "")),
        checksum=value.get("checksum") if isinstance(value.get("checksum"), str) else None,
        manual_review_required=value.get("manual_review_required", True)
        if isinstance(value.get("manual_review_required", True), bool)
        else True,
        manual_review_minimum=value.get("manual_review_minimum", 10)
        if isinstance(value.get("manual_review_minimum", 10), int)
        and not isinstance(value.get("manual_review_minimum", 10), bool)
        else 10,
    )


def _event_data(event: Any) -> tuple[str, Mapping[str, Any]]:
    if is_dataclass(event):
        event_type = getattr(event, "type", "")
        data = getattr(event, "data", {})
    elif isinstance(event, Mapping):
        event_type = event.get("type", "")
        data = event.get("data", event)
    else:
        event_type = getattr(event, "type", "")
        data = getattr(event, "data", {})
    if not isinstance(event_type, str) or not isinstance(data, Mapping):
        return "", {}
    return event_type, data


async def _iterate_events(stream: Any):
    if hasattr(stream, "__aiter__"):
        async for event in stream:
            yield event
        return
    if isinstance(stream, Iterable):
        for event in stream:
            yield event


def _coerce_result(value: Any, index: int) -> EvaluationCaseResult:
    if isinstance(value, EvaluationCaseResult):
        return replace(
            value,
            generation_available=_generation_available(
                answered=value.answered,
                refused=value.refused,
                reason_code=value.reason_code,
            ),
        )
    if isinstance(value, Mapping):
        get = value.get
    else:
        get = lambda key, default=None: getattr(value, key, default)
    citation_ids = get("citation_source_ids", get("citations", ()))
    if isinstance(citation_ids, list):
        citation_ids = tuple(
            item.get("reference_id") if isinstance(item, Mapping) else str(item)
            for item in citation_ids
            if (isinstance(item, Mapping) and isinstance(item.get("reference_id"), str)) or isinstance(item, str)
        )
    elif not isinstance(citation_ids, tuple):
        citation_ids = ()
    refused = bool(get("refused", False))
    answered = bool(get("answered", not refused))
    no_answer = bool(get("no_answer", False))
    reason_code = str(get("reason_code")) if get("reason_code") is not None else None
    generation_available = _generation_available(
        answered=answered,
        refused=refused,
        reason_code=reason_code,
    )
    return EvaluationCaseResult(
        case_id=str(get("case_id", index)),
        question=str(get("question", "")),
        type=str(get("type", "")),
        answered=answered,
        refused=refused,
        no_answer=no_answer,
        reference_points_covered=bool(get("reference_points_covered", get("grounded", False))),
        citation_valid=bool(get("citation_valid", get("citation_ok", False))),
        citation_source_ids=tuple(item for item in citation_ids if isinstance(item, str)),
        first_status_latency_ms=_number_or_none(get("first_status_latency_ms")),
        answer_duration_ms=_number_or_none(get("answer_duration_ms")),
        reason_code=reason_code,
        generation_available=generation_available,
        source_visibility=bool(get("source_visibility", False)),
        candidate_count=_non_negative_int_or_none(get("candidate_count")),
        evidence_count=_non_negative_int_or_none(get("evidence_count")),
        highest_rerank_score=_number_or_none(get("highest_rerank_score")),
        retrieval_status=_safe_retrieval_status(get("retrieval_status")),
        retrieval_reason_code=_safe_reason_code(get("retrieval_reason_code")),
    )


def _generation_available(*, answered: bool, refused: bool, reason_code: str | None) -> bool:
    """Return whether the case produced a usable answer-or-refusal outcome."""

    return (answered and not refused and reason_code is None) or (
        refused and reason_code == _MODEL_REFUSAL_REASON
    )


def _covers_reference_points(answer: str, points: Sequence[str]) -> bool:
    if not isinstance(answer, str) or not answer.strip() or not points:
        return False
    normalised = answer.casefold()
    return all(isinstance(point, str) and point.strip().casefold() in normalised for point in points)


def _is_no_answer_type(value: str) -> bool:
    return isinstance(value, str) and value.strip().casefold() in _NO_ANSWER_TYPES


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return float(value)


def _non_negative_int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _safe_retrieval_status(value: Any) -> str | None:
    return value if value in {"ready", "refused", "unavailable"} else None


def _safe_reason_code(value: Any) -> str | None:
    return value if value in {
        "INSUFFICIENT_EVIDENCE",
        "RETRIEVAL_UNAVAILABLE",
    } else None


def _safe_retrieval_diagnostics(value: Any) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    candidate_count = _non_negative_int_or_none(value.get("candidate_count"))
    evidence_count = _non_negative_int_or_none(value.get("evidence_count"))
    highest_score = _number_or_none(value.get("highest_rerank_score"))
    status = _safe_retrieval_status(value.get("status"))
    reason_code = _safe_reason_code(value.get("reason_code"))
    if candidate_count is None or evidence_count is None or status is None:
        return None
    return {
        "candidate_count": candidate_count,
        "evidence_count": evidence_count,
        "highest_rerank_score": highest_score,
        "status": status,
        "reason_code": reason_code,
    }


def _diagnostic_fields(value: Mapping[str, object] | None) -> dict[str, object]:
    if value is None:
        return {}
    return {
        "candidate_count": value["candidate_count"],
        "evidence_count": value["evidence_count"],
        "highest_rerank_score": value["highest_rerank_score"],
        "retrieval_status": value["status"],
        "retrieval_reason_code": value["reason_code"],
    }


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _average(values: Sequence[float | None]) -> float | None:
    numbers = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(numbers) / len(numbers) if numbers else None


def _milliseconds(seconds: float) -> float:
    return round(max(0.0, seconds) * 1000, 3)


def _checksum(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "EVALUATION_TYPES",
    "EvaluationCase",
    "EvaluationCaseResult",
    "EvaluationGeneration",
    "EvaluationReport",
    "EvaluationService",
    "EvaluationSet",
    "calculate_report",
]
