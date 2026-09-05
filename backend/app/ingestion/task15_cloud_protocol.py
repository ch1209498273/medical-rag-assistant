"""Fail-closed cloud Embedding protocol for the Task 15 private stage."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.domain.models import Chunk, SourceRef
from app.ingestion.staging_chunker import StagingChunkCandidate
from app.ingestion.task15_models import StagingChunkRecord, StagingManifest
from app.providers.siliconflow import SiliconFlowClient
from app.settings import get_settings, require_cloud_keys

APPROVED_PROVIDER = "siliconflow"
APPROVED_MODEL = "BAAI/bge-m3"
VECTOR_DIMENSION = 1024


@dataclass(frozen=True, slots=True)
class Task15CloudAuthorization:
    """Explicit authorization facts for one bounded cloud operation."""

    provider: str
    model: str
    max_calls: int
    retries: int
    save_raw_responses: bool
    private_output_dir: Path
    authorization_id: str


def parse_authorization(path: Path) -> Task15CloudAuthorization:
    """Parse an authorization file while rejecting secrets and unknown shapes."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cloud authorization is invalid") from error
    if not isinstance(payload, dict):
        raise TypeError("cloud authorization must be an object")
    if any(
        isinstance(key, str)
        and any(marker in key.casefold() for marker in ("api_key", "apikey", "secret", "token"))
        for key in payload
    ):
        raise ValueError("cloud authorization must not contain secrets")
    required = {
        "provider",
        "model",
        "max_calls",
        "retries",
        "save_raw_responses",
        "private_output_dir",
        "authorization_id",
    }
    if set(payload) != required:
        raise ValueError("cloud authorization fields are invalid")
    private_dir = payload["private_output_dir"]
    if not isinstance(private_dir, str) or not private_dir.strip():
        raise ValueError("private_output_dir is invalid")
    authorization = Task15CloudAuthorization(
        provider=str(payload["provider"]),
        model=str(payload["model"]),
        max_calls=payload["max_calls"],  # type: ignore[arg-type]
        retries=payload["retries"],  # type: ignore[arg-type]
        save_raw_responses=payload["save_raw_responses"],  # type: ignore[arg-type]
        private_output_dir=Path(private_dir),
        authorization_id=str(payload["authorization_id"]),
    )
    validate_authorization(authorization, expected_calls=0)
    return authorization


def validate_authorization(
    auth: Task15CloudAuthorization,
    *,
    expected_calls: int,
) -> None:
    """Validate provider, budget, retry and privacy gates before network setup."""

    if auth.provider != APPROVED_PROVIDER:
        raise ValueError("provider is not approved")
    if auth.model != APPROVED_MODEL:
        raise ValueError("model is not approved")
    if isinstance(expected_calls, bool) or not isinstance(expected_calls, int) or expected_calls < 0:
        raise ValueError("expected_calls is invalid")
    if isinstance(auth.max_calls, bool) or not isinstance(auth.max_calls, int) or auth.max_calls < expected_calls:
        raise ValueError("cloud call budget is insufficient")
    if auth.retries != 0:
        raise ValueError("retries must be zero")
    if auth.save_raw_responses is not False:
        raise ValueError("raw response saving must be disabled")
    if not isinstance(auth.authorization_id, str) or not auth.authorization_id.strip():
        raise ValueError("authorization_id is invalid")
    if not _looks_private(auth.private_output_dir):
        raise ValueError("private_output_dir must be private")


async def run_authorized_embedding(
    manifest: StagingManifest,
    auth: Task15CloudAuthorization,
    *,
    approved_backend: str | None = None,
    provider: Any = None,
    vector_store: Any = None,
) -> Path:
    """Embed one explicitly approved candidate backend with zero retries."""

    if approved_backend not in {"docling", "legacy"}:
        raise ValueError("approved_backend must be docling or legacy")
    candidates = _load_candidates(manifest, approved_backend)
    validate_authorization(auth, expected_calls=len(candidates))
    auth.private_output_dir.mkdir(parents=True, exist_ok=True)

    owns_provider = provider is None
    resolved_provider = provider
    if resolved_provider is None:
        settings = get_settings()
        require_cloud_keys(settings)
        resolved_provider = SiliconFlowClient(
            settings.siliconflow_api_key,
            settings.siliconflow_base_url,
            embedding_model=APPROVED_MODEL,
            reranker_model=settings.reranker_model,
            ocr_model=settings.ocr_model,
            max_retries=0,
        )

    vectors: list[list[float]] = []
    chunks: list[Chunk] = []
    failure_code: str | None = None
    calls = 0
    try:
        for candidate in candidates:
            calls += 1
            try:
                response = await resolved_provider.embed([candidate.text])
                vector = _validate_single_vector(response)
            except Exception as error:  # noqa: BLE001 - aggregate only, never raw text
                failure_code = error.__class__.__name__.casefold()
                break
            vectors.append(vector)
            chunks.append(
                Chunk(
                    chunk_id=candidate.record.chunk_id,
                    text=candidate.text,
                    source=candidate.source,
                    content_hash=candidate.record.content_hash,
                )
            )
        complete = failure_code is None and len(vectors) == len(candidates)
        vector_store_written = False
        if complete and vector_store is not None and chunks:
            vector_store.upsert(manifest.source_version, chunks, vectors)
            vector_store_written = True
        summary = {
            "run_id": manifest.run_id,
            "source_version": manifest.source_version,
            "provider": APPROVED_PROVIDER,
            "model": APPROVED_MODEL,
            "authorization_id": auth.authorization_id,
            "approved_backend": approved_backend,
            "status": "complete" if complete else "blocked",
            "candidate_count": len(candidates),
            "call_count": calls,
            "success_count": len(vectors),
            "failure_count": 0 if complete else 1,
            "failure_code": failure_code,
            "retries": 0,
            "original_response_saved": False,
            "vector_dimension": VECTOR_DIMENSION,
            "vector_store_written": vector_store_written,
            "created_at": datetime.now(UTC).isoformat(),
        }
        output = auth.private_output_dir / f"task15-embedding-{manifest.run_id}.json"
        output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return output
    finally:
        if owns_provider:
            close = getattr(resolved_provider, "aclose", None)
            if callable(close):
                result = close()
                if hasattr(result, "__await__"):
                    await result


def _load_candidates(
    manifest: StagingManifest,
    approved_backend: str,
) -> tuple[StagingChunkCandidate, ...]:
    manifest_path = Path(manifest.output_root) / "chunk-manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("chunk manifest is invalid") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("files"), list):
        raise TypeError("chunk manifest is invalid")
    candidates: list[StagingChunkCandidate] = []
    for item in payload["files"]:
        if not isinstance(item, dict):
            raise TypeError("chunk manifest file entry is invalid")
        details = item.get(approved_backend)
        if not isinstance(details, Mapping):
            continue
        relative = details.get("path")
        count = details.get("count")
        if not isinstance(relative, str) or not isinstance(count, int):
            raise TypeError("chunk manifest candidate path is invalid")
        if count == 0:
            continue
        candidate_path = Path(relative)
        if not candidate_path.is_absolute():
            candidate_path = Path(manifest.output_root) / candidate_path
        try:
            lines = candidate_path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise ValueError("private candidate file is unavailable") from error
        if len(lines) != count:
            raise ValueError("candidate count does not match manifest")
        for line in lines:
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("private candidate record is invalid") from error
            candidates.append(_candidate_from_private_record(raw))
    if not candidates:
        raise ValueError("approved backend has no candidates")
    return tuple(candidates)


def _candidate_from_private_record(raw: object) -> StagingChunkCandidate:
    if not isinstance(raw, dict):
        raise TypeError("private candidate record is invalid")
    text = raw.get("text")
    source_payload = raw.get("source")
    record_payload = raw.get("record")
    if not isinstance(text, str) or not text.strip() or not isinstance(source_payload, dict) or not isinstance(record_payload, dict):
        raise ValueError("private candidate record is invalid")
    try:
        record = StagingChunkRecord(
            chunk_id=str(record_payload["chunk_id"]),
            source_version=str(record_payload["source_version"]),
            source_relative_path=str(record_payload["source_relative_path"]),
            source_sha256=str(record_payload["source_sha256"]),
            heading_path=tuple(record_payload["heading_path"]),
            content_kind=str(record_payload["content_kind"]),
            provenance_digest=str(record_payload["provenance_digest"]),
            content_hash=str(record_payload["content_hash"]),
            token_count=int(record_payload["token_count"]),
            parser_backend=str(record_payload["parser_backend"]),
            ordinal=int(record_payload["ordinal"]),
        )
        source = SourceRef(
            file_name=str(source_payload["file_name"]),
            heading_path=tuple(source_payload["heading_path"]),
            page=_optional_int(source_payload.get("page")),
            page_end=_optional_int(source_payload.get("page_end")),
            paragraph_start=_optional_int(source_payload.get("paragraph_start")),
            paragraph_end=_optional_int(source_payload.get("paragraph_end")),
            table_id=_optional_table_id(source_payload.get("table_id")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("private candidate record is invalid") from error
    if record.content_hash != _digest(text.strip()):
        raise ValueError("private candidate content hash is invalid")
    return StagingChunkCandidate(record=record, text=text.strip(), source=source)


def _validate_single_vector(value: object) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 1:
        raise ValueError("embedding response count is invalid")
    vector = value[0]
    if not isinstance(vector, (list, tuple)) or len(vector) != VECTOR_DIMENSION:
        raise ValueError("embedding vector dimension is invalid")
    result = [float(item) for item in vector]
    if any(not math.isfinite(item) for item in result):
        raise ValueError("embedding vector contains a non-finite value")
    return result


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("source position is invalid")
    return value


def _optional_table_id(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.startswith("table-"):
        raise TypeError("source table_id must be a table-N value or null")
    suffix = value.removeprefix("table-")
    if not suffix.isdigit() or int(suffix) < 1:
        raise ValueError("source table_id must be a table-N value or null")
    return value


def _looks_private(path: Path) -> bool:
    parts = {part.casefold() for part in Path(path).resolve().parts}
    return bool(parts & {"private", "private-output", "knowledge-refresh"})


def _digest(value: str) -> str:
    from hashlib import sha256

    return sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "Task15CloudAuthorization",
    "parse_authorization",
    "run_authorized_embedding",
    "validate_authorization",
]
