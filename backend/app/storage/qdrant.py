"""Qdrant Local Mode implementation of version-filtered vector retrieval."""

from __future__ import annotations

import logging
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.domain.models import Chunk, SourceRef
from app.domain.ports import SearchHit

LOGGER = logging.getLogger(__name__)


class QdrantLocalVectorStore:
    """Store 1024-dimensional policy embeddings in a local durable collection."""

    collection_name = "policy_chunks_v1"
    vector_dimension = 1024

    def __init__(self, data_path: Path, *, create_if_missing: bool = True) -> None:
        """Open the local store, optionally without creating any state.

        Production ingestion keeps the historical default of creating the
        Qdrant directory and collection.  Offline evaluation passes
        ``create_if_missing=False`` so a missing snapshot fails closed without
        leaving a newly-created database behind.
        """

        self._readonly_tempdir: tempfile.TemporaryDirectory[str] | None = None
        client_path = data_path
        if create_if_missing:
            data_path.mkdir(parents=True, exist_ok=True)
        else:
            if not data_path.is_dir():
                raise ValueError("qdrant data path is unavailable")
            # Qdrant Local Mode opens its lock and collection SQLite files in
            # read/write mode during construction.  Mirror the source first so
            # those unavoidable implementation details never touch R1 input.
            self._readonly_tempdir = tempfile.TemporaryDirectory(
                prefix="task14-qdrant-readonly-"
            )
            client_path = Path(self._readonly_tempdir.name) / "snapshot"
            try:
                shutil.copytree(data_path, client_path)
            except BaseException:
                self._cleanup_readonly_mirror()
                raise
        try:
            client = QdrantClient(path=str(client_path))
        except BaseException:
            self._cleanup_readonly_mirror()
            raise
        self._client = client
        try:
            if not self._client.collection_exists(self.collection_name):
                if not create_if_missing:
                    raise ValueError("qdrant collection is unavailable")
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=self.vector_dimension,
                        distance=qmodels.Distance.COSINE,
                    ),
                )
        except BaseException:
            try:
                client.close()
            except BaseException:  # noqa: BLE001 - preserve collection error
                LOGGER.warning(
                    "qdrant constructor cleanup failed",
                    extra={"stage": "qdrant_constructor_cleanup"},
                )
            finally:
                self._cleanup_readonly_mirror()
            raise

    def upsert(
        self,
        version_id: str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        """Write every chunk with the source data needed to validate a citation."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("duplicate chunk_id values are not allowed per version")
        if any(len(vector) != self.vector_dimension for vector in vectors):
            raise ValueError(f"vectors must have {self.vector_dimension} dimensions")
        points = [
            qmodels.PointStruct(
                id=str(uuid5(NAMESPACE_URL, f"{version_id}:{chunk.chunk_id}")),
                vector=list(vector),
                payload=self._payload(version_id, chunk),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        if points:
            self._client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def search(
        self,
        query_vector: Sequence[float],
        active_versions: set[str],
        limit: int,
    ) -> list[SearchHit]:
        """Search only caller-approved active versions, failing safe for an empty set."""
        if not active_versions or limit <= 0:
            return []
        if len(query_vector) != self.vector_dimension:
            raise ValueError(f"query vector must have {self.vector_dimension} dimensions")
        response = self._client.query_points(
            collection_name=self.collection_name,
            query=list(query_vector),
            query_filter=qmodels.Filter(
                should=[
                    qmodels.FieldCondition(
                        key="version_id",
                        match=qmodels.MatchAny(any=sorted(active_versions)),
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [self._search_hit(point.payload or {}, point.score) for point in response.points]

    def snapshot_active_chunks(
        self, active_versions: set[str], limit: int = 10_000
    ) -> list[SearchHit]:
        """Read a bounded, version-filtered source snapshot without mutating Qdrant."""

        if not active_versions or limit <= 0:
            return []
        if limit > 10_000:
            raise ValueError("snapshot limit exceeds the hard bound")
        records, _ = self._client.scroll(
            collection_name=self.collection_name,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="version_id",
                        match=qmodels.MatchAny(any=sorted(active_versions)),
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [self._search_hit(record.payload or {}, 0.0) for record in records]

    def snapshot_all_chunks(self, limit: int = 10_000) -> list[SearchHit]:
        """Read a bounded local snapshot across versions for mismatch diagnostics."""

        if limit <= 0:
            return []
        if limit > 10_000:
            raise ValueError("snapshot limit exceeds the hard bound")
        records, _ = self._client.scroll(
            collection_name=self.collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [self._search_hit(record.payload or {}, 0.0) for record in records]

    def collection_info(self):
        """Expose read-only collection metadata for health checks and integration tests."""
        return self._client.get_collection(self.collection_name)

    def close(self) -> None:
        try:
            self._client.close()
        finally:
            self._cleanup_readonly_mirror()

    def _cleanup_readonly_mirror(self) -> None:
        tempdir = self._readonly_tempdir
        if tempdir is None:
            return
        self._readonly_tempdir = None
        try:
            tempdir.cleanup()
        except BaseException:  # noqa: BLE001 - preserve the original error
            LOGGER.warning(
                "qdrant readonly mirror cleanup failed",
                extra={"stage": "qdrant_readonly_mirror_cleanup"},
            )

    def delete_version(self, version_id: str) -> None:
        """Remove vectors from a timed-out unpublished version."""
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="version_id",
                            match=qmodels.MatchValue(value=version_id),
                        )
                    ]
                )
            ),
            wait=True,
        )

    @staticmethod
    def _payload(version_id: str, chunk: Chunk) -> dict[str, object]:
        return {
            "version_id": version_id,
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "content_hash": chunk.content_hash,
            "file_name": chunk.source.file_name,
            "heading_path": list(chunk.source.heading_path),
            "page": chunk.source.page,
            "page_end": chunk.source.page_end,
            "paragraph_start": chunk.source.paragraph_start,
            "paragraph_end": chunk.source.paragraph_end,
            "table_id": chunk.source.table_id,
        }

    @staticmethod
    def _search_hit(payload: dict[str, object], score: float) -> SearchHit:
        version_id = _required_payload_text(payload, "version_id")
        chunk_id = _required_payload_text(payload, "chunk_id")
        content_hash = _required_payload_text(payload, "content_hash")
        text = _required_payload_text(payload, "text")
        file_name = _required_payload_text(payload, "file_name")
        heading_path = _required_heading_path(payload.get("heading_path"))
        source = SourceRef(
            file_name=file_name,
            heading_path=heading_path,
            page=_optional_int(payload.get("page"), "page"),
            page_end=_optional_int(payload.get("page_end"), "page_end"),
            paragraph_start=_optional_int(
                payload.get("paragraph_start"), "paragraph_start"
            ),
            paragraph_end=_optional_int(payload.get("paragraph_end"), "paragraph_end"),
            table_id=_optional_table_id(payload.get("table_id")),
        )
        return SearchHit(
            version_id=version_id,
            chunk=Chunk(
                chunk_id=chunk_id,
                text=text,
                content_hash=content_hash,
                source=source,
            ),
            score=score,
        )


def _required_payload_text(payload: dict[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"invalid qdrant payload field: {field}")
    return value


def _required_heading_path(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("invalid qdrant payload field: heading_path")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError("invalid qdrant payload field: heading_path")
    return tuple(value)


def _optional_int(value: object, field: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(  # noqa: TRY004 - malformed untrusted payload is a value error
            f"invalid qdrant payload field: {field}"
        )
    return value


def _optional_table_id(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.startswith("table-"):
        raise ValueError("invalid qdrant payload field: table_id")
    suffix = value.removeprefix("table-")
    if not suffix.isdigit() or int(suffix) < 1:
        raise ValueError("invalid qdrant payload field: table_id")
    return value
