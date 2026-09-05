"""Deterministic lexical retrieval over a bounded active chunk snapshot.

The index is request-local and read-only with respect to the durable stores.  A
query term contributes ``idf * (1 + log(tf)) * field_boost`` for each field in
which it occurs.  ``idf`` is ``log((N + 1) / (df + 1)) + 1``; text has a boost
of ``1.0`` and heading text has a boost of ``2.0``.  All terms and tie-break
keys are processed deterministically.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from app.domain.models import Chunk, SourceRef
from app.domain.ports import SearchHit

_MAX_SNAPSHOT_LIMIT = 10_000
_MAX_RESULT_LIMIT = 20
_TEXT_FIELD_BOOST = 1.0
_HEADING_FIELD_BOOST = 2.0
_TOKEN_PATTERN = re.compile(
    r"[A-Za-z0-9]+|[㐀-䶿一-鿿豈-﫿]+"
)
_CJK_PATTERN = re.compile(r"[㐀-䶿一-鿿豈-﫿]+")


class ActiveChunkSnapshot(Protocol):
    """Read-only source of chunks belonging to caller-approved versions."""

    def snapshot_active_chunks(
        self, active_versions: set[str], limit: int = 10_000
    ) -> list[SearchHit]: ...


@dataclass(frozen=True, slots=True)
class _IndexedChunk:
    hit: SearchHit
    heading_counts: MappingProxyType[str, int]
    text_counts: MappingProxyType[str, int]


class ActiveLexicalIndex:
    """Build and search an in-memory lexical index for active chunks.

    The provider is called once for the currently active version set.  A later
    search with a different set replaces the in-memory snapshot and rebuilds
    postings exactly once for that set.  No persistent store is mutated.
    """

    def __init__(
        self,
        snapshot_provider: ActiveChunkSnapshot,
        *,
        snapshot_limit: int = 10_000,
    ) -> None:
        if (
            not isinstance(snapshot_limit, int)
            or isinstance(snapshot_limit, bool)
            or not 1 <= snapshot_limit <= _MAX_SNAPSHOT_LIMIT
        ):
            raise ValueError(
                f"snapshot_limit must be between 1 and {_MAX_SNAPSHOT_LIMIT}"
            )
        if not callable(getattr(snapshot_provider, "snapshot_active_chunks", None)):
            raise ValueError(  # noqa: TRY004 - this boundary reports a clear value error
                "snapshot provider must expose snapshot_active_chunks"
            )
        self._snapshot_provider = snapshot_provider
        self._snapshot_limit = snapshot_limit
        self._indexed_versions: frozenset[str] | None = None
        self._indexed_chunks: tuple[_IndexedChunk, ...] = ()
        self._postings: dict[str, tuple[int, ...]] = {}
        self._document_frequency: dict[str, int] = {}

    def search(
        self,
        queries: Sequence[str],
        active_versions: set[str],
        limit: int = 20,
    ) -> list[SearchHit]:
        """Return deterministically ranked lexical hits from active chunks."""

        if not isinstance(limit, int) or isinstance(limit, bool):
            raise ValueError("limit must be an integer")  # noqa: TRY004
        if limit <= 0:
            return []
        effective_limit = min(limit, _MAX_RESULT_LIMIT)

        versions = _normalise_active_versions(active_versions)
        if not versions:
            self._clear_index()
            return []

        query_tokens = _query_tokens(queries)
        if not query_tokens:
            return []

        self._ensure_snapshot(versions)
        document_count = len(self._indexed_chunks)
        if not document_count:
            return []

        scored: list[tuple[float, SearchHit]] = []
        for indexed in self._indexed_chunks:
            score = self._score(indexed, query_tokens, document_count)
            if score > 0.0 and indexed.hit.version_id in versions:
                scored.append((score, indexed.hit))

        scored.sort(
            key=lambda item: (
                -item[0],
                item[1].version_id,
                item[1].chunk.chunk_id,
            )
        )
        return [
            SearchHit(version_id=hit.version_id, chunk=hit.chunk, score=score)
            for score, hit in scored[:effective_limit]
        ]

    def _ensure_snapshot(self, active_versions: frozenset[str]) -> None:
        if self._indexed_versions == active_versions:
            return

        try:
            snapshot = self._snapshot_provider.snapshot_active_chunks(
                set(active_versions), limit=self._snapshot_limit
            )
        except Exception as exc:
            raise ValueError("active chunk snapshot provider failed") from exc

        if isinstance(snapshot, (str, bytes, bytearray)) or not isinstance(
            snapshot, Sequence
        ):
            raise ValueError("active chunk snapshot must be a sequence")  # noqa: TRY004
        if len(snapshot) > self._snapshot_limit:
            raise ValueError("active chunk snapshot exceeds snapshot_limit")

        indexed_chunks, postings, document_frequency = _build_index(
            snapshot, active_versions
        )
        self._indexed_versions = active_versions
        self._indexed_chunks = tuple(indexed_chunks)
        self._postings = postings
        self._document_frequency = document_frequency

    def _clear_index(self) -> None:
        self._indexed_versions = frozenset()
        self._indexed_chunks = ()
        self._postings = {}
        self._document_frequency = {}

    def _score(
        self,
        indexed: _IndexedChunk,
        query_tokens: frozenset[str],
        document_count: int,
    ) -> float:
        score = 0.0
        for token in sorted(query_tokens):
            if token not in self._document_frequency:
                continue
            document_frequency = self._document_frequency[token]
            idf = math.log(
                (document_count + 1) / (document_frequency + 1)
            ) + 1.0
            heading_tf = indexed.heading_counts.get(token, 0)
            text_tf = indexed.text_counts.get(token, 0)
            if heading_tf:
                score += idf * (1.0 + math.log(heading_tf)) * _HEADING_FIELD_BOOST
            if text_tf:
                score += idf * (1.0 + math.log(text_tf)) * _TEXT_FIELD_BOOST
        return score


def tokenize_lexical(text: str) -> tuple[str, ...]:
    """Tokenize ASCII/numeric words and contiguous Chinese character bigrams."""

    if not isinstance(text, str):
        raise ValueError("lexical text must be a string")  # noqa: TRY004

    tokens: list[str] = []
    for match in _TOKEN_PATTERN.finditer(text):
        value = match.group(0)
        if _CJK_PATTERN.fullmatch(value):
            tokens.extend(value[index : index + 2] for index in range(len(value) - 1))
        else:
            tokens.append(value.casefold())
    return tuple(tokens)


def _normalise_active_versions(active_versions: set[str]) -> frozenset[str]:
    if isinstance(active_versions, (str, bytes, bytearray)):
        raise ValueError("active_versions must be a set of strings")  # noqa: TRY004
    try:
        versions = frozenset(active_versions)
    except TypeError as exc:
        raise ValueError("active_versions must be an iterable of strings") from exc
    if any(not isinstance(version, str) or not version.strip() for version in versions):
        raise ValueError("active_versions must contain non-empty strings")
    return versions


def _query_tokens(queries: Sequence[str]) -> frozenset[str]:
    if isinstance(queries, (str, bytes, bytearray)) or not isinstance(queries, Sequence):
        raise ValueError("queries must be a sequence of strings")  # noqa: TRY004
    tokens: set[str] = set()
    for query in queries:
        if not isinstance(query, str):
            raise ValueError("queries must contain strings")  # noqa: TRY004
        tokens.update(tokenize_lexical(query))
    return frozenset(tokens)


def _build_index(
    snapshot: Sequence[SearchHit], active_versions: frozenset[str]
) -> tuple[
    list[_IndexedChunk],
    dict[str, tuple[int, ...]],
    dict[str, int],
]:
    indexed_chunks: list[_IndexedChunk] = []
    document_frequency: Counter[str] = Counter()
    postings: defaultdict[str, list[int]] = defaultdict(list)
    seen_ids: set[tuple[str, str]] = set()

    for position, raw_hit in enumerate(snapshot):
        hit = _validate_and_canonicalise_hit(raw_hit, position)
        if hit.version_id not in active_versions:
            continue
        identity = (hit.version_id, hit.chunk.chunk_id)
        if identity in seen_ids:
            raise ValueError("active chunk snapshot contains duplicate chunk identity")
        seen_ids.add(identity)

        heading_text = " ".join(hit.chunk.source.heading_path)
        heading_counts = Counter(tokenize_lexical(heading_text))
        text_counts = Counter(tokenize_lexical(hit.chunk.text))
        indexed = _IndexedChunk(
            hit=hit,
            heading_counts=MappingProxyType(dict(heading_counts)),
            text_counts=MappingProxyType(dict(text_counts)),
        )
        chunk_index = len(indexed_chunks)
        indexed_chunks.append(indexed)
        for token in sorted(set(heading_counts) | set(text_counts)):
            postings[token].append(chunk_index)
            document_frequency[token] += 1

    return (
        indexed_chunks,
        {token: tuple(indices) for token, indices in postings.items()},
        dict(document_frequency),
    )


def _validate_and_canonicalise_hit(raw_hit: object, position: int) -> SearchHit:
    prefix = f"invalid active chunk snapshot field at index {position}: "
    version_id = getattr(raw_hit, "version_id", None)
    if not isinstance(version_id, str) or not version_id.strip():
        raise ValueError(prefix + "version_id")

    raw_chunk = getattr(raw_hit, "chunk", None)
    if raw_chunk is None:
        raise ValueError(prefix + "chunk")
    chunk_id = getattr(raw_chunk, "chunk_id", None)
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise ValueError(prefix + "chunk_id")
    text = getattr(raw_chunk, "text", None)
    if not isinstance(text, str):
        raise ValueError(prefix + "text")  # noqa: TRY004
    content_hash = getattr(raw_chunk, "content_hash", None)
    if not isinstance(content_hash, str) or not content_hash.strip():
        raise ValueError(prefix + "content_hash")

    raw_source = getattr(raw_chunk, "source", None)
    if raw_source is None:
        raise ValueError(prefix + "source")
    file_name = getattr(raw_source, "file_name", None)
    if not isinstance(file_name, str) or not file_name.strip():
        raise ValueError(prefix + "file_name")
    raw_heading_path = getattr(raw_source, "heading_path", None)
    if isinstance(raw_heading_path, (str, bytes, bytearray)) or not isinstance(
        raw_heading_path, Sequence
    ):
        raise ValueError(prefix + "heading_path")  # noqa: TRY004
    heading_path = tuple(raw_heading_path)
    if any(not isinstance(item, str) or not item.strip() for item in heading_path):
        raise ValueError(prefix + "heading_path")

    raw_score = getattr(raw_hit, "score", None)
    if (
        isinstance(raw_score, bool)
        or not isinstance(raw_score, (int, float))
        or not math.isfinite(float(raw_score))
    ):
        raise ValueError(prefix + "score")

    if isinstance(raw_hit, SearchHit) and isinstance(raw_chunk, Chunk):
        return raw_hit

    source = SourceRef(
        file_name=file_name,
        heading_path=heading_path,
        page=_optional_int(raw_source, "page", prefix),
        page_end=_optional_int(raw_source, "page_end", prefix),
        paragraph_start=_optional_int(raw_source, "paragraph_start", prefix),
        paragraph_end=_optional_int(raw_source, "paragraph_end", prefix),
        table_id=_optional_table_id(raw_source, prefix),
    )
    return SearchHit(
        version_id=version_id,
        chunk=Chunk(
            chunk_id=chunk_id,
            text=text,
            source=source,
            content_hash=content_hash,
        ),
        score=float(raw_score),
    )


def _optional_int(raw_source: object, field: str, prefix: str) -> int | None:
    value = getattr(raw_source, field, None)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(prefix + field)  # noqa: TRY004
    return value


def _optional_table_id(raw_source: object, prefix: str) -> str | None:
    value = getattr(raw_source, "table_id", None)
    if value is None:
        return None
    if not isinstance(value, str) or not value.startswith("table-"):
        raise ValueError(prefix + "table_id")
    suffix = value.removeprefix("table-")
    if not suffix.isdigit() or int(suffix) < 1:
        raise ValueError(prefix + "table_id")
    return value


__all__ = ["ActiveChunkSnapshot", "ActiveLexicalIndex", "tokenize_lexical"]
