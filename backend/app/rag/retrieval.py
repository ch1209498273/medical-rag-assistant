"""Vector recall, diversity filtering, reranking, and relevance gating."""

from __future__ import annotations

import re
from collections.abc import Sequence
from difflib import SequenceMatcher

from app.domain.ports import DocumentRepository, SearchHit, VectorStore
from app.rag.models import CandidateSet, Evidence, RetrievalResult

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_WHITESPACE = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w\u4e00-\u9fff]+", re.UNICODE)


def clean_question(question: str) -> str:
    """Remove C0 controls and reject requests outside the supported size."""

    if not isinstance(question, str):
        raise TypeError("question must be a string")
    cleaned = _CONTROL_CHARS.sub("", question).strip()
    if not cleaned:
        raise ValueError("question must not be empty")
    if len(cleaned) > 1000:
        raise ValueError("question must not exceed 1000 characters")
    return cleaned


class RetrievalService:
    """Retrieve active policy chunks and enforce a minimum relevance score."""

    def __init__(
        self,
        *,
        repository: DocumentRepository,
        vector_store: VectorStore,
        embedder,
        reranker,
        retrieval_limit: int = 20,
        rerank_limit: int = 6,
        relevance_threshold: float = 0.35,
    ) -> None:
        if retrieval_limit < 1 or retrieval_limit > 20:
            raise ValueError("retrieval_limit must be between 1 and 20")
        if rerank_limit < 1 or rerank_limit > 20:
            raise ValueError("rerank_limit must be between 1 and 20")
        if rerank_limit > retrieval_limit:
            raise ValueError("rerank_limit must not exceed retrieval_limit")
        if not 0.0 <= relevance_threshold <= 1.0:
            raise ValueError("relevance_threshold must be between 0 and 1")
        self.repository = repository
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = reranker
        self.retrieval_limit = retrieval_limit
        self.rerank_limit = rerank_limit
        self.relevance_threshold = relevance_threshold

    async def retrieve_candidates(self, question: str) -> CandidateSet:
        """Recall and deduplicate only chunks from currently active versions."""

        cleaned_question = clean_question(question)
        vectors = await self.embedder.embed([cleaned_question])
        if not isinstance(vectors, Sequence) or not vectors:
            return CandidateSet(question=cleaned_question)
        query_vector = vectors[0]
        active_versions = set(self.repository.active_version_ids())
        if not active_versions:
            return CandidateSet(question=cleaned_question)
        hits = self.vector_store.search(
            query_vector,
            active_versions,
            self.retrieval_limit,
        )
        active_hits = [hit for hit in hits if hit.version_id in active_versions][:20]
        retained = _deduplicate_hits(active_hits)
        return CandidateSet(question=cleaned_question, hits=tuple(retained))

    async def rerank(
        self, question: str, candidates: CandidateSet
    ) -> RetrievalResult:
        """Rerank candidates and refuse when the strongest score is too low."""

        cleaned_question = clean_question(question)
        if cleaned_question != candidates.question:
            # The candidate set is request-scoped; do not silently rank a set
            # recalled for another question.
            raise ValueError("candidate question does not match request")
        if not candidates.hits:
            return RetrievalResult(
                status="refused", reason_code="INSUFFICIENT_EVIDENCE"
            )

        documents = [hit.chunk.text for hit in candidates.hits]
        top_n = min(self.rerank_limit, 6, len(documents))
        ranked = await self.reranker.rerank(
            cleaned_question,
            documents,
            top_n,
        )
        if not ranked:
            return RetrievalResult(
                status="refused", reason_code="INSUFFICIENT_EVIDENCE"
            )

        validated_ranked: list[tuple[float, int]] = []
        for item in ranked:
            index = getattr(item, "index", None)
            score = getattr(item, "score", None)
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index < len(candidates.hits)
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
            ):
                continue
            validated_ranked.append((float(score), index))

        validated_ranked.sort(key=lambda value: value[0], reverse=True)
        evidence: list[Evidence] = []
        for score, index in validated_ranked[:6]:
            evidence.append(
                Evidence(
                    reference_id=f"S{len(evidence) + 1}",
                    hit=candidates.hits[index],
                    rerank_score=score,
                )
            )
        if not evidence or evidence[0].rerank_score < self.relevance_threshold:
            return RetrievalResult(
                status="refused", reason_code="INSUFFICIENT_EVIDENCE"
            )
        return RetrievalResult(status="ready", evidence=tuple(evidence))

    async def retrieve(self, question: str) -> RetrievalResult:
        """Run candidate recall followed by reranking in order."""

        candidates = await self.retrieve_candidates(question)
        return await self.rerank(candidates.question, candidates)


def _deduplicate_hits(hits: Sequence[SearchHit]) -> list[SearchHit]:
    """Keep high-scoring, non-overlapping chunks with heading diversity caps."""

    retained: list[SearchHit] = []
    hashes: set[str] = set()
    heading_counts: dict[tuple[str, tuple[str, ...]], int] = {}
    for hit in hits:
        chunk = hit.chunk
        content_hash = str(chunk.content_hash)
        if content_hash in hashes:
            continue
        normalised = _normalise_text(chunk.text)
        heading_key = (chunk.source.file_name, tuple(chunk.source.heading_path))
        if any(
            _is_duplicate_text(normalised, _normalise_text(existing.chunk.text))
            for existing in retained
        ):
            continue
        if heading_counts.get(heading_key, 0) >= 2:
            continue
        hashes.add(content_hash)
        heading_counts[heading_key] = heading_counts.get(heading_key, 0) + 1
        retained.append(hit)
    return retained


def _normalise_text(text: str) -> str:
    value = _WHITESPACE.sub(" ", str(text).casefold()).strip()
    return _NON_WORD.sub("", value)


def _is_duplicate_text(left: str, right: str) -> bool:
    if left == right:
        return True
    return SequenceMatcher(None, left, right).ratio() >= 0.85


__all__ = ["RetrievalService", "clean_question"]
