"""Vector recall, diversity filtering, reranking, and relevance gating."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from datetime import UTC, date, datetime
from difflib import SequenceMatcher
from typing import Literal

from app.chat.safety import validate_audience_scope
from app.domain.ports import (
    AudienceScope,
    DocumentRepository,
    EligibilityResult,
    SearchHit,
    VectorStore,
)
from app.evaluation.task14d_models import (
    BranchCandidateFact,
    DedupDropFact,
    FusionCandidateFact,
    Task14DRetrievalTrace,
)
from app.rag.lexical import ActiveLexicalIndex
from app.rag.models import (
    CandidateSet,
    Evidence,
    RetrievalDiagnosticCandidate,
    RetrievalDiagnosticMetadata,
    RetrievalDiagnosticTrace,
    RetrievalResult,
)
from app.rag.query import build_query_variants, clean_question

_WHITESPACE = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w\u4e00-\u9fff]+", re.UNICODE)

RetrievalStrategy = Literal["vector", "lexical", "hybrid", "hybrid_normalized"]
RetrievalFusionProfile = Literal["balanced", "vector_dominant"]
_MAX_CANDIDATE_LIMIT = 20
_DIAGNOSTIC_TRACE_SCHEMA_VERSION = "retrieval-diagnostic-v1"
_DIAGNOSTIC_SELECTOR_NAME = "reranker"
_DIAGNOSTIC_SELECTOR_VERSION = "v1"
_RRF_WEIGHTS_BY_PROFILE: dict[str, tuple[float, float]] = {
    "balanced": (1.0, 1.0),
    "vector_dominant": (3.0, 1.0),
}


def merge_ranked_hits(
    vector_hits: Sequence[SearchHit],
    lexical_hits: Sequence[SearchHit],
    *,
    limit: int,
    rrf_k: int = 60,
    vector_weight: float = 1.0,
    lexical_weight: float = 1.0,
) -> list[SearchHit]:
    """Merge vector and lexical rankings using weighted RRF.

    A hit contributes ``weight / (rrf_k + rank)`` in each branch, where rank
    is one-based.  Content hashes are the primary document identity so that a
    chunk returned by both branches is represented once with its scores
    added.  The result is always bounded by the existing candidate budget.
    """

    merged, _, _ = merge_ranked_hits_with_diagnostics(
        vector_hits,
        lexical_hits,
        limit=limit,
        rrf_k=rrf_k,
        vector_weight=vector_weight,
        lexical_weight=lexical_weight,
    )
    return merged


def merge_ranked_hits_with_diagnostics(
    vector_hits: Sequence[SearchHit],
    lexical_hits: Sequence[SearchHit],
    *,
    limit: int,
    rrf_k: int = 60,
    vector_weight: float = 1.0,
    lexical_weight: float = 1.0,
) -> tuple[list[SearchHit], tuple[FusionCandidateFact, ...], tuple[DedupDropFact, ...]]:
    """Merge ranked branches and retain private branch/RRF attribution facts."""

    _validate_rrf_k(rrf_k)
    vector_weight = _validate_rrf_weight("vector_weight", vector_weight)
    lexical_weight = _validate_rrf_weight("lexical_weight", lexical_weight)
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    effective_limit = min(limit, _MAX_CANDIDATE_LIMIT)

    merged: dict[tuple[str, ...], tuple[float, SearchHit, int | None, int | None]] = {}
    drops: list[DedupDropFact] = []
    for branch, branch_weight, ranked_hits in (
        ("vector", vector_weight, vector_hits),
        ("lexical", lexical_weight, lexical_hits),
    ):
        seen: set[tuple[str, ...]] = set()
        rank = 0
        for hit in ranked_hits:
            key = _hit_identity(hit)
            candidate_id = _diagnostic_candidate_id(hit)
            if key in seen:
                drops.append(
                    DedupDropFact(
                        candidate_id=candidate_id,
                        reason="rrf_content_hash_merge",
                        stage="rrf",
                        kept_candidate_id=candidate_id,
                        source_id=getattr(hit.chunk, "chunk_id", None),
                    )
                )
                continue
            seen.add(key)
            rank += 1
            contribution = branch_weight / (rrf_k + rank)
            if key in merged:
                score, retained_hit, vector_rank, lexical_rank = merged[key]
                if branch == "vector":
                    vector_rank = rank
                else:
                    lexical_rank = rank
                merged[key] = (score + contribution, retained_hit, vector_rank, lexical_rank)
            else:
                merged[key] = (
                    contribution,
                    hit,
                    rank if branch == "vector" else None,
                    rank if branch == "lexical" else None,
                )

    ranked = sorted(
        merged.values(),
        key=lambda item: (-item[0], _stable_hit_key(item[1])),
    )
    fusion_facts: list[FusionCandidateFact] = []
    merged_hits: list[SearchHit] = []
    for fused_rank, (score, hit, vector_rank, lexical_rank) in enumerate(ranked, start=1):
        if vector_rank is not None and lexical_rank is not None:
            membership = "shared"
        elif vector_rank is not None:
            membership = "vector_only"
        else:
            membership = "lexical_only"
        fusion_facts.append(
            FusionCandidateFact(
                candidate_id=_diagnostic_candidate_id(hit),
                membership=membership,
                vector_rank=vector_rank,
                lexical_rank=lexical_rank,
                fused_rank=fused_rank,
                rrf_score=score,
                source_id=getattr(hit.chunk, "chunk_id", None),
            )
        )
        if fused_rank <= effective_limit:
            merged_hits.append(
                SearchHit(version_id=hit.version_id, chunk=hit.chunk, score=score)
            )
        else:
            drops.append(
                DedupDropFact(
                    candidate_id=_diagnostic_candidate_id(hit),
                    reason="top20_truncation",
                    stage="rrf",
                    source_id=getattr(hit.chunk, "chunk_id", None),
                )
            )
    return merged_hits, tuple(fusion_facts), tuple(drops)


def _validate_rrf_k(rrf_k: int) -> None:
    if not isinstance(rrf_k, int) or isinstance(rrf_k, bool) or rrf_k <= 0:
        raise ValueError("rrf_k must be a positive integer")


def _validate_rrf_weight(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(  # noqa: TRY004 - all invalid weights share one contract
            f"{name} must be a finite positive number"
        )
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be a finite positive number"
        ) from error
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


def rrf_weights_for_profile(
    fusion_profile: RetrievalFusionProfile,
) -> tuple[float, float]:
    """Return the fixed vector/lexical weights for a safe fusion profile."""

    if not isinstance(fusion_profile, str) or fusion_profile not in _RRF_WEIGHTS_BY_PROFILE:
        raise ValueError("unsupported fusion profile")
    return _RRF_WEIGHTS_BY_PROFILE[fusion_profile]


def _hit_identity(hit: SearchHit) -> tuple[str, ...]:
    content_hash = getattr(hit.chunk, "content_hash", None)
    if isinstance(content_hash, str) and content_hash:
        return ("content_hash", content_hash)
    return (
        "chunk",
        str(hit.version_id),
        str(getattr(hit.chunk, "chunk_id", "")),
    )


def _diagnostic_candidate_id(hit: SearchHit) -> str:
    """Return a stable private identity without copying candidate content."""

    content_hash = getattr(hit.chunk, "content_hash", None)
    if isinstance(content_hash, str) and content_hash.strip():
        return content_hash.strip()
    return f"{hit.version_id}:{getattr(hit.chunk, 'chunk_id', '')}"


def _branch_candidate_facts(
    hits: Sequence[SearchHit], branch: str
) -> tuple[BranchCandidateFact, ...]:
    if branch not in {"vector", "lexical"}:
        raise ValueError("unsupported diagnostic branch")
    return tuple(
        BranchCandidateFact(
            candidate_id=_diagnostic_candidate_id(hit),
            branch=branch,  # type: ignore[arg-type]
            rank=rank,
            retrieval_score=hit.score,
            content_hash=str(hit.chunk.content_hash),
            source_id=getattr(hit.chunk, "chunk_id", None),
        )
        for rank, hit in enumerate(hits, start=1)
    )


def _stable_hit_key(hit: SearchHit) -> tuple[str, str, str, str]:
    chunk = hit.chunk
    source = chunk.source
    heading = "/".join(str(item) for item in source.heading_path)
    return (
        str(hit.version_id),
        str(chunk.chunk_id),
        str(chunk.content_hash),
        f"{source.file_name}\x00{heading}\x00{chunk.text}",
    )


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
        retrieval_strategy: RetrievalStrategy = "vector",
        lexical_index: ActiveLexicalIndex | None = None,
        rrf_k: int = 60,
        fusion_profile: RetrievalFusionProfile = "balanced",
        vector_weight: float | None = None,
        lexical_weight: float | None = None,
        diagnostic_trace: bool = False,
        task14d_diagnostics: bool = False,
        post_fusion_dedup: bool = True,
        require_business_eligibility: bool = True,
    ) -> None:
        if retrieval_limit < 1 or retrieval_limit > 20:
            raise ValueError("retrieval_limit must be between 1 and 20")
        if rerank_limit < 1 or rerank_limit > 20:
            raise ValueError("rerank_limit must be between 1 and 20")
        if rerank_limit > retrieval_limit:
            raise ValueError("rerank_limit must not exceed retrieval_limit")
        if not 0.0 <= relevance_threshold <= 1.0:
            raise ValueError("relevance_threshold must be between 0 and 1")
        _validate_rrf_k(rrf_k)
        profile_vector_weight, profile_lexical_weight = rrf_weights_for_profile(
            fusion_profile
        )
        if (vector_weight is None) != (lexical_weight is None):
            raise ValueError(
                "vector_weight and lexical_weight must be provided together"
            )
        if vector_weight is not None and lexical_weight is not None:
            resolved_vector_weight = _validate_rrf_weight(
                "vector_weight", vector_weight
            )
            resolved_lexical_weight = _validate_rrf_weight(
                "lexical_weight", lexical_weight
            )
            if (
                resolved_vector_weight != profile_vector_weight
                or resolved_lexical_weight != profile_lexical_weight
            ):
                raise ValueError("weights do not match fusion profile")
        else:
            resolved_vector_weight = profile_vector_weight
            resolved_lexical_weight = profile_lexical_weight
        if retrieval_strategy not in {"vector", "lexical", "hybrid", "hybrid_normalized"}:
            raise ValueError(
                "retrieval_strategy must be vector, lexical, hybrid, or hybrid_normalized"
            )
        if retrieval_strategy in {"lexical", "hybrid", "hybrid_normalized"} and lexical_index is None:
            raise ValueError(
                "lexical_index is required for hybrid retrieval strategies"
            )
        if not isinstance(diagnostic_trace, bool):
            raise TypeError("diagnostic_trace must be a boolean")
        if not isinstance(task14d_diagnostics, bool):
            raise TypeError("task14d_diagnostics must be a boolean")
        if not isinstance(post_fusion_dedup, bool):
            raise TypeError("post_fusion_dedup must be a boolean")
        if not isinstance(require_business_eligibility, bool):
            raise TypeError("require_business_eligibility must be a boolean")
        self.repository = repository
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = reranker
        self.retrieval_limit = retrieval_limit
        self.rerank_limit = rerank_limit
        self.relevance_threshold = relevance_threshold
        self.retrieval_strategy = retrieval_strategy
        self.lexical_index = lexical_index
        self.rrf_k = rrf_k
        self.fusion_profile = fusion_profile
        self.retrieval_fusion_profile = fusion_profile
        self.vector_weight = resolved_vector_weight
        self.lexical_weight = resolved_lexical_weight
        self.diagnostic_trace = diagnostic_trace
        self.task14d_diagnostics = task14d_diagnostics
        self.post_fusion_dedup = post_fusion_dedup
        self._task14d_trace: Task14DRetrievalTrace | None = None
        self.require_business_eligibility = require_business_eligibility

    @property
    def task14d_trace(self) -> Task14DRetrievalTrace | None:
        """Return the latest text-free Task 14D trace, when enabled."""

        return self._task14d_trace

    async def retrieve_candidates(
        self,
        question: str,
        audience_scope: AudienceScope = "unspecified",
        as_of: date | None = None,
    ) -> CandidateSet:
        """Recall only chunks from business-eligible technical versions.

        Older injected repositories used by the demo/tests may only expose
        ``active_version_ids``.  They retain that compatibility path; the
        production repository exposes ``eligible_version_ids`` and therefore
        fails closed when no approved, in-date, scope-matching version exists.
        """

        cleaned_question = clean_question(question)
        audience_scope = validate_audience_scope(audience_scope)
        self._task14d_trace = (
            Task14DRetrievalTrace() if self.task14d_diagnostics else None
        )
        if as_of is None:
            as_of = datetime.now(UTC).date()
        if not isinstance(as_of, date):
            raise TypeError("as_of must be a date")

        # The public demo uses fictional, pre-bundled documents and a
        # deterministic offline provider.  It deliberately has no business
        # metadata-management step; the application assembly opts into this
        # explicit compatibility mode only for that runtime.  Real/default
        # retrieval remains fail-closed through ``eligible_version_ids``.
        if not self.require_business_eligibility:
            active_versions = set(self.repository.active_version_ids())
            exclusion_reason = None
            eligibility_method = None
        else:
            eligibility_method = getattr(self.repository, "eligible_version_ids", None)
            active_versions, exclusion_reason = await self._eligible_versions(
                audience_scope, as_of
            )
        # The production repository exposes business eligibility, so an empty
        # allowlist can short-circuit before any provider call.  Legacy fakes
        # only expose technical active IDs; preserve their historical
        # embedding observation before returning an empty candidate set.
        if callable(eligibility_method) and not active_versions:
            return CandidateSet(
                question=cleaned_question,
                reason_code=exclusion_reason,
            )

        if not active_versions:
            return CandidateSet(question=cleaned_question)
        active_vector_hits: list[SearchHit] = []
        vector_facts: tuple[BranchCandidateFact, ...] = ()
        if self.retrieval_strategy != "lexical":
            vectors = await self.embedder.embed([cleaned_question])
            if not isinstance(vectors, Sequence) or not vectors:
                return CandidateSet(question=cleaned_question)
            query_vector = vectors[0]
            hits = self.vector_store.search(
                query_vector,
                active_versions,
                self.retrieval_limit,
            )
            active_vector_hits = [
                hit for hit in hits if hit.version_id in active_versions
            ][: _MAX_CANDIDATE_LIMIT]
            if self.task14d_diagnostics:
                vector_facts = _branch_candidate_facts(active_vector_hits, "vector")

        active_lexical_hits: list[SearchHit] = []
        lexical_facts: tuple[BranchCandidateFact, ...] = ()
        if self.retrieval_strategy == "vector":
            combined_hits = active_vector_hits
        else:
            if self.lexical_index is None:  # Defensive guard for mutable integrations.
                raise ValueError(
                    "lexical_index is required for hybrid retrieval strategies"
                )
            if self.retrieval_strategy in {"lexical", "hybrid"}:
                lexical_queries = (cleaned_question,)
            else:
                lexical_queries = build_query_variants(
                    cleaned_question, use_aliases=True
                ).lexical_queries
            lexical_hits = self.lexical_index.search(
                lexical_queries,
                active_versions,
                self.retrieval_limit,
            )
            active_lexical_hits = [
                hit
                for hit in lexical_hits
                if hit.version_id in active_versions
            ][:_MAX_CANDIDATE_LIMIT]
            if self.task14d_diagnostics:
                lexical_facts = _branch_candidate_facts(
                    active_lexical_hits, "lexical"
                )
            if self.retrieval_strategy == "lexical":
                combined_hits = active_lexical_hits
            elif active_lexical_hits:
                combined_hits = merge_ranked_hits(
                    active_vector_hits,
                    active_lexical_hits,
                    limit=self.retrieval_limit,
                    rrf_k=self.rrf_k,
                    vector_weight=self.vector_weight,
                    lexical_weight=self.lexical_weight,
                )
            else:
                combined_hits = active_vector_hits

        fusion_facts: tuple[FusionCandidateFact, ...] = ()
        fusion_drops: tuple[DedupDropFact, ...] = ()
        if self.task14d_diagnostics and self.retrieval_strategy == "hybrid":
            _, fusion_facts, fusion_drops = merge_ranked_hits_with_diagnostics(
                active_vector_hits,
                active_lexical_hits,
                limit=self.retrieval_limit,
                rrf_k=self.rrf_k,
                vector_weight=self.vector_weight,
                lexical_weight=self.lexical_weight,
            )
        if self.task14d_diagnostics:
            retained, post_fusion_drops = deduplicate_hits_with_diagnostics(
                combined_hits,
                post_fusion_dedup=self.post_fusion_dedup,
                limit=_MAX_CANDIDATE_LIMIT,
            )
            self._task14d_trace = Task14DRetrievalTrace(
                vector=vector_facts,
                lexical=lexical_facts,
                fusion=fusion_facts,
                drops=tuple(fusion_drops) + tuple(post_fusion_drops),
            )
        else:
            retained = _deduplicate_hits(combined_hits)[:_MAX_CANDIDATE_LIMIT]
        return CandidateSet(question=cleaned_question, hits=tuple(retained))

    async def _eligible_versions(
        self, audience_scope: AudienceScope, as_of: date
    ) -> tuple[set[str], str | None]:
        eligibility_method = getattr(self.repository, "eligible_version_ids", None)
        if callable(eligibility_method):
            result = eligibility_method(audience_scope, as_of)
            # A custom repository may make the boundary asynchronous; support
            # it without weakening the formal synchronous protocol.
            if hasattr(result, "__await__"):
                result = await result
            if not isinstance(result, EligibilityResult):
                raise TypeError("eligible_version_ids returned an invalid result")
            version_ids = result.version_ids
            if not isinstance(version_ids, frozenset):
                raise TypeError("eligible version ids are invalid")
            if not all(isinstance(version_id, str) and version_id for version_id in version_ids):
                raise TypeError("eligible version ids are invalid")
            return set(version_ids), result.exclusion_reason

        # Compatibility for old in-memory fakes and non-formal management
        # integrations.  The real SQLite repository always takes the branch
        # above once schema v5 is available.
        return set(self.repository.active_version_ids()), None

    async def rerank(
        self, question: str, candidates: CandidateSet
    ) -> RetrievalResult:
        """Rerank candidates and refuse when the strongest score is too low."""

        cleaned_question = clean_question(question)
        if cleaned_question != candidates.question:
            # The candidate set is request-scoped; do not silently rank a set
            # recalled for another question.
            raise ValueError("candidate question does not match request")
        ranked_candidates = (
            candidates.hits[:_MAX_CANDIDATE_LIMIT]
            if self.diagnostic_trace
            else candidates.hits
        )
        if not ranked_candidates:
            return RetrievalResult(
                status="refused",
                reason_code="INSUFFICIENT_EVIDENCE",
                diagnostic_trace=(
                    self._diagnostic_trace_for(
                        ranked_candidates,
                        rerank_scores={},
                        selected_ranks={},
                        all_candidate_scores_captured=True,
                    )
                    if self.diagnostic_trace
                    else None
                ),
            )

        documents = [hit.chunk.text for hit in ranked_candidates]
        top_n = (
            len(documents)
            if self.diagnostic_trace
            else min(self.rerank_limit, 6, len(documents))
        )
        ranked = await self.reranker.rerank(
            cleaned_question,
            documents,
            top_n,
        )
        if not ranked:
            return RetrievalResult(
                status="refused",
                reason_code="INSUFFICIENT_EVIDENCE",
                diagnostic_trace=(
                    self._diagnostic_trace_for(
                        ranked_candidates,
                        rerank_scores={},
                        selected_ranks={},
                        all_candidate_scores_captured=False,
                    )
                    if self.diagnostic_trace
                    else None
                ),
            )

        validated_ranked: list[tuple[float, int]] = []
        rerank_scores: dict[int, float] = {}
        invalid_score = False
        response_count = 0
        for item in ranked:
            response_count += 1
            index = getattr(item, "index", None)
            score = getattr(item, "score", None)
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index < len(ranked_candidates)
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
            ):
                invalid_score = True
                continue
            try:
                normalised_score = float(score)
            except (OverflowError, TypeError, ValueError):
                invalid_score = True
                continue
            if self.diagnostic_trace and not math.isfinite(normalised_score):
                invalid_score = True
                continue
            validated_ranked.append((normalised_score, index))
            if self.diagnostic_trace:
                rerank_scores[index] = normalised_score

        validated_ranked.sort(key=lambda value: value[0], reverse=True)
        evidence: list[Evidence] = []
        selected_ranks: dict[int, int] = {}
        for score, index in validated_ranked[:6]:
            selected_ranks.setdefault(index, len(evidence) + 1)
            evidence.append(
                Evidence(
                    reference_id=f"S{len(evidence) + 1}",
                    hit=ranked_candidates[index],
                    rerank_score=score,
                )
            )
        if not evidence or evidence[0].rerank_score < self.relevance_threshold:
            return RetrievalResult(
                status="refused",
                reason_code="INSUFFICIENT_EVIDENCE",
                diagnostic_trace=(
                    self._diagnostic_trace_for(
                        ranked_candidates,
                        rerank_scores=rerank_scores,
                        selected_ranks={},
                        all_candidate_scores_captured=(
                            response_count == len(ranked_candidates)
                            and len(rerank_scores) == len(ranked_candidates)
                            and not invalid_score
                        ),
                    )
                    if self.diagnostic_trace
                    else None
                ),
            )
        return RetrievalResult(
            status="ready",
            evidence=tuple(evidence),
            diagnostic_trace=(
                self._diagnostic_trace_for(
                    ranked_candidates,
                    rerank_scores=rerank_scores,
                    selected_ranks=selected_ranks,
                    all_candidate_scores_captured=(
                        response_count == len(ranked_candidates)
                        and len(rerank_scores) == len(ranked_candidates)
                        and not invalid_score
                    ),
                )
                if self.diagnostic_trace
                else None
            ),
        )

    def _diagnostic_trace_for(
        self,
        candidates: Sequence[SearchHit],
        *,
        rerank_scores: dict[int, float],
        selected_ranks: dict[int, int],
        all_candidate_scores_captured: bool,
    ) -> RetrievalDiagnosticTrace:
        records: list[RetrievalDiagnosticCandidate] = []
        for index, hit in enumerate(candidates):
            chunk_id = getattr(getattr(hit, "chunk", None), "chunk_id", "")
            source_id = chunk_id.strip() if isinstance(chunk_id, str) else ""
            raw_retrieval_score = getattr(hit, "score", None)
            retrieval_score: float | None
            if isinstance(raw_retrieval_score, bool) or not isinstance(
                raw_retrieval_score, (int, float)
            ):
                retrieval_score = None
            else:
                try:
                    value = float(raw_retrieval_score)
                except (OverflowError, TypeError, ValueError):
                    retrieval_score = None
                else:
                    retrieval_score = value if math.isfinite(value) else None
            candidate_index = index + 1
            records.append(
                RetrievalDiagnosticCandidate(
                    source_id=source_id,
                    candidate_rank=candidate_index,
                    retrieval_score=retrieval_score,
                    rerank_score=rerank_scores.get(index),
                    selected=index in selected_ranks,
                    selected_rank=selected_ranks.get(index),
                )
            )
        metadata = RetrievalDiagnosticMetadata(
            schema_version=_DIAGNOSTIC_TRACE_SCHEMA_VERSION,
            retrieval_strategy=self.retrieval_strategy,
            selector_name=_DIAGNOSTIC_SELECTOR_NAME,
            selector_version=_DIAGNOSTIC_SELECTOR_VERSION,
            retrieval_limit=self.retrieval_limit,
            rerank_limit=self.rerank_limit,
            relevance_threshold=self.relevance_threshold,
            rrf_k=self.rrf_k,
            all_candidate_scores_captured=all_candidate_scores_captured,
            fusion_profile=(
                self.fusion_profile
                if self.retrieval_strategy != "vector"
                else None
            ),
            vector_weight=(
                self.vector_weight if self.retrieval_strategy != "vector" else None
            ),
            lexical_weight=(
                self.lexical_weight if self.retrieval_strategy != "vector" else None
            ),
        )
        return RetrievalDiagnosticTrace(metadata=metadata, candidates=tuple(records))

    async def retrieve(
        self,
        question: str,
        audience_scope: AudienceScope = "unspecified",
        as_of: date | None = None,
    ) -> RetrievalResult:
        """Run candidate recall followed by reranking in order."""

        candidates = await self.retrieve_candidates(question, audience_scope, as_of)
        return await self.rerank(candidates.question, candidates)


def deduplicate_hits_with_diagnostics(
    hits: Sequence[SearchHit],
    *,
    post_fusion_dedup: bool,
    limit: int = _MAX_CANDIDATE_LIMIT,
) -> tuple[list[SearchHit], tuple[DedupDropFact, ...]]:
    """Apply the production post-RRF rules and explain every diagnostic drop."""

    if not isinstance(post_fusion_dedup, bool):
        raise TypeError("post_fusion_dedup must be a boolean")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    effective_limit = min(limit, _MAX_CANDIDATE_LIMIT)
    if not post_fusion_dedup:
        retained = list(hits[:effective_limit])
        drops = tuple(
            DedupDropFact(
                candidate_id=_diagnostic_candidate_id(hit),
                reason="top20_truncation",
                stage="post_rrf",
                source_id=getattr(hit.chunk, "chunk_id", None),
            )
            for hit in hits[effective_limit:]
        )
        return retained, drops

    retained: list[SearchHit] = []
    drops: list[DedupDropFact] = []
    hashes: set[str] = set()
    kept_by_hash: dict[str, str] = {}
    heading_counts: dict[tuple[str, tuple[str, ...]], int] = {}
    heading_first: dict[tuple[str, tuple[str, ...]], str] = {}
    normalised_retained: list[tuple[str, str]] = []
    for hit in hits:
        candidate_id = _diagnostic_candidate_id(hit)
        chunk = hit.chunk
        content_hash = str(chunk.content_hash)
        if content_hash in hashes:
            drops.append(
                DedupDropFact(
                    candidate_id=candidate_id,
                    reason="exact_duplicate",
                    stage="post_rrf",
                    kept_candidate_id=kept_by_hash.get(content_hash),
                    source_id=getattr(chunk, "chunk_id", None),
                )
            )
            continue
        normalised = _normalise_text(chunk.text)
        near_duplicate_id = next(
            (
                kept_id
                for kept_id, retained_text in normalised_retained
                if _is_duplicate_text(normalised, retained_text)
            ),
            None,
        )
        if near_duplicate_id is not None:
            drops.append(
                DedupDropFact(
                    candidate_id=candidate_id,
                    reason="near_duplicate",
                    stage="post_rrf",
                    kept_candidate_id=near_duplicate_id,
                    source_id=getattr(chunk, "chunk_id", None),
                )
            )
            continue
        heading_key = (chunk.source.file_name, tuple(chunk.source.heading_path))
        if heading_counts.get(heading_key, 0) >= 2:
            drops.append(
                DedupDropFact(
                    candidate_id=candidate_id,
                    reason="heading_cap",
                    stage="post_rrf",
                    kept_candidate_id=heading_first.get(heading_key),
                    source_id=getattr(chunk, "chunk_id", None),
                )
            )
            continue
        if len(retained) >= effective_limit:
            drops.append(
                DedupDropFact(
                    candidate_id=candidate_id,
                    reason="top20_truncation",
                    stage="post_rrf",
                    source_id=getattr(chunk, "chunk_id", None),
                )
            )
            continue
        hashes.add(content_hash)
        kept_by_hash[content_hash] = candidate_id
        heading_counts[heading_key] = heading_counts.get(heading_key, 0) + 1
        heading_first.setdefault(heading_key, candidate_id)
        normalised_retained.append((candidate_id, normalised))
        retained.append(hit)
    return retained, tuple(drops)


def _deduplicate_hits(hits: Sequence[SearchHit]) -> list[SearchHit]:
    """Keep high-scoring, non-overlapping chunks with heading diversity caps."""

    retained, _ = deduplicate_hits_with_diagnostics(
        hits,
        post_fusion_dedup=True,
        limit=len(hits) if hits else 1,
    )
    return retained


def _normalise_text(text: str) -> str:
    value = _WHITESPACE.sub(" ", str(text).casefold()).strip()
    return _NON_WORD.sub("", value)


def _is_duplicate_text(left: str, right: str) -> bool:
    if left == right:
        return True
    return SequenceMatcher(None, left, right).ratio() >= 0.85


__all__ = [
    "RetrievalFusionProfile",
    "RetrievalService",
    "clean_question",
    "deduplicate_hits_with_diagnostics",
    "merge_ranked_hits",
    "merge_ranked_hits_with_diagnostics",
    "rrf_weights_for_profile",
]
