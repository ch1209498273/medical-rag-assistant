"""Provider-independent helpers for diagnosing the retrieval layer.

The helpers in this module deliberately operate on safe evaluation facts and
on an in-memory local snapshot.  They never call a model provider and never
include document text in the returned summaries.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.evaluation.industry_metrics import (
    ndcg_at_k,
    rate_metric,
    reciprocal_rank,
)

_NO_ANSWER_TYPES = frozenset(
    {"no_answer", "no-answer", "no answer", "无答案", "无依据", "refusal", "unanswerable"}
)
_REQUIRED_TYPES = (
    "fact",
    "procedure",
    "condition",
    "prohibition",
    "synthesis",
    "no_answer",
)
_TOKEN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+|[A-Za-z0-9]+", re.UNICODE)


@dataclass(frozen=True)
class SourceAlignment:
    """Classify a question's expected source IDs against the active snapshot."""

    exact: tuple[str, ...] = ()
    alias: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()

    @property
    def expected_count(self) -> int:
        return len(self.exact) + len(self.alias) + len(self.missing)

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_count": self.expected_count,
            "exact_count": len(self.exact),
            "alias_count": len(self.alias),
            "missing_count": len(self.missing),
        }


def classify_expected_sources(
    expected_ids: Sequence[str],
    active_ids: Iterable[str],
    *,
    aliases: Mapping[str, str] | None = None,
) -> SourceAlignment:
    """Classify unique expected IDs as exact, safely aliased, or missing."""

    active = {value.strip() for value in active_ids if isinstance(value, str) and value.strip()}
    alias_map = {
        key.strip(): value.strip()
        for key, value in (aliases or {}).items()
        if isinstance(key, str)
        and key.strip()
        and isinstance(value, str)
        and value.strip()
    }
    seen: set[str] = set()
    exact: list[str] = []
    alias: list[str] = []
    missing: list[str] = []
    for value in expected_ids:
        if not isinstance(value, str):
            continue
        item = value.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        if item in active:
            exact.append(item)
        elif alias_map.get(item) in active:
            alias.append(item)
        else:
            missing.append(item)
    return SourceAlignment(tuple(exact), tuple(alias), tuple(missing))


def summarize_source_alignment(
    cases: Sequence[Mapping[str, Any]],
    active_ids: Iterable[str],
    *,
    aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return aggregate expected-source alignment facts without source IDs."""

    active = tuple(active_ids)
    total = exact = aliased = missing = cases_with_missing = 0
    for case in cases:
        source_ids = case.get("source_ids", ())
        if not isinstance(source_ids, Sequence) or isinstance(source_ids, (str, bytes)):
            source_ids = ()
        alignment = classify_expected_sources(source_ids, active, aliases=aliases)
        total += alignment.expected_count
        exact += len(alignment.exact)
        aliased += len(alignment.alias)
        missing += len(alignment.missing)
        cases_with_missing += bool(alignment.missing)
    return {
        "active_source_count": len(set(active)),
        "expected_source_count": total,
        "exact_source_count": exact,
        "alias_source_count": aliased,
        "missing_source_count": missing,
        "cases_with_missing_sources": cases_with_missing,
        "all_expected_sources_present": missing == 0,
    }


def inspect_chunk_integrity(chunks: Sequence[Any]) -> dict[str, int]:
    """Inspect bounded chunk metadata and return counts only."""

    chunk_ids: list[str] = []
    content_hashes: list[str] = []
    files: set[str] = set()
    blank_text = missing_file = missing_heading = 0
    for item in chunks:
        fields = _chunk_fields(item)
        chunk_id = fields["chunk_id"]
        content_hash = fields["content_hash"]
        text = fields["text"]
        file_name = fields["file_name"]
        heading = fields["heading_path"]
        if chunk_id:
            chunk_ids.append(chunk_id)
        if content_hash:
            content_hashes.append(content_hash)
        if not text:
            blank_text += 1
        if not file_name:
            missing_file += 1
        else:
            files.add(file_name)
        if not heading:
            missing_heading += 1
    return {
        "total_chunks": len(chunks),
        "unique_chunk_ids": len(set(chunk_ids)),
        "duplicate_chunk_id_count": len(chunk_ids) - len(set(chunk_ids)),
        "unique_content_hashes": len(set(content_hashes)),
        "duplicate_content_hash_count": len(content_hashes) - len(set(content_hashes)),
        "blank_text_count": blank_text,
        "missing_file_name_count": missing_file,
        "missing_heading_count": missing_heading,
        "active_file_count": len(files),
    }


def _chunk_fields(item: Any) -> dict[str, Any]:
    value = item
    if isinstance(value, Mapping) and isinstance(value.get("chunk"), Mapping):
        value = value["chunk"]
    elif not isinstance(value, Mapping) and hasattr(value, "chunk"):
        value = value.chunk
    if isinstance(value, Mapping):
        chunk_id = value.get("chunk_id", "")
        text = value.get("text", value.get("content", ""))
        content_hash = value.get("content_hash", "")
        source = value.get("source")
        file_name = value.get("file_name", "")
        heading = value.get("heading_path", ())
        if isinstance(source, Mapping):
            file_name = source.get("file_name", file_name)
            heading = source.get("heading_path", heading)
    else:
        chunk_id = getattr(value, "chunk_id", "")
        text = getattr(value, "text", "")
        content_hash = getattr(value, "content_hash", "")
        source = getattr(value, "source", None)
        file_name = getattr(source, "file_name", "")
        heading = getattr(source, "heading_path", ())
    if not isinstance(chunk_id, str):
        chunk_id = ""
    if not isinstance(text, str):
        text = ""
    if not isinstance(content_hash, str):
        content_hash = ""
    if not isinstance(file_name, str):
        file_name = ""
    if not isinstance(heading, (list, tuple)):
        heading = ()
    heading_values = tuple(str(value).strip() for value in heading if str(value).strip())
    return {
        "chunk_id": chunk_id.strip(),
        "text": text.strip(),
        "content_hash": content_hash.strip(),
        "file_name": file_name.strip(),
        "heading_path": heading_values,
    }


@dataclass(frozen=True)
class LexicalHit:
    """A local lexical hit; text is intentionally not retained."""

    chunk_id: str
    score: float


class LexicalIndex:
    """Deterministic, lightweight lexical comparator for local diagnosis."""

    def __init__(self, chunks: Sequence[Any]) -> None:
        records: list[tuple[str, frozenset[str]]] = []
        document_frequency: Counter[str] = Counter()
        for item in chunks:
            fields = _chunk_fields(item)
            if not fields["chunk_id"] or not fields["text"]:
                continue
            tokens = frozenset(_tokenize(fields["text"]))
            if not tokens:
                continue
            records.append((fields["chunk_id"], tokens))
            document_frequency.update(tokens)
        self._records = tuple(records)
        total = len(records)
        self._idf = {
            token: math.log((total + 1) / (count + 1)) + 1.0
            for token, count in document_frequency.items()
        }

    def search(self, question: str, *, limit: int = 20) -> tuple[LexicalHit, ...]:
        if not isinstance(question, str) or not question.strip():
            return ()
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        query_tokens = set(_tokenize(question))
        if not query_tokens:
            return ()
        scored: list[LexicalHit] = []
        for chunk_id, tokens in self._records:
            score = sum(self._idf.get(token, 1.0) for token in query_tokens & tokens)
            if score > 0.0:
                scored.append(LexicalHit(chunk_id, score))
        scored.sort(key=lambda hit: (-hit.score, hit.chunk_id))
        return tuple(scored[:limit])


def lexical_rank(question: str, chunks: Sequence[Any], *, limit: int = 20) -> tuple[str, ...]:
    """Return deterministic local lexical IDs for a single question."""

    return tuple(hit.chunk_id for hit in LexicalIndex(chunks).search(question, limit=limit))


def _tokenize(value: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(value.casefold()):
        group = match.group(0)
        if not group:
            continue
        if re.fullmatch(r"[A-Za-z0-9]+", group):
            tokens.append(group)
            continue
        tokens.extend(group)
        if len(group) > 1:
            tokens.extend(group[index : index + 2] for index in range(len(group) - 1))
    return tuple(dict.fromkeys(tokens))


def summarize_c1_by_type(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Summarize C1 outcomes by type without copying questions or IDs."""

    raw_results = report.get("results", ())
    if not isinstance(raw_results, Sequence) or isinstance(raw_results, (str, bytes)):
        raise TypeError("report results are invalid")
    groups: dict[str, list[Mapping[str, Any]]] = {key: [] for key in _REQUIRED_TYPES}
    for item in raw_results:
        if not isinstance(item, Mapping):
            continue
        case_type = _canonical_type(item.get("type"))
        if case_type in groups:
            groups[case_type].append(item)
    return {case_type: _summarize_outcomes(items) for case_type, items in groups.items()}


def _summarize_outcomes(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    answerable = [item for item in items if not bool(item.get("no_answer"))]
    no_answer = [item for item in items if bool(item.get("no_answer"))]
    recall20_hits = recall20_total = recall6_hits = recall6_total = 0
    precision6_hits = precision6_total = 0
    mrr_values: list[float] = []
    ndcg_values: list[float] = []
    for item in answerable:
        expected = _string_ids(item.get("expected_source_ids"))
        candidates = _string_ids(item.get("candidate_source_ids"))
        evidence = _string_ids(item.get("evidence_source_ids"))
        expected_set = set(expected)
        candidate20 = tuple(dict.fromkeys(candidates))[:20]
        evidence6 = tuple(dict.fromkeys(evidence))[:6]
        recall20_hits += len(set(candidate20) & expected_set)
        recall20_total += len(expected_set)
        recall6_hits += len(set(evidence6) & expected_set)
        recall6_total += len(expected_set)
        precision6_hits += len(set(evidence6) & expected_set)
        precision6_total += len(evidence6)
        mrr_values.append(reciprocal_rank(candidate20, expected))
        if expected_set:
            ndcg_values.append(ndcg_at_k(evidence6, {source_id: 1.0 for source_id in expected_set}, 6))
    return {
        "case_count": len(items),
        "answerable_count": len(answerable),
        "no_answer_count": len(no_answer),
        "answer_availability": rate_metric(sum(bool(item.get("answered")) for item in items), len(items)).to_dict(),
        "answerable_over_refusal": rate_metric(sum(bool(item.get("refused")) for item in answerable), len(answerable)).to_dict(),
        "grounded_answer_success": rate_metric(
            sum(
                bool(item.get("answered"))
                and bool(item.get("reference_points_covered"))
                and bool(item.get("citation_valid"))
                for item in answerable
            ),
            len(answerable),
        ).to_dict(),
        "recall_at_20": rate_metric(recall20_hits, recall20_total).to_dict(),
        "recall_at_6": rate_metric(recall6_hits, recall6_total).to_dict(),
        "precision_at_6": rate_metric(precision6_hits, precision6_total).to_dict(),
        "mrr_at_20": sum(mrr_values) / len(mrr_values) if mrr_values else 0.0,
        "mrr_denominator": len(mrr_values),
        "ndcg_at_6": sum(ndcg_values) / len(ndcg_values) if ndcg_values else None,
        "ndcg_denominator": len(ndcg_values),
        "no_answer_refusal": rate_metric(sum(bool(item.get("refused")) for item in no_answer), len(no_answer)).to_dict(),
        "negative_evidence_rate": rate_metric(
            sum(_positive_int(item.get("evidence_count")) > 0 for item in no_answer), len(no_answer)
        ).to_dict(),
    }


def summarize_lexical_metrics(
    cases: Sequence[Mapping[str, Any]],
    chunks: Sequence[Any],
    *,
    retrieval_k: int = 20,
    evidence_k: int = 6,
) -> dict[str, Any]:
    """Calculate the local lexical comparator using the same source labels."""

    index = LexicalIndex(chunks)
    groups: dict[str, list[Mapping[str, Any]]] = {key: [] for key in _REQUIRED_TYPES}
    for case in cases:
        case_type = _canonical_type(case.get("type"))
        if case_type in groups:
            groups[case_type].append(case)
    summaries = {
        case_type: _summarize_ranked_cases(items, index, retrieval_k, evidence_k)
        for case_type, items in groups.items()
    }
    summaries["overall"] = _summarize_ranked_cases(cases, index, retrieval_k, evidence_k)
    return summaries


def _summarize_ranked_cases(
    cases: Sequence[Mapping[str, Any]],
    index: LexicalIndex,
    retrieval_k: int,
    evidence_k: int,
) -> dict[str, Any]:
    answerable = [case for case in cases if not _is_no_answer(case.get("type"))]
    no_answer = [case for case in cases if _is_no_answer(case.get("type"))]
    recall20_hits = recall20_total = recall6_hits = recall6_total = 0
    precision6_hits = precision6_total = 0
    mrr_values: list[float] = []
    ndcg_values: list[float] = []
    for case in answerable:
        expected = _string_ids(case.get("source_ids"))
        hits = index.search(str(case.get("question", "")), limit=retrieval_k)
        candidate_ids = tuple(hit.chunk_id for hit in hits)
        evidence_ids = candidate_ids[:evidence_k]
        expected_set = set(expected)
        recall20_hits += len(set(candidate_ids[:20]) & expected_set)
        recall20_total += len(expected_set)
        recall6_hits += len(set(evidence_ids) & expected_set)
        recall6_total += len(expected_set)
        precision6_hits += len(set(evidence_ids) & expected_set)
        precision6_total += len(evidence_ids)
        mrr_values.append(reciprocal_rank(candidate_ids, expected))
        if expected_set:
            ndcg_values.append(ndcg_at_k(evidence_ids, {source_id: 1.0 for source_id in expected_set}, evidence_k))
    contamination = []
    for case in no_answer:
        hits = index.search(str(case.get("question", "")), limit=evidence_k)
        contamination.append(any(hit.score > 0.0 for hit in hits))
    return {
        "case_count": len(cases),
        "answerable_count": len(answerable),
        "no_answer_count": len(no_answer),
        "recall_at_20": rate_metric(recall20_hits, recall20_total).to_dict(),
        "recall_at_6": rate_metric(recall6_hits, recall6_total).to_dict(),
        "precision_at_6": rate_metric(precision6_hits, precision6_total).to_dict(),
        "mrr_at_20": sum(mrr_values) / len(mrr_values) if mrr_values else 0.0,
        "mrr_denominator": len(mrr_values),
        "ndcg_at_6": sum(ndcg_values) / len(ndcg_values) if ndcg_values else None,
        "ndcg_denominator": len(ndcg_values),
        "negative_evidence_rate": rate_metric(sum(contamination), len(contamination)).to_dict(),
    }


def _canonical_type(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    key = value.strip().casefold()
    aliases = {
        "事实": "fact",
        "事实查询": "fact",
        "流程": "procedure",
        "流程步骤": "procedure",
        "适用条件": "condition",
        "禁止": "prohibition",
        "禁止事项": "prohibition",
        "跨段归纳": "synthesis",
        "cross_section": "synthesis",
        "无答案": "no_answer",
        "无依据": "no_answer",
        "no-answer": "no_answer",
        "no answer": "no_answer",
        "refusal": "no_answer",
        "unanswerable": "no_answer",
    }
    return aliases.get(key, key)


def _is_no_answer(value: Any) -> bool:
    return _canonical_type(value) in _NO_ANSWER_TYPES or _canonical_type(value) == "no_answer"


def _string_ids(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(dict.fromkeys(item.strip() for item in value if isinstance(item, str) and item.strip()))


def _positive_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)


__all__ = [
    "LexicalHit",
    "LexicalIndex",
    "SourceAlignment",
    "classify_expected_sources",
    "inspect_chunk_integrity",
    "lexical_rank",
    "summarize_c1_by_type",
    "summarize_lexical_metrics",
    "summarize_source_alignment",
]
