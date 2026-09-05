"""Deterministic, request-local query validation and retrieval variants."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_WHITESPACE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,.;:!?，。！？；：、）》】』」’”])")

# Keep this table deliberately small and ordered.  It is a retrieval aid, not
# a semantic rewrite, and changing it should be an explicit experiment.
QUERY_ALIAS_VERSION = "v1"
QUERY_ALIASES: tuple[tuple[str, str], ...] = (
    ("血透", "血液透析"),
    ("HD", "血液透析"),
    ("HDF", "血液透析滤过"),
    ("CRRT", "连续性肾脏替代治疗"),
    ("AVF", "动静脉内瘘"),
    ("CVC", "中心静脉导管"),
)


@dataclass(frozen=True)
class QueryVariants:
    """The original question and deterministic lexical retrieval queries."""

    original: str
    normalized: str
    lexical_queries: tuple[str, ...]
    aliases_applied: tuple[str, ...]


def clean_question(question: str) -> str:
    """Remove legacy controls and enforce the existing question boundary."""

    if not isinstance(question, str):
        raise TypeError("question must be a string")
    cleaned = _CONTROL_CHARS.sub("", question).strip()
    if not cleaned:
        raise ValueError("question must not be empty")
    if len(cleaned) > 1000:
        raise ValueError("question must not exceed 1000 characters")
    return cleaned


def build_query_variants(question: str, *, use_aliases: bool) -> QueryVariants:
    """Build stable local query variants without changing the user question."""

    original = clean_question(question)
    normalized = _normalize_query(original)
    if not use_aliases:
        return QueryVariants(
            original=original,
            normalized=normalized,
            lexical_queries=(original,),
            aliases_applied=(),
        )

    aliased = normalized
    aliases_applied: list[str] = []
    for alias, canonical in QUERY_ALIASES:
        replacement = _replace_alias(aliased, alias, canonical)
        if replacement != aliased:
            aliases_applied.append(f"{alias}→{canonical}")
            aliased = replacement

    lexical_queries = _stable_unique(
        (original, normalized, aliased)
    )
    return QueryVariants(
        original=original,
        normalized=normalized,
        lexical_queries=lexical_queries,
        aliases_applied=tuple(aliases_applied),
    )


def _normalize_query(value: str) -> str:
    """Normalize width, punctuation, and runs of whitespace for lookup only."""

    normalized = unicodedata.normalize("NFKC", value)
    normalized = _WHITESPACE.sub(" ", normalized).strip()
    return _SPACE_BEFORE_PUNCTUATION.sub(r"\1", normalized)


def _replace_alias(value: str, alias: str, canonical: str) -> str:
    if alias.isascii():
        pattern = re.compile(
            rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])",
            re.IGNORECASE,
        )
        return pattern.sub(canonical, value)
    return value.replace(alias, canonical)


def _stable_unique(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return tuple(unique)


__all__ = [
    "QUERY_ALIASES",
    "QUERY_ALIAS_VERSION",
    "QueryVariants",
    "build_query_variants",
    "clean_question",
]
