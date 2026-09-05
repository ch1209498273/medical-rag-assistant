"""Deterministic, network-free providers for the public demo and tests."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from app.providers.siliconflow import RankedItem

ScenarioType = Literal[
    "fact",
    "procedure",
    "condition",
    "prohibition",
    "synthesis",
    "no_answer",
    "followup",
]

_SCENARIO_TYPES = frozenset(
    {
        "fact",
        "procedure",
        "condition",
        "prohibition",
        "synthesis",
        "no_answer",
        "followup",
    }
)
_TOP_LEVEL_FIELDS = frozenset({"schema_version", "scenarios"})
_SCENARIO_FIELDS = frozenset(
    {
        "scenario_id",
        "type",
        "question",
        "standalone_question",
        "answer",
        "refused",
        "evidence_terms",
    }
)
_QUESTION_TAG_RE = re.compile(
    r"<current-question>(?P<question>.*?)</current-question>\s*\Z", re.DOTALL
)
_ANSWER_QUESTION_RE = re.compile(r"\A\s*问题：(?P<question>[^\r\n]*)")
_REWRITER_PROMPT_RE = re.compile(
    r"\A.*?<conversation-history>.*?</conversation-history>\s*"
    r"<current-question>(?P<question>.*?)</current-question>\s*\Z",
    re.DOTALL,
)
_TOKENS_RE = re.compile(r"[\u4e00-\u9fff]|[a-z0-9]+")
_WHITESPACE_RE = re.compile(r"\s+")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


@dataclass(frozen=True)
class DemoScenario:
    """One exact-match question and its deterministic demo answer."""

    scenario_id: str
    type: ScenarioType
    question: str
    standalone_question: str
    answer: str
    refused: bool
    evidence_terms: tuple[str, ...]


def load_demo_scenarios(path: str | Path) -> tuple[DemoScenario, ...]:
    """Load and validate the version-one scenario catalog from JSON."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise ValueError("demo scenario file is invalid") from error
    if not isinstance(payload, Mapping):
        raise TypeError("demo scenario document must be an object")
    if set(payload) != _TOP_LEVEL_FIELDS:
        raise ValueError("demo scenario document fields are invalid")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError("demo scenario schema_version must be 1")
    raw_scenarios = payload.get("scenarios")
    if not isinstance(raw_scenarios, list):
        raise TypeError("demo scenarios must be a list")

    scenarios: list[DemoScenario] = []
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    for raw in raw_scenarios:
        scenario = _parse_scenario(raw)
        normalized_questions = {
            _normalize_question(scenario.question),
            _normalize_question(scenario.standalone_question),
        }
        if scenario.scenario_id in seen_ids:
            raise ValueError("demo scenario_id values must be unique")
        if seen_questions.intersection(normalized_questions):
            raise ValueError("demo scenario questions must be unique")
        seen_ids.add(scenario.scenario_id)
        seen_questions.update(normalized_questions)
        scenarios.append(scenario)
    return tuple(scenarios)


class DemoKnowledgeProvider:
    """Use stable hash vectors and lexical cosine scores without network I/O."""

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        values = list(texts)
        if any(not isinstance(text, str) for text in values):
            raise ValueError("embedding inputs must be strings")
        return [_vector(text) for text in values]

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int,
    ) -> list[RankedItem]:
        values = list(documents)
        if not isinstance(query, str) or any(
            not isinstance(document, str) for document in values
        ):
            raise ValueError("rerank query and documents must be strings")
        if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n <= 0:
            raise ValueError("top_n must be a positive integer")
        if not values:
            return []

        query_vector = _vector(query)
        scored = [
            (index, _dot(query_vector, _vector(document)))
            for index, document in enumerate(values)
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [
            RankedItem(index=index, document=values[index], score=score)
            for index, score in scored[: min(top_n, len(values))]
        ]

    async def ocr_page(self, image_bytes: bytes) -> str:
        """Return no OCR text; the offline provider never pretends to read pixels."""

        if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
            raise ValueError("image_bytes must be non-empty bytes")
        return ""

    async def aclose(self) -> None:
        """The offline provider owns no resources."""


class DemoLanguageModel:
    """Answer only exact catalog questions and fail closed for all others."""

    def __init__(self, scenarios: Sequence[DemoScenario]) -> None:
        values = tuple(scenarios)
        if any(not isinstance(scenario, DemoScenario) for scenario in values):
            raise ValueError("demo scenarios must contain DemoScenario values")
        by_question: dict[str, DemoScenario] = {}
        for scenario in values:
            if not isinstance(scenario.answer, str) or not isinstance(
                scenario.refused, bool
            ):
                raise TypeError("demo scenario answer contract is invalid")
            if scenario.refused is False and not scenario.answer.strip():
                raise ValueError("non-refused demo scenario answer must be non-empty")
            if scenario.refused is True and scenario.answer.strip():
                raise ValueError("refused demo scenario answer must be empty")
            for question in (scenario.question, scenario.standalone_question):
                normalized = _normalize_question(question)
                if normalized in by_question and by_question[normalized] != scenario:
                    raise ValueError("demo scenario questions must be unique")
                by_question[normalized] = scenario
        self._scenarios = values
        self._by_question = by_question

    async def complete_json(
        self, messages: Sequence[Mapping[str, Any]]
    ) -> dict[str, str]:
        """Return the one structured shape requested by supported app prompts."""

        system_content = _message_content(messages, role="system")
        if _prompt_kind(system_content) == "rewriter":
            question = _extract_rewriter_question(messages)
            if question is None:
                raise ValueError("demo rewriter prompt is invalid")
            scenario = self._by_question.get(_normalize_question(question))
            standalone = scenario.standalone_question if scenario else question
            return {"standalone_question": standalone}
        if _prompt_kind(system_content) == "reference":
            return {"reference_answer": ""}
        raise ValueError("unsupported demo structured prompt")

    async def aclose(self) -> None:
        """The offline provider owns no resources."""

    def stream_answer(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        json_output: bool = False,
        max_tokens: int | None = None,
        thinking: Mapping[str, str] | None = None,
    ) -> AsyncIterator[str]:
        """Yield one formal two-field JSON answer for an exact scenario."""

        return self._stream_answer(messages)

    async def _stream_answer(
        self, messages: Sequence[Mapping[str, Any]]
    ) -> AsyncIterator[str]:
        try:
            system_content = _message_content(messages, role="system")
            question = (
                _extract_answer_question(messages)
                if _prompt_kind(system_content) == "answer"
                else None
            )
            scenario = (
                self._by_question.get(_normalize_question(question))
                if question is not None
                else None
            )
        except (TypeError, ValueError):
            scenario = None
        if scenario is None or scenario.refused:
            payload = {"refused": True, "answer": ""}
        else:
            payload = {"refused": False, "answer": scenario.answer}
        yield json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _parse_scenario(raw: object) -> DemoScenario:
    if not isinstance(raw, Mapping):
        raise TypeError("demo scenario must be an object")
    if set(raw) != _SCENARIO_FIELDS:
        raise ValueError("demo scenario fields are invalid")
    scenario_id = raw.get("scenario_id")
    scenario_type = raw.get("type")
    question = raw.get("question")
    standalone_question = raw.get("standalone_question")
    answer = raw.get("answer")
    refused = raw.get("refused")
    evidence_terms = raw.get("evidence_terms")
    if not isinstance(scenario_id, str) or not scenario_id.strip():
        raise ValueError("demo scenario_id must be a non-empty string")
    if not isinstance(scenario_type, str) or scenario_type not in _SCENARIO_TYPES:
        raise ValueError("demo scenario type is invalid")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("demo scenario question must be a non-empty string")
    if not isinstance(standalone_question, str) or not standalone_question.strip():
        raise ValueError(
            "demo scenario standalone_question must be a non-empty string"
        )
    if not isinstance(answer, str):
        raise TypeError("demo scenario answer must be a string")
    if not isinstance(refused, bool):
        raise TypeError("demo scenario refused must be a boolean")
    if not isinstance(evidence_terms, list) or not evidence_terms:
        raise ValueError("demo scenario evidence_terms must be strings")
    normalized_terms: list[str] = []
    seen_terms: set[str] = set()
    for term in evidence_terms:
        if not isinstance(term, str) or not term.strip():
            raise ValueError("demo scenario evidence_terms must be strings")
        normalized_term = term.strip()
        term_key = normalized_term.casefold()
        if term_key in seen_terms:
            raise ValueError("demo scenario evidence_terms must be unique")
        seen_terms.add(term_key)
        normalized_terms.append(normalized_term)
    normalized_answer = answer.strip()
    if refused is False and not normalized_answer:
        raise ValueError("non-refused demo scenario answer must be non-empty")
    if refused is True and normalized_answer:
        raise ValueError("refused demo scenario answer must be empty")
    return DemoScenario(
        scenario_id=scenario_id.strip(),
        type=scenario_type,
        question=question.strip(),
        standalone_question=standalone_question.strip(),
        answer=normalized_answer,
        refused=refused,
        evidence_terms=tuple(normalized_terms),
    )


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_TOKENS_RE.findall(text.casefold()))


def _vector(text: str) -> list[float]:
    values = [0.0] * 1024
    for token in _tokens(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        values[int.from_bytes(digest[:4], "big") % 1024] += 1.0
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _normalize_question(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("demo question must be a string")
    if _CONTROL_RE.search(value):
        raise ValueError("demo question contains control characters")
    return _WHITESPACE_RE.sub(" ", value.casefold()).strip()


def _message_content(
    messages: Sequence[Mapping[str, Any]], *, role: str
) -> str:
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise TypeError("demo messages must be a sequence")
    if len(messages) != 2:
        raise ValueError("demo messages must contain one system and one user message")
    if [message.get("role") for message in messages if isinstance(message, Mapping)] != [
        "system",
        "user",
    ]:
        raise ValueError("demo messages must contain one system and one user message")
    if any(not isinstance(message, Mapping) for message in messages):
        raise TypeError("demo messages must contain mappings")
    content = messages[0 if role == "system" else 1].get("content")
    if not isinstance(content, str):
        raise TypeError("demo message content must be a string")
    if role not in {"system", "user"}:
        raise ValueError("demo message role is invalid")
    return content


def _prompt_kind(system_content: str) -> Literal["answer", "rewriter", "reference", "unknown"]:
    markers = {
        "answer": "refused" in system_content and "answer" in system_content,
        "rewriter": "standalone_question" in system_content,
        "reference": "reference_answer" in system_content,
    }
    if sum(markers.values()) != 1:
        return "unknown"
    return next(kind for kind, present in markers.items() if present)


def _extract_answer_question(
    messages: Sequence[Mapping[str, Any]],
) -> str | None:
    user_content = _message_content(messages, role="user")
    if _QUESTION_TAG_RE.search(user_content):
        return None
    answer_question = _ANSWER_QUESTION_RE.match(user_content)
    if answer_question:
        return answer_question.group("question").strip()
    return None


def _extract_rewriter_question(
    messages: Sequence[Mapping[str, Any]],
) -> str | None:
    user_content = _message_content(messages, role="user")
    if any(
        user_content.count(marker) != 1
        for marker in (
            "<conversation-history>",
            "</conversation-history>",
            "<current-question>",
            "</current-question>",
        )
    ):
        return None
    tagged = _REWRITER_PROMPT_RE.match(user_content)
    if tagged and tagged.group("question").strip():
        return tagged.group("question").strip()
    return None


__all__ = [
    "DemoKnowledgeProvider",
    "DemoLanguageModel",
    "DemoScenario",
    "load_demo_scenarios",
]
