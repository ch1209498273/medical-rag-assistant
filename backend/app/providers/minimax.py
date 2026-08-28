"""Optional MiniMax evaluation generator and DeepSeek fallback."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from app.errors import ProviderError
from app.providers.deepseek import DeepSeekClient
from app.providers.http import ProviderHttpClient
from app.settings import is_configured_key

MINIMAX_EVALUATION_TIMEOUT_SECONDS = 180.0
MINIMAX_EVALUATION_TEMPERATURE = 0.25
DEEPSEEK_EVALUATION_TIMEOUT_SECONDS = 180.0
DEEPSEEK_EVALUATION_MAX_OUTPUT_TOKENS = 8192
DEEPSEEK_EVALUATION_INSTRUCTIONS = (
    "Return only a JSON object that conforms to the supplied evaluation schema. "
    "Do not include explanations or fields outside that schema."
)


def _evaluation_quota_instruction(context: Any) -> str:
    """Return the fixed type plan without constraining other requests."""

    if not isinstance(context, Mapping) or context.get("count") != 30:
        return ""
    return (
        " 当 context.count == 30 时，必须严格按以下固定顺序生成 cases 数组，共 30 题，"
        "每类恰好 5 题：1–5 题 type=fact；6–10 题 type=procedure；"
        "11–15 题 type=condition；16–20 题 type=prohibition；"
        "21–25 题 type=synthesis；26–30 题 type=no_answer。"
        "cases 数组必须严格按此顺序返回，不得重排、遗漏或添加题目。"
    )

_EVALUATION_CASES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "cases": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "minLength": 1,
                        "pattern": r"\S",
                    },
                    "reference_points": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "string",
                            "minLength": 1,
                            "pattern": r"\S",
                        },
                    },
                    "source_ids": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "string",
                            "minLength": 1,
                            "pattern": r"\S",
                        },
                    },
                    "type": {
                        "type": "string",
                        "enum": [
                            "fact",
                            "procedure",
                            "condition",
                            "prohibition",
                            "synthesis",
                            "no_answer",
                        ],
                    },
                },
                "required": [
                    "question",
                    "reference_points",
                    "source_ids",
                    "type",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["cases"],
    "additionalProperties": False,
}

EVALUATION_TYPE_ALIASES = {
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


@dataclass
class EvaluationCase:
    """One candidate question and the source evidence expected in its answer."""

    question: str
    reference_points: list[str]
    source_ids: list[str]
    type: str


@dataclass(frozen=True)
class EvaluationGeneration:
    """Provider-owned metadata and immutable candidate cases.

    The generation adapter, rather than model output, supplies ``provider``
    and ``model_id``.  ``cases`` is normalised to a tuple so a provider cannot
    mutate a generation after it has been validated or persisted.  A few
    sequence methods intentionally keep compatibility with the Task 4 list
    return shape while callers migrate to the richer value object.
    """

    cases: tuple[EvaluationCase, ...]
    provider: str
    model_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases", tuple(self.cases))
        if not self.provider.strip() or not self.model_id.strip():
            raise ValueError("evaluation generation metadata is required")

    def __iter__(self):
        return iter(self.cases)

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index):
        return self.cases[index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cases": [
                {
                    "question": case.question,
                    "reference_points": list(case.reference_points),
                    "source_ids": list(case.source_ids),
                    "type": case.type,
                }
                for case in self.cases
            ],
            "provider": self.provider,
            "model_id": self.model_id,
        }

    def __eq__(self, other: object) -> bool:
        if isinstance(other, EvaluationGeneration):
            return (
                self.cases == other.cases
                and self.provider == other.provider
                and self.model_id == other.model_id
            )
        if isinstance(other, (list, tuple)):
            return self.cases == tuple(other)
        return NotImplemented


class EvaluationGenerator(Protocol):
    async def generate(self, context: Any) -> EvaluationGeneration: ...


class MiniMaxClient(ProviderHttpClient):
    """OpenAI-compatible MiniMax client used only for offline evaluation setup."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.minimaxi.com/v1",
        *,
        model: str = "MiniMax-M2.7",
        transport: Any = None,
        sleep: Any = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        super().__init__(
            "minimax",
            api_key,
            base_url,
            transport=transport,
            sleep=sleep,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.model = model

    async def complete_json(self, context: Any) -> Any:
        if isinstance(context, str):
            context_text = context
        else:
            try:
                context_text = json.dumps(context, ensure_ascii=False)
            except (TypeError, ValueError) as error:
                raise ProviderError(
                    "minimax", "generate", 200, False, "evaluation context is not serializable"
                ) from error
        response = await self.request_json(
            "generate",
            "/chat/completions",
            {
                "model": self.model,
                "stream": False,
                "thinking": {"type": "disabled"},
                "max_completion_tokens": 8192,
                "temperature": MINIMAX_EVALUATION_TEMPERATURE,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "只输出 JSON 对象，格式必须为 {\"cases\":[...]}。每题必须包含 question、reference_points、"
                            "source_ids、type 四个字段，不要添加解释。type 只能是 fact、"
                            "procedure、condition、prohibition、synthesis 或 no_answer；"
                            "覆盖事实查询、流程步骤、适用条件、禁止事项、跨段归纳和无答案六类。"
                            f"{_evaluation_quota_instruction(context)}"
                        ),
                    },
                    {"role": "user", "content": context_text},
                ],
            },
        )
        content = _extract_message_content(response)
        if isinstance(content, (list, dict)):
            return content
        if not isinstance(content, str) or not content.strip():
            raise ProviderError(
                "minimax", "generate", 200, False, "MiniMax response did not contain structured text"
            )
        try:
            return json.loads(content)
        except (TypeError, ValueError) as error:
            raise ProviderError(
                "minimax", "generate", 200, False, "MiniMax response was not valid structured JSON"
            ) from error


class MiniMaxEvaluationGenerator:
    """Strictly validate MiniMax-produced cases before they enter evaluation data."""

    provider = "minimax"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.minimaxi.com/v1",
        *,
        model: str = "MiniMax-M2.7",
        client: MiniMaxClient | None = None,
        transport: Any = None,
        sleep: Any = None,
        timeout: float = MINIMAX_EVALUATION_TIMEOUT_SECONDS,
    ) -> None:
        self.client = client or MiniMaxClient(
            api_key or "",
            base_url,
            model=model,
            transport=transport,
            sleep=sleep,
            timeout=timeout,
            max_retries=0,
        )
        self.model_id = getattr(self.client, "model", model)

    async def generate(self, context: Any) -> EvaluationGeneration:
        payload = await self.client.complete_json(context)
        return EvaluationGeneration(
            tuple(parse_evaluation_cases(payload, provider="minimax")),
            provider="minimax",
            model_id=self.model_id,
        )


class DeepSeekResponsesEvaluationClient(ProviderHttpClient):
    """DeepSeek Responses client used only for offline evaluation fallback."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        *,
        model: str = "deepseek-v4-flash",
        transport: Any = None,
        sleep: Any = None,
        timeout: float = DEEPSEEK_EVALUATION_TIMEOUT_SECONDS,
        max_retries: int = 0,
    ) -> None:
        super().__init__(
            "deepseek",
            api_key,
            base_url,
            transport=transport,
            sleep=sleep,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.model = model
        self.model_id = model

    async def complete_json(self, context: Any) -> Any:
        if isinstance(context, str):
            context_text = context
        else:
            try:
                context_text = json.dumps(context, ensure_ascii=False)
            except (OverflowError, RecursionError, TypeError, ValueError) as error:
                raise ProviderError(
                    "deepseek",
                    "generate",
                    None,
                    False,
                    "evaluation context is not serializable",
                ) from error

        response = await self.request_json(
            "generate",
            "/responses",
            {
                "model": self.model,
                "input": context_text,
                "instructions": (
                    DEEPSEEK_EVALUATION_INSTRUCTIONS
                    + _evaluation_quota_instruction(context)
                ),
                "stream": False,
                "reasoning": {"effort": "none"},
                "max_output_tokens": DEEPSEEK_EVALUATION_MAX_OUTPUT_TOKENS,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "evaluation_cases",
                        "schema": _EVALUATION_CASES_SCHEMA,
                    }
                },
            },
        )
        return _extract_deepseek_responses_json(response)


class DeepSeekResponsesEvaluationGenerator:
    """Validate DeepSeek Responses evaluation output with the canonical parser."""

    provider = "deepseek"

    def __init__(self, client: Any) -> None:
        self.client = client
        self.model_id = (
            getattr(client, "model_id", None)
            or getattr(client, "model", "deepseek-v4-flash")
        )

    async def generate(self, context: Any) -> EvaluationGeneration:
        payload = await self.client.complete_json(context)
        return EvaluationGeneration(
            tuple(parse_evaluation_cases(payload, provider="deepseek")),
            provider="deepseek",
            model_id=self.model_id,
        )


class DeepSeekEvaluationGenerator:
    """Use the formal-answer provider for evaluation generation when MiniMax is absent."""

    def __init__(self, client: DeepSeekClient) -> None:
        self.client = client
        self.provider = "deepseek"
        self.model_id = getattr(client, "model", "deepseek-v4-flash")

    async def generate(self, context: Any) -> EvaluationGeneration:
        if isinstance(context, str):
            context_text = context
        else:
            context_text = json.dumps(context, ensure_ascii=False)
        messages = [
            {
                "role": "system",
                "content": (
                    "只输出 JSON 对象，且顶层只能有 cases 字段，格式必须为 {\"cases\":[...]}。"
                    "每题必须且只能包含 question、reference_points、source_ids、type 四个字段，"
                    "不要添加解释。type 只能是 fact、"
                    "procedure、condition、prohibition、synthesis 或 no_answer；"
                    "覆盖事实查询、流程步骤、适用条件、禁止事项、跨段归纳和无答案六类。"
                ),
            },
            {"role": "user", "content": context_text},
        ]
        if hasattr(self.client, "complete_json"):
            payload = await self.client.complete_json(messages)
        else:
            content = await self.client.complete(messages)  # type: ignore[attr-defined]
            try:
                payload = json.loads(content)
            except (TypeError, ValueError) as error:
                raise ProviderError(
                    "deepseek", "generate", 200, False, "DeepSeek evaluation JSON was invalid"
                ) from error
        return EvaluationGeneration(
            tuple(parse_evaluation_cases(payload, provider="deepseek")),
            provider="deepseek",
            model_id=self.model_id,
        )


class FallbackEvaluationGenerator:
    """Try configured evaluation generators in an explicit, safe order.

    Only provider-boundary failures are eligible for fallback.  The public
    error after the chain is exhausted is intentionally aggregate and never
    contains upstream response text.
    """

    def __init__(self, generators: list[EvaluationGenerator] | tuple[EvaluationGenerator, ...]):
        self.generators = tuple(generators)

    async def generate(self, context: Any) -> EvaluationGeneration:
        if not self.generators:
            raise ProviderError(
                "evaluation",
                "generate",
                None,
                True,
                "evaluation generation unavailable",
            )
        failures: list[ProviderError] = []
        for generator in self.generators:
            try:
                result = await generator.generate(context)
                return _normalise_generation(result, generator)
            except ProviderError as error:
                failures.append(error)
            except (TimeoutError, ConnectionError, OSError):
                failures.append(
                    ProviderError(
                        _generator_provider(generator),
                        "generate",
                        None,
                        True,
                        "provider request timed out or was unavailable",
                    )
                )
            except Exception:  # noqa: BLE001 - provider boundary fails closed
                failures.append(
                    ProviderError(
                        _generator_provider(generator),
                        "generate",
                        None,
                        False,
                        "structured evaluation response was invalid",
                    )
                )
        # Do not join ``failures``: ProviderError messages can still carry
        # provider-specific details and are not part of the API contract.
        raise ProviderError(
            "evaluation",
            "generate",
            None,
            any(error.retryable for error in failures),
            "evaluation generation unavailable",
        ) from None


def create_evaluation_generator(
    minimax_api_key: str = "",
    *,
    minimax_base_url: str = "https://api.minimaxi.com/v1",
    minimax_model: str = "MiniMax-M2.7",
    deepseek_api_key: str = "",
    deepseek_base_url: str = "https://api.deepseek.com",
    deepseek_model: str = "deepseek-v4-flash",
    minimax_client: MiniMaxClient | None = None,
    deepseek_client: DeepSeekClient | None = None,
    deepseek_responses_client: DeepSeekResponsesEvaluationClient | None = None,
    transport: Any = None,
    sleep: Any = None,
) -> EvaluationGenerator:
    """Build the isolated MiniMax -> DeepSeek evaluation chain."""

    generators: list[EvaluationGenerator] = []
    if is_configured_key(minimax_api_key) or minimax_client is not None:
        generators.append(
            MiniMaxEvaluationGenerator(
                minimax_api_key,
                minimax_base_url,
                model=minimax_model,
                client=minimax_client,
                transport=transport,
                sleep=sleep,
            )
        )
    if (
        is_configured_key(deepseek_api_key)
        or deepseek_client is not None
        or deepseek_responses_client is not None
    ):
        if deepseek_responses_client is not None:
            generators.append(
                DeepSeekResponsesEvaluationGenerator(deepseek_responses_client)
            )
        elif deepseek_client is not None:
            generators.append(DeepSeekEvaluationGenerator(deepseek_client))
        else:
            fallback = DeepSeekResponsesEvaluationClient(
                deepseek_api_key,
                deepseek_base_url,
                model=deepseek_model,
                transport=transport,
                sleep=sleep,
                timeout=DEEPSEEK_EVALUATION_TIMEOUT_SECONDS,
                max_retries=0,
            )
            generators.append(DeepSeekResponsesEvaluationGenerator(fallback))

    # Preserve the established direct-adapter contract for callers that inject
    # one fake DeepSeek client, while normal configuration always gets a chain.
    if (
        deepseek_client is not None
        and deepseek_responses_client is None
        and len(generators) == 1
    ):
        return generators[0]
    return FallbackEvaluationGenerator(generators)


# Descriptive alias for callers that prefer a builder verb.
build_evaluation_generator = create_evaluation_generator


def parse_evaluation_cases(payload: Any, *, provider: str) -> list[EvaluationCase]:
    """Normalise one known envelope alias, then strictly validate every case."""

    if isinstance(payload, Mapping):
        if set(payload) == {"cases"}:
            payload = payload["cases"]
        elif set(payload) == {"questions"}:
            payload = payload["questions"]
    if not isinstance(payload, list) or not payload:
        raise _invalid(provider, "evaluation response must be a non-empty list")

    required = {"question", "reference_points", "source_ids", "type"}
    cases: list[EvaluationCase] = []
    for item in payload:
        if not isinstance(item, Mapping) or set(item) != required:
            raise _invalid(provider, "evaluation case has an invalid structure")
        question = item["question"]
        reference_points = item["reference_points"]
        source_ids = item["source_ids"]
        case_type = item["type"]
        if (
            not isinstance(question, str)
            or not question.strip()
            or not isinstance(reference_points, list)
            or not reference_points
            or any(not isinstance(point, str) or not point.strip() for point in reference_points)
            or not isinstance(source_ids, list)
            or not source_ids
            or any(not isinstance(source_id, str) or not source_id.strip() for source_id in source_ids)
            or not isinstance(case_type, str)
            or not case_type.strip()
        ):
            raise _invalid(provider, "evaluation case fields have invalid types")
        normalized_type = EVALUATION_TYPE_ALIASES.get(case_type.strip().casefold())
        if normalized_type is None:
            raise _invalid(provider, "evaluation case type is unsupported")
        cases.append(
            EvaluationCase(
                question=question.strip(),
                reference_points=[point.strip() for point in reference_points],
                source_ids=[source_id.strip() for source_id in source_ids],
                type=normalized_type,
            )
        )
    return cases


def _extract_message_content(response: Any) -> Any:
    if not isinstance(response, Mapping):
        raise _invalid("minimax", "evaluation response has an invalid structure")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise _invalid("minimax", "evaluation response has an invalid structure")
    message = choices[0].get("message")
    if not isinstance(message, Mapping) or "content" not in message:
        raise _invalid("minimax", "evaluation response has an invalid structure")
    return message["content"]


def _extract_deepseek_responses_json(response: Any) -> Any:
    if not isinstance(response, Mapping) or response.get("status") != "completed":
        raise _deepseek_responses_invalid()
    output = response.get("output")
    if not isinstance(output, list) or not output:
        raise _deepseek_responses_invalid()

    fragments: list[str] = []
    for item in output:
        if not isinstance(item, Mapping):
            raise _deepseek_responses_invalid()
        if (
            item.get("type") != "message"
            or item.get("role") != "assistant"
            or item.get("status") != "completed"
        ):
            raise _deepseek_responses_invalid()
        content = item.get("content")
        if not isinstance(content, list):
            raise _deepseek_responses_invalid()
        for block in content:
            if not isinstance(block, Mapping) or block.get("type") != "output_text":
                raise _deepseek_responses_invalid()
            text = block.get("text")
            if not isinstance(text, str):
                raise _deepseek_responses_invalid()
            if text.strip():
                fragments.append(text)

    if not fragments:
        raise _deepseek_responses_invalid()
    try:
        return json.loads("".join(fragments))
    except (TypeError, ValueError) as error:
        raise _deepseek_responses_invalid() from error


def _deepseek_responses_invalid() -> ProviderError:
    return ProviderError(
        "deepseek",
        "generate",
        200,
        False,
        "DeepSeek Responses evaluation response was invalid",
    )


def _invalid(provider: str, message: str) -> ProviderError:
    return ProviderError(provider, "generate", 200, False, message)


def _normalise_generation(result: Any, generator: Any) -> EvaluationGeneration:
    if isinstance(result, EvaluationGeneration):
        return result
    if isinstance(result, (list, tuple)):
        return EvaluationGeneration(
            tuple(result),
            provider=_generator_provider(generator),
            model_id=_generator_model(generator),
        )
    raise ProviderError(
        _generator_provider(generator),
        "generate",
        200,
        False,
        "structured evaluation response was invalid",
    )


def _generator_provider(generator: Any) -> str:
    provider = getattr(generator, "provider", None)
    if isinstance(provider, str) and provider.strip():
        return provider.strip()
    client = getattr(generator, "client", None)
    provider = getattr(client, "provider", None)
    return provider.strip() if isinstance(provider, str) and provider.strip() else "evaluation"


def _generator_model(generator: Any) -> str:
    model = getattr(generator, "model_id", None)
    if isinstance(model, str) and model.strip():
        return model.strip()
    client = getattr(generator, "client", None)
    model = getattr(client, "model", None)
    return model.strip() if isinstance(model, str) and model.strip() else "unknown"
