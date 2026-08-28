"""Contract tests for MiniMax evaluation generation and fallback."""

from __future__ import annotations

import json
from dataclasses import dataclass

import app.providers.minimax as minimax_provider
import pytest
from app.errors import ProviderError
from app.providers.deepseek import DeepSeekClient
from app.providers.minimax import (
    DeepSeekEvaluationGenerator,
    EvaluationCase,
    EvaluationGeneration,
    FallbackEvaluationGenerator,
    MiniMaxEvaluationGenerator,
    create_evaluation_generator,
    parse_evaluation_cases,
)


@dataclass
class JsonTransport:
    payload: object
    status_code: int = 200

    def __post_init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def request(self, method: str, url: str, **kwargs: object):
        self.calls.append({"method": method, "url": url, **kwargs})
        return type(
            "Response",
            (),
            {"status_code": self.status_code, "json": lambda _: self.payload},
        )()


@dataclass
class SequencedTransport:
    responses: list[tuple[int, object]]

    def __post_init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def request(self, method: str, url: str, **kwargs: object):
        self.calls.append({"method": method, "url": url, **kwargs})
        status_code, payload = self.responses[len(self.calls) - 1]
        return type(
            "Response",
            (),
            {"status_code": status_code, "json": lambda _: payload},
        )()


@dataclass
class TimeoutTransport:
    def __post_init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def request(self, method: str, url: str, **kwargs: object):
        self.calls.append({"method": method, "url": url, **kwargs})
        raise TimeoutError("synthetic evaluation timeout")


def valid_payload() -> list[dict[str, object]]:
    return [
        {
            "question": "交接班前需要核对哪些事项？",
            "reference_points": ["核对患者身份", "核对设备状态"],
            "source_ids": ["S1", "S2"],
            "type": "procedure",
        }
    ]


def valid_enveloped_payload() -> dict[str, object]:
    return {"cases": valid_payload()}


RESPONSES_CONTEXT = "虚构制度片段"
SENSITIVE_RESPONSE_CANARY = "responses-sensitive-canary"


def responses_completed_payload(payload: object) -> dict[str, object]:
    return {
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps(payload, ensure_ascii=False),
                    }
                ],
            }
        ],
    }


def responses_completed_payload_with_canary(payload: object) -> dict[str, object]:
    response = responses_completed_payload(payload)
    output = response["output"]
    assert isinstance(output, list) and output
    message = output[0]
    assert isinstance(message, dict)
    message["id"] = SENSITIVE_RESPONSE_CANARY
    return response


def responses_generator(transport: JsonTransport):
    client_type = getattr(
        minimax_provider, "DeepSeekResponsesEvaluationClient", None
    )
    generator_type = getattr(
        minimax_provider, "DeepSeekResponsesEvaluationGenerator", None
    )
    assert client_type is not None, "DeepSeek Responses client is not implemented"
    assert generator_type is not None, "DeepSeek Responses generator is not implemented"
    return generator_type(client_type("deepseek-key", transport=transport))


def responses_failure_payload(kind: str) -> dict[str, object]:
    if kind == "non_completed_status":
        payload = responses_completed_payload(valid_enveloped_payload())
        payload["status"] = "in_progress"
        payload["error"] = {"message": SENSITIVE_RESPONSE_CANARY}
        return payload
    if kind == "missing_output_text":
        return {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "refusal",
                            "refusal": SENSITIVE_RESPONSE_CANARY,
                        }
                    ],
                }
            ],
        }
    if kind == "empty_output_text":
        payload = responses_completed_payload_with_canary(valid_enveloped_payload())
        message = payload["output"][0]  # type: ignore[index]
        assert isinstance(message, dict)
        message["content"][0]["text"] = ""  # type: ignore[index]
        return payload
    if kind == "invalid_json":
        return {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": (
                                '{"cases":["'
                                + SENSITIVE_RESPONSE_CANARY
                            ),
                        }
                    ],
                }
            ],
        }
    if kind == "extra_case_field":
        case = dict(valid_payload()[0])
        case["unexpected"] = SENSITIVE_RESPONSE_CANARY
        return responses_completed_payload({"cases": [case]})
    if kind == "empty_cases":
        return responses_completed_payload_with_canary({"cases": []})
    if kind == "empty_reference_points":
        case = dict(valid_payload()[0])
        case["reference_points"] = []
        return responses_completed_payload_with_canary({"cases": [case]})
    if kind == "empty_source_ids":
        case = dict(valid_payload()[0])
        case["source_ids"] = []
        return responses_completed_payload_with_canary({"cases": [case]})
    if kind == "invalid_source_id_type":
        case = dict(valid_payload()[0])
        case["source_ids"] = [123]
        return responses_completed_payload_with_canary({"cases": [case]})
    if kind == "whitespace_question":
        case = dict(valid_payload()[0])
        case["question"] = " \t"
        return responses_completed_payload_with_canary({"cases": [case]})
    if kind == "whitespace_reference_point":
        case = dict(valid_payload()[0])
        case["reference_points"] = [" \n"]
        return responses_completed_payload_with_canary({"cases": [case]})
    if kind == "whitespace_source_id":
        case = dict(valid_payload()[0])
        case["source_ids"] = [" \r"]
        return responses_completed_payload_with_canary({"cases": [case]})
    raise AssertionError(f"unknown Responses failure kind: {kind}")


@pytest.mark.asyncio
async def test_minimax_evaluation_uses_long_timeout_without_retries() -> None:
    transport = TimeoutTransport()
    generator = MiniMaxEvaluationGenerator("key", transport=transport, sleep=lambda _: None)

    with pytest.raises(ProviderError) as raised:
        await generator.generate("虚构制度片段")

    assert raised.value.safe_message == "provider request timed out or was unavailable"
    assert len(transport.calls) == 1
    assert transport.calls[0]["timeout"] == 180.0


@pytest.mark.asyncio
async def test_minimax_request_disables_thinking_and_bounds_object_output() -> None:
    transport = JsonTransport(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(valid_enveloped_payload(), ensure_ascii=False)
                    }
                }
            ]
        }
    )
    generator = MiniMaxEvaluationGenerator("key", transport=transport)

    await generator.generate("虚构制度片段")

    request_json = transport.calls[0]["json"]
    assert isinstance(request_json, dict)
    assert request_json["thinking"] == {"type": "disabled"}
    assert request_json["max_completion_tokens"] == 8192
    assert request_json["temperature"] == 0.25
    assert request_json["stream"] is False
    system_prompt = request_json["messages"][0]["content"]
    assert '{"cases":[...]}' in system_prompt
    assert "question、reference_points、source_ids、type" in system_prompt
    assert "fact、procedure、condition、prohibition、synthesis 或 no_answer" in system_prompt


@pytest.mark.asyncio
async def test_minimax_generator_accepts_cases_object_envelope() -> None:
    transport = JsonTransport(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(valid_enveloped_payload(), ensure_ascii=False)
                    }
                }
            ]
        }
    )
    generator = MiniMaxEvaluationGenerator("key", transport=transport)

    cases = await generator.generate("虚构制度片段")

    assert cases == [
        EvaluationCase(
            question="交接班前需要核对哪些事项？",
            reference_points=["核对患者身份", "核对设备状态"],
            source_ids=["S1", "S2"],
            type="procedure",
        )
    ]


def test_parser_normalises_only_the_known_questions_envelope_alias() -> None:
    cases = parse_evaluation_cases({"questions": valid_payload()}, provider="deepseek")

    assert cases == [
        EvaluationCase(
            question="交接班前需要核对哪些事项？",
            reference_points=["核对患者身份", "核对设备状态"],
            source_ids=["S1", "S2"],
            type="procedure",
        )
    ]

    with pytest.raises(ProviderError):
        parse_evaluation_cases(
            {"questions": valid_payload(), "untrusted_wrapper": []}, provider="deepseek"
        )


@pytest.mark.asyncio
async def test_minimax_generator_rejects_malformed_structured_text_safely() -> None:
    transport = JsonTransport(
        {"choices": [{"message": {"content": "not valid structured JSON"}}]}
    )
    generator = MiniMaxEvaluationGenerator("key", transport=transport)

    with pytest.raises(ProviderError) as raised:
        await generator.generate("虚构制度片段")

    assert raised.value.safe_message == "MiniMax response was not valid structured JSON"


@pytest.mark.asyncio
async def test_minimax_generator_strictly_parses_evaluation_cases() -> None:
    transport = JsonTransport(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(valid_payload(), ensure_ascii=False)
                    }
                }
            ]
        }
    )
    generator = MiniMaxEvaluationGenerator("key", transport=transport)

    cases = await generator.generate("虚构制度片段")

    assert cases == [
        EvaluationCase(
            question="交接班前需要核对哪些事项？",
            reference_points=["核对患者身份", "核对设备状态"],
            source_ids=["S1", "S2"],
            type="procedure",
        )
    ]
    assert transport.calls[0]["url"] == "https://api.minimaxi.com/v1/chat/completions"


@pytest.mark.asyncio
async def test_minimax_generator_rejects_missing_or_wrong_fields() -> None:
    malformed = [
        {
            "question": "问题",
            "reference_points": ["依据"],
            "source_ids": [1],
            "type": "fact",
        }
    ]
    transport = JsonTransport(
        {"choices": [{"message": {"content": json.dumps(malformed)}}]}
    )

    with pytest.raises(ProviderError, match="evaluation"):
        await MiniMaxEvaluationGenerator("key", transport=transport).generate("片段")


@pytest.mark.asyncio
async def test_deepseek_responses_generator_builds_schema_request_and_generation() -> None:
    transport = JsonTransport(responses_completed_payload(valid_enveloped_payload()))
    generator = responses_generator(transport)

    generation = await generator.generate(RESPONSES_CONTEXT)

    assert generation == EvaluationGeneration(
        (
            EvaluationCase(
                question="交接班前需要核对哪些事项？",
                reference_points=["核对患者身份", "核对设备状态"],
                source_ids=["S1", "S2"],
                type="procedure",
            ),
        ),
        provider="deepseek",
        model_id="deepseek-v4-flash",
    )

    request = transport.calls[0]
    assert request["url"] == "https://api.deepseek.com/responses"
    assert request["timeout"] == 180.0
    request_json = request["json"]
    assert isinstance(request_json, dict)
    assert request_json["model"] == "deepseek-v4-flash"
    assert request_json["input"] == RESPONSES_CONTEXT
    assert set(request_json) == {
        "model",
        "input",
        "instructions",
        "stream",
        "reasoning",
        "max_output_tokens",
        "text",
    }
    assert "temperature" not in request_json
    assert "messages" not in request_json
    assert "response_format" not in request_json
    assert isinstance(request_json["instructions"], str)
    assert request_json["instructions"]
    assert request_json["stream"] is False
    assert request_json["reasoning"] == {"effort": "none"}
    assert request_json["max_output_tokens"] == 8192

    text = request_json["text"]
    assert isinstance(text, dict)
    response_format = text["format"]
    assert isinstance(response_format, dict)
    assert response_format["type"] == "json_schema"
    assert response_format["name"] == "evaluation_cases"
    schema = response_format["schema"]
    assert isinstance(schema, dict)
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["cases"]

    properties = schema["properties"]
    assert isinstance(properties, dict)
    assert set(properties) == {"cases"}
    cases_schema = properties["cases"]
    assert isinstance(cases_schema, dict)
    assert cases_schema["type"] == "array"
    assert cases_schema["minItems"] == 1
    case_schema = cases_schema["items"]
    assert isinstance(case_schema, dict)
    assert case_schema["type"] == "object"
    assert case_schema["additionalProperties"] is False
    assert case_schema["required"] == [
        "question",
        "reference_points",
        "source_ids",
        "type",
    ]
    case_properties = case_schema["properties"]
    assert isinstance(case_properties, dict)
    assert set(case_properties) == {
        "question",
        "reference_points",
        "source_ids",
        "type",
    }
    question_schema = case_properties["question"]
    assert isinstance(question_schema, dict)
    assert question_schema["type"] == "string"
    assert question_schema["minLength"] == 1
    assert question_schema["pattern"] == r"\S"
    reference_points_schema = case_properties["reference_points"]
    assert isinstance(reference_points_schema, dict)
    assert reference_points_schema["type"] == "array"
    assert reference_points_schema["minItems"] == 1
    reference_point_schema = reference_points_schema["items"]
    assert isinstance(reference_point_schema, dict)
    assert reference_point_schema["type"] == "string"
    assert reference_point_schema["minLength"] == 1
    assert reference_point_schema["pattern"] == r"\S"
    source_ids_schema = case_properties["source_ids"]
    assert isinstance(source_ids_schema, dict)
    assert source_ids_schema["type"] == "array"
    assert source_ids_schema["minItems"] == 1
    source_id_schema = source_ids_schema["items"]
    assert isinstance(source_id_schema, dict)
    assert source_id_schema["type"] == "string"
    assert source_id_schema["minLength"] == 1
    assert source_id_schema["pattern"] == r"\S"
    type_schema = case_properties["type"]
    assert isinstance(type_schema, dict)
    assert type_schema["type"] == "string"
    assert type_schema["enum"] == [
        "fact",
        "procedure",
        "condition",
        "prohibition",
        "synthesis",
        "no_answer",
    ]

    client = generator.client
    assert client.timeout == 180.0
    assert client.max_retries == 0


def _assert_exact_thirty_type_plan(instructions: object) -> None:
    assert isinstance(instructions, str)
    expected_plan = (
        "当 context.count == 30 时，必须严格按以下固定顺序生成 cases 数组，共 30 题，"
        "每类恰好 5 题：1–5 题 type=fact；6–10 题 type=procedure；"
        "11–15 题 type=condition；16–20 题 type=prohibition；"
        "21–25 题 type=synthesis；26–30 题 type=no_answer。"
        "cases 数组必须严格按此顺序返回，不得重排、遗漏或添加题目。"
    )
    assert expected_plan in instructions


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["minimax", "deepseek"])
@pytest.mark.parametrize(
    ("context", "requires_exact_quota"),
    [
        (
            {
                "count": 30,
                "question_types": [
                    "fact",
                    "procedure",
                    "condition",
                    "prohibition",
                    "synthesis",
                    "no_answer",
                ],
                "sources": [],
            },
            True,
        ),
        ({"count": 2, "question_types": ["fact"], "sources": []}, False),
        ("虚构制度片段", False),
    ],
)
async def test_evaluation_adapters_apply_exact_type_quota_only_for_count_30(
    provider: str,
    context: object,
    requires_exact_quota: bool,
) -> None:
    if provider == "minimax":
        transport = JsonTransport(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                valid_enveloped_payload(), ensure_ascii=False
                            )
                        }
                    }
                ]
            }
        )
        client = minimax_provider.MiniMaxClient("minimax-key", transport=transport)
    else:
        transport = JsonTransport(responses_completed_payload(valid_enveloped_payload()))
        client = minimax_provider.DeepSeekResponsesEvaluationClient(
            "deepseek-key", transport=transport
        )

    await client.complete_json(context)

    request_json = transport.calls[0]["json"]
    assert isinstance(request_json, dict)
    instructions = (
        request_json["messages"][0]["content"]
        if provider == "minimax"
        else request_json["instructions"]
    )
    if requires_exact_quota:
        _assert_exact_thirty_type_plan(instructions)
    else:
        assert isinstance(instructions, str)
        assert "context.count == 30" not in instructions


@pytest.mark.asyncio
async def test_deepseek_responses_serializes_context_and_safely_rejects_unserializable_input() -> None:
    transport = JsonTransport(responses_completed_payload(valid_enveloped_payload()))
    client = minimax_provider.DeepSeekResponsesEvaluationClient(
        "deepseek-key", transport=transport
    )

    await client.complete_json({"count": 2, "question_types": ["fact"]})

    request_json = transport.calls[0]["json"]
    assert isinstance(request_json, dict)
    assert request_json["input"] == '{"count": 2, "question_types": ["fact"]}'

    with pytest.raises(ProviderError) as raised:
        await client.complete_json(object())

    assert raised.value.safe_message == "evaluation context is not serializable"
    assert len(transport.calls) == 1


@pytest.mark.asyncio
async def test_deepseek_responses_rejects_deeply_nested_context_safely() -> None:
    transport = JsonTransport(responses_completed_payload(valid_enveloped_payload()))
    client = minimax_provider.DeepSeekResponsesEvaluationClient(
        "deepseek-key", transport=transport
    )
    nested_context: object = "leaf"
    for _ in range(10_000):
        nested_context = [nested_context]

    with pytest.raises(ProviderError) as raised:
        await client.complete_json(nested_context)

    assert raised.value.safe_message == "evaluation context is not serializable"
    assert not transport.calls


@pytest.mark.asyncio
async def test_deepseek_responses_joins_only_completed_assistant_output_text_in_order() -> None:
    encoded = json.dumps(valid_enveloped_payload(), ensure_ascii=False)
    split_at = len(encoded) // 2
    transport = JsonTransport(
        {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": encoded[:split_at]},
                    ],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": encoded[split_at:]},
                    ],
                },
            ],
        }
    )
    generator = responses_generator(transport)

    generation = await generator.generate(RESPONSES_CONTEXT)

    assert generation.provider == "deepseek"
    assert generation.cases[0].source_ids == ["S1", "S2"]


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_output_item", [{"type": "reasoning"}, "unknown-output-item"])
async def test_deepseek_responses_rejects_unknown_output_item_before_later_valid_message(
    invalid_output_item: object,
) -> None:
    valid_response = responses_completed_payload(valid_enveloped_payload())
    valid_output = valid_response["output"]
    assert isinstance(valid_output, list) and valid_output
    transport = JsonTransport(
        {
            "status": "completed",
            "output": [invalid_output_item, valid_output[0]],
        }
    )
    generator = responses_generator(transport)

    with pytest.raises(ProviderError) as raised:
        await generator.generate(RESPONSES_CONTEXT)

    assert SENSITIVE_RESPONSE_CANARY not in str(raised.value)
    assert len(transport.calls) == 1


@pytest.mark.asyncio
async def test_factory_accepts_explicit_responses_client_seam() -> None:
    transport = JsonTransport(responses_completed_payload(valid_enveloped_payload()))
    responses_client = minimax_provider.DeepSeekResponsesEvaluationClient(
        "deepseek-key", transport=transport
    )

    generator = create_evaluation_generator(
        minimax_api_key="",
        deepseek_responses_client=responses_client,
    )

    assert isinstance(generator, FallbackEvaluationGenerator)
    assert len(generator.generators) == 1
    assert isinstance(
        generator.generators[0], minimax_provider.DeepSeekResponsesEvaluationGenerator
    )
    await generator.generate(RESPONSES_CONTEXT)
    assert [call["url"] for call in transport.calls] == [
        "https://api.deepseek.com/responses"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response_kind",
    [
        "non_completed_status",
        "missing_output_text",
        "empty_output_text",
        "invalid_json",
        "extra_case_field",
        "empty_cases",
        "empty_reference_points",
        "empty_source_ids",
        "invalid_source_id_type",
        "whitespace_question",
        "whitespace_reference_point",
        "whitespace_source_id",
    ],
)
async def test_deepseek_responses_rejects_malformed_shapes_without_leaking_response(
    response_kind: str,
) -> None:
    transport = JsonTransport(responses_failure_payload(response_kind))
    generator = responses_generator(transport)

    with pytest.raises(ProviderError) as raised:
        await generator.generate(RESPONSES_CONTEXT)

    assert SENSITIVE_RESPONSE_CANARY not in str(raised.value)
    assert len(transport.calls) == 1


@pytest.mark.asyncio
async def test_deepseek_fallback_prompt_requires_exact_cases_envelope() -> None:
    transport = JsonTransport(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(valid_enveloped_payload(), ensure_ascii=False)
                    }
                }
            ]
        }
    )
    generator = DeepSeekEvaluationGenerator(DeepSeekClient("deepseek-key", transport=transport))

    await generator.generate("片段")

    request_json = transport.calls[0]["json"]
    assert isinstance(request_json, dict)
    system_prompt = request_json["messages"][0]["content"]
    assert system_prompt == (
        "只输出 JSON 对象，且顶层只能有 cases 字段，格式必须为 {\"cases\":[...]}。"
        "每题必须且只能包含 question、reference_points、source_ids、type 四个字段，"
        "不要添加解释。type 只能是 fact、procedure、condition、prohibition、synthesis 或 no_answer；"
        "覆盖事实查询、流程步骤、适用条件、禁止事项、跨段归纳和无答案六类。"
    )


@pytest.mark.asyncio
async def test_deepseek_fallback_accepts_cases_object_envelope() -> None:
    transport = JsonTransport(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(valid_enveloped_payload(), ensure_ascii=False)
                    }
                }
            ]
        }
    )
    generator = DeepSeekEvaluationGenerator(DeepSeekClient("deepseek-key", transport=transport))

    cases = await generator.generate("片段")

    assert cases == [
        EvaluationCase(
            question="交接班前需要核对哪些事项？",
            reference_points=["核对患者身份", "核对设备状态"],
            source_ids=["S1", "S2"],
            type="procedure",
        )
    ]


@pytest.mark.asyncio
async def test_factory_without_minimax_key_returns_deepseek_generator() -> None:
    deepseek_transport = JsonTransport(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(valid_payload(), ensure_ascii=False)
                    }
                }
            ]
        }
    )
    deepseek_client = DeepSeekClient("deepseek-key", transport=deepseek_transport)

    generator = create_evaluation_generator(
        minimax_api_key="",
        deepseek_client=deepseek_client,
    )

    assert isinstance(generator, DeepSeekEvaluationGenerator)
    cases = await generator.generate("片段")
    assert cases[0].source_ids == ["S1", "S2"]
    assert (
        deepseek_transport.calls[0]["url"]
        == "https://api.deepseek.com/chat/completions"
    )
    assert deepseek_transport.calls[0]["json"]["stream"] is False  # type: ignore[index]


@pytest.mark.asyncio
async def test_factory_deepseek_fallback_uses_responses_evaluation_adapter() -> None:
    transport = JsonTransport(responses_completed_payload(valid_enveloped_payload()))

    generator_kwargs = {
        "minimax_api_key": "",
        "deepseek_api_key": "fixture",
        "transport": transport,
    }
    generator = create_evaluation_generator(**generator_kwargs)

    assert isinstance(generator, FallbackEvaluationGenerator)
    assert len(generator.generators) == 1
    fallback = generator.generators[0]
    client_type = getattr(
        minimax_provider, "DeepSeekResponsesEvaluationClient", None
    )
    generator_type = getattr(
        minimax_provider, "DeepSeekResponsesEvaluationGenerator", None
    )
    assert client_type is not None, "DeepSeek Responses client is not implemented"
    assert generator_type is not None, "DeepSeek Responses generator is not implemented"
    assert isinstance(fallback, generator_type)
    assert isinstance(fallback.client, client_type)

    await generator.generate(RESPONSES_CONTEXT)

    assert [call["url"] for call in transport.calls] == [
        "https://api.deepseek.com/responses"
    ]


@pytest.mark.asyncio
async def test_factory_falls_back_from_minimax_to_deepseek_responses_in_order() -> None:
    transport = SequencedTransport(
        [
            (400, {"error": {"message": "synthetic MiniMax failure"}}),
            (200, responses_completed_payload(valid_enveloped_payload())),
        ]
    )

    generator_kwargs = {
        "minimax_api_key": "fixture",
        "deepseek_api_key": "fixture",
        "transport": transport,
    }
    generator = create_evaluation_generator(**generator_kwargs)

    generation = await generator.generate(RESPONSES_CONTEXT)

    assert generation.provider == "deepseek"
    assert [call["url"] for call in transport.calls] == [
        "https://api.minimaxi.com/v1/chat/completions",
        "https://api.deepseek.com/responses",
    ]
