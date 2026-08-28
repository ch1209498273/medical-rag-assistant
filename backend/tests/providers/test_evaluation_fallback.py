from __future__ import annotations

from types import SimpleNamespace

import pytest
from app.errors import ProviderError
from app.providers.minimax import (
    EvaluationCase,
    EvaluationGeneration,
    FallbackEvaluationGenerator,
    MiniMaxEvaluationGenerator,
    create_evaluation_generator,
)


def valid_case(source_id: str = "S1") -> EvaluationCase:
    return EvaluationCase(
        question="问题",
        reference_points=["要点"],
        source_ids=[source_id],
        type="fact",
    )


class FailingGenerator:
    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.calls = 0

    async def generate(self, context):
        self.calls += 1
        raise ProviderError(self.provider, "generate", 503, True, "safe failure")


class SuccessfulGenerator:
    def __init__(self, provider: str, model_id: str) -> None:
        self.provider = provider
        self.model_id = model_id
        self.calls = 0

    async def generate(self, context):
        self.calls += 1
        return EvaluationGeneration((valid_case(),), self.provider, self.model_id)


@pytest.mark.asyncio
async def test_generation_falls_back_from_minimax_to_deepseek():
    minimax = FailingGenerator("minimax")
    deepseek = SuccessfulGenerator("deepseek", "deepseek-v4-flash")
    generator = FallbackEvaluationGenerator([minimax, deepseek])

    result = await generator.generate({"sources": ["S1"]})

    assert result.provider == "deepseek"
    assert result.model_id == "deepseek-v4-flash"
    assert len(result.cases) == 1
    assert deepseek.calls == 1


@pytest.mark.asyncio
async def test_fallback_error_does_not_echo_upstream_response() -> None:
    class LeakyFailure:
        async def generate(self, context):
            raise ProviderError(
                "minimax",
                "generate",
                429,
                True,
                "upstream private response should never be returned",
            )

    with pytest.raises(ProviderError) as caught:
        await FallbackEvaluationGenerator([LeakyFailure()]).generate("context")

    assert caught.value.safe_message == "evaluation generation unavailable"
    assert "private" not in str(caught.value)


@pytest.mark.asyncio
async def test_unsupported_type_falls_through_to_next_provider() -> None:
    class JsonTransport:
        async def request(self, method: str, url: str, **kwargs: object):
            return type(
                "Response",
                (),
                {
                    "status_code": 200,
                    "json": lambda _: {
                        "choices": [
                            {
                                "message": {
                                    "content": (
                                        '[{"question":"问题","reference_points":["要点"],'
                                        '"source_ids":["S1"],"type":"unsupported"}]'
                                    )
                                }
                            }
                        ]
                    },
                },
            )()

    invalid = MiniMaxEvaluationGenerator("fictional-key", transport=JsonTransport())
    successful = SuccessfulGenerator("deepseek", "deepseek-v4-flash")
    result = await FallbackEvaluationGenerator([invalid, successful]).generate("context")

    assert result.provider == "deepseek"
    assert successful.calls == 1


def test_factory_skips_placeholder_keys_without_constructing_providers() -> None:
    generator = create_evaluation_generator(
        minimax_api_key="your-key-here",
        deepseek_api_key="example",
    )

    assert isinstance(generator, FallbackEvaluationGenerator)
    assert generator.generators == ()


def test_factory_orders_supported_evaluation_fallback_as_minimax_then_deepseek() -> None:
    generator = create_evaluation_generator(
        minimax_client=SimpleNamespace(model="MiniMax-M2.7"),
        deepseek_client=SimpleNamespace(model="deepseek-v4-flash"),
    )

    assert isinstance(generator, FallbackEvaluationGenerator)
    assert [item.provider for item in generator.generators] == ["minimax", "deepseek"]
