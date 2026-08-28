import asyncio
import json
import logging
from io import StringIO
from types import SimpleNamespace

import pytest
from app.errors import ProviderError
from app.main import (
    APP_VERSION,
    build_evaluation_runtime,
    configure_logging,
    create_app,
    require_cloud_keys,
)
from app.settings import Settings
from fastapi.testclient import TestClient


def test_health_reports_missing_keys_without_exposing_values(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)

    settings = Settings(_env_file=None)
    response = TestClient(
        create_app(settings=settings, document_service=object())
    ).get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "runtime_mode": "cloud",
        "providers": {
        "deepseek": "missing",
        "siliconflow": "missing",
        "minimax": "optional_missing",
        },
    }
    assert set(response.json()) == {"runtime_mode", "providers"}
    assert "api_key" not in response.text.lower()


def test_app_metadata_uses_public_name_and_application_version():
    app = create_app(Settings(_env_file=None), document_service=object())

    assert app.title == "Medical Knowledge Q&A Assistant"
    assert app.version == APP_VERSION


def test_health_reports_configured_required_provider_without_returning_canary(monkeypatch):
    canary = "sk-test-canary-123456789"
    monkeypatch.setenv("DEEPSEEK_API_KEY", canary)
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.setenv("MINIMAX_API_KEY", "your-key-here")

    settings = Settings(_env_file=None)
    response = TestClient(
        create_app(settings=settings, document_service=object())
    ).get("/api/health")

    assert response.json() == {
        "runtime_mode": "cloud",
        "providers": {
        "deepseek": "configured",
        "siliconflow": "missing",
        "minimax": "optional_missing",
        },
    }
    assert canary not in response.text
    assert "api_key" not in response.text.lower()


def test_legacy_provider_key_is_ignored_by_health_and_evaluation_runtime(
    monkeypatch, tmp_path
):
    legacy_key_name = "TAO" + "TOKEN_API_KEY"
    monkeypatch.setenv(legacy_key_name, "legacy-provider-key")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    settings = Settings(_env_file=None, private_data_dir=tmp_path)

    response = TestClient(
        create_app(
            settings=settings,
            document_service=object(),
            rag_service=object(),
            evaluation_service=object(),
        )
    ).get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "runtime_mode": "cloud",
        "providers": {
        "deepseek": "missing",
        "siliconflow": "missing",
        "minimax": "optional_missing",
        },
    }

    class NoCallTransport:
        def __init__(self):
            self.calls = 0

        async def request(self, method: str, url: str, **kwargs: object):
            self.calls += 1
            raise AssertionError("legacy provider request was attempted")

    transport = NoCallTransport()
    runtime = build_evaluation_runtime(
        settings,
        SimpleNamespace(repository=object(), vector_store=object()),
        object(),
        transport=transport,
    )
    try:
        with pytest.raises(ProviderError, match="evaluation generation unavailable"):
            asyncio.run(runtime.service.generator.generate("context"))
        assert transport.calls == 0
    finally:
        asyncio.run(runtime.close())


def test_settings_secret_fields_are_not_repr_or_serialized():
    canary = "sk-test-canary-123456789"
    settings_kwargs = {
        "deepseek_api_key": canary,
        "siliconflow_api_key": "sf-test-canary-123456789",
        "minimax_api_key": "mm-test-canary-123456789",
    }
    settings = Settings(**settings_kwargs)

    rendered = repr(settings) + str(settings)
    serialized = json.dumps(settings.model_dump(), default=str)
    serialized_json = settings.model_dump_json()

    assert canary not in rendered
    assert canary not in serialized
    assert canary not in serialized_json
    assert "deepseek_api_key" not in serialized
    assert "siliconflow_api_key" not in serialized
    assert "minimax_api_key" not in serialized


def test_require_cloud_keys_reports_missing_state_without_exposing_values():
    canary = "sk-cloud-canary-123456789"
    settings_kwargs = {"deepseek_api_key": canary}
    settings = Settings(
        _env_file=None,
        siliconflow_api_key="",
        **settings_kwargs,
    )

    with pytest.raises(
        RuntimeError, match="required cloud providers are not configured"
    ) as error:
        require_cloud_keys(settings)

    assert canary not in str(error.value)


def test_cors_allows_only_explicit_local_frontend_origin():
    client = TestClient(create_app())
    allowed = client.options(
        "/api/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    denied = client.options(
        "/api/health",
        headers={
            "Origin": "https://untrusted.example",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "access-control-allow-origin" not in denied.headers
    assert allowed.headers.get("access-control-allow-credentials") != "true"


def test_safe_logger_does_not_emit_a_bare_provider_token():
    token = "sk-" + "1234567890abcdef"
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = configure_logging()
    logger.addHandler(handler)
    try:
        logger.info(f"provider response {token}")
    finally:
        logger.removeHandler(handler)

    assert token not in stream.getvalue()
