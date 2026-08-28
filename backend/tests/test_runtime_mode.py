import pytest
from app.main import create_app
from app.settings import Settings
from fastapi.testclient import TestClient
from pydantic import ValidationError


def test_demo_health_never_implies_keys_are_required():
    settings = Settings(_env_file=None, app_runtime_mode="demo")
    payload = TestClient(create_app(settings, document_service=object())).get("/api/health").json()
    assert payload == {
        "runtime_mode": "demo",
        "providers": {
            "deepseek": "not_required",
            "siliconflow": "not_required",
            "minimax": "not_required",
        },
    }


def test_runtime_mode_rejects_unknown_values():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, app_runtime_mode="automatic")
