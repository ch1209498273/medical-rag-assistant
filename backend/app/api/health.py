"""Non-sensitive service health endpoint."""

from typing import Annotated, Literal

from app.settings import RuntimeMode, Settings, get_settings
from fastapi import APIRouter, Depends
from pydantic import BaseModel

ProviderState = Literal["configured", "missing", "optional_missing", "not_required"]


class HealthResponse(BaseModel):
    """Only status fields are exposed; credentials are not part of this model."""

    runtime_mode: RuntimeMode
    providers: dict[str, ProviderState]


router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    """Report provider availability without revealing credential metadata."""

    if settings.app_runtime_mode == "demo":
        providers: dict[str, ProviderState] = {
            "deepseek": "not_required",
            "siliconflow": "not_required",
            "minimax": "not_required",
        }
    else:
        providers = {
            "deepseek": settings.provider_status("deepseek"),
            "siliconflow": settings.provider_status("siliconflow"),
            "minimax": settings.provider_status("minimax"),
        }
    return HealthResponse(runtime_mode=settings.app_runtime_mode, providers=providers)
