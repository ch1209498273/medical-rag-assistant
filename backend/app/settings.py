"""Application configuration with a deliberately non-leaky representation."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SECRET_FIELDS = frozenset(
    {
        "deepseek_api_key",
        "siliconflow_api_key",
        "minimax_api_key",
    }
)
PLACEHOLDER_KEYS = frozenset(
    {"", "example", "placeholder", "your-key-here", "your_key_here", "changeme"}
)
DEFAULT_ALLOWED_ORIGINS = (
    "http://127.0.0.1:5173",
    "http://localhost:5173",
)
APPROVED_FREE_SILICONFLOW_MODELS = MappingProxyType(
    {
        "embedding_model": "BAAI/bge-m3",
        "reranker_model": "BAAI/bge-reranker-v2-m3",
        "ocr_model": "PaddlePaddle/PaddleOCR-VL-1.5",
    }
)

# Conservative local-only resource defaults.  The upper bounds are deliberately
# part of the schema so an environment file cannot turn ingestion or generation
# into an unbounded resource consumer.
DEFAULT_INGESTION_MAX_FILE_BYTES = 8 * 1024 * 1024
DEFAULT_INGESTION_MAX_PDF_PAGES = 200
DEFAULT_INGESTION_MAX_DOCX_PARAGRAPHS = 10_000
DEFAULT_INGESTION_MAX_DOCX_UNCOMPRESSED_BYTES = 32 * 1024 * 1024
DEFAULT_INGESTION_MAX_TEXT_CHARS = 500_000
DEFAULT_INGESTION_FILE_TIMEOUT_SECONDS = 30.0
DEFAULT_INGESTION_SCAN_MAX_FILES = 100
DEFAULT_DEEPSEEK_STREAM_MAX_BYTES = 1 * 1024 * 1024
DEFAULT_DEEPSEEK_STREAM_MAX_CHARS = 200_000
DEFAULT_DEEPSEEK_STREAM_MAX_EVENTS = 2_048
DEFAULT_DEEPSEEK_STREAM_TIMEOUT_SECONDS = 60.0

RuntimeMode = Literal["demo", "cloud"]


class Settings(BaseSettings):
    """Runtime settings.

    Provider keys remain available to backend adapters as strings, but are
    omitted from the model's normal representation and serialization.  This
    prevents accidental inclusion in diagnostics, API responses, or logs.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    deepseek_api_key: str = Field(default="", repr=False)
    app_runtime_mode: RuntimeMode = "cloud"
    demo_questions_path: Path = Path("../demo/questions.json")
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    stage_b_verification_enabled: bool = False
    siliconflow_api_key: str = Field(default="", repr=False)
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    siliconflow_free_only: bool = True
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    ocr_model: str = "PaddlePaddle/PaddleOCR-VL-1.5"
    retrieval_limit: int = Field(default=20, ge=1, le=20)
    rerank_limit: int = Field(default=6, ge=1, le=20)
    relevance_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    chat_history_max_turns: int = Field(default=5, ge=1, le=20)
    chat_history_max_chars: int = Field(default=6000, ge=500, le=6000)
    ingestion_max_file_bytes: int = Field(
        default=DEFAULT_INGESTION_MAX_FILE_BYTES, ge=1, le=64 * 1024 * 1024
    )
    ingestion_max_pdf_pages: int = Field(default=DEFAULT_INGESTION_MAX_PDF_PAGES, ge=1, le=3000)
    ingestion_max_pdf_pixels: int = Field(default=20_000_000, ge=1, le=100_000_000)
    ingestion_max_pdf_render_bytes: int = Field(
        default=4 * 1024 * 1024, ge=1, le=16 * 1024 * 1024
    )
    ingestion_max_docx_paragraphs: int = Field(
        default=DEFAULT_INGESTION_MAX_DOCX_PARAGRAPHS, ge=1, le=100_000
    )
    ingestion_max_docx_uncompressed_bytes: int = Field(
        default=DEFAULT_INGESTION_MAX_DOCX_UNCOMPRESSED_BYTES,
        ge=1,
        le=256 * 1024 * 1024,
    )
    ingestion_max_docx_zip_entries: int = Field(default=2_000, ge=1, le=10_000)
    ingestion_max_text_chars: int = Field(
        default=DEFAULT_INGESTION_MAX_TEXT_CHARS, ge=1, le=5_000_000
    )
    ingestion_file_timeout_seconds: float = Field(
        default=DEFAULT_INGESTION_FILE_TIMEOUT_SECONDS, gt=0.0, le=1800.0
    )
    ingestion_scan_max_files: int = Field(default=DEFAULT_INGESTION_SCAN_MAX_FILES, ge=1, le=1000)
    deepseek_stream_max_bytes: int = Field(
        default=DEFAULT_DEEPSEEK_STREAM_MAX_BYTES, ge=1, le=8 * 1024 * 1024
    )
    deepseek_stream_max_chars: int = Field(
        default=DEFAULT_DEEPSEEK_STREAM_MAX_CHARS, ge=1, le=1_000_000
    )
    deepseek_stream_max_events: int = Field(
        default=DEFAULT_DEEPSEEK_STREAM_MAX_EVENTS, ge=1, le=10_000
    )
    deepseek_stream_timeout_seconds: float = Field(
        default=DEFAULT_DEEPSEEK_STREAM_TIMEOUT_SECONDS, gt=0.0, le=300.0
    )
    minimax_api_key: str = Field(default="", repr=False)
    minimax_base_url: str = "https://api.minimaxi.com/v1"
    minimax_model: str = "MiniMax-M2.7"
    source_documents_dir: Path = Path("../资料")
    private_data_dir: Path = Path("../data/private")
    qdrant_path: Path = Path("../data/qdrant")
    sqlite_path: Path = Path("../data/app.sqlite3")

    # The server is intentionally local by default.  Origins are explicit;
    # callers may set a comma-separated list for a trusted local network.
    host: str = "127.0.0.1"
    port: int = 8000
    allowed_origins: str = ",".join(DEFAULT_ALLOWED_ORIGINS)

    @model_validator(mode="after")
    def validate_siliconflow_models(self) -> Settings:
        if self.siliconflow_free_only:
            for field, approved in APPROVED_FREE_SILICONFLOW_MODELS.items():
                if getattr(self, field) != approved:
                    raise ValueError(
                        f"{field} must use the approved free SiliconFlow model"
                    )
        if self.rerank_limit > self.retrieval_limit:
            raise ValueError("rerank_limit must not exceed retrieval_limit")
        backend_directory = PROJECT_ROOT / "backend"
        for field in (
            "source_documents_dir",
            "private_data_dir",
            "qdrant_path",
            "sqlite_path",
            "demo_questions_path",
        ):
            value = getattr(self, field)
            resolved = value.resolve() if value.is_absolute() else (backend_directory / value).resolve()
            setattr(self, field, resolved)
        return self

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Serialize without provider credentials, even when not requested."""

        kwargs["exclude"] = _merge_secret_exclude(kwargs.get("exclude"))
        return super().model_dump(*args, **kwargs)

    def model_dump_json(self, *args: Any, **kwargs: Any) -> str:
        """JSON serialization counterpart to :meth:`model_dump`."""

        kwargs["exclude"] = _merge_secret_exclude(kwargs.get("exclude"))
        return super().model_dump_json(*args, **kwargs)

    def provider_status(self, provider: str) -> str:
        """Return a non-sensitive configuration status for one provider."""

        if provider not in {"deepseek", "siliconflow", "minimax"}:
            raise ValueError(f"unknown provider: {provider}")
        value = getattr(self, f"{provider}_api_key")
        if not is_configured_key(value):
            return "optional_missing" if provider == "minimax" else "missing"
        return "configured"

    def cors_origins(self) -> tuple[str, ...]:
        """Return an explicit, non-wildcard CORS origin allow list."""

        origins = tuple(
            origin.strip()
            for origin in self.allowed_origins.split(",")
            if origin.strip() and origin.strip() != "*"
        )
        return origins or DEFAULT_ALLOWED_ORIGINS


def _merge_secret_exclude(exclude: Any) -> Any:
    """Merge mandatory credential exclusions into a Pydantic exclude value."""

    if exclude is None:
        return set(SECRET_FIELDS)
    if isinstance(exclude, Mapping):
        merged = dict(exclude)
        merged.update({field: True for field in SECRET_FIELDS})
        return merged
    if isinstance(exclude, set):
        return exclude | SECRET_FIELDS
    return set(SECRET_FIELDS)


def is_configured_key(value: str) -> bool:
    """Return whether a provider key is non-empty and not an example value."""

    return isinstance(value, str) and value.strip().casefold() not in PLACEHOLDER_KEYS


def require_cloud_keys(settings: Settings) -> None:
    """Fail closed when a required Cloud provider credential is unavailable."""

    if any(
        settings.provider_status(provider) != "configured"
        for provider in ("deepseek", "siliconflow")
    ):
        raise RuntimeError("required cloud providers are not configured")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process configuration for non-request-scoped consumers."""

    return Settings()
