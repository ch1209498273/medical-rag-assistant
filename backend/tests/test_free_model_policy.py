import pytest
from app.settings import Settings
from pydantic import ValidationError


def test_siliconflow_free_only_defaults_to_approved_models() -> None:
    settings = Settings(_env_file=None)
    assert settings.siliconflow_free_only is True
    assert settings.stage_b_verification_enabled is False
    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.reranker_model == "BAAI/bge-reranker-v2-m3"
    assert settings.ocr_model == "PaddlePaddle/PaddleOCR-VL-1.5"


def test_stage_b_verification_can_be_opted_in_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("STAGE_B_VERIFICATION_ENABLED", "true")

    settings = Settings(_env_file=None)

    assert settings.stage_b_verification_enabled is True


def test_retrieval_limit_rejects_values_above_specified_hard_cap() -> None:
    with pytest.raises(ValidationError, match="less than or equal to 20"):
        Settings(_env_file=None, retrieval_limit=21)


def test_large_local_corpus_can_use_bounded_pdf_and_timeout_limits() -> None:
    settings = Settings(
        _env_file=None,
        ingestion_max_pdf_pages=2_500,
        ingestion_file_timeout_seconds=900,
    )

    assert settings.ingestion_max_pdf_pages == 2_500
    assert settings.ingestion_file_timeout_seconds == 900


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ingestion_max_file_bytes", 65 * 1024 * 1024),
        ("ingestion_max_pdf_pages", 3001),
        ("ingestion_max_docx_paragraphs", 100001),
        ("ingestion_max_docx_uncompressed_bytes", 257 * 1024 * 1024),
        ("ingestion_max_docx_zip_entries", 10001),
        ("ingestion_max_text_chars", 5_000_001),
        ("ingestion_file_timeout_seconds", 1801),
        ("ingestion_scan_max_files", 1001),
        ("deepseek_stream_max_bytes", 8 * 1024 * 1024 + 1),
        ("deepseek_stream_max_chars", 1_000_001),
        ("deepseek_stream_max_events", 10_001),
        ("deepseek_stream_timeout_seconds", 301),
    ],
)
def test_security_resource_limits_have_non_expandable_hard_bounds(
    field: str, value: int
) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


@pytest.mark.parametrize(
    ("field", "model"),
    [
        ("embedding_model", "Pro/BAAI/bge-m3"),
        ("reranker_model", "Pro/BAAI/bge-reranker-v2-m3"),
        ("ocr_model", "some-paid-ocr"),
    ],
)
def test_siliconflow_free_only_rejects_unapproved_or_pro_models(
    field: str, model: str
) -> None:
    with pytest.raises(ValidationError, match="approved free SiliconFlow model"):
        Settings(_env_file=None, **{field: model})
