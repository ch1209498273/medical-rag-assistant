from __future__ import annotations

from pathlib import Path

from app.feedback.redaction import contains_unredacted_identifier


def test_privacy_audit_rejects_provider_and_identifier_fields(sample_case, tmp_path: Path) -> None:
    payload = sample_case.to_dict()
    serialized = str(payload)
    assert "provider_response" not in serialized
    assert "reasoning_content" not in serialized
    assert "api_key" not in serialized
    assert not contains_unredacted_identifier(sample_case.question_redacted)
    assert not contains_unredacted_identifier(sample_case.answer_redacted)
