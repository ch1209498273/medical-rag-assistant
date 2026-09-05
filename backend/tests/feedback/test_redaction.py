from __future__ import annotations

from app.feedback.redaction import (
    REDACTOR_VERSION,
    contains_unredacted_identifier,
    redact_interaction,
    redact_text,
)


def test_redacts_typed_identifiers_and_preserves_medical_terms() -> None:
    result = redact_text(
        "患者姓名：张三，手机号 13812345678，身份证 110101199001011234，"
        "病案号 MRN-20260903，住址：珠海市香洲区某路。透析中低血压应先评估。"
    )

    assert result.status == "passed"
    assert result.version == REDACTOR_VERSION
    assert "张三" not in result.text
    assert "13812345678" not in result.text
    assert "110101199001011234" not in result.text
    assert "MRN-20260903" not in result.text
    assert "珠海市香洲区某路" not in result.text
    assert "透析中低血压" in result.text
    assert {"patient_name", "phone", "id_number", "case_number", "address"}.issubset(
        set(result.flags)
    )
    assert not contains_unredacted_identifier(result.text)


def test_redacts_dates_email_and_device_labels() -> None:
    result = redact_text(
        "检查日期：2026年9月3日，邮箱 user@example.com，床号 A12，"
        "透析机编号 HD-009。"
    )

    assert result.status == "passed"
    assert "2026年9月3日" not in result.text
    assert "user@example.com" not in result.text
    assert "A12" not in result.text
    assert "HD-009" not in result.text
    assert {"date", "email", "bed_number", "device_number"}.issubset(set(result.flags))


def test_same_entity_uses_same_placeholder_within_one_interaction() -> None:
    question_result, answer_result, flags = redact_interaction(
        "患者姓名：张三的透析参数是什么？",
        "患者姓名：张三目前应由护士李四按制度复核。",
    )

    assert question_result.status == "passed"
    assert answer_result.status == "passed"
    assert "张三" not in question_result.text + answer_result.text
    assert question_result.text.count("[患者-1]") == 1
    assert answer_result.text.count("[患者-1]") == 1
    assert "[人员-1]" in answer_result.text
    assert "patient_name" in flags
    assert "staff_name" in flags


def test_redacts_pharmacist_names_as_staff_identifiers() -> None:
    result = redact_text("药师姓名：王芳应按制度复核药品。")

    assert result.status == "passed"
    assert "王芳" not in result.text
    assert "[人员-1]" in result.text
    assert "staff_name" in result.flags


def test_attachment_and_encoding_markers_fail_closed() -> None:
    attachment = redact_text("请查看附件：患者检查结果.jpg")
    invalid = redact_text("文本含无法解析字符 �")

    assert attachment.status == "blocked"
    assert invalid.status in {"review_required", "blocked"}
    assert attachment.text != "患者检查结果.jpg"
    assert invalid.text != "文本含无法解析字符 �"
    assert not contains_unredacted_identifier(attachment.text)


def test_empty_non_text_and_unresolved_identifier_never_pass() -> None:
    assert redact_text("").status in {"review_required", "blocked"}
    assert redact_text(None).status in {"review_required", "blocked"}  # type: ignore[arg-type]
    unresolved = redact_text("请处理这条疑似个人标识 AB123456789012345678")
    assert unresolved.status in {"review_required", "blocked"}
    assert not contains_unredacted_identifier(unresolved.text)


def test_token_like_url_is_removed() -> None:
    result = redact_text("联系 https://example.com/reset?token=abc123XYZ987")

    assert result.status == "passed"
    assert "abc123XYZ987" not in result.text
    assert "token" in result.flags or "contact" in result.flags
