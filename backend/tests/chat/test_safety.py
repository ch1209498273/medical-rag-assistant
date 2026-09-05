import pytest
from app.chat.safety import is_safe_answer_text, validate_citations


def _windows_path(*parts: str) -> str:
    return "C:" + chr(92) + chr(92).join(parts)


def _unc_path(*parts: str) -> str:
    return chr(92) * 2 + chr(92).join(parts)


def citation_with_excerpt(excerpt: str) -> dict[str, object]:
    return {
        "reference_id": "S1",
        "file_name": "虚构制度.docx",
        "heading_path": ["制度"],
        "page": None,
        "page_end": None,
        "paragraph_start": 1,
        "paragraph_end": 1,
        "excerpt": excerpt,
    }


@pytest.mark.parametrize(
    "text",
    [
        "Kt/V",
        "mL/min",
        "本次透析 Kt/V 为 1.2，血流量为 250 mL/min。",
    ],
)
def test_is_safe_answer_text_allows_dialysis_slash_units(text):
    assert is_safe_answer_text(text)


def test_validate_citations_allows_dialysis_slash_excerpt():
    excerpt = "本次透析 Kt/V 为 1.2，血流量为 250 mL/min。"

    citations = validate_citations([citation_with_excerpt(excerpt)])

    assert citations[0]["excerpt"] == excerpt


def test_validate_citations_accepts_docx_table_locator_without_paragraph_range():
    citation = citation_with_excerpt("表格中的要求")
    citation["paragraph_start"] = None
    citation["paragraph_end"] = None
    citation["table_id"] = "table-7"

    citations = validate_citations([citation])

    assert citations[0]["table_id"] == "table-7"


def test_validate_citations_normalizes_display_whitespace():
    citation = citation_with_excerpt("第一行\r\n第二行\t。")

    citations = validate_citations([citation])

    assert citations[0]["excerpt"] == "第一行 第二行 。"


@pytest.mark.parametrize(
    "excerpt",
    [
        "/" + "private/secret.docx",
        _windows_path("private", "secret.docx"),
        "包含 " + "sk-test-secret" + " 的正文",
        "包含 " + "Bearer-test-secret" + " 的正文",
        "包含 " + "api-key=" + "secret" + " 的正文",
    ],
)
def test_validate_citations_rejects_paths_and_credentials(excerpt):
    with pytest.raises(ValueError):
        validate_citations([citation_with_excerpt(excerpt)])


@pytest.mark.parametrize(
    "text",
    [
        _windows_path("private", "provider.log"),
        "C:" + "/" + "private/provider.log",
        "/" + "var/log/provider.log",
        _unc_path("server", "share", "provider.log"),
    ],
)
def test_is_safe_answer_text_rejects_absolute_paths(text):
    assert not is_safe_answer_text(text)


@pytest.mark.parametrize(
    "text",
    [
        "路径C:" + chr(92) + "private" + chr(92) + "provider.log",
        "xC:" + chr(92) + "private" + chr(92) + "provider.log",
        "路径C:" + "/" + "private/provider.log",
        "xC:" + "/" + "private/provider.log",
    ],
)
def test_is_safe_answer_text_rejects_embedded_windows_drive_paths(text):
    assert not is_safe_answer_text(text)


def test_reference_answer_is_allowed_in_a_separate_field_on_answered_messages():
    from app.chat.models import ChatMessage
    from app.chat.safety import normalize_chat_message

    refused = normalize_chat_message(
        ChatMessage(
            "a1",
            "s1",
            "assistant",
            "依据不足",
            "refused",
            None,
            (),
            "INSUFFICIENT_EVIDENCE",
            "2026-08-26T00:00:00Z",
            "u1",
            "通用学习参考",
        )
    )
    answered = normalize_chat_message(
        {
            "message_id": "a2",
            "session_id": "s1",
            "role": "assistant",
            "content": "正式回答",
            "status": "answered",
            "rewritten_question": None,
            "citations": [],
            "reason_code": None,
            "created_at": "2026-08-26T00:00:00Z",
            "reply_to_message_id": "u2",
            "reference_answer": "单独标注的通用参考",
        }
    )

    assert refused.reference_answer == "通用学习参考"
    assert answered.status == "answered"
    assert answered.content == "正式回答"
    assert answered.reference_answer == "单独标注的通用参考"
