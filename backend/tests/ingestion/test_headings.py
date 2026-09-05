from xml.etree import ElementTree

from app.ingestion.headings import (
    docx_heading_level,
    normalise_heading_title,
    resolve_pdf_toc,
    update_heading_path,
)

_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def test_normalise_heading_title_is_deterministic_and_bounded() -> None:
    assert normalise_heading_title("  Ａ　Ｂ\nＣ  ") == "A B C"
    assert normalise_heading_title("") is None
    assert normalise_heading_title("标题\x00") is None
    assert normalise_heading_title("x" * 201) is None


def test_update_heading_path_keeps_only_known_levels() -> None:
    assert update_heading_path((), 1, "总则") == ("总则",)
    assert update_heading_path(("总则",), 2, "设备") == ("总则", "设备")
    assert update_heading_path(("总则", "设备"), 1, "培训") == ("培训",)
    assert update_heading_path(("总则",), 4, "异常层级") == ("总则", "异常层级")
    assert update_heading_path(("总则",), 0, "无效") is None


def test_resolve_pdf_toc_maps_explicit_entries_to_pages() -> None:
    resolution = resolve_pdf_toc(
        [[1, "总则", 1], [2, "设备管理", 3]],
        page_count=4,
    )

    assert resolution.paths_by_page == (
        ("总则",),
        ("总则",),
        ("总则", "设备管理"),
        ("总则", "设备管理"),
    )
    assert resolution.valid_entry_count == 2
    assert resolution.invalid_entry_count == 0
    assert resolution.ambiguous_page_count == 0


def test_resolve_pdf_toc_fails_closed_for_invalid_and_ambiguous_entries() -> None:
    resolution = resolve_pdf_toc(
        [
            [0, "bad level", 1],
            [1, "", 1],
            [1, "bad page", 0],
            [True, "bool level", 1],
            [1, "第一章", 2],
            [2, "页内小节", 2],
        ],
        page_count=3,
    )

    assert resolution.paths_by_page == (
        ("正文",),
        ("正文",),
        ("第一章", "页内小节"),
    )
    assert resolution.valid_entry_count == 2
    assert resolution.invalid_entry_count == 4
    assert resolution.ambiguous_page_count == 1


def test_docx_heading_level_accepts_only_explicit_xml_signals() -> None:
    heading = ElementTree.fromstring(
        f'<w:p xmlns:w="{_W}"><w:pPr><w:pStyle w:val="Heading2"/></w:pPr></w:p>'
    )
    outline = ElementTree.fromstring(
        f'<w:p xmlns:w="{_W}"><w:pPr><w:outlineLvl w:val="1"/></w:pPr></w:p>'
    )
    visual_only = ElementTree.fromstring(
        f'<w:p xmlns:w="{_W}"><w:pPr><w:rPr><w:b/></w:rPr></w:pPr></w:p>'
    )

    assert docx_heading_level(heading) == 2
    assert docx_heading_level(outline) == 2
    assert docx_heading_level(visual_only) is None
