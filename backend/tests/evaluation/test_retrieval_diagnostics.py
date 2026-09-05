from __future__ import annotations

from app.evaluation.retrieval_diagnostics import (
    classify_expected_sources,
    inspect_chunk_integrity,
    lexical_rank,
    summarize_c1_by_type,
)


def test_classify_expected_sources_separates_exact_alias_and_missing() -> None:
    result = classify_expected_sources(
        ["exact", "alias-key", "missing", "exact"],
        {"exact", "resolved"},
        aliases={"alias-key": "resolved"},
    )

    assert result.exact == ("exact",)
    assert result.alias == ("alias-key",)
    assert result.missing == ("missing",)


def test_inspect_chunk_integrity_reports_duplicates_and_blank_metadata() -> None:
    chunks = [
        {"chunk_id": "a", "text": "内容", "content_hash": "h1", "file_name": "a.pdf", "heading_path": ["一"]},
        {"chunk_id": "b", "text": "内容", "content_hash": "h1", "file_name": "", "heading_path": []},
        {"chunk_id": "b", "text": "", "content_hash": "h2", "file_name": "b.pdf", "heading_path": ["二"]},
    ]

    result = inspect_chunk_integrity(chunks)

    assert result["total_chunks"] == 3
    assert result["unique_chunk_ids"] == 2
    assert result["duplicate_chunk_id_count"] == 1
    assert result["duplicate_content_hash_count"] == 1
    assert result["blank_text_count"] == 1
    assert result["missing_file_name_count"] == 1
    assert result["missing_heading_count"] == 1


def test_lexical_rank_is_deterministic_and_prefers_matching_chinese_terms() -> None:
    chunks = [
        {"chunk_id": "unrelated", "text": "员工培训签到流程", "content_hash": "u"},
        {"chunk_id": "target", "text": "透析患者测量血压，每小时记录一次", "content_hash": "t"},
        {"chunk_id": "other", "text": "透析患者营养指导", "content_hash": "o"},
    ]

    first = lexical_rank("透析患者血压频率", chunks, limit=2)
    second = lexical_rank("透析患者血压频率", chunks, limit=2)

    assert first == second
    assert first[0] == "target"


def test_summarize_c1_by_type_returns_counts_only() -> None:
    report = {
        "results": [
            {
                "type": "fact",
                "answered": True,
                "refused": False,
                "no_answer": False,
                "reference_points_covered": True,
                "citation_valid": True,
                "source_visibility": True,
                "expected_source_ids": ["a"],
                "candidate_source_ids": ["a", "x"],
                "evidence_source_ids": ["a"],
                "evidence_count": 1,
            },
            {
                "type": "no_answer",
                "answered": True,
                "refused": False,
                "no_answer": True,
                "reference_points_covered": True,
                "citation_valid": True,
                "source_visibility": True,
                "expected_source_ids": ["n"],
                "candidate_source_ids": ["n"],
                "evidence_source_ids": ["n"],
                "evidence_count": 1,
            },
        ]
    }

    result = summarize_c1_by_type(report)

    assert result["fact"]["answer_availability"]["numerator"] == 1
    assert result["fact"]["answer_availability"]["denominator"] == 1
    assert result["no_answer"]["negative_evidence_rate"]["numerator"] == 1
    assert "question" not in str(result)
    assert "candidate_source_ids" not in str(result)
