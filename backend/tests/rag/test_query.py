from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from app.rag.query import QueryVariants, build_query_variants, clean_question
from app.rag.retrieval import clean_question as retrieval_clean_question


def test_clean_question_rejects_empty_input_and_preserves_type_boundary():
    with pytest.raises(TypeError):
        clean_question(None)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        clean_question(" \t\n")


@pytest.mark.parametrize("control", ["\x00", "\x1f", "\x7f"])
def test_clean_question_removes_legacy_control_characters(control: str):
    assert clean_question(f"透析{control}流程") == "透析流程"


def test_clean_question_keeps_the_existing_1000_character_limit():
    assert len(clean_question("问" * 1000)) == 1000
    with pytest.raises(ValueError):
        clean_question("问" * 1001)


def test_normalized_query_unifies_full_width_punctuation_and_whitespace():
    variants = build_query_variants("  透析　，  低血压？  ", use_aliases=False)

    assert variants.original == "透析　，  低血压？"
    assert variants.normalized == "透析, 低血压?"
    assert variants.lexical_queries == (variants.original,)
    assert variants.aliases_applied == ()


def test_aliases_disabled_returns_only_the_cleaned_original_query():
    variants = build_query_variants("血透 HD CRRT", use_aliases=False)

    assert variants.lexical_queries == (variants.original,)
    assert variants.aliases_applied == ()
    assert variants.original == "血透 HD CRRT"


def test_aliases_enabled_retains_original_and_has_stable_order():
    question = "血透 HD CRRT"
    first = build_query_variants(question, use_aliases=True)
    second = build_query_variants(question, use_aliases=True)

    assert first == second
    assert first.lexical_queries[0] == first.original
    assert first.lexical_queries[-1] == "血液透析 血液透析 连续性肾脏替代治疗"
    assert first.aliases_applied == (
        "血透→血液透析",
        "HD→血液透析",
        "CRRT→连续性肾脏替代治疗",
    )


def test_query_variants_is_frozen_and_legacy_import_is_an_alias():
    variants = build_query_variants("透析流程", use_aliases=False)

    assert isinstance(variants, QueryVariants)
    with pytest.raises(FrozenInstanceError):
        variants.original = "另一个问题"  # type: ignore[misc]
    assert retrieval_clean_question is clean_question
