from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
from app.domain.models import Chunk, SourceRef
from app.domain.ports import SearchHit
from app.rag.lexical import ActiveLexicalIndex, tokenize_lexical


def make_hit(
    chunk_id: str,
    text: str,
    *,
    version_id: str = "v1",
    heading_path: tuple[str, ...] = ("制度",),
) -> SearchHit:
    return SearchHit(
        version_id=version_id,
        chunk=Chunk(
            chunk_id=chunk_id,
            text=text,
            source=SourceRef(file_name="虚构制度.docx", heading_path=heading_path),
            content_hash=f"hash-{version_id}-{chunk_id}",
        ),
        score=0.0,
    )


@dataclass
class FakeSnapshotProvider:
    snapshots: dict[frozenset[str], list[SearchHit]]
    calls: list[tuple[set[str], int]] = field(default_factory=list)
    error: Exception | None = None

    def snapshot_active_chunks(
        self, active_versions: set[str], limit: int = 10000
    ) -> list[SearchHit]:
        self.calls.append((set(active_versions), limit))
        if self.error is not None:
            raise self.error
        return list(self.snapshots.get(frozenset(active_versions), ()))


def test_tokenize_lexical_emits_ascii_numeric_words_and_chinese_bigrams() -> None:
    assert tokenize_lexical("CRRT 2.0：血液透析") == (
        "crrt",
        "2",
        "0",
        "血液",
        "液透",
        "透析",
    )


def test_heading_match_has_a_higher_score_than_text_only_match() -> None:
    provider = FakeSnapshotProvider(
        snapshots={
            frozenset({"v1"}): [
                make_hit("heading", "流程说明", heading_path=("透析",)),
                make_hit("text", "透析流程说明", heading_path=("流程",)),
            ]
        }
    )

    results = ActiveLexicalIndex(provider).search(["透析"], {"v1"})

    assert [result.chunk.chunk_id for result in results] == ["heading", "text"]
    assert results[0].score > results[1].score


def test_equal_scores_are_sorted_by_version_then_chunk_id() -> None:
    provider = FakeSnapshotProvider(
        snapshots={
            frozenset({"v1", "v2"}): [
                make_hit("c2", "透析", version_id="v1"),
                make_hit("c1", "透析", version_id="v2"),
                make_hit("c1", "透析", version_id="v1"),
            ]
        }
    )

    results = ActiveLexicalIndex(provider).search(["透析"], {"v1", "v2"})

    assert [(result.version_id, result.chunk.chunk_id) for result in results] == [
        ("v1", "c1"),
        ("v1", "c2"),
        ("v2", "c1"),
    ]


def test_empty_active_versions_returns_no_hits_without_loading_snapshot() -> None:
    provider = FakeSnapshotProvider(snapshots={})

    assert ActiveLexicalIndex(provider).search(["透析"], set()) == []
    assert provider.calls == []


def test_results_only_include_active_versions() -> None:
    provider = FakeSnapshotProvider(
        snapshots={
            frozenset({"v1"}): [
                make_hit("active", "透析", version_id="v1"),
                make_hit("stale", "透析", version_id="v2"),
            ]
        }
    )

    results = ActiveLexicalIndex(provider).search(["透析"], {"v1"})

    assert [(result.version_id, result.chunk.chunk_id) for result in results] == [
        ("v1", "active")
    ]


def test_search_hard_truncates_to_requested_limit() -> None:
    hits = [make_hit(f"c{i}", "透析") for i in range(5)]
    provider = FakeSnapshotProvider(snapshots={frozenset({"v1"}): hits})

    results = ActiveLexicalIndex(provider).search(["透析"], {"v1"}, limit=2)

    assert len(results) == 2


def test_search_has_a_twenty_hit_hard_cap_even_for_a_larger_limit() -> None:
    hits = [make_hit(f"c{i:02d}", "透析") for i in range(25)]
    provider = FakeSnapshotProvider(snapshots={frozenset({"v1"}): hits})

    results = ActiveLexicalIndex(provider).search(["透析"], {"v1"}, limit=100)

    assert len(results) <= 20
    assert [result.chunk.chunk_id for result in results] == [
        f"c{i:02d}" for i in range(20)
    ]


def test_snapshot_is_loaded_once_per_active_set_and_refreshed_on_change() -> None:
    provider = FakeSnapshotProvider(
        snapshots={
            frozenset({"v1"}): [make_hit("one", "透析", version_id="v1")],
            frozenset({"v2"}): [make_hit("two", "透析", version_id="v2")],
        }
    )
    index = ActiveLexicalIndex(provider)

    index.search(["透析"], {"v1"})
    index.search(["透析"], {"v1"})
    index.search(["透析"], {"v2"})
    index.search(["透析"], {"v2"})

    assert provider.calls == [
        ({"v1"}, 10000),
        ({"v2"}, 10000),
    ]


def test_snapshot_over_limit_raises_clear_value_error() -> None:
    provider = FakeSnapshotProvider(
        snapshots={frozenset({"v1"}): [make_hit("c1", "透析"), make_hit("c2", "透析")]}
    )

    with pytest.raises(ValueError, match="snapshot"):
        ActiveLexicalIndex(provider, snapshot_limit=1).search(["透析"], {"v1"})


def test_malformed_snapshot_field_raises_value_error_without_printing_text(
    capsys: pytest.CaptureFixture[str],
) -> None:
    malformed = SearchHit(
        version_id="v1",
        chunk=SimpleNamespace(
            chunk_id="bad",
            text=None,
            source=SourceRef(file_name="虚构制度.docx", heading_path=("制度",)),
            content_hash="hash-bad",
        ),
        score=0.0,
    )
    provider = FakeSnapshotProvider(snapshots={frozenset({"v1"}): [malformed]})

    with pytest.raises(ValueError, match="text"):
        ActiveLexicalIndex(provider).search(["透析"], {"v1"})

    assert capsys.readouterr().out == ""


def test_snapshot_provider_exception_is_wrapped_as_value_error() -> None:
    provider = FakeSnapshotProvider(snapshots={}, error=RuntimeError("private failure"))

    with pytest.raises(ValueError, match="snapshot provider"):
        ActiveLexicalIndex(provider).search(["透析"], {"v1"})
