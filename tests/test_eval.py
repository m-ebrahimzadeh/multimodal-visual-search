"""Evaluation metric tests.

Metrics are checked against hand-computed values. A retrieval metric that is
subtly wrong produces a plausible-looking table, so these are worth pinning
precisely rather than asserting "greater than zero".
"""

from __future__ import annotations

import numpy as np
import pytest

from vsearch.encoders.base import l2_normalize
from vsearch.eval import (
    Judged,
    average_precision,
    evaluate,
    evaluate_image_to_image,
    flickr_text_queries,
    label_relevance_groups,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    to_markdown_table,
)
from vsearch.index import FaissStore

DIM = 8


def _judged(ranked: str, relevant: str) -> Judged:
    """Compact helper: 'abc' -> ids ['a','b','c']."""
    return Judged(ranked=list(ranked), relevant=frozenset(relevant))


# --- guards ----------------------------------------------------------------


def test_query_without_relevant_items_is_rejected() -> None:
    """Scoring it as 0 understates the system; as 1 overstates it."""
    with pytest.raises(ValueError, match="no relevant items"):
        Judged(ranked=["a"], relevant=frozenset())


# --- recall ----------------------------------------------------------------


def test_recall_is_a_hit_rate_with_one_relevant_item() -> None:
    """The Flickr30k text->image protocol."""
    assert recall_at_k(_judged("xyza", "a"), 10) == 1.0
    assert recall_at_k(_judged("xyza", "a"), 3) == 0.0
    assert recall_at_k(_judged("axyz", "a"), 1) == 1.0


def test_recall_is_a_fraction_with_many_relevant_items() -> None:
    assert recall_at_k(_judged("abcd", "ac"), 4) == 1.0
    assert recall_at_k(_judged("abcd", "ac"), 2) == 0.5


# --- precision -------------------------------------------------------------


def test_precision_divides_by_k_not_by_results_returned() -> None:
    """A short result list must not score as though it were full."""
    assert precision_at_k(Judged(ranked=["a"], relevant=frozenset("a")), 10) == pytest.approx(0.1)


def test_precision_counts_relevant_in_top_k() -> None:
    assert precision_at_k(_judged("abcd", "ac"), 4) == 0.5


def test_precision_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        precision_at_k(_judged("ab", "a"), 0)


# --- reciprocal rank -------------------------------------------------------


@pytest.mark.parametrize(
    ("ranked", "expected"),
    [("abc", 1.0), ("bac", 0.5), ("bca", 1 / 3), ("bcd", 0.0)],
)
def test_reciprocal_rank(ranked: str, expected: float) -> None:
    assert reciprocal_rank(_judged(ranked, "a")) == pytest.approx(expected)


# --- average precision -----------------------------------------------------


def test_average_precision_hand_computed() -> None:
    """Relevant at ranks 1 and 3: (1/1 + 2/3) / 2."""
    assert average_precision(_judged("abc", "ac")) == pytest.approx((1.0 + 2 / 3) / 2)


def test_average_precision_is_one_for_a_perfect_ranking() -> None:
    assert average_precision(_judged("abcd", "ab")) == pytest.approx(1.0)


def test_average_precision_is_zero_when_nothing_relevant_is_found() -> None:
    assert average_precision(_judged("xyz", "a")) == 0.0


def test_average_precision_normalizes_by_reachable_items() -> None:
    """With 5 relevant items but only 2 results, dividing by 5 would punish a
    system that retrieved everything it possibly could."""
    judged = Judged(ranked=["a", "b"], relevant=frozenset("abcde"))
    assert average_precision(judged) == pytest.approx(1.0)


# --- nDCG ------------------------------------------------------------------


def test_ndcg_is_one_for_a_perfect_ranking() -> None:
    assert ndcg_at_k(_judged("abcd", "ab"), 4) == pytest.approx(1.0)


def test_ndcg_penalizes_lower_placement() -> None:
    top = ndcg_at_k(_judged("abcd", "a"), 4)
    lower = ndcg_at_k(_judged("bcad", "a"), 4)
    assert top == pytest.approx(1.0)
    assert 0 < lower < top


def test_ndcg_hand_computed() -> None:
    """One relevant item at rank 2: (1/log2(3)) / (1/log2(2))."""
    assert ndcg_at_k(_judged("ba", "a"), 2) == pytest.approx(1 / np.log2(3))


# --- aggregation -----------------------------------------------------------


def test_evaluate_averages_over_queries() -> None:
    metrics = evaluate([_judged("abc", "a"), _judged("bca", "a")], ks=(1,))
    assert metrics.queries == 2
    assert metrics.recall[1] == pytest.approx(0.5)
    assert metrics.mrr == pytest.approx((1.0 + 1 / 3) / 2)


def test_evaluate_handles_an_empty_query_set() -> None:
    assert evaluate([]).queries == 0


def test_metrics_serialize_with_string_keys() -> None:
    """JSON object keys must be strings, or ints round-trip as '1'."""
    payload = evaluate([_judged("abc", "a")], ks=(1, 5)).to_dict()
    assert set(payload["recall"]) == {"1", "5"}  # type: ignore[arg-type]


def test_summary_row_has_expected_columns() -> None:
    row = evaluate([_judged("abc", "a")]).summary_row()
    assert {"queries", "R@1", "R@5", "R@10", "MRR", "mAP", "nDCG@10"} <= set(row)


def test_markdown_table_renders_rows() -> None:
    rows = {"clip": evaluate([_judged("abc", "a")]).summary_row()}
    table = to_markdown_table(rows, (1, 5, 10))
    assert "| config |" in table
    assert "| clip |" in table


def test_markdown_table_handles_no_rows() -> None:
    assert to_markdown_table({}, (1,)) == "_no results_"


# --- protocols -------------------------------------------------------------


def _store_with(payloads: list[dict[str, object]]) -> FaissStore:
    store = FaissStore(dim=DIM)
    vectors = l2_normalize(np.eye(DIM, dtype=np.float32)[: len(payloads)])
    store.add([f"item-{i}" for i in range(len(payloads))], vectors, payloads)
    return store


def test_flickr_queries_expand_captions_per_image() -> None:
    store = _store_with(
        [
            {"split": "test", "captions": ["a dog", "a puppy"]},
            {"split": "test", "captions": ["a cat"]},
        ]
    )
    queries = flickr_text_queries(store, split="test")
    assert len(queries) == 3
    assert {q.target_id for q in queries} == {"item-0", "item-1"}


def test_flickr_queries_restrict_to_the_canonical_split() -> None:
    """Evaluating over all 31k images gives a number that looks similar but
    means something else."""
    store = _store_with(
        [
            {"split": "train", "captions": ["a dog"]},
            {"split": "test", "captions": ["a cat"]},
        ]
    )
    queries = flickr_text_queries(store, split="test")
    assert [q.target_id for q in queries] == ["item-1"]


def test_flickr_queries_can_span_all_splits() -> None:
    store = _store_with(
        [{"split": "train", "captions": ["a dog"]}, {"split": "test", "captions": ["a cat"]}]
    )
    assert len(flickr_text_queries(store, split=None)) == 2


def test_relevance_groups_key_on_both_fields() -> None:
    store = _store_with(
        [
            {"articleType": "Shirts", "baseColour": "Blue"},
            {"articleType": "Shirts", "baseColour": "Blue"},
            {"articleType": "Shirts", "baseColour": "Red"},
        ]
    )
    groups = label_relevance_groups(store)
    assert groups[("Shirts", "Blue")] == {"item-0", "item-1"}
    assert groups[("Shirts", "Red")] == {"item-2"}


def test_items_missing_a_relevance_field_are_skipped() -> None:
    store = _store_with([{"articleType": "Shirts"}, {"articleType": "Shirts", "baseColour": "Red"}])
    assert list(label_relevance_groups(store)) == [("Shirts", "Red")]


def test_image_to_image_excludes_the_query_itself() -> None:
    """The query is in the index, so a self-match would inflate every metric."""
    payloads: list[dict[str, object]] = [
        {"articleType": "Shirts", "baseColour": "Blue"} for _ in range(4)
    ]
    store = _store_with(payloads)
    ids = ["item-0", "item-1"]
    judged = list(evaluate_image_to_image(store, ids, store.vectors_for(ids), k=3))

    assert len(judged) == 2
    for query_id, item in zip(ids, judged, strict=True):
        assert query_id not in item.ranked
        assert query_id not in item.relevant


def test_image_to_image_skips_items_with_no_peers() -> None:
    """A unique item is unscoreable, not a failure."""
    store = _store_with(
        [
            {"articleType": "Shirts", "baseColour": "Blue"},
            {"articleType": "Watches", "baseColour": "Silver"},
        ]
    )
    ids = ["item-0", "item-1"]
    assert list(evaluate_image_to_image(store, ids, store.vectors_for(ids), k=2)) == []


def test_vectors_for_round_trips_stored_embeddings() -> None:
    store = _store_with([{"a": 1}, {"a": 2}])
    vectors = store.vectors_for(["item-0", "item-1"])
    assert vectors.shape == (2, DIM)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), np.ones(2), atol=1e-5)


def test_vectors_for_rejects_unknown_ids() -> None:
    store = _store_with([{"a": 1}])
    with pytest.raises(KeyError, match="not in this index"):
        store.vectors_for(["nope"])
