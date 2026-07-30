"""Vector store tests.

Uses clustered synthetic vectors rather than uniform random ones: real
embeddings are clustered, and uniform-random high-dimensional data is an
unrepresentatively hard case for a graph index like HNSW.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vsearch.encoders.base import l2_normalize
from vsearch.index import FaissStore, SearchHit

DIM = 32


def _clustered(n: int, dim: int = DIM, clusters: int = 12, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(clusters, dim))
    assignment = rng.integers(0, clusters, size=n)
    vectors = centres[assignment] + 0.15 * rng.normal(size=(n, dim))
    return l2_normalize(vectors)


def _store(n: int = 200, backend: str = "flat", seed: int = 0) -> tuple[FaissStore, np.ndarray]:
    vectors = _clustered(n, seed=seed)
    store = FaissStore(dim=DIM, backend=backend)  # type: ignore[arg-type]
    payloads = [
        {
            "colour": ["Red", "Blue", "Green"][i % 3],
            "gender": ["Men", "Women"][i % 2],
            "position": i,
        }
        for i in range(n)
    ]
    store.add([f"item-{i}" for i in range(n)], vectors, payloads)
    return store, vectors


# --- construction ----------------------------------------------------------


def test_rejects_non_positive_dim() -> None:
    with pytest.raises(ValueError, match="dim must be positive"):
        FaissStore(dim=0)


def test_len_and_repr() -> None:
    store, _ = _store(50)
    assert len(store) == 50
    assert "n=50" in repr(store)


def test_empty_store_returns_empty_results() -> None:
    store = FaissStore(dim=DIM)
    results = store.search(_clustered(3), k=5)
    assert results == [[], [], []]


# --- validation ------------------------------------------------------------


def test_add_rejects_dim_mismatch() -> None:
    store = FaissStore(dim=DIM)
    with pytest.raises(ValueError, match="does not match index dim"):
        store.add(["a"], l2_normalize(np.ones((1, DIM + 1))))


def test_add_rejects_id_count_mismatch() -> None:
    store = FaissStore(dim=DIM)
    with pytest.raises(ValueError, match="ids for"):
        store.add(["a", "b"], _clustered(3))


def test_add_rejects_payload_count_mismatch() -> None:
    store = FaissStore(dim=DIM)
    with pytest.raises(ValueError, match="payloads for"):
        store.add(["a", "b"], _clustered(2), [{"x": 1}])


def test_add_rejects_duplicate_ids() -> None:
    """Duplicates would make id->position ambiguous and corrupt lookups."""
    store = FaissStore(dim=DIM)
    store.add(["a", "b"], _clustered(2))
    with pytest.raises(ValueError, match="must be unique"):
        store.add(["b"], _clustered(1))


def test_search_rejects_bad_k() -> None:
    store, _ = _store(10)
    with pytest.raises(ValueError, match="k must be positive"):
        store.search(_clustered(1), k=0)


def test_search_rejects_dim_mismatch() -> None:
    store, _ = _store(10)
    with pytest.raises(ValueError, match="query dim"):
        store.search(l2_normalize(np.ones((1, DIM + 3))), k=3)


# --- exact search ----------------------------------------------------------


def test_self_query_returns_itself_first() -> None:
    store, vectors = _store(200)
    results = store.search(vectors[7:8], k=5)
    assert results[0][0].id == "item-7"
    assert results[0][0].score == pytest.approx(1.0, abs=1e-4)


def test_scores_are_cosine_and_descending() -> None:
    store, vectors = _store(200)
    hits = store.search(vectors[3:4], k=10)[0]
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0001 <= s <= 1.0001 for s in scores)


def test_flat_matches_brute_force_numpy() -> None:
    """IndexFlatIP on normalised vectors must equal an argmax over dot products."""
    store, vectors = _store(300)
    query = vectors[42:43]
    expected = np.argsort(-(vectors @ query[0]))[:5]
    actual = [int(hit.payload["position"]) for hit in store.search(query, k=5)[0]]
    assert actual == expected.tolist()


def test_payloads_are_returned_with_hits() -> None:
    store, vectors = _store(50)
    hit = store.search(vectors[0:1], k=1)[0][0]
    assert hit.payload["colour"] in {"Red", "Blue", "Green"}


def test_multiple_queries_return_one_row_each() -> None:
    store, vectors = _store(100)
    results = store.search(vectors[:4], k=3)
    assert len(results) == 4
    assert all(len(row) == 3 for row in results)


# --- the -1 padding trap ---------------------------------------------------


def test_k_larger_than_corpus_is_not_padded() -> None:
    """FAISS pads short rows with id=-1/score=-inf; those must never leak."""
    store, vectors = _store(5)
    hits = store.search(vectors[0:1], k=50)[0]
    assert len(hits) == 5
    assert all(hit.id.startswith("item-") for hit in hits)
    assert all(np.isfinite(hit.score) for hit in hits)


def test_selective_filter_is_not_padded() -> None:
    store, vectors = _store(60)
    hits = store.search(vectors[0:1], k=50, where={"position": 3})[0]
    assert len(hits) == 1
    assert hits[0].id == "item-3"
    assert np.isfinite(hits[0].score)


# --- filtering -------------------------------------------------------------


def test_filter_restricts_to_matching_items() -> None:
    store, vectors = _store(120)
    hits = store.search(vectors[0:1], k=10, where={"colour": "Red"})[0]
    assert hits
    assert all(hit.payload["colour"] == "Red" for hit in hits)


def test_filter_values_are_ored_within_a_field() -> None:
    store, vectors = _store(120)
    hits = store.search(vectors[0:1], k=20, where={"colour": ["Red", "Blue"]})[0]
    assert {hit.payload["colour"] for hit in hits} <= {"Red", "Blue"}
    assert len(hits) == 20


def test_filter_fields_are_anded() -> None:
    store, vectors = _store(120)
    hits = store.search(vectors[0:1], k=10, where={"colour": "Red", "gender": "Men"})[0]
    assert all(h.payload["colour"] == "Red" and h.payload["gender"] == "Men" for h in hits)


def test_prefilter_still_fills_k() -> None:
    """The point of pre-filtering: a selective facet still returns k results.

    Post-filtering an unfiltered top-k would return roughly k/3 here.
    """
    store, vectors = _store(300)
    hits = store.search(vectors[0:1], k=25, where={"colour": "Red"})[0]
    assert len(hits) == 25


def test_filter_matching_nothing_returns_empty() -> None:
    store, vectors = _store(60)
    assert store.search(vectors[0:1], k=5, where={"colour": "Chartreuse"})[0] == []


def test_unknown_filter_field_raises() -> None:
    """A typo'd facet must be loud, not silently match everything."""
    store, vectors = _store(60)
    with pytest.raises(KeyError, match="cannot filter on"):
        store.search(vectors[0:1], k=5, where={"colr": "Red"})


def test_empty_filter_behaves_like_no_filter() -> None:
    store, vectors = _store(80)
    assert len(store.search(vectors[0:1], k=7, where={})[0]) == 7


# --- facets ----------------------------------------------------------------


def test_filterable_fields_are_discovered() -> None:
    store, _ = _store(20)
    assert set(store.filterable_fields) == {"colour", "gender", "position"}


def test_facet_values_are_listed() -> None:
    store, _ = _store(20)
    assert store.facet_values("colour") == ["Blue", "Green", "Red"]


def test_unknown_facet_lookup_raises() -> None:
    store, _ = _store(20)
    with pytest.raises(KeyError, match="unknown facet"):
        store.facet_values("nope")


def test_list_valued_payloads_are_stored_but_not_filterable() -> None:
    """Flickr30k carries five captions per image: text, not a facet."""
    store = FaissStore(dim=DIM)
    store.add(["a"], _clustered(1), [{"captions": ["a dog", "a cat"], "split": "test"}])
    assert store.filterable_fields == ["split"]
    assert store.get("a") is not None
    assert store.get("a").payload["captions"] == ["a dog", "a cat"]  # type: ignore[union-attr]


def test_nan_values_are_not_indexed_as_facets() -> None:
    """NaN != NaN, so each would become its own facet key."""
    store = FaissStore(dim=DIM)
    store.add(
        ["a", "b"],
        _clustered(2),
        [{"year": float("nan"), "usage": "Casual"}, {"year": 2012.0, "usage": "Casual"}],
    )
    assert store.facet_values("year") == [2012.0]


def test_excluded_fields_are_stored_but_not_faceted() -> None:
    """Free-text columns would build one singleton set per row."""
    store = FaissStore(dim=DIM, facet_exclude=["productDisplayName"])
    store.add(
        ["a"],
        _clustered(1),
        [{"productDisplayName": "Titan Women Silver Watch", "baseColour": "Silver"}],
    )
    assert store.filterable_fields == ["baseColour"]
    hit = store.get("a")
    assert hit is not None
    assert hit.payload["productDisplayName"] == "Titan Women Silver Watch"


def test_facet_exclusion_survives_save_load(tmp_path: Path) -> None:
    store = FaissStore(dim=DIM, facet_exclude=["name"])
    store.add(["a"], _clustered(1), [{"name": "unique thing", "colour": "Red"}])
    store.save(tmp_path / "idx")

    restored = FaissStore.load(tmp_path / "idx")
    assert restored.filterable_fields == ["colour"]


def test_none_values_are_not_indexed_as_facets() -> None:
    store = FaissStore(dim=DIM)
    store.add(["a"], _clustered(1), [{"season": None, "usage": "Casual"}])
    assert store.filterable_fields == ["usage"]


# --- get -------------------------------------------------------------------


def test_get_returns_hit_for_known_id() -> None:
    store, _ = _store(20)
    hit = store.get("item-5")
    assert isinstance(hit, SearchHit)
    assert hit.payload["position"] == 5


def test_get_returns_none_for_unknown_id() -> None:
    store, _ = _store(20)
    assert store.get("missing") is None


# --- HNSW ------------------------------------------------------------------


def test_hnsw_recall_against_exact() -> None:
    """The approximate index must stay close to ground truth."""
    vectors = _clustered(2000, seed=1)
    ids = [f"item-{i}" for i in range(len(vectors))]

    exact = FaissStore(dim=DIM, backend="flat")
    exact.add(ids, vectors)
    approx = FaissStore(dim=DIM, backend="hnsw")
    approx.add(ids, vectors)

    queries = vectors[:50]
    k = 10
    overlap = 0
    for exact_row, approx_row in zip(
        exact.search(queries, k=k), approx.search(queries, k=k), strict=True
    ):
        overlap += len({h.id for h in exact_row} & {h.id for h in approx_row})
    recall = overlap / (len(queries) * k)
    assert recall >= 0.95, f"HNSW recall@{k} was {recall:.3f}"


def test_hnsw_supports_filtering() -> None:
    store, vectors = _store(400, backend="hnsw")
    hits = store.search(vectors[0:1], k=10, where={"colour": "Red"})[0]
    assert hits
    assert all(hit.payload["colour"] == "Red" for hit in hits)


def test_hnsw_ef_search_is_raised_to_k() -> None:
    """efSearch below k cannot return k candidates, so it is raised.

    Asserted on the parameters rather than on a result count: FAISS builds
    HNSW graphs with OpenMP threading, so edge insertion order varies between
    runs and an exact result count at a high k/N ratio is genuinely flaky.
    """
    store, _ = _store(100, backend="hnsw")
    assert store._search_params(200).efSearch >= 200
    assert store._search_params(8).efSearch >= 8


def test_hnsw_fills_k_at_a_realistic_ratio() -> None:
    """The regime the demo actually runs in: k far smaller than the corpus."""
    store, vectors = _store(2000, backend="hnsw")
    assert len(store.search(vectors[0:1], k=50)[0]) == 50


# --- persistence -----------------------------------------------------------


@pytest.mark.parametrize("backend", ["flat", "hnsw"])
def test_save_load_round_trip(tmp_path: Path, backend: str) -> None:
    store, vectors = _store(150, backend=backend)
    before = store.search(vectors[:3], k=5)

    store.save(tmp_path / "idx")
    restored = FaissStore.load(tmp_path / "idx")

    assert len(restored) == len(store)
    assert restored.dim == store.dim
    assert restored.backend == store.backend
    assert restored.filterable_fields == store.filterable_fields

    after = restored.search(vectors[:3], k=5)
    for row_before, row_after in zip(before, after, strict=True):
        assert [h.id for h in row_before] == [h.id for h in row_after]
        assert [round(h.score, 5) for h in row_before] == [round(h.score, 5) for h in row_after]


def test_loaded_store_supports_filtering(tmp_path: Path) -> None:
    """Facets are rebuilt on load rather than serialised."""
    store, vectors = _store(120)
    store.save(tmp_path / "idx")
    restored = FaissStore.load(tmp_path / "idx")
    hits = restored.search(vectors[0:1], k=10, where={"colour": "Red"})[0]
    assert hits
    assert all(hit.payload["colour"] == "Red" for hit in hits)


def test_load_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no index found"):
        FaissStore.load(tmp_path / "absent")


def test_load_detects_sidecar_mismatch(tmp_path: Path) -> None:
    """A truncated sidecar must fail loudly, not silently misalign ids."""
    store, _ = _store(40)
    directory = tmp_path / "idx"
    store.save(directory)

    sidecar = directory / "store.jsonl"
    lines = sidecar.read_text(encoding="utf-8").splitlines()
    sidecar.write_text("\n".join(lines[:-5]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="corrupt"):
        FaissStore.load(directory)
