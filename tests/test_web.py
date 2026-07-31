"""Static-bundle export and deployment-verification tests.

The bundle's whole reason to exist is that the deployed demo ranks by the same
vectors the evaluation tables were measured on. So the central test here is not
that the files appear -- it is that scoring the exported bytes reproduces
FAISS's ranking exactly. Everything else guards a way that guarantee can be
quietly lost: a byte order, a row order, a NaN that makes the JSON unparseable,
a deployment left serving an older export.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from vsearch.encoders.base import l2_normalize
from vsearch.index import FaissStore
from vsearch.web import DEFAULT_EXAMPLES, export_bundle, verify_deployment
from vsearch.web.export import (
    ATTRIBUTION_FILENAME,
    CORPUS_FILENAME,
    EMBEDDINGS_FILENAME,
    EXAMPLES_FILENAME,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

DIM = 16


class _StubEncoder:
    """Deterministic stand-in for a text encoder.

    The export only needs `encode_text`; loading real CLIP weights to check
    that six strings become six rows would make this suite download a model.
    """

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim
        self.seen: list[str] = []

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        self.seen.extend(texts)
        rng = np.random.default_rng(len(texts))
        return l2_normalize(rng.normal(size=(len(texts), self.dim)))


def _run_dir(
    tmp_path: Path,
    *,
    n: int = 12,
    with_images: bool = True,
    payloads: list[dict[str, object]] | None = None,
) -> tuple[Path, FaissStore, np.ndarray]:
    rng = np.random.default_rng(0)
    vectors = l2_normalize(rng.normal(size=(n, DIM)))
    ids = [f"item-{i}" for i in range(n)]
    resolved = payloads or [
        {"productDisplayName": f"Product {i}", "baseColour": ["Red", "Blue"][i % 2]}
        for i in range(n)
    ]

    store = FaissStore(dim=DIM)
    store.add(ids, vectors, resolved)

    run_dir = tmp_path / "run"
    store.save(run_dir / "index")
    (run_dir / "manifest.json").write_text(
        json.dumps({"corpus": "demo", "encoder": "clip", "model_id": "openai/clip-test"}),
        encoding="utf-8",
    )

    if with_images:
        images = run_dir / "images"
        images.mkdir(parents=True, exist_ok=True)
        for identifier in ids:
            (images / f"{identifier}.jpg").write_bytes(b"\xff\xd8\xff\xd9")

    return run_dir, store, vectors


def _load_bundle(destination: Path) -> tuple[dict[str, object], np.ndarray]:
    corpus = json.loads((destination / CORPUS_FILENAME).read_text(encoding="utf-8"))
    raw = np.frombuffer((destination / EMBEDDINGS_FILENAME).read_bytes(), dtype="<f4")
    return corpus, raw.reshape(int(corpus["count"]), int(corpus["dim"]))


# --- the guarantee ---------------------------------------------------------


def test_exported_vectors_reproduce_faiss_ranking(tmp_path: Path) -> None:
    """Scoring the bundle must give the same order as searching the index.

    This is the property the deployed demo rests on. If it ever fails, the page
    still renders a ranked list -- just a different one from the metrics.
    """
    run_dir, store, _ = _run_dir(tmp_path, n=40)
    export_bundle(run_dir, tmp_path / "out", encoder=None)
    corpus, matrix = _load_bundle(tmp_path / "out")

    ids = [item["id"] for item in corpus["items"]]  # type: ignore[index]
    rng = np.random.default_rng(7)
    for _ in range(10):
        query = l2_normalize(rng.normal(size=(1, DIM))).astype(np.float32)
        from_bundle = [ids[i] for i in np.argsort(-(matrix @ query[0]))[:5]]
        from_faiss = [hit.id for hit in store.search(query, k=5)[0]]
        assert from_bundle == from_faiss


def test_row_order_is_the_join_key(tmp_path: Path) -> None:
    """corpus.json row i must describe embeddings.bin row i.

    Nothing in the binary identifies a row, so a reordering here would attach
    every label to the wrong vector without any file failing to parse.
    """
    run_dir, store, vectors = _run_dir(tmp_path, n=8)
    export_bundle(run_dir, tmp_path / "out", encoder=None)
    corpus, matrix = _load_bundle(tmp_path / "out")

    for position, item in enumerate(corpus["items"]):  # type: ignore[arg-type]
        assert item["id"] == store.ids()[position]
        np.testing.assert_allclose(matrix[position], vectors[position], atol=1e-6)


def test_embeddings_are_little_endian_float32(tmp_path: Path) -> None:
    """The width and byte order are a wire format, not a local convention."""
    run_dir, _, vectors = _run_dir(tmp_path, n=4)
    export_bundle(run_dir, tmp_path / "out", encoder=None)

    payload = (tmp_path / "out" / EMBEDDINGS_FILENAME).read_bytes()
    assert len(payload) == 4 * DIM * 4
    np.testing.assert_allclose(
        np.frombuffer(payload, dtype="<f4").reshape(4, DIM), vectors, atol=1e-6
    )


def test_rejects_unnormalised_vectors(tmp_path: Path) -> None:
    """A dot product over non-unit rows is not a cosine, and would not error."""
    rng = np.random.default_rng(0)
    store = FaissStore(dim=DIM)
    store.add(["a", "b"], (3.0 * rng.normal(size=(2, DIM))).astype(np.float32))
    run_dir = tmp_path / "run"
    store.save(run_dir / "index")

    with pytest.raises(ValueError, match="not unit-norm"):
        export_bundle(run_dir, tmp_path / "out", encoder=None)


def test_rejects_an_empty_index(tmp_path: Path) -> None:
    store = FaissStore(dim=DIM)
    run_dir = tmp_path / "run"
    store.save(run_dir / "index")

    with pytest.raises(ValueError, match="no vectors"):
        export_bundle(run_dir, tmp_path / "out", encoder=None)


# --- JSON validity ---------------------------------------------------------


def test_nan_payloads_become_null(tmp_path: Path) -> None:
    """`json.dumps` emits a bare NaN, which `JSON.parse` rejects outright.

    The fashion corpus has a float `year` with gaps, so one missing value would
    otherwise make the entire corpus file unparseable in the browser.
    """
    payloads: list[dict[str, object]] = [
        {"productDisplayName": "With year", "year": 2011.0},
        {"productDisplayName": "Missing year", "year": float("nan")},
    ]
    run_dir, _, _ = _run_dir(tmp_path, n=2, payloads=payloads)
    export_bundle(run_dir, tmp_path / "out", encoder=None)

    text = (tmp_path / "out" / CORPUS_FILENAME).read_text(encoding="utf-8")
    assert "NaN" not in text
    corpus = json.loads(text)  # would raise on a bare NaN
    assert corpus["items"][1]["payload"]["year"] is None


def test_nan_facet_values_become_null(tmp_path: Path) -> None:
    """Facets are serialised separately from payloads and need the same guard."""
    payloads: list[dict[str, object]] = [
        {"productDisplayName": "With year", "year": 2011.0},
        {"productDisplayName": "Missing year", "year": float("nan")},
    ]
    run_dir, _, _ = _run_dir(tmp_path, n=2, payloads=payloads)
    export_bundle(run_dir, tmp_path / "out", encoder=None)

    corpus = json.loads((tmp_path / "out" / CORPUS_FILENAME).read_text(encoding="utf-8"))
    assert all(value is None or np.isfinite(value) for value in corpus["facets"].get("year", []))


# --- assets ----------------------------------------------------------------


def test_thumbnails_are_copied_and_referenced_bundle_relative(tmp_path: Path) -> None:
    run_dir, _, _ = _run_dir(tmp_path, n=5)
    bundle = export_bundle(run_dir, tmp_path / "out", encoder=None)

    assert bundle.images == 5
    corpus, _ = _load_bundle(tmp_path / "out")
    for item in corpus["items"]:  # type: ignore[arg-type]
        # Relative to the bundle, not to the page: the page knows where it
        # mounted the bundle, the bundle does not.
        assert item["image"] == f"images/{item['id']}.jpg"
        assert (tmp_path / "out" / str(item["image"])).exists()


def test_missing_thumbnails_are_reported_not_fatal(tmp_path: Path) -> None:
    run_dir, _, _ = _run_dir(tmp_path, n=3, with_images=False)
    bundle = export_bundle(run_dir, tmp_path / "out", encoder=None)

    assert bundle.images == 0
    corpus, _ = _load_bundle(tmp_path / "out")
    assert all(item["image"] is None for item in corpus["items"])  # type: ignore[arg-type]


def test_attribution_names_the_source(tmp_path: Path) -> None:
    run_dir, _, _ = _run_dir(tmp_path, n=2)
    export_bundle(run_dir, tmp_path / "out", encoder=None, source="https://example.invalid/dataset")
    text = (tmp_path / "out" / ATTRIBUTION_FILENAME).read_text(encoding="utf-8")
    assert "https://example.invalid/dataset" in text


def test_title_falls_back_to_the_id_without_a_text_field(tmp_path: Path) -> None:
    run_dir, _, _ = _run_dir(
        tmp_path, n=2, payloads=[{"baseColour": "Red"}, {"baseColour": "Blue"}]
    )
    export_bundle(run_dir, tmp_path / "out", encoder=None)

    corpus, _ = _load_bundle(tmp_path / "out")
    assert corpus["title_field"] is None
    assert [item["title"] for item in corpus["items"]] == ["item-0", "item-1"]  # type: ignore[arg-type]


# --- examples --------------------------------------------------------------


def test_examples_are_embedded_when_an_encoder_is_given(tmp_path: Path) -> None:
    run_dir, _, _ = _run_dir(tmp_path, n=4)
    encoder = _StubEncoder()
    bundle = export_bundle(run_dir, tmp_path / "out", encoder=encoder)  # type: ignore[arg-type]

    assert bundle.examples == len(DEFAULT_EXAMPLES)
    assert encoder.seen == list(DEFAULT_EXAMPLES)

    payload = json.loads((tmp_path / "out" / EXAMPLES_FILENAME).read_text(encoding="utf-8"))
    assert [entry["text"] for entry in payload["examples"]] == list(DEFAULT_EXAMPLES)
    assert all(len(entry["vector"]) == DIM for entry in payload["examples"])


def test_examples_are_omitted_without_an_encoder(tmp_path: Path) -> None:
    """The export must run on a machine holding no model weights."""
    run_dir, _, _ = _run_dir(tmp_path, n=4)
    bundle = export_bundle(run_dir, tmp_path / "out", encoder=None)

    assert bundle.examples == 0
    payload = json.loads((tmp_path / "out" / EXAMPLES_FILENAME).read_text(encoding="utf-8"))
    assert payload["examples"] == []


def test_example_vectors_are_usable_for_ranking(tmp_path: Path) -> None:
    """A precomputed vector must rank identically to the same vector re-encoded.

    These are the demo's instant path and its parity ground truth. If they
    disagree with the index, the page answers example queries with one ranking
    and free text with another, and measures the encoder against the wrong
    baseline.
    """
    run_dir, store, _ = _run_dir(tmp_path, n=20)
    export_bundle(run_dir, tmp_path / "out", encoder=_StubEncoder())  # type: ignore[arg-type]

    corpus, matrix = _load_bundle(tmp_path / "out")
    ids = [item["id"] for item in corpus["items"]]  # type: ignore[index]
    payload = json.loads((tmp_path / "out" / EXAMPLES_FILENAME).read_text(encoding="utf-8"))

    for entry in payload["examples"]:
        query = np.asarray(entry["vector"], dtype=np.float32)[None, :]
        assert [ids[i] for i in np.argsort(-(matrix @ query[0]))[:3]] == [
            hit.id for hit in store.search(query, k=3)[0]
        ]


# --- deployment verification -----------------------------------------------
#
# `verify_deployment` fetches over HTTP. These stub the fetch with a reader over
# a real exported bundle on disk, so what gets verified is the actual file
# format rather than a hand-built fixture that could drift from the exporter.


def _serve(bundle: Path, monkeypatch: pytest.MonkeyPatch, *, corrupt: object = None) -> None:
    """Point `verify_deployment`'s fetch at a local bundle directory."""
    from vsearch.web import verify

    def fake_fetch(url: str) -> bytes:
        name = url.rsplit("/", 1)[-1]
        if corrupt is not None and name == corrupt[0]:  # type: ignore[index]
            return corrupt[1]  # type: ignore[index]
        return (bundle / name).read_bytes()

    monkeypatch.setattr(verify, "_fetch", fake_fetch)


def test_a_faithful_deployment_verifies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir, _, _ = _run_dir(tmp_path, n=24)
    export_bundle(run_dir, tmp_path / "out", encoder=None)
    _serve(tmp_path / "out", monkeypatch)

    deployment = verify_deployment("https://example.invalid", run_dir, probes=5)

    assert deployment.ok
    assert deployment.identical
    assert deployment.ids_match
    assert deployment.max_abs_difference == 0.0
    assert deployment.mismatches == 0


def test_stale_vectors_are_caught(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A deployment left serving an older export still parses and still ranks.

    This is the failure the command exists for: nothing about it is visible on
    the page, which renders a plausible grid of the wrong results.
    """
    run_dir, _, _ = _run_dir(tmp_path, n=16)
    export_bundle(run_dir, tmp_path / "out", encoder=None)

    stale = l2_normalize(np.random.default_rng(99).normal(size=(16, DIM)))
    _serve(
        tmp_path / "out",
        monkeypatch,
        corrupt=(EMBEDDINGS_FILENAME, stale.astype("<f4").tobytes()),
    )

    deployment = verify_deployment("https://example.invalid", run_dir, probes=5)

    assert not deployment.ok
    assert not deployment.identical
    assert deployment.max_abs_difference > 0
    assert deployment.mismatches == 5


def test_reordered_ids_are_caught(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Row order is the join key, so an id list out of index order is fatal."""
    run_dir, _, _ = _run_dir(tmp_path, n=8)
    export_bundle(run_dir, tmp_path / "out", encoder=None)

    corpus = json.loads((tmp_path / "out" / CORPUS_FILENAME).read_text(encoding="utf-8"))
    corpus["items"].reverse()
    _serve(
        tmp_path / "out",
        monkeypatch,
        corrupt=(CORPUS_FILENAME, json.dumps(corpus).encode("utf-8")),
    )

    deployment = verify_deployment("https://example.invalid", run_dir, probes=3)

    assert not deployment.ids_match
    assert not deployment.ok


def test_a_truncated_upload_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Half an embeddings.bin would otherwise reshape into a silently wrong index."""
    run_dir, _, _ = _run_dir(tmp_path, n=10)
    export_bundle(run_dir, tmp_path / "out", encoder=None)
    truncated = (tmp_path / "out" / EMBEDDINGS_FILENAME).read_bytes()[: 5 * DIM * 4]
    _serve(tmp_path / "out", monkeypatch, corrupt=(EMBEDDINGS_FILENAME, truncated))

    with pytest.raises(RuntimeError, match="bytes but"):
        verify_deployment("https://example.invalid", run_dir)


def test_a_deployment_of_a_different_run_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Comparing against the wrong run directory must say so, not crash."""
    deployed, _, _ = _run_dir(tmp_path / "a", n=12)
    export_bundle(deployed, tmp_path / "out", encoder=None)
    other, _, _ = _run_dir(tmp_path / "b", n=20)
    _serve(tmp_path / "out", monkeypatch)

    with pytest.raises(RuntimeError, match="re-export and redeploy"):
        verify_deployment("https://example.invalid", other)


def test_a_non_http_url_is_rejected(tmp_path: Path) -> None:
    run_dir, _, _ = _run_dir(tmp_path, n=4)
    with pytest.raises(ValueError, match="http"):
        verify_deployment("file:///etc/passwd", run_dir)
