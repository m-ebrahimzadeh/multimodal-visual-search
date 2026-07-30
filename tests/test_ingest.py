"""Ingestion tests.

A fake corpus and a stub encoder stand in for the real ones, so resumability,
sharding and index assembly are all exercised without network or weights.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from vsearch.encoders.base import BaseEncoder, EncoderSpec, Modality
from vsearch.index import FaissStore
from vsearch.ingest import (
    ArtifactLayout,
    ImageRecord,
    IngestConfig,
    Manifest,
    available_corpora,
    build_index,
    get_corpus_spec,
    ingest,
)
from vsearch.ingest import loaders as loaders_module
from vsearch.ingest import pipeline as pipeline_module

DIM = 8


class _StubEncoder(BaseEncoder):
    """Embeds an image as a deterministic function of its width."""

    def __init__(self) -> None:
        spec = EncoderSpec(name="stub", model_id="stub/stub", modality=Modality.MULTIMODAL, dim=DIM)
        super().__init__(spec, device="cpu", batch_size=4)
        self.calls = 0

    def _embed_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        self.calls += 1
        rows = [[float(img.width)] + [float(i)] * (DIM - 1) for i, img in enumerate(images)]
        return torch.tensor(rows)

    def _embed_texts(self, texts: Sequence[str]) -> torch.Tensor:
        return torch.ones((len(texts), DIM))


def _fake_corpus(count: int, *, fail_after: int | None = None) -> tuple[object, list[str]]:
    """Build a replacement for iter_corpus over `count` synthetic records."""
    produced: list[str] = []

    def fake_iter(
        name: str,
        *,
        limit: int | None = None,
        skip: int = 0,
        split_filter: str | None = None,
        token: str | None = None,
        streaming: bool = False,
    ) -> Iterator[ImageRecord]:
        for position in range(skip, count):
            if limit is not None and position - skip >= limit:
                return
            if fail_after is not None and len(produced) >= fail_after:
                msg = "simulated interruption"
                raise RuntimeError(msg)
            identifier = f"item-{position}"
            produced.append(identifier)
            yield ImageRecord(
                id=identifier,
                image=Image.new("RGB", (10 + position, 12)),
                payload={"colour": ["Red", "Blue"][position % 2], "position": position},
            )

    return fake_iter, produced


# --- Manifest --------------------------------------------------------------


def _manifest(**overrides: object) -> Manifest:
    base = {
        "fingerprint": "abc",
        "corpus": "fashion",
        "encoder": "clip",
        "model_id": "openai/clip-vit-base-patch32",
        "dim": DIM,
        "shard_size": 4,
    }
    return Manifest(**{**base, **overrides})  # type: ignore[arg-type]


def test_next_shard_on_fresh_manifest() -> None:
    assert _manifest().next_shard() == 0


def test_next_shard_after_contiguous_run() -> None:
    assert _manifest(completed_shards=[0, 1, 2]).next_shard() == 3


def test_next_shard_resumes_from_first_gap() -> None:
    """Resuming from max+1 would silently skip the missing shard."""
    assert _manifest(completed_shards=[0, 1, 3, 4]).next_shard() == 2


def test_manifest_json_round_trip() -> None:
    original = _manifest(completed_shards=[0, 1], records=8, exhausted=True)
    assert Manifest.from_json(original.to_json()) == original


# --- IngestConfig fingerprint ---------------------------------------------


def test_fingerprint_is_stable() -> None:
    a = IngestConfig(corpus="fashion", encoder="clip")
    b = IngestConfig(corpus="fashion", encoder="clip")
    assert a.fingerprint() == b.fingerprint()


@pytest.mark.parametrize(
    ("field", "value"),
    [("corpus", "flickr30k"), ("encoder", "siglip2"), ("limit", 10), ("split_filter", "test")],
)
def test_fingerprint_changes_with_content_settings(field: str, value: object) -> None:
    base = IngestConfig(corpus="fashion", encoder="clip")
    changed = IngestConfig(**{**base.__dict__, field: value})  # type: ignore[arg-type]
    assert base.fingerprint() != changed.fingerprint()


def test_fingerprint_ignores_backend_and_thumbnails() -> None:
    """These affect packaging, not embedding content, so they must not
    invalidate shards already computed."""
    base = IngestConfig(corpus="fashion", encoder="clip")
    repacked = IngestConfig(
        corpus="fashion", encoder="clip", index_backend="hnsw", thumbnail_size=128
    )
    assert base.fingerprint() == repacked.fingerprint()


def test_run_name_combines_corpus_and_encoder() -> None:
    assert IngestConfig(corpus="fashion", encoder="clip").run_name == "fashion__clip"


# --- layout ----------------------------------------------------------------


def test_layout_paths_are_zero_padded(tmp_path: Path) -> None:
    layout = ArtifactLayout(tmp_path / "run")
    assert layout.embeddings_path(7).name == "shard_00007.npy"
    assert layout.records_path(12).name == "shard_00012.jsonl"


def test_manifest_write_is_atomic(tmp_path: Path) -> None:
    """No .tmp file may survive; a truncated manifest breaks resume."""
    layout = ArtifactLayout(tmp_path / "run")
    layout.ensure()
    layout.write_manifest(_manifest())
    assert layout.manifest_path.exists()
    assert not list(layout.root.glob("*.tmp"))
    assert layout.load_manifest() == _manifest()


def test_load_manifest_returns_none_when_absent(tmp_path: Path) -> None:
    assert ArtifactLayout(tmp_path / "run").load_manifest() is None


# --- end-to-end ingest -----------------------------------------------------


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    count: int,
    *,
    shard_size: int = 4,
    fail_after: int | None = None,
    limit: int | None = None,
) -> tuple[Manifest, _StubEncoder, list[str]]:
    fake_iter, produced = _fake_corpus(count, fail_after=fail_after)
    monkeypatch.setattr(pipeline_module, "iter_corpus", fake_iter)
    encoder = _StubEncoder()
    config = IngestConfig(corpus="fashion", encoder="clip", shard_size=shard_size, limit=limit)
    manifest = ingest(config, artifacts_dir=tmp_path, encoder=encoder)
    return manifest, encoder, produced


def test_ingest_indexes_every_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, _, _ = _run(tmp_path, monkeypatch, count=10)
    assert manifest.records == 10
    assert manifest.exhausted is True

    store = FaissStore.load(tmp_path / "fashion__clip" / "index")
    assert len(store) == 10


def test_ingest_writes_a_final_partial_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """10 records at shard_size 4 must produce 3 shards, not 2."""
    manifest, _, _ = _run(tmp_path, monkeypatch, count=10, shard_size=4)
    assert manifest.completed_shards == [0, 1, 2]


def test_ingest_writes_thumbnails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _run(tmp_path, monkeypatch, count=6)
    thumbnails = sorted((tmp_path / "fashion__clip" / "images").glob("*.jpg"))
    assert len(thumbnails) == 6
    with Image.open(thumbnails[0]) as image:
        assert image.mode == "RGB"


def test_ingest_respects_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, _, _ = _run(tmp_path, monkeypatch, count=100, limit=7)
    assert manifest.records == 7


def test_indexed_payloads_are_filterable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _run(tmp_path, monkeypatch, count=12)
    store = FaissStore.load(tmp_path / "fashion__clip" / "index")
    query = np.zeros((1, DIM), dtype=np.float32)
    query[0, 0] = 1.0
    hits = store.search(query, k=5, where={"colour": "Red"})[0]
    assert hits
    assert all(hit.payload["colour"] == "Red" for hit in hits)


# --- resumability ----------------------------------------------------------


def test_interrupted_ingest_keeps_completed_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _run(tmp_path, monkeypatch, count=20, shard_size=4, fail_after=9)

    layout = ArtifactLayout(tmp_path / "fashion__clip")
    manifest = layout.load_manifest()
    assert manifest is not None
    # Two full shards committed before the failure at record 9.
    assert manifest.completed_shards == [0, 1]
    assert manifest.records == 8


def test_resume_continues_and_does_not_redo_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(RuntimeError):
        _run(tmp_path, monkeypatch, count=20, shard_size=4, fail_after=9)

    # Second attempt: no failure, same corpus.
    fake_iter, produced = _fake_corpus(20)
    monkeypatch.setattr(pipeline_module, "iter_corpus", fake_iter)
    config = IngestConfig(corpus="fashion", encoder="clip", shard_size=4)
    manifest = ingest(config, artifacts_dir=tmp_path, encoder=_StubEncoder())

    assert manifest.records == 20
    assert manifest.completed_shards == [0, 1, 2, 3, 4]
    # The resumed run must not re-emit the first 8 records.
    assert produced[0] == "item-8"
    assert len(produced) == 12

    store = FaissStore.load(tmp_path / "fashion__clip" / "index")
    assert len(store) == 20
    assert len(set(store.ids())) == 20


def test_a_different_encoder_gets_its_own_run_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Corpus and encoder are part of run_name, so they cannot collide."""
    _run(tmp_path, monkeypatch, count=8)

    fake_iter, _ = _fake_corpus(8)
    monkeypatch.setattr(pipeline_module, "iter_corpus", fake_iter)
    config = IngestConfig(corpus="fashion", encoder="siglip2", shard_size=4)
    ingest(config, artifacts_dir=tmp_path, encoder=_StubEncoder())

    assert (tmp_path / "fashion__clip" / "index").is_dir()
    assert (tmp_path / "fashion__siglip2" / "index").is_dir()


def test_resume_rejects_changed_shard_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The dangerous case: same run directory, different shard geometry.

    Resume computes skip = start_shard * shard_size, so changing shard_size
    between runs would silently skip or duplicate records.
    """
    _run(tmp_path, monkeypatch, count=8, shard_size=4)

    fake_iter, _ = _fake_corpus(8)
    monkeypatch.setattr(pipeline_module, "iter_corpus", fake_iter)
    config = IngestConfig(corpus="fashion", encoder="clip", shard_size=8)
    with pytest.raises(ValueError, match="different settings"):
        ingest(config, artifacts_dir=tmp_path, encoder=_StubEncoder())


def test_resume_rejects_changed_split_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same run name, different subset of the corpus."""
    _run(tmp_path, monkeypatch, count=8, shard_size=4)

    fake_iter, _ = _fake_corpus(8)
    monkeypatch.setattr(pipeline_module, "iter_corpus", fake_iter)
    config = IngestConfig(corpus="fashion", encoder="clip", shard_size=4, split_filter="test")
    with pytest.raises(ValueError, match="different settings"):
        ingest(config, artifacts_dir=tmp_path, encoder=_StubEncoder())


# --- index assembly --------------------------------------------------------


def test_build_index_detects_corrupt_shard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Embedding/record counts must agree, or ids silently misalign."""
    _run(tmp_path, monkeypatch, count=8, shard_size=4)
    layout = ArtifactLayout(tmp_path / "fashion__clip")
    manifest = layout.load_manifest()
    assert manifest is not None

    records_path = layout.records_path(0)
    lines = records_path.read_text(encoding="utf-8").splitlines()
    records_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="corrupt"):
        build_index(layout, manifest)


def test_build_index_can_repack_as_hnsw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Shards are backend-agnostic, so re-indexing needs no re-embedding."""
    _run(tmp_path, monkeypatch, count=12, shard_size=4)
    layout = ArtifactLayout(tmp_path / "fashion__clip")
    manifest = layout.load_manifest()
    assert manifest is not None

    store = build_index(layout, manifest, backend="hnsw")
    assert store.backend == "hnsw"
    assert len(store) == 12


def test_shard_files_are_valid_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _run(tmp_path, monkeypatch, count=6, shard_size=4)
    layout = ArtifactLayout(tmp_path / "fashion__clip")
    with layout.records_path(0).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    assert len(rows) == 4
    assert all("id" in row and "payload" in row for row in rows)


def test_shard_embeddings_are_normalized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _run(tmp_path, monkeypatch, count=6, shard_size=4)
    layout = ArtifactLayout(tmp_path / "fashion__clip")
    embeddings = np.load(layout.embeddings_path(0))
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), np.ones(4), atol=1e-5)


# --- corpus specs ----------------------------------------------------------


def test_available_corpora() -> None:
    assert available_corpora() == ["fashion", "flickr30k"]


def test_unknown_corpus_raises() -> None:
    with pytest.raises(KeyError, match="unknown corpus"):
        get_corpus_spec("nope")


def test_flickr30k_pins_the_parquet_conversion_branch() -> None:
    """nlphuji/flickr30k ships a loading script, which datasets v3 refuses
    with "Dataset scripts are no longer supported". The auto-converted parquet
    branch has the same content and loads normally."""
    assert get_corpus_spec("flickr30k").revision == "refs/convert/parquet"


def test_fashion_needs_no_revision_pin() -> None:
    assert get_corpus_spec("fashion").revision is None


def test_flickr30k_declares_its_internal_split_column() -> None:
    """The whole dataset arrives as one Hub split; the real assignment is a
    column, and that is what makes the canonical benchmark reachable."""
    spec = get_corpus_spec("flickr30k")
    assert spec.split == "test"
    assert spec.split_column == "split"
    assert spec.caption_column == "caption"


def test_fashion_declares_the_filter_facets() -> None:
    spec = get_corpus_spec("fashion")
    assert "articleType" in spec.payload_columns
    assert "baseColour" in spec.payload_columns
    assert spec.caption_column is None


def test_free_text_columns_are_declared_non_facets() -> None:
    """productDisplayName is unique per product; faceting it is dead weight."""
    fashion = get_corpus_spec("fashion")
    assert "productDisplayName" in fashion.text_columns
    assert "productDisplayName" in fashion.payload_columns
    assert "filename" in get_corpus_spec("flickr30k").text_columns


def test_built_index_excludes_free_text_facets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_iter, _ = _fake_corpus(4)

    def with_text(*args: object, **kwargs: object) -> Iterator[ImageRecord]:
        for record in fake_iter(*args, **kwargs):  # type: ignore[operator]
            record.payload["productDisplayName"] = f"unique name {record.id}"
            yield record

    monkeypatch.setattr(pipeline_module, "iter_corpus", with_text)
    config = IngestConfig(corpus="fashion", encoder="clip", shard_size=4)
    ingest(config, artifacts_dir=tmp_path, encoder=_StubEncoder())

    store = FaissStore.load(tmp_path / "fashion__clip" / "index")
    assert "productDisplayName" not in store.filterable_fields
    assert "colour" in store.filterable_fields


def test_captions_are_normalized_to_a_list() -> None:
    spec = get_corpus_spec("flickr30k")
    payload = loaders_module._build_payload(
        spec, {"caption": ["a dog", "a cat"], "filename": "x.jpg", "split": "test"}
    )
    assert payload["captions"] == ["a dog", "a cat"]
    assert payload["split"] == "test"


def test_scalar_caption_is_wrapped_in_a_list() -> None:
    spec = get_corpus_spec("flickr30k")
    payload = loaders_module._build_payload(spec, {"caption": "a lone caption"})
    assert payload["captions"] == ["a lone caption"]


def test_missing_payload_columns_are_skipped() -> None:
    spec = get_corpus_spec("fashion")
    payload = loaders_module._build_payload(spec, {"baseColour": "Red"})
    assert payload == {"baseColour": "Red"}
