"""UI construction tests.

The Gradio app is built and its callbacks are exercised directly. Rendering
is Gradio's problem; what matters here is that the callbacks return what the
gallery expects and degrade sensibly on bad input.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from vsearch.encoders.base import BaseEncoder, EncoderSpec, Modality, l2_normalize
from vsearch.index import FaissStore
from vsearch.search import IndexHandle, SearchResponse, SearchResult, SearchService
from vsearch.ui.app import (
    MAX_FACET_CARDINALITY,
    _facet_choices,
    _status,
    _to_gallery,
    build_ui,
    image_roots,
)

DIM = 8


class _FakeEncoder(BaseEncoder):
    def __init__(self, name: str, modality: Modality) -> None:
        spec = EncoderSpec(name=name, model_id=f"fake/{name}", modality=modality, dim=DIM)
        super().__init__(spec, device="cpu", batch_size=8)

    def _embed_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        return torch.tensor([[1.0] + [0.0] * (DIM - 1) for _ in images])

    def _embed_texts(self, texts: Sequence[str]) -> torch.Tensor:
        return torch.tensor([[1.0] + [0.0] * (DIM - 1) for _ in texts])


@pytest.fixture
def images_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "images"
    directory.mkdir()
    for i in range(DIM):
        Image.new("RGB", (8, 8)).save(directory / f"item-{i}.jpg")
    return directory


def _handle(name: str, modality: Modality, images_dir: Path | None) -> IndexHandle:
    store = FaissStore(dim=DIM, facet_exclude=["productDisplayName"])
    store.add(
        [f"item-{i}" for i in range(DIM)],
        l2_normalize(np.eye(DIM, dtype=np.float32)),
        [
            {
                "baseColour": "Red" if i % 2 else "Blue",
                "gender": "Men" if i < 4 else "Women",
                "productDisplayName": f"thing {i}",
            }
            for i in range(DIM)
        ],
    )
    handle = IndexHandle(encoder=name, store=store, images_dir=images_dir)
    handle._encoder = _FakeEncoder(name, modality)
    return handle


@pytest.fixture
def service(images_dir: Path) -> SearchService:
    return SearchService(
        {
            "clip": _handle("clip", Modality.MULTIMODAL, images_dir),
            "dinov3": _handle("dinov3", Modality.VISION, images_dir),
        }
    )


# --- helpers ---------------------------------------------------------------


def test_gallery_pairs_paths_with_scored_captions(images_dir: Path) -> None:
    response = SearchResponse(
        results=[
            SearchResult(
                rank=1,
                id="item-0",
                score=0.3123,
                payload={"productDisplayName": "Red Boot"},
                thumbnail=images_dir / "item-0.jpg",
            )
        ],
        encoder="clip",
        took_ms=12.0,
        total_indexed=8,
    )
    ((path, caption),) = _to_gallery(response)
    assert path.endswith("item-0.jpg")
    assert "0.312" in caption
    assert "Red Boot" in caption
    assert "cos" in caption


def test_gallery_labels_fused_scores_differently(images_dir: Path) -> None:
    """A fused score is an RRF score, not a cosine; the caption must not lie."""
    response = SearchResponse(
        results=[
            SearchResult(
                rank=1, id="a", score=0.016, payload={}, thumbnail=images_dir / "item-0.jpg"
            )
        ],
        encoder="clip+dinov3",
        took_ms=5.0,
        total_indexed=8,
        fused=True,
    )
    ((_, caption),) = _to_gallery(response)
    assert "rrf" in caption


def test_gallery_drops_results_without_thumbnails() -> None:
    """An empty tile in the grid reads as a bug."""
    response = SearchResponse(
        results=[SearchResult(rank=1, id="a", score=0.5, payload={}, thumbnail=None)],
        encoder="clip",
        took_ms=1.0,
        total_indexed=1,
    )
    assert _to_gallery(response) == []


def test_status_names_the_score_scale() -> None:
    plain = SearchResponse(results=[], encoder="clip", took_ms=42.0, total_indexed=1000)
    fused = SearchResponse(
        results=[], encoder="clip+dinov3", took_ms=42.0, total_indexed=1000, fused=True
    )
    assert "cosine similarity" in _status(plain)
    assert "RRF score" in _status(fused)
    assert "1,000" in _status(plain)


# --- facet selection -------------------------------------------------------


def test_facet_choices_are_offered(service: SearchService) -> None:
    choices = _facet_choices(service)
    assert "baseColour" in choices
    assert choices["baseColour"] == ["Blue", "Red"]


def test_facet_choices_exclude_free_text(service: SearchService) -> None:
    assert "productDisplayName" not in _facet_choices(service)


def test_facet_choices_drop_high_cardinality_fields(images_dir: Path) -> None:
    """A dropdown with thousands of entries is not a usable control."""
    count = MAX_FACET_CARDINALITY + 50
    store = FaissStore(dim=DIM)
    rng = np.random.default_rng(0)
    store.add(
        [f"item-{i}" for i in range(count)],
        l2_normalize(rng.normal(size=(count, DIM))),
        # sku is unique per row (over the cap); colour is constant (no use as
        # a filter); size has three values and is genuinely useful.
        [
            {"sku": f"sku-{i}", "colour": "Red", "size": ["S", "M", "L"][i % 3]}
            for i in range(count)
        ],
    )
    handle = IndexHandle(encoder="clip", store=store, images_dir=images_dir)
    handle._encoder = _FakeEncoder("clip", Modality.MULTIMODAL)

    choices = _facet_choices(SearchService({"clip": handle}))
    assert list(choices) == ["size"]


# --- allowed paths ---------------------------------------------------------


def test_image_roots_are_absolute(service: SearchService, images_dir: Path) -> None:
    """Gradio sandboxes the filesystem; without these the grid renders empty."""
    roots = image_roots(service)
    assert roots == [str(images_dir.resolve())]
    assert Path(roots[0]).is_absolute()


def test_image_roots_are_empty_without_thumbnails() -> None:
    handle = _handle("clip", Modality.MULTIMODAL, None)
    assert image_roots(SearchService({"clip": handle})) == []


# --- construction ----------------------------------------------------------


def test_build_ui_returns_blocks(service: SearchService) -> None:
    import gradio as gr

    assert isinstance(build_ui(service), gr.Blocks)


def test_build_ui_works_without_facets(images_dir: Path) -> None:
    store = FaissStore(dim=DIM)
    store.add(["a"], l2_normalize(np.ones((1, DIM), dtype=np.float32)))
    handle = IndexHandle(encoder="clip", store=store, images_dir=images_dir)
    handle._encoder = _FakeEncoder("clip", Modality.MULTIMODAL)
    assert build_ui(SearchService({"clip": handle})) is not None
