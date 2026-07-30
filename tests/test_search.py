"""Search service and fusion tests."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from vsearch.encoders import TextNotSupportedError
from vsearch.encoders.base import BaseEncoder, EncoderSpec, Modality, l2_normalize
from vsearch.index import FaissStore
from vsearch.index.base import SearchHit
from vsearch.search import (
    IndexHandle,
    IndexNotLoadedError,
    SearchResult,
    SearchService,
    build_service,
    discover_indexes,
    reciprocal_rank_fusion,
)

DIM = 8


def _hits(ids: Sequence[str], base: float = 0.9) -> list[SearchHit]:
    return [SearchHit(id=i, score=base - n * 0.01, payload={"n": n}) for n, i in enumerate(ids)]


# --- reciprocal rank fusion ------------------------------------------------


def test_rrf_rewards_agreement_between_encoders() -> None:
    """An item both rankings like should beat one only the first likes."""
    a = _hits(["x", "agreed", "y"])
    b = _hits(["z", "agreed", "w"])
    fused = reciprocal_rank_fusion([a, b])
    assert fused[0].id == "agreed"


def test_rrf_ignores_score_magnitude() -> None:
    """The whole point: a wide-scoring encoder must not dominate a narrow one."""
    wide = [SearchHit(id="wide", score=0.99, payload={})]
    narrow = [SearchHit(id="narrow", score=0.21, payload={})]
    fused = reciprocal_rank_fusion([wide, narrow])
    # Both sat at rank 1 in their own list, so both get 1/(60+1) despite the
    # 0.99 vs 0.21 raw-score gap.
    assert [hit.score for hit in fused] == pytest.approx([1 / 61, 1 / 61])


def test_rrf_respects_weights() -> None:
    fused = reciprocal_rank_fusion(
        [[SearchHit(id="a", score=1.0, payload={})], [SearchHit(id="b", score=1.0, payload={})]],
        weights=[3.0, 1.0],
    )
    assert fused[0].id == "a"


def test_rrf_rejects_mismatched_weights() -> None:
    with pytest.raises(ValueError, match="weights for"):
        reciprocal_rank_fusion([_hits(["a"])], weights=[1.0, 2.0])


def test_rrf_truncates_to_k() -> None:
    assert len(reciprocal_rank_fusion([_hits(list("abcdef"))], k=3)) == 3


def test_rrf_is_deterministic_on_ties() -> None:
    """Two runs over identical input must not disagree on ordering."""
    first = reciprocal_rank_fusion([_hits(["a"]), _hits(["b"])])
    second = reciprocal_rank_fusion([_hits(["a"]), _hits(["b"])])
    assert [h.id for h in first] == [h.id for h in second]


def test_rrf_preserves_payloads() -> None:
    fused = reciprocal_rank_fusion([[SearchHit(id="a", score=0.5, payload={"colour": "Red"})]])
    assert fused[0].payload["colour"] == "Red"


def test_rrf_handles_empty_rankings() -> None:
    assert reciprocal_rank_fusion([[], []]) == []


# --- service scaffolding ---------------------------------------------------


class _FakeEncoder(BaseEncoder):
    """Maps text and images onto known basis vectors."""

    def __init__(self, name: str, modality: Modality, axis: int) -> None:
        spec = EncoderSpec(name=name, model_id=f"fake/{name}", modality=modality, dim=DIM)
        super().__init__(spec, device="cpu", batch_size=8)
        self.axis = axis

    def _embed_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        rows = []
        for image in images:
            row = [0.0] * DIM
            row[image.width % DIM] = 1.0
            rows.append(row)
        return torch.tensor(rows)

    def _embed_texts(self, texts: Sequence[str]) -> torch.Tensor:
        rows = []
        for text in texts:
            row = [0.0] * DIM
            row[len(text) % DIM] = 1.0
            rows.append(row)
        return torch.tensor(rows)


def _basis(index: int) -> np.ndarray:
    vector = np.zeros((1, DIM), dtype=np.float32)
    vector[0, index] = 1.0
    return vector


def _handle(
    name: str,
    modality: Modality = Modality.MULTIMODAL,
    *,
    images_dir: Path | None = None,
) -> IndexHandle:
    store = FaissStore(dim=DIM)
    vectors = l2_normalize(np.eye(DIM, dtype=np.float32))
    store.add(
        [f"item-{i}" for i in range(DIM)],
        vectors,
        [
            {"colour": "Red" if i % 2 else "Blue", "productDisplayName": f"thing {i}"}
            for i in range(DIM)
        ],
    )
    handle = IndexHandle(encoder=name, store=store, images_dir=images_dir)
    handle._encoder = _FakeEncoder(name, modality, axis=0)
    return handle


def _service(**kwargs: object) -> SearchService:
    return SearchService(
        {
            "clip": _handle("clip", Modality.MULTIMODAL),
            "dinov3": _handle("dinov3", Modality.VISION),
        },
        **kwargs,  # type: ignore[arg-type]
    )


# --- service behaviour -----------------------------------------------------


def test_service_requires_an_index() -> None:
    with pytest.raises(ValueError, match="at least one index"):
        SearchService({})


def test_text_encoders_excludes_vision_only() -> None:
    service = _service()
    assert service.encoders == ["clip", "dinov3"]
    assert service.text_encoders == ["clip"]


def test_text_search_returns_ranked_results() -> None:
    response = _service().search_text("abc", k=3)
    assert len(response) == 3
    assert [r.rank for r in response.results] == [1, 2, 3]
    assert response.encoder == "clip"
    assert response.took_ms >= 0


def test_text_search_rejects_vision_only_encoder() -> None:
    with pytest.raises(TextNotSupportedError, match="vision-only"):
        _service().search_text("a query", encoder="dinov3")


def test_text_search_rejects_empty_query() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _service().search_text("   ")


def test_text_search_defaults_to_a_multimodal_encoder() -> None:
    """dinov3 sorts first alphabetically but cannot serve text."""
    assert _service().search_text("query").encoder == "clip"


def test_unknown_encoder_raises() -> None:
    with pytest.raises(IndexNotLoadedError, match="no index loaded"):
        _service().search_text("query", encoder="siglip2")


def test_image_search_uses_any_encoder() -> None:
    response = _service().search_image(Image.new("RGB", (3, 3)), k=2, encoder="dinov3")
    assert response.encoder == "dinov3"
    assert len(response) == 2


def test_search_applies_metadata_filters() -> None:
    response = _service().search_text("abc", k=4, where={"colour": "Red"})
    assert all(r.payload["colour"] == "Red" for r in response.results)


def test_fused_image_search_marks_itself_fused() -> None:
    response = _service().search_image(Image.new("RGB", (3, 3)), k=4, fuse=["clip", "dinov3"])
    assert response.fused is True
    assert response.encoder == "clip+dinov3"
    assert len(response) == 4


def test_facets_list_values_for_dropdowns() -> None:
    service = SearchService({"clip": _handle("clip")})
    facets = service.facets()
    assert sorted(facets["colour"]) == ["Blue", "Red"]


def test_warmup_reports_which_encoders_loaded() -> None:
    assert _service().warmup() == ["clip", "dinov3"]


def test_warmup_survives_an_encoder_that_will_not_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cold demo beats no demo, so a failed warmup must not abort startup."""
    service = _service()
    broken = service.handle("dinov3")
    broken._encoder = None
    monkeypatch.setattr(
        type(broken),
        "resolve_encoder",
        lambda self, **kwargs: (_ for _ in ()).throw(OSError("weights unreachable")),
    )
    assert service.warmup() == []


def test_took_ms_excludes_model_loading() -> None:
    """Timing starts after the encoder resolves, so a cold load is not
    reported as query latency."""
    response = _service().search_text("abc", k=2)
    assert response.took_ms < 1000


def test_stats_report_index_shape() -> None:
    stats = _service().stats()
    assert stats["text_encoders"] == ["clip"]
    assert stats["indexes"]["clip"]["count"] == DIM
    assert stats["indexes"]["clip"]["dim"] == DIM


# --- result presentation ---------------------------------------------------


def test_title_prefers_product_name() -> None:
    result = SearchResult(rank=1, id="x", score=0.5, payload={"productDisplayName": "Red Boot"})
    assert result.title == "Red Boot"


def test_title_falls_back_to_first_caption() -> None:
    result = SearchResult(rank=1, id="x", score=0.5, payload={"captions": ["a dog runs"]})
    assert result.title == "a dog runs"


def test_title_falls_back_to_id() -> None:
    assert SearchResult(rank=1, id="x", score=0.5, payload={}).title == "x"


def test_thumbnail_is_none_when_missing(tmp_path: Path) -> None:
    handle = _handle("clip", images_dir=tmp_path)
    assert handle.thumbnail_for("item-0") is None


def test_thumbnail_is_found_when_present(tmp_path: Path) -> None:
    Image.new("RGB", (4, 4)).save(tmp_path / "item-0.jpg")
    handle = _handle("clip", images_dir=tmp_path)
    assert handle.thumbnail_for("item-0") == tmp_path / "item-0.jpg"


# --- discovery -------------------------------------------------------------


def test_discover_finds_built_indexes(tmp_path: Path) -> None:
    store = FaissStore(dim=DIM)
    store.add(["a"], _basis(0))
    store.save(tmp_path / "fashion__clip" / "index")
    (tmp_path / "fashion__clip" / "images").mkdir(parents=True)

    handles = discover_indexes(tmp_path, "fashion")
    assert list(handles) == ["clip"]
    assert handles["clip"].images_dir is not None


def test_discover_skips_runs_without_a_built_index(tmp_path: Path) -> None:
    (tmp_path / "fashion__clip" / "shards").mkdir(parents=True)
    assert discover_indexes(tmp_path, "fashion") == {}


def test_discover_ignores_other_corpora(tmp_path: Path) -> None:
    store = FaissStore(dim=DIM)
    store.add(["a"], _basis(0))
    store.save(tmp_path / "flickr30k__clip" / "index")
    assert discover_indexes(tmp_path, "fashion") == {}


def test_discover_returns_empty_for_missing_dir(tmp_path: Path) -> None:
    assert discover_indexes(tmp_path / "absent", "fashion") == {}


def test_build_service_explains_how_to_fix_a_missing_index(tmp_path: Path) -> None:
    with pytest.raises(IndexNotLoadedError, match="vsearch ingest"):
        build_service(tmp_path, "fashion")
