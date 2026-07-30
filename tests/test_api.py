"""API tests.

Driven through httpx's ASGI transport, so no server process is started and
no port is bound.
"""

from __future__ import annotations

import io
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from vsearch.api import main as api_main
from vsearch.api.main import app
from vsearch.encoders.base import BaseEncoder, EncoderSpec, Modality, l2_normalize
from vsearch.index import FaissStore
from vsearch.search import IndexHandle, SearchService

DIM = 8


class _FakeEncoder(BaseEncoder):
    def __init__(self, name: str, modality: Modality) -> None:
        spec = EncoderSpec(name=name, model_id=f"fake/{name}", modality=modality, dim=DIM)
        super().__init__(spec, device="cpu", batch_size=8)

    def _embed_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        return torch.tensor([[1.0] + [0.0] * (DIM - 1) for _ in images])

    def _embed_texts(self, texts: Sequence[str]) -> torch.Tensor:
        return torch.tensor([[1.0] + [0.0] * (DIM - 1) for _ in texts])


def _handle(name: str, modality: Modality, images_dir: Path | None) -> IndexHandle:
    store = FaissStore(dim=DIM, facet_exclude=["productDisplayName"])
    store.add(
        [f"item-{i}" for i in range(DIM)],
        l2_normalize(np.eye(DIM, dtype=np.float32)),
        [
            {"baseColour": "Red" if i % 2 else "Blue", "productDisplayName": f"thing {i}"}
            for i in range(DIM)
        ],
    )
    handle = IndexHandle(encoder=name, store=store, images_dir=images_dir)
    handle._encoder = _FakeEncoder(name, modality)
    return handle


@pytest.fixture
def images_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "images"
    directory.mkdir()
    for i in range(DIM):
        Image.new("RGB", (8, 8), color=(i * 20, 40, 60)).save(directory / f"item-{i}.jpg")
    return directory


@pytest.fixture
def client(images_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A client whose app already has a loaded service."""
    service = SearchService(
        {
            "clip": _handle("clip", Modality.MULTIMODAL, images_dir),
            "dinov3": _handle("dinov3", Modality.VISION, images_dir),
        }
    )
    monkeypatch.setattr(api_main.state, "service", service, raising=False)
    monkeypatch.setattr(api_main.state, "error", None, raising=False)
    monkeypatch.setattr(api_main.state, "corpus", "fashion", raising=False)
    # Bypass lifespan so the fixture's service is not overwritten at startup.
    with TestClient(app) as test_client:
        monkeypatch.setattr(api_main.state, "service", service, raising=False)
        yield test_client


@pytest.fixture
def degraded_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A client with no index loaded."""
    with TestClient(app) as test_client:
        monkeypatch.setattr(api_main.state, "service", None, raising=False)
        monkeypatch.setattr(api_main.state, "error", "no built index", raising=False)
        yield test_client


def _png(colour: tuple[int, int, int] = (200, 30, 30)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), color=colour).save(buffer, format="PNG")
    return buffer.getvalue()


# --- meta ------------------------------------------------------------------


def test_root_points_at_the_docs(client: TestClient) -> None:
    body = client.get("/").json()
    assert body["docs"] == "/docs"


def test_health_reports_loaded_indexes(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["encoders"] == ["clip", "dinov3"]
    assert body["text_encoders"] == ["clip"]
    assert body["indexes"]["clip"]["count"] == DIM


def test_health_is_degraded_not_dead_without_an_index(degraded_client: TestClient) -> None:
    """A Space that crash-loops is far harder to diagnose than one that says why."""
    response = degraded_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["detail"] == "no built index"


def test_search_returns_503_when_degraded(degraded_client: TestClient) -> None:
    response = degraded_client.post("/search/text", json={"query": "boots"})
    assert response.status_code == 503
    assert "no built index" in response.json()["detail"]


def test_facets_lists_filterable_fields(client: TestClient) -> None:
    body = client.get("/facets").json()
    assert sorted(body["facets"]["baseColour"]) == ["Blue", "Red"]
    assert "productDisplayName" not in body["facets"]


# --- text search -----------------------------------------------------------


def test_text_search_returns_ranked_results(client: TestClient) -> None:
    body = client.post("/search/text", json={"query": "red boots", "k": 3}).json()
    assert body["count"] == 3
    assert [r["rank"] for r in body["results"]] == [1, 2, 3]
    assert body["encoder"] == "clip"
    assert body["fused"] is False


def test_text_search_results_carry_titles_and_thumbnails(client: TestClient) -> None:
    body = client.post("/search/text", json={"query": "red boots", "k": 1}).json()
    result = body["results"][0]
    assert result["title"].startswith("thing ")
    assert result["thumbnail_url"] == f"/thumbnail/clip/{result['id']}"


def test_text_search_applies_filters(client: TestClient) -> None:
    body = client.post(
        "/search/text", json={"query": "boots", "k": 4, "filters": {"baseColour": "Red"}}
    ).json()
    assert all(r["payload"]["baseColour"] == "Red" for r in body["results"])


def test_text_search_rejects_empty_query(client: TestClient) -> None:
    assert client.post("/search/text", json={"query": ""}).status_code == 422


def test_text_search_rejects_out_of_range_k(client: TestClient) -> None:
    assert client.post("/search/text", json={"query": "x", "k": 0}).status_code == 422
    assert client.post("/search/text", json={"query": "x", "k": 9999}).status_code == 422


def test_text_search_rejects_vision_only_encoder(client: TestClient) -> None:
    response = client.post("/search/text", json={"query": "x", "encoder": "dinov3"})
    assert response.status_code == 400
    assert "vision-only" in response.json()["detail"]


def test_text_search_reports_unknown_encoder(client: TestClient) -> None:
    response = client.post("/search/text", json={"query": "x", "encoder": "siglip2"})
    assert response.status_code == 404


def test_text_search_reports_unknown_facet(client: TestClient) -> None:
    """A typo'd filter must be a 400, not a silent full-corpus search."""
    response = client.post("/search/text", json={"query": "x", "filters": {"colr": "Red"}})
    assert response.status_code == 400
    assert "cannot filter on" in response.json()["detail"]


# --- image search ----------------------------------------------------------


def test_image_search_accepts_an_upload(client: TestClient) -> None:
    response = client.post(
        "/search/image", files={"file": ("q.png", _png(), "image/png")}, data={"k": 3}
    )
    assert response.status_code == 200
    assert response.json()["count"] == 3


def test_image_search_can_fuse_encoders(client: TestClient) -> None:
    body = client.post(
        "/search/image",
        files={"file": ("q.png", _png(), "image/png")},
        data={"k": 4, "fuse": "clip,dinov3"},
    ).json()
    assert body["fused"] is True
    assert body["encoder"] == "clip+dinov3"


def test_image_search_rejects_a_non_image(client: TestClient) -> None:
    response = client.post(
        "/search/image", files={"file": ("x.png", b"definitely not a png", "image/png")}
    )
    assert response.status_code == 400
    assert "could not decode" in response.json()["detail"]


def test_image_search_rejects_an_empty_upload(client: TestClient) -> None:
    response = client.post("/search/image", files={"file": ("x.png", b"", "image/png")})
    assert response.status_code == 400


def test_image_search_rejects_an_oversized_upload(client: TestClient) -> None:
    """A public demo will eventually be handed an enormous file."""
    payload = b"\x89PNG\r\n\x1a\n" + b"0" * (api_main.MAX_UPLOAD_BYTES + 10)
    response = client.post("/search/image", files={"file": ("big.png", payload, "image/png")})
    assert response.status_code == 413


# --- thumbnails ------------------------------------------------------------


def test_thumbnail_is_served(client: TestClient) -> None:
    response = client.get("/thumbnail/clip/item-0")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"


def test_thumbnail_404s_for_unknown_item(client: TestClient) -> None:
    assert client.get("/thumbnail/clip/item-999").status_code == 404


def test_thumbnail_404s_for_unknown_encoder(client: TestClient) -> None:
    assert client.get("/thumbnail/nope/item-0").status_code == 404


@pytest.mark.parametrize(
    "attack",
    [
        "..%2f..%2f..%2f.env",
        "..%5c..%5c.env",
        "%2e%2e%2f%2e%2e%2fsecret",
    ],
)
def test_thumbnail_refuses_path_traversal(client: TestClient, attack: str) -> None:
    """An id is a path component; escaping the images dir must be impossible."""
    response = client.get(f"/thumbnail/clip/{attack}")
    assert response.status_code == 404


def test_thumbnail_handle_refuses_escape_directly(images_dir: Path) -> None:
    """Checked in the service, so a future caller cannot reintroduce the hole."""
    secret = images_dir.parent / "secret.jpg"
    secret.write_bytes(b"top secret")
    handle = _handle("clip", Modality.MULTIMODAL, images_dir)
    assert handle.thumbnail_for("../secret") is None


def test_thumbnail_fused_encoder_resolves_to_first(client: TestClient) -> None:
    assert client.get("/thumbnail/clip+dinov3/item-0").status_code == 200
