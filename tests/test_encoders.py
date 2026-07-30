"""Encoder tests.

Fast tests use a stub backbone so batching, ordering, RGB coercion and
normalisation are verified without downloading weights. Tests that need real
checkpoints are marked ``slow`` and excluded from CI by default.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
import torch
from huggingface_hub.errors import GatedRepoError
from PIL import Image

from vsearch.encoders import (
    ENCODERS,
    BaseEncoder,
    EncoderSpec,
    Modality,
    TextNotSupportedError,
    available_encoders,
    get_spec,
    l2_normalize,
    load_encoder,
)
from vsearch.encoders import registry as registry_module

STUB_DIM = 2


def _spec(modality: Modality = Modality.MULTIMODAL) -> EncoderSpec:
    return EncoderSpec(name="stub", model_id="stub/stub", modality=modality, dim=STUB_DIM)


class _StubEncoder(BaseEncoder):
    """Records what it was handed, and encodes inputs recoverably.

    Images embed as ``(width, 1)`` and texts as ``(len, 1)``. After L2
    normalisation the *ratio* of the two components still recovers the input,
    which is what lets us assert that batching preserves order.
    """

    def __init__(self, modality: Modality = Modality.MULTIMODAL, batch_size: int = 3) -> None:
        super().__init__(_spec(modality), device="cpu", batch_size=batch_size)
        self.image_batches: list[int] = []
        self.text_batches: list[int] = []
        self.modes_seen: list[str] = []

    def _embed_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        self.image_batches.append(len(images))
        self.modes_seen.extend(img.mode for img in images)
        return torch.tensor([[float(img.width), 1.0] for img in images])

    def _embed_texts(self, texts: Sequence[str]) -> torch.Tensor:
        self.text_batches.append(len(texts))
        return torch.tensor([[float(len(t)), 1.0] for t in texts])


def _images(widths: Sequence[int], mode: str = "RGB") -> list[Image.Image]:
    return [Image.new(mode, (w, 4)) for w in widths]


# --- l2_normalize ----------------------------------------------------------


def test_normalize_produces_unit_rows() -> None:
    out = l2_normalize(np.array([[3.0, 4.0], [1.0, 0.0]]))
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0, 1.0], atol=1e-6)


def test_normalize_leaves_zero_rows_as_zero() -> None:
    """A zero row must not become NaN -- one NaN poisons an entire index."""
    out = l2_normalize(np.array([[0.0, 0.0], [3.0, 4.0]]))
    assert not np.isnan(out).any()
    np.testing.assert_array_equal(out[0], [0.0, 0.0])


def test_normalize_returns_contiguous_float32() -> None:
    """FAISS requires contiguous float32; float64 fails obscurely inside it."""
    out = l2_normalize(np.array([[1.0, 2.0]], dtype=np.float64))
    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]


def test_normalize_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="2-D"):
        l2_normalize(np.array([1.0, 2.0]))


# --- batching --------------------------------------------------------------


def test_encode_image_empty_returns_zero_by_dim() -> None:
    out = _StubEncoder().encode_image([])
    assert out.shape == (0, STUB_DIM)
    assert out.dtype == np.float32


def test_encode_text_empty_returns_zero_by_dim() -> None:
    out = _StubEncoder().encode_text([])
    assert out.shape == (0, STUB_DIM)


def test_images_are_split_into_batches() -> None:
    encoder = _StubEncoder(batch_size=3)
    encoder.encode_image(_images([1, 2, 3, 4, 5, 6, 7]))
    assert encoder.image_batches == [3, 3, 1]


def test_texts_are_split_into_batches() -> None:
    encoder = _StubEncoder(batch_size=2)
    encoder.encode_text(["a", "bb", "ccc", "dddd", "eeeee"])
    assert encoder.text_batches == [2, 2, 1]


def test_batching_preserves_input_order() -> None:
    """A reordering bug here is silent and ruins every downstream ranking."""
    widths = [10, 20, 30, 40, 50, 60, 70]
    out = _StubEncoder(batch_size=3).encode_image(_images(widths))
    assert out.shape == (len(widths), STUB_DIM)
    # Rows are (w, 1) scaled to unit length, so component ratio recovers w.
    np.testing.assert_allclose(out[:, 0] / out[:, 1], widths, rtol=1e-5)


def test_all_rows_are_normalized_across_batch_boundaries() -> None:
    out = _StubEncoder(batch_size=2).encode_image(_images([5, 15, 25, 35, 45]))
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), np.ones(5), atol=1e-6)


@pytest.mark.parametrize("mode", ["L", "RGBA", "P", "CMYK"])
def test_non_rgb_images_are_converted(mode: str) -> None:
    """Real corpora contain palette, alpha and CMYK images."""
    encoder = _StubEncoder()
    encoder.encode_image(_images([8], mode=mode))
    assert encoder.modes_seen == ["RGB"]


def test_rgb_images_are_passed_through() -> None:
    encoder = _StubEncoder()
    encoder.encode_image(_images([8], mode="RGB"))
    assert encoder.modes_seen == ["RGB"]


# --- modality guards -------------------------------------------------------


def test_vision_encoder_refuses_text() -> None:
    encoder = _StubEncoder(modality=Modality.VISION)
    assert encoder.supports_text is False
    with pytest.raises(TextNotSupportedError, match="vision-only"):
        encoder.encode_text(["a query"])


def test_multimodal_encoder_accepts_text() -> None:
    assert _StubEncoder(modality=Modality.MULTIMODAL).supports_text is True


# --- registry --------------------------------------------------------------


def test_registry_keys_match_spec_names() -> None:
    for key, spec in ENCODERS.items():
        assert key == spec.name


def test_registry_dims_are_positive() -> None:
    assert all(spec.dim > 0 for spec in ENCODERS.values())


def test_declared_fallbacks_resolve() -> None:
    for spec in ENCODERS.values():
        if spec.fallback is not None:
            assert spec.fallback in ENCODERS
            # A fallback must be reachable without a licence, or it is useless.
            assert ENCODERS[spec.fallback].gated is False


def test_fallback_matches_modality_and_dim() -> None:
    """Swapping in a different-dim fallback would break a prebuilt index."""
    for spec in ENCODERS.values():
        if spec.fallback is not None:
            replacement = ENCODERS[spec.fallback]
            assert replacement.modality is spec.modality
            assert replacement.dim == spec.dim


def test_siglip_declares_max_length_padding() -> None:
    """Regression guard for a silent-corruption bug.

    SigLIP was trained with fixed 64-token padding. Under "longest" it still
    returns embeddings -- just subtly wrong ones -- so nothing would fail
    except retrieval quality.
    """
    assert get_spec("siglip2").text_padding == "max_length"


def test_clip_is_multimodal_and_dino_is_vision() -> None:
    assert get_spec("clip").modality is Modality.MULTIMODAL
    assert get_spec("dinov3").modality is Modality.VISION


def test_unknown_encoder_lists_the_valid_names() -> None:
    with pytest.raises(KeyError, match="unknown encoder"):
        get_spec("clipp")


def test_available_encoders_is_sorted() -> None:
    names = available_encoders()
    assert names == sorted(names)
    assert "clip" in names


# --- gated fallback behaviour ---------------------------------------------


def test_gated_encoder_falls_back_when_unreachable(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Licence not accepted must degrade the demo, not 500 it."""
    built: list[str] = []

    def fake_build(spec: EncoderSpec, device: str, batch_size: int, token: str | None) -> object:
        built.append(spec.name)
        if spec.gated:
            raise GatedRepoError("access to model is restricted")
        return object()

    monkeypatch.setattr(registry_module, "_build", fake_build)
    load_encoder("dinov3")

    assert built == ["dinov3", "dinov2"]
    assert "falling back" in caplog.text


def test_fallback_can_be_disabled_for_benchmarking(monkeypatch: pytest.MonkeyPatch) -> None:
    """Benchmarks must never silently measure a different model."""

    def fake_build(spec: EncoderSpec, device: str, batch_size: int, token: str | None) -> object:
        raise GatedRepoError("access to model is restricted")

    monkeypatch.setattr(registry_module, "_build", fake_build)
    with pytest.raises(GatedRepoError):
        load_encoder("dinov3", allow_fallback=False)


def test_ungated_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """A network blip on an ungated model is a real error, not a fallback."""

    def fake_build(spec: EncoderSpec, device: str, batch_size: int, token: str | None) -> object:
        raise OSError("connection reset")

    monkeypatch.setattr(registry_module, "_build", fake_build)
    with pytest.raises(OSError, match="connection reset"):
        load_encoder("clip")


# --- real checkpoints ------------------------------------------------------


@pytest.mark.slow
def test_clip_round_trip() -> None:
    encoder = load_encoder("clip", device="cpu", batch_size=4)
    images = encoder.encode_image(_images([64, 64, 64]))
    texts = encoder.encode_text(["a red boot", "a blue sky"])

    assert images.shape == (3, encoder.dim)
    assert texts.shape == (2, encoder.dim)
    np.testing.assert_allclose(np.linalg.norm(images, axis=1), np.ones(3), atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(texts, axis=1), np.ones(2), atol=1e-5)


@pytest.mark.slow
def test_clip_is_deterministic() -> None:
    """eval() must be on; otherwise dropout makes indexes irreproducible."""
    encoder = load_encoder("clip", device="cpu", batch_size=4)
    first = encoder.encode_text(["red leather ankle boots"])
    second = encoder.encode_text(["red leather ankle boots"])
    np.testing.assert_allclose(first, second, atol=1e-6)


@pytest.mark.slow
def test_clip_batch_size_does_not_change_embeddings() -> None:
    """Ingest and query use different batch sizes; they must agree."""
    texts = ["a red boot", "a blue sky", "a green field", "a yellow car", "a black bag"]
    small = load_encoder("clip", device="cpu", batch_size=1).encode_text(texts)
    large = load_encoder("clip", device="cpu", batch_size=5).encode_text(texts)
    np.testing.assert_allclose(small, large, atol=1e-5)


@pytest.mark.slow
def test_clip_semantics_are_sane() -> None:
    """Matching text should out-score mismatched text on the same image."""
    encoder = load_encoder("clip", device="cpu")
    red = Image.new("RGB", (224, 224), color=(220, 20, 20))
    image_vec = encoder.encode_image([red])[0]
    text_vecs = encoder.encode_text(["a solid red image", "a solid blue image"])
    assert float(text_vecs[0] @ image_vec) > float(text_vecs[1] @ image_vec)


@pytest.mark.slow
def test_dino_reports_declared_dim_and_registers() -> None:
    encoder = load_encoder("dinov3", device="cpu", batch_size=2)
    out = encoder.encode_image(_images([224, 224]))
    assert out.shape == (2, encoder.dim)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), np.ones(2), atol=1e-5)


@pytest.mark.slow
def test_dino_refuses_text() -> None:
    with pytest.raises(TextNotSupportedError):
        load_encoder("dinov3", device="cpu").encode_text(["a query"])
