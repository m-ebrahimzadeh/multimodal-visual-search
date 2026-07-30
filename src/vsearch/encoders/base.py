"""Encoder abstraction shared by every backbone.

The central invariant: **every embedding leaving an encoder is float32 and
L2-normalised**. That is enforced once, here, rather than trusted to each
implementation. Two things depend on it:

* Cosine similarity becomes a plain inner product, so FAISS ``IndexFlatIP``
  yields cosine scores with no extra work (see ``vsearch.index``).
* Scores from different encoders land on a comparable [-1, 1] scale, which is
  what makes fusing or A/B-ing two backbones meaningful.

FAISS also *requires* contiguous float32; returning float64 from here would
fail deep inside the index with a far less obvious message.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch
    from PIL.Image import Image

Embeddings = NDArray[np.float32]


class Modality(StrEnum):
    """What a backbone can encode."""

    MULTIMODAL = "multimodal"
    """Shared text+image space (CLIP, SigLIP) -- supports text->image search."""

    VISION = "vision"
    """Image-only (DINOv3) -- stronger image->image, but no text tower."""


class TextNotSupportedError(RuntimeError):
    """Raised when text is passed to a vision-only encoder."""


@dataclass(frozen=True)
class EncoderSpec:
    """Declarative description of a backbone.

    Training-time contracts (text padding, register-token count) live here as
    data rather than as ``if "siglip" in name`` checks scattered through call
    sites. Getting one wrong does not raise -- it silently degrades retrieval
    quality -- so they are stated once, explicitly, next to the model id.
    """

    name: str
    model_id: str
    modality: Modality
    dim: int

    text_padding: Literal["longest", "max_length"] = "longest"
    """SigLIP *must* use ``max_length``: it was trained with fixed 64-token
    padding, and ``longest`` produces quietly wrong text embeddings."""

    pooling: Literal["cls", "mean"] = "cls"
    """Vision-only pooling. DINOv3 docs recommend the CLS token for retrieval."""

    gated: bool = False
    """Requires accepting a licence on the Hub plus an ``HF_TOKEN``."""

    fallback: str | None = None
    """Encoder name to load when this one is unreachable (e.g. licence not
    accepted). Lets a public demo degrade instead of returning a 500."""

    @property
    def supports_text(self) -> bool:
        return self.modality is Modality.MULTIMODAL


def l2_normalize(vectors: NDArray[np.floating]) -> Embeddings:
    """Scale each row to unit length, returning contiguous float32.

    Zero rows would divide by zero; they are left as zeros instead, which
    scores 0 against everything -- a neutral result rather than a NaN that
    silently poisons an entire index.
    """
    array = np.ascontiguousarray(vectors, dtype=np.float32)
    if array.ndim != 2:
        msg = f"expected a 2-D (n, dim) array, got shape {array.shape}"
        raise ValueError(msg)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    np.divide(array, norms, out=array, where=norms > 0)
    return array


def _batched(items: Sequence[object], size: int) -> Iterator[slice]:
    for start in range(0, len(items), size):
        yield slice(start, min(start + size, len(items)))


class BaseEncoder(ABC):
    """Base class handling batching, RGB coercion, and normalisation.

    Subclasses implement only the model-specific forward pass.
    """

    def __init__(self, spec: EncoderSpec, device: str, batch_size: int = 32) -> None:
        self.spec = spec
        self.device = device
        self.batch_size = batch_size

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def dim(self) -> int:
        return self.spec.dim

    @property
    def supports_text(self) -> bool:
        return self.spec.supports_text

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.spec.name!r}, "
            f"model_id={self.spec.model_id!r}, device={self.device!r}, dim={self.dim})"
        )

    # -- Public API ---------------------------------------------------------

    def encode_image(self, images: Sequence[Image]) -> Embeddings:
        """Embed images. Returns ``(len(images), dim)``, L2-normalised."""
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)

        # Corpora contain palette PNGs, RGBA, CMYK and grayscale. Processors
        # handle some of these inconsistently, so coerce up front rather than
        # discover a channel-order bug via mediocre recall later.
        rgb = [img if img.mode == "RGB" else img.convert("RGB") for img in images]

        chunks = [self._embed_images(rgb[window]) for window in _batched(rgb, self.batch_size)]
        return l2_normalize(self._to_numpy(chunks))

    def encode_text(self, texts: Sequence[str]) -> Embeddings:
        """Embed text. Returns ``(len(texts), dim)``, L2-normalised.

        Raises ``TextNotSupportedError`` on vision-only backbones.
        """
        if not self.supports_text:
            msg = (
                f"{self.spec.name!r} is a vision-only encoder ({self.spec.model_id}) "
                "and has no text tower; use a multimodal encoder for text search."
            )
            raise TextNotSupportedError(msg)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        chunks = [self._embed_texts(texts[window]) for window in _batched(texts, self.batch_size)]
        return l2_normalize(self._to_numpy(chunks))

    # -- Subclass hooks -----------------------------------------------------

    @abstractmethod
    def _embed_images(self, images: Sequence[Image]) -> torch.Tensor:
        """Forward one batch of RGB images to a ``(batch, dim)`` tensor."""

    @abstractmethod
    def _embed_texts(self, texts: Sequence[str]) -> torch.Tensor:
        """Forward one batch of strings to a ``(batch, dim)`` tensor."""

    # -- Internals ----------------------------------------------------------

    @staticmethod
    def _to_numpy(chunks: Sequence[torch.Tensor]) -> NDArray[np.floating]:
        import torch

        stacked = torch.cat(list(chunks), dim=0)
        # float32 because bf16/fp16 batches from a GPU run cannot go straight
        # to numpy, and FAISS wants float32 regardless.
        return stacked.detach().to(dtype=torch.float32, device="cpu").numpy()
