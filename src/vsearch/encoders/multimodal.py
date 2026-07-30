"""CLIP / SigLIP encoders -- a shared text+image embedding space.

Both families expose ``get_image_features`` / ``get_text_features`` and project
into one space, so a single implementation covers them. The difference that
matters is text padding, which is declared per-model in the ``EncoderSpec``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from vsearch.encoders.base import BaseEncoder, EncoderSpec

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch
    from PIL.Image import Image


class MultimodalEncoder(BaseEncoder):
    """Text+image encoder backed by a CLIP- or SigLIP-family checkpoint."""

    def __init__(
        self,
        spec: EncoderSpec,
        device: str,
        batch_size: int = 32,
        token: str | None = None,
    ) -> None:
        super().__init__(spec, device, batch_size)

        import torch
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        # transformers ships py.typed but leaves the Auto* factories untyped.
        # use_fast selects the torchvision image processor over the PIL one --
        # materially quicker for batch preprocessing, which is a real share of
        # ingest wall-clock. Set here so ingest and query preprocess alike;
        # mixing the two would put a small systematic offset between indexed
        # and query vectors.
        self._processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            spec.model_id, token=token, use_fast=True
        )
        model = AutoModel.from_pretrained(spec.model_id, token=token)
        # eval() disables dropout. Without it, repeated calls return slightly
        # different vectors and an index is no longer reproducible.
        self._model = model.eval().to(device)

    def _embed_images(self, images: Sequence[Image]) -> torch.Tensor:
        inputs = self._processor(images=list(images), return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            features = self._model.get_image_features(**inputs)
        return cast("torch.Tensor", features)

    def _embed_texts(self, texts: Sequence[str]) -> torch.Tensor:
        inputs = self._processor(
            text=list(texts),
            # Declared per-model: SigLIP was trained with fixed-length padding
            # and returns quietly wrong vectors under "longest".
            padding=self.spec.text_padding,
            # CLIP caps at 77 tokens; without this a long caption raises
            # mid-ingest after an expensive run has already started.
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with self._torch.inference_mode():
            features = self._model.get_text_features(**inputs)
        return cast("torch.Tensor", features)
