"""DINOv3 / DINOv2 encoders -- self-supervised, image-only.

Pure-vision features are the stronger signal for "find me more like this":
they capture visual structure without being pulled toward whatever a caption
happened to mention. The cost is no text tower, so these cannot serve
text->image search at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from vsearch.encoders.base import BaseEncoder, EncoderSpec, TextNotSupportedError

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch
    from PIL.Image import Image


class VisionEncoder(BaseEncoder):
    """Image-only encoder backed by a DINO-family checkpoint."""

    def __init__(
        self,
        spec: EncoderSpec,
        device: str,
        batch_size: int = 32,
        token: str | None = None,
    ) -> None:
        super().__init__(spec, device, batch_size)

        import torch
        from transformers import AutoImageProcessor, AutoModel

        self._torch = torch
        # transformers ships py.typed but leaves the Auto* factories untyped.
        # use_fast picks the torchvision processor over the PIL one; see the
        # note in multimodal.py on why it is set at construction.
        self._processor = AutoImageProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            spec.model_id, token=token, use_fast=True
        )
        model = AutoModel.from_pretrained(spec.model_id, token=token)
        self._model = model.eval().to(device)

        # DINOv3 prepends 4 register tokens after CLS; DINOv2 has none. Read
        # it from the config rather than hardcoding, so both work unchanged.
        self._num_register_tokens = int(getattr(self._model.config, "num_register_tokens", 0))

    @property
    def num_register_tokens(self) -> int:
        return self._num_register_tokens

    def _embed_images(self, images: Sequence[Image]) -> torch.Tensor:
        inputs = self._processor(images=list(images), return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            outputs = self._model(**inputs)

        hidden = outputs.last_hidden_state  # (batch, 1 + registers + patches, dim)
        if self.spec.pooling == "cls":
            return cast("torch.Tensor", hidden[:, 0, :])

        # Mean over *patch* tokens only. The obvious-looking hidden[:, 1:, :]
        # would fold DINOv3's register tokens into the embedding -- they are
        # global scratch space, not image content, and including them degrades
        # retrieval silently.
        first_patch = 1 + self._num_register_tokens
        return cast("torch.Tensor", hidden[:, first_patch:, :].mean(dim=1))

    def _embed_texts(self, texts: Sequence[str]) -> torch.Tensor:
        # Unreachable via encode_text(), which guards on supports_text first.
        # Kept explicit so a future direct caller fails loudly.
        msg = f"{self.spec.name!r} has no text tower"
        raise TextNotSupportedError(msg)


__all__ = ["EncoderSpec", "VisionEncoder"]
