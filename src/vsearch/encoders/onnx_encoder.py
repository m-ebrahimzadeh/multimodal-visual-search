"""ONNX Runtime encoder.

Implements the same ``BaseEncoder`` contract as the torch backends, so an
exported model is a drop-in swap: the batching, RGB coercion and L2
normalisation in the base class are shared, and only the forward pass differs.
That is what makes a torch-vs-ONNX comparison a fair one.

Preprocessing deliberately still uses the Hugging Face processor. Only the
forward pass is exported, so any difference measured between backends is the
graph and its precision, not a different resize or normalisation.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from vsearch.encoders.base import BaseEncoder, EncoderSpec, TextNotSupportedError

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch
    from PIL.Image import Image

IMAGE_MODEL_FILENAME = "image_encoder.onnx"
TEXT_MODEL_FILENAME = "text_encoder.onnx"


def _session(path: Path, threads: int | None) -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads is not None:
        # Pinned so benchmark rows are comparable; ORT otherwise sizes thread
        # pools from the host, and a 12-thread laptop would not be measuring
        # the same thing as a 2-vCPU Space.
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


class OnnxEncoder(BaseEncoder):
    """Encoder backed by exported ONNX graphs."""

    def __init__(
        self,
        spec: EncoderSpec,
        model_dir: Path,
        *,
        batch_size: int = 32,
        threads: int | None = None,
        token: str | None = None,
    ) -> None:
        super().__init__(spec, device="cpu", batch_size=batch_size)

        import torch
        from transformers import AutoProcessor

        self._torch = torch
        self._model_dir = Path(model_dir)

        image_path = self._model_dir / IMAGE_MODEL_FILENAME
        if not image_path.exists():
            msg = f"no exported image encoder at {image_path}; run `vsearch export` first"
            raise FileNotFoundError(msg)
        self._image_session = _session(image_path, threads)

        text_path = self._model_dir / TEXT_MODEL_FILENAME
        self._text_session = (
            _session(text_path, threads) if spec.supports_text and text_path.exists() else None
        )

        self._processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            spec.model_id, token=token, use_fast=True
        )

    @property
    def model_dir(self) -> Path:
        return self._model_dir

    def _embed_images(self, images: Sequence[Image]) -> torch.Tensor:
        # "pt", not "np": the fast (torchvision) processor only emits torch
        # tensors. Converting here also keeps preprocessing byte-identical to
        # the torch backend, which is what makes the parity number meaningful.
        inputs = self._processor(images=list(images), return_tensors="pt")
        pixel_values = np.ascontiguousarray(inputs["pixel_values"].numpy(), dtype=np.float32)
        output = self._image_session.run(None, {"pixel_values": pixel_values})[0]
        return self._torch.from_numpy(np.asarray(output, dtype=np.float32))

    def _embed_texts(self, texts: Sequence[str]) -> torch.Tensor:
        if self._text_session is None:
            msg = f"{self.spec.name!r} has no exported text encoder at {self._model_dir}"
            raise TextNotSupportedError(msg)

        encoded = self._processor(
            text=list(texts),
            padding=self.spec.text_padding,
            truncation=True,
            return_tensors="pt",
        )
        feeds = {
            node.name: np.ascontiguousarray(encoded[node.name].numpy(), dtype=np.int64)
            for node in self._text_session.get_inputs()
            if node.name in encoded
        }
        output = self._text_session.run(None, feeds)[0]
        return self._torch.from_numpy(np.asarray(output, dtype=np.float32))
