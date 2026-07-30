"""Export encoders to ONNX and quantize them to INT8.

Only the forward pass is exported; preprocessing stays on the Hugging Face
processor. That keeps a torch-vs-ONNX comparison honest -- any measured
difference is the graph and its precision, not a different resize.

The image and text towers are exported separately. They are independent at
query time (a text search never runs the vision tower), so a combined graph
would force loading and optimising weights that will not be used.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vsearch.encoders.base import EncoderSpec, Modality
from vsearch.encoders.onnx_encoder import IMAGE_MODEL_FILENAME, TEXT_MODEL_FILENAME
from vsearch.encoders.registry import get_spec

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

logger = logging.getLogger(__name__)

OPSET = 17
"""Opset 17 has native LayerNormalization, which keeps transformer graphs
compact and lets ORT fuse them; earlier opsets decompose it into a dozen ops."""

OUTPUT_NAME = "out_embedding"
"""Graph output name. Deliberately not "embedding": CLIP's vision model has an
`embeddings` submodule and the exporter emits a node under that name, so the
obvious choice collides and the saved graph fails to load."""


def graph_size_bytes(model_path: Path) -> int:
    """On-disk size of a graph including any external weight sidecar.

    The dynamo exporter writes large tensors to a separate ``.onnx_data``
    file, so stat()-ing the .onnx alone reports only the graph -- a few KB --
    and would make an fp32 export look smaller than its INT8 counterpart.
    """
    if not model_path.exists():
        return 0
    total = model_path.stat().st_size
    for sidecar in model_path.parent.glob(f"{model_path.name}*"):
        if sidecar != model_path and sidecar.is_file():
            total += sidecar.stat().st_size
    return total


@dataclass(frozen=True)
class ExportResult:
    """Where an export landed and how big it is."""

    encoder: str
    model_dir: Path
    precision: str
    image_bytes: int
    text_bytes: int = 0

    @property
    def total_bytes(self) -> int:
        return self.image_bytes + self.text_bytes

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 * 1024)


def _build_wrappers(spec: EncoderSpec, model: Any) -> tuple[Any, Any | None]:
    """Wrap a model so its forward *is* the embedding we index.

    Pooling lives inside the exported graph rather than in Python, so the
    ONNX output matches the torch encoder exactly. Doing it outside would
    make the two backends silently diverge for DINOv3.
    """
    from torch import nn

    if spec.modality is Modality.MULTIMODAL:

        class ImageTower(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = model

            def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
                return self.model.get_image_features(pixel_values=pixel_values)  # type: ignore[no-any-return]

        class TextTower(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = model

            def forward(
                self, input_ids: torch.Tensor, attention_mask: torch.Tensor
            ) -> torch.Tensor:
                return self.model.get_text_features(  # type: ignore[no-any-return]
                    input_ids=input_ids, attention_mask=attention_mask
                )

        return ImageTower().eval(), TextTower().eval()

    num_registers = int(getattr(model.config, "num_register_tokens", 0))
    pooling = spec.pooling

    class VisionTower(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            hidden = self.model(pixel_values=pixel_values).last_hidden_state
            if pooling == "cls":
                return hidden[:, 0, :]  # type: ignore[no-any-return]
            # Skip CLS and the register tokens; see vsearch.encoders.vision.
            return hidden[:, 1 + num_registers :, :].mean(dim=1)  # type: ignore[no-any-return]

    return VisionTower().eval(), None


def _strip_stale_shapes(source: Path) -> Path:
    """Drop intermediate shape annotations before quantization.

    The dynamo exporter records value_info for intermediate tensors that can
    disagree with what ONNX re-infers, and quantize_dynamic runs shape
    inference first. On CLIP that surfaces as "Inferred shape and existing
    shape differ in dimension 0: (768) vs (512)" -- hidden size versus
    projection dim. The graph is valid; only the annotations are stale, so
    clearing them lets inference re-derive them consistently.
    """
    import onnx

    model = onnx.load(str(source))
    del model.graph.value_info[:]
    # Repopulate from scratch. Clearing alone leaves quantization unable to
    # type intermediate tensors ("Unable to find data type for weight_name"),
    # so the annotations must be rebuilt, not merely removed. strict_mode is
    # off because a single un-inferable node should not abort the export.
    model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    prepared = source.with_name(f"{source.stem}.reshaped.onnx")
    onnx.save(model, str(prepared))
    return prepared


def _quantize(source: Path, target: Path) -> None:
    """Dynamic INT8 quantization of weights.

    Dynamic (not static) because it needs no calibration dataset: activations
    are quantized per-batch at runtime. For transformer encoders the win is
    mostly in the weight-heavy matmuls, which is exactly what this covers.
    """
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    prepared = _strip_stale_shapes(source)
    try:
        quantize_dynamic(
            model_input=str(prepared),
            model_output=str(target),
            weight_type=QuantType.QInt8,
            # Backstop for any tensor shape inference still cannot type; these
            # graphs are entirely fp32 before quantization.
            extra_options={"DefaultTensorType": int(onnx.TensorProto.FLOAT)},
        )
    finally:
        prepared.unlink(missing_ok=True)


def export_encoder(
    name: str,
    output_root: Path,
    *,
    quantize: bool = True,
    token: str | None = None,
    opset: int = OPSET,
) -> list[ExportResult]:
    """Export one encoder to ONNX fp32, and optionally also INT8."""
    import torch
    from transformers import AutoModel

    spec = get_spec(name)
    # allow_fallback is not offered here: exporting dinov2 into a directory
    # labelled dinov3 would silently mislabel every benchmark row.
    model = AutoModel.from_pretrained(spec.model_id, token=token).eval()
    image_tower, text_tower = _build_wrappers(spec, model)

    fp32_dir = output_root / f"{name}-fp32"
    fp32_dir.mkdir(parents=True, exist_ok=True)

    image_path = fp32_dir / IMAGE_MODEL_FILENAME
    dummy_pixels = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        image_tower,
        (dummy_pixels,),
        str(image_path),
        input_names=["pixel_values"],
        # Not "embedding": CLIP has an internal `embeddings` module and the
        # exporter emits a node by that name, producing "Duplicate definition
        # of name" when the graph is loaded.
        output_names=[OUTPUT_NAME],
        # Batch must be dynamic: ingest runs batches of 32+, a query runs 1,
        # and a fixed axis would force a separate export per batch size.
        dynamic_axes={"pixel_values": {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
        opset_version=opset,
    )
    logger.info("Exported image tower -> %s", image_path)

    text_bytes = 0
    if text_tower is not None:
        text_path = fp32_dir / TEXT_MODEL_FILENAME
        length = 64 if spec.text_padding == "max_length" else 16
        dummy_ids = torch.ones(1, length, dtype=torch.long)
        torch.onnx.export(
            text_tower,
            (dummy_ids, torch.ones(1, length, dtype=torch.long)),
            str(text_path),
            input_names=["input_ids", "attention_mask"],
            # Not "embedding": CLIP has an internal `embeddings` module and the
            # exporter emits a node by that name, producing "Duplicate definition
            # of name" when the graph is loaded.
            output_names=[OUTPUT_NAME],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                OUTPUT_NAME: {0: "batch"},
            },
            opset_version=opset,
        )
        text_bytes = graph_size_bytes(text_path)
        logger.info("Exported text tower -> %s", text_path)

    results = [
        ExportResult(
            encoder=name,
            model_dir=fp32_dir,
            precision="fp32",
            image_bytes=graph_size_bytes(image_path),
            text_bytes=text_bytes,
        )
    ]

    if quantize:
        int8_dir = output_root / f"{name}-int8"
        int8_dir.mkdir(parents=True, exist_ok=True)
        _quantize(image_path, int8_dir / IMAGE_MODEL_FILENAME)
        int8_text_bytes = 0
        if text_tower is not None:
            _quantize(fp32_dir / TEXT_MODEL_FILENAME, int8_dir / TEXT_MODEL_FILENAME)
            int8_text_bytes = graph_size_bytes(int8_dir / TEXT_MODEL_FILENAME)
        results.append(
            ExportResult(
                encoder=name,
                model_dir=int8_dir,
                precision="int8",
                image_bytes=graph_size_bytes(int8_dir / IMAGE_MODEL_FILENAME),
                text_bytes=int8_text_bytes,
            )
        )
        logger.info(
            "Quantized to INT8: %.1f MB -> %.1f MB", results[0].total_mb, results[1].total_mb
        )

    return results


def clean_exports(output_root: Path) -> None:
    """Remove exported models."""
    if output_root.is_dir():
        shutil.rmtree(output_root)
