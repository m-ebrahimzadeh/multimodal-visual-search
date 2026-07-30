"""Encoders: CLIP/SigLIP (text+image) and DINOv3/DINOv2 (image-only)."""

from vsearch.encoders.base import (
    BaseEncoder,
    Embeddings,
    EncoderSpec,
    Modality,
    TextNotSupportedError,
    l2_normalize,
)
from vsearch.encoders.multimodal import MultimodalEncoder
from vsearch.encoders.registry import (
    ENCODERS,
    available_encoders,
    get_spec,
    load_encoder,
)
from vsearch.encoders.vision import VisionEncoder

__all__ = [
    "ENCODERS",
    "BaseEncoder",
    "Embeddings",
    "EncoderSpec",
    "Modality",
    "MultimodalEncoder",
    "TextNotSupportedError",
    "VisionEncoder",
    "available_encoders",
    "get_spec",
    "l2_normalize",
    "load_encoder",
]
