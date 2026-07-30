"""Encoder registry: names to backbones, with graceful degradation.

Model choices and their training-time contracts are declared here in one
table. ``load_encoder`` is the only construction path, so the gated-model
fallback cannot be forgotten at a call site.
"""

from __future__ import annotations

import logging

from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
)

from vsearch.config import get_settings, resolve_device
from vsearch.encoders.base import BaseEncoder, EncoderSpec, Modality
from vsearch.encoders.multimodal import MultimodalEncoder
from vsearch.encoders.vision import VisionEncoder

logger = logging.getLogger(__name__)

# Errors meaning "this checkpoint is not reachable for this user". transformers
# wraps hub failures in OSError, so it is included deliberately -- without it
# the gated-model fallback never fires and a public demo 500s instead.
_UNREACHABLE = (
    GatedRepoError,
    RepositoryNotFoundError,
    LocalEntryNotFoundError,
    HfHubHTTPError,
    OSError,
)

ENCODERS: dict[str, EncoderSpec] = {
    "clip": EncoderSpec(
        name="clip",
        model_id="openai/clip-vit-base-patch32",
        modality=Modality.MULTIMODAL,
        dim=512,
        text_padding="longest",
    ),
    "siglip2": EncoderSpec(
        name="siglip2",
        model_id="google/siglip2-base-patch16-224",
        modality=Modality.MULTIMODAL,
        dim=768,
        # Non-negotiable: SigLIP was trained with fixed 64-token padding.
        text_padding="max_length",
    ),
    "dinov3": EncoderSpec(
        name="dinov3",
        model_id="facebook/dinov3-vits16-pretrain-lvd1689m",
        modality=Modality.VISION,
        dim=384,
        pooling="cls",
        gated=True,
        fallback="dinov2",
    ),
    "dinov2": EncoderSpec(
        name="dinov2",
        model_id="facebook/dinov2-small",
        modality=Modality.VISION,
        dim=384,
        pooling="cls",
    ),
}


def available_encoders() -> list[str]:
    """Registered encoder names, sorted."""
    return sorted(ENCODERS)


def get_spec(name: str) -> EncoderSpec:
    """Look up a spec, with a helpful message on typos."""
    try:
        return ENCODERS[name]
    except KeyError:
        msg = f"unknown encoder {name!r}; available: {', '.join(available_encoders())}"
        raise KeyError(msg) from None


def _build(spec: EncoderSpec, device: str, batch_size: int, token: str | None) -> BaseEncoder:
    if spec.modality is Modality.MULTIMODAL:
        return MultimodalEncoder(spec, device=device, batch_size=batch_size, token=token)
    return VisionEncoder(spec, device=device, batch_size=batch_size, token=token)


def load_encoder(
    name: str,
    *,
    device: str | None = None,
    batch_size: int | None = None,
    token: str | None = None,
    allow_fallback: bool = True,
) -> BaseEncoder:
    """Load an encoder by registry name.

    When a gated checkpoint is unreachable -- the licence has not been accepted
    or no ``HF_TOKEN`` is present -- and the spec declares a ``fallback``, the
    fallback is loaded and a warning is logged. That keeps a public demo alive
    on an ungated backbone instead of failing outright, while still making the
    substitution obvious in logs and in ``/health``.

    Pass ``allow_fallback=False`` for benchmarking and evaluation, where
    silently measuring a different model than the one named would be worse
    than an error.
    """
    settings = get_settings()
    spec = get_spec(name)
    resolved_device = device if device is not None else resolve_device(settings.device)
    resolved_batch = batch_size if batch_size is not None else settings.batch_size
    resolved_token = token if token is not None else settings.token

    try:
        return _build(spec, resolved_device, resolved_batch, resolved_token)
    except _UNREACHABLE as exc:
        if not (allow_fallback and spec.fallback):
            raise

        hint = (
            f" Accept the licence at https://huggingface.co/{spec.model_id} "
            "and set HF_TOKEN to use it."
            if spec.gated
            else ""
        )
        logger.warning(
            "Encoder %r (%s) is unreachable (%s: %s); falling back to %r.%s",
            spec.name,
            spec.model_id,
            type(exc).__name__,
            exc,
            spec.fallback,
            hint,
        )
        fallback_spec = get_spec(spec.fallback)
        return _build(fallback_spec, resolved_device, resolved_batch, resolved_token)
