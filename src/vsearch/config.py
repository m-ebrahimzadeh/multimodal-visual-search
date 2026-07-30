"""Central configuration for vsearch.

Settings are read from environment variables and an optional ``.env`` file at
the project root. Environment variables use the ``VSEARCH_`` prefix, with one
deliberate exception: ``HF_TOKEN`` keeps its conventional unprefixed name so
the same variable serves this project, the ``huggingface_hub`` client, and a
Hugging Face Space secret without duplication.

Nothing here imports torch at module scope. Config is pulled in by CLI help
paths and unit tests where paying torch's multi-second import would be waste;
``resolve_device`` imports it lazily instead.
"""

from __future__ import annotations

import enum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/vsearch/config.py -> src/vsearch -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Device(enum.StrEnum):
    """Compute backend selection."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


def resolve_device(requested: Device | str = Device.AUTO) -> str:
    """Resolve ``auto`` to the best backend actually available on this machine.

    An explicit request is honoured verbatim and never silently downgraded --
    if someone asks for ``cuda`` on a machine without it, the resulting torch
    error is far more useful than a quiet fallback to CPU that makes a
    benchmark look mysteriously slow.
    """
    device = Device(requested)
    if device is not Device.AUTO:
        return device.value

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Settings(BaseSettings):
    """Runtime settings, loaded from env vars and ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="VSEARCH_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # --- Compute -----------------------------------------------------------
    device: Device = Device.AUTO
    batch_size: int = Field(default=32, ge=1, description="Encoder batch size.")
    num_workers: int = Field(default=4, ge=0, description="Image decode workers.")

    # --- Model selection (names resolved by vsearch.encoders.registry) ------
    text_encoder: str = Field(default="clip", description="Encoder for text->image search.")
    image_encoder: str = Field(default="dinov3", description="Encoder for image->image search.")

    # --- Retrieval ---------------------------------------------------------
    index_backend: Literal["flat", "hnsw"] = "flat"
    top_k: int = Field(default=24, ge=1, le=500)

    # --- Filesystem layout -------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    results_dir: Path = PROJECT_ROOT / "results"

    # --- Hugging Face ------------------------------------------------------
    hf_token: SecretStr | None = Field(default=None, validation_alias="HF_TOKEN")
    artifact_repo: str = Field(
        default="",
        description="Hub dataset repo holding prebuilt index artifacts, e.g. 'user/vsearch-index'.",
    )

    @property
    def token(self) -> str | None:
        """The HF token as a plain string, or None when unset.

        Kept behind a property so the raw value is never the default repr of a
        Settings object -- an accidental ``print(settings)`` or a logged
        traceback would otherwise leak it.
        """
        return self.hf_token.get_secret_value() if self.hf_token is not None else None

    def ensure_dirs(self) -> None:
        """Create the working directories this run will write to."""
        for directory in (self.data_dir, self.artifacts_dir, self.results_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton.

    Cached because ``Settings()`` re-reads and re-parses ``.env`` on every
    instantiation, and this is called from request handlers.
    """
    return Settings()
