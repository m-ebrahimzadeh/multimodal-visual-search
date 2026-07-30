"""Tests for configuration loading and device resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from vsearch.config import Device, Settings, resolve_device


def test_defaults_are_sane() -> None:
    settings = Settings()
    assert settings.batch_size >= 1
    assert settings.top_k >= 1
    assert settings.index_backend in {"flat", "hnsw"}


def test_env_prefix_is_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VSEARCH_BATCH_SIZE", "64")
    monkeypatch.setenv("VSEARCH_TEXT_ENCODER", "siglip2")
    settings = Settings()
    assert settings.batch_size == 64
    assert settings.text_encoder == "siglip2"


def test_hf_token_is_read_without_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """HF_TOKEN is intentionally unprefixed to match the HF ecosystem."""
    monkeypatch.setenv("HF_TOKEN", "hf_dummy_value")
    settings = Settings()
    assert settings.token == "hf_dummy_value"


def test_token_is_not_exposed_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stray print() or logged traceback must not leak the token."""
    monkeypatch.setenv("HF_TOKEN", "hf_super_secret_value")
    settings = Settings()
    assert "hf_super_secret_value" not in repr(settings)
    assert "hf_super_secret_value" not in str(settings)


def test_token_is_none_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    # Bypass the .env file so a developer's real token does not fail this test.
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.token is None


def test_batch_size_must_be_positive() -> None:
    with pytest.raises(ValueError):
        Settings(batch_size=0)


@pytest.mark.parametrize("explicit", ["cpu", "cuda", "mps"])
def test_explicit_device_is_never_downgraded(explicit: str) -> None:
    """An explicit request must pass through, even if unavailable here.

    A silent fallback would make a benchmark look inexplicably slow rather
    than failing loudly at model load.
    """
    assert resolve_device(explicit) == explicit


def test_auto_device_resolves_to_a_real_backend() -> None:
    assert resolve_device(Device.AUTO) in {"cpu", "cuda", "mps"}


def test_ensure_dirs_creates_paths(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        results_dir=tmp_path / "results",
    )
    settings.ensure_dirs()
    assert settings.data_dir.is_dir()
    assert settings.artifacts_dir.is_dir()
    assert settings.results_dir.is_dir()
