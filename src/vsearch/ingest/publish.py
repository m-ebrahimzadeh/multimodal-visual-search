"""Publish and retrieve index artifacts via the Hugging Face Hub.

This is the seam that lets ingestion and serving run on different machines:
embed on a Colab GPU, publish the result, and have a CPU-only Space pull a
prebuilt index at startup instead of re-embedding tens of thousands of images.

Artifacts go to a *dataset* repo rather than a model repo -- they are data
(vectors, metadata, thumbnails), not weights.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Intermediate shards are excluded by default. IndexFlatIP already stores the
# raw vectors, so shipping both would roughly double the artifact for nothing.
# They are only needed to rebuild the index under a *different* backend.
DEFAULT_IGNORE = ("shards/*", "*.tmp", "*.npy")


def push_artifacts(
    run_dir: Path,
    repo_id: str,
    *,
    token: str,
    private: bool = True,
    include_shards: bool = False,
    commit_message: str | None = None,
) -> str:
    """Upload one ingest run to a Hub dataset repo. Returns the repo URL."""
    from huggingface_hub import HfApi

    if not run_dir.is_dir():
        msg = f"no ingest run at {run_dir}"
        raise FileNotFoundError(msg)
    if not (run_dir / "index").is_dir():
        msg = f"{run_dir} has no built index/ directory; run ingest first"
        raise FileNotFoundError(msg)

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

    ignore = None if include_shards else list(DEFAULT_IGNORE)
    api.upload_folder(
        folder_path=str(run_dir),
        repo_id=repo_id,
        repo_type="dataset",
        ignore_patterns=ignore,
        commit_message=commit_message or f"Publish index artifacts from {run_dir.name}",
    )

    url = f"https://huggingface.co/datasets/{repo_id}"
    logger.info("Published %s to %s", run_dir.name, url)
    return url


def pull_artifacts(
    repo_id: str,
    destination: Path,
    *,
    token: str | None = None,
    revision: str | None = None,
) -> Path:
    """Download a published index into ``destination``. Returns the local path.

    ``revision`` pins a commit or tag, so a deployed Space can be held on a
    known-good index while a new one is being built.
    """
    from huggingface_hub import snapshot_download

    destination.mkdir(parents=True, exist_ok=True)
    local = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(destination),
        token=token,
        revision=revision,
    )
    logger.info("Pulled %s into %s", repo_id, local)
    return Path(local)
