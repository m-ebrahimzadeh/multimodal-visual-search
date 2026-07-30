"""Resumable batched ingestion.

Embedding a corpus takes minutes on a GPU and hours on a laptop CPU, so the
pipeline is built to survive interruption: work is committed in shards, and a
re-run picks up from the first missing shard instead of starting over.

Layout of an ingest run::

    artifacts/<corpus>__<encoder>/
      manifest.json          run config, fingerprint, completed shards
      shards/
        shard_00000.npy      embeddings for this shard
        shard_00000.jsonl    ids + payloads, one record per line
      images/<id>.jpg        thumbnails for the results grid
      index/                 the built FaissStore

Shards are written *before* the manifest records them, so a crash mid-write
leaves an orphan shard that is simply overwritten on resume -- never a
manifest claiming data that does not exist.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vsearch.encoders import BaseEncoder, load_encoder
from vsearch.encoders.base import Embeddings
from vsearch.index import FaissStore
from vsearch.index.faiss_store import Backend
from vsearch.ingest.loaders import ImageRecord, get_corpus_spec, iter_corpus

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.json"
THUMBNAIL_QUALITY = 85


@dataclass(frozen=True)
class IngestConfig:
    """Everything that determines what an ingest run produces."""

    corpus: str
    encoder: str
    shard_size: int = 2048
    limit: int | None = None
    split_filter: str | None = None
    thumbnail_size: int = 256
    index_backend: Backend = "flat"
    streaming: bool = False

    def fingerprint(self) -> str:
        """Hash of the settings that affect embedding *content*.

        Deliberately excludes device and batch size: resuming a Colab GPU run
        on a laptop CPU is a supported workflow, and batch size is asserted
        not to change embeddings (see test_clip_batch_size_does_not_change_
        embeddings). Changing the corpus or encoder, by contrast, invalidates
        every shard already written.
        """
        material = json.dumps(
            {
                "corpus": self.corpus,
                "encoder": self.encoder,
                "limit": self.limit,
                "split_filter": self.split_filter,
                "shard_size": self.shard_size,
            },
            sort_keys=True,
        )
        return hashlib.sha256(material.encode()).hexdigest()[:16]

    @property
    def run_name(self) -> str:
        return f"{self.corpus}__{self.encoder}"


@dataclass
class Manifest:
    """Crash-safe record of what an ingest run has completed."""

    fingerprint: str
    corpus: str
    encoder: str
    model_id: str
    dim: int
    shard_size: int
    completed_shards: list[int] = field(default_factory=list)
    records: int = 0
    exhausted: bool = False
    """True once the corpus iterator ran out -- distinguishes 'finished' from
    'stopped early', which a shard count alone cannot tell you."""

    def next_shard(self) -> int:
        """First shard index not yet committed.

        Resumes from the first *gap* rather than the highest index, so a
        partially-written run can never skip over missing data.
        """
        done = set(self.completed_shards)
        index = 0
        while index in done:
            index += 1
        return index

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> Manifest:
        return cls(**json.loads(text))


class ArtifactLayout:
    """Filesystem layout for one ingest run."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.shards = root / "shards"
        self.images = root / "images"
        self.index = root / "index"

    def ensure(self) -> None:
        for directory in (self.root, self.shards, self.images, self.index):
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_FILENAME

    def embeddings_path(self, shard: int) -> Path:
        return self.shards / f"shard_{shard:05d}.npy"

    def records_path(self, shard: int) -> Path:
        return self.shards / f"shard_{shard:05d}.jsonl"

    def load_manifest(self) -> Manifest | None:
        if not self.manifest_path.exists():
            return None
        return Manifest.from_json(self.manifest_path.read_text(encoding="utf-8"))

    def write_manifest(self, manifest: Manifest) -> None:
        """Write atomically so a crash cannot leave a truncated manifest.

        A half-written manifest is worse than none: it would claim shards that
        do not exist and make the run unresumable.
        """
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(manifest.to_json(), encoding="utf-8")
        temporary.replace(self.manifest_path)


def _save_thumbnail(record: ImageRecord, directory: Path, size: int) -> None:
    """Write a small JPEG for the results grid.

    Done here because the image is already decoded; re-decoding the corpus
    later just to make thumbnails would double the expensive part of ingest.
    """
    # convert() already returns a new image; copy() keeps the RGB path from
    # mutating the caller's image, since thumbnail() resizes in place.
    source = record.image
    image = source.convert("RGB") if source.mode != "RGB" else source.copy()
    image.thumbnail((size, size))
    image.save(directory / f"{record.id}.jpg", format="JPEG", quality=THUMBNAIL_QUALITY)


def _write_shard(
    layout: ArtifactLayout,
    shard: int,
    records: Sequence[ImageRecord],
    embeddings: Embeddings,
    thumbnail_size: int,
) -> None:
    np.save(layout.embeddings_path(shard), embeddings)
    with layout.records_path(shard).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps({"id": record.id, "payload": record.payload}) + "\n")
    for record in records:
        _save_thumbnail(record, layout.images, thumbnail_size)


def _encode_shard(encoder: BaseEncoder, records: Sequence[ImageRecord]) -> Embeddings:
    return encoder.encode_image([record.image for record in records])


def ingest(
    config: IngestConfig,
    *,
    artifacts_dir: Path,
    device: str | None = None,
    batch_size: int | None = None,
    token: str | None = None,
    encoder: BaseEncoder | None = None,
) -> Manifest:
    """Embed a corpus into shards, then build the index.

    Safe to re-run: completed shards are skipped. Raises if the run directory
    holds shards produced under a different corpus or encoder, since mixing
    embeddings from two models into one index yields silent nonsense.
    """
    layout = ArtifactLayout(artifacts_dir / config.run_name)
    layout.ensure()

    if encoder is None:
        encoder = load_encoder(config.encoder, device=device, batch_size=batch_size, token=token)

    manifest = layout.load_manifest()
    if manifest is None:
        manifest = Manifest(
            fingerprint=config.fingerprint(),
            corpus=config.corpus,
            encoder=config.encoder,
            model_id=encoder.spec.model_id,
            dim=encoder.dim,
            shard_size=config.shard_size,
        )
    elif manifest.fingerprint != config.fingerprint():
        msg = (
            f"{layout.root} already holds shards produced with different settings "
            f"(fingerprint {manifest.fingerprint}, shard_size {manifest.shard_size}); "
            f"this run is {config.fingerprint()} with shard_size {config.shard_size}. "
            "Resume computes skip = completed_shards * shard_size, so continuing "
            "would silently skip or duplicate records. Delete the run directory to "
            "start over, or restore the original settings to resume."
        )
        raise ValueError(msg)

    start_shard = manifest.next_shard()
    skip = start_shard * config.shard_size
    if start_shard:
        logger.info(
            "Resuming %s from shard %d (%d records already done)",
            config.run_name,
            start_shard,
            skip,
        )

    records: list[ImageRecord] = []
    shard = start_shard

    stream = iter_corpus(
        config.corpus,
        limit=None if config.limit is None else max(config.limit - skip, 0),
        split_filter=config.split_filter,
        token=token,
        streaming=config.streaming,
        skip=skip,
    )

    for record in stream:
        records.append(record)
        if len(records) < config.shard_size:
            continue
        _commit(layout, manifest, config, encoder, shard, records)
        shard += 1
        records = []

    if records:
        _commit(layout, manifest, config, encoder, shard, records)

    # Reached only by running the corpus iterator to completion.
    manifest.exhausted = True
    layout.write_manifest(manifest)

    build_index(layout, manifest, backend=config.index_backend)
    return manifest


def _commit(
    layout: ArtifactLayout,
    manifest: Manifest,
    config: IngestConfig,
    encoder: BaseEncoder,
    shard: int,
    records: Sequence[ImageRecord],
) -> None:
    """Encode, persist, then record the shard -- in that order."""
    embeddings = _encode_shard(encoder, records)
    _write_shard(layout, shard, records, embeddings, config.thumbnail_size)

    manifest.completed_shards.append(shard)
    manifest.completed_shards.sort()
    manifest.records += len(records)
    layout.write_manifest(manifest)
    logger.info("Committed shard %d (%d records)", shard, len(records))


def build_index(
    layout: ArtifactLayout,
    manifest: Manifest,
    *,
    backend: Backend = "flat",
) -> FaissStore:
    """Assemble committed shards into a searchable index and persist it."""
    # The corpus declares which payload columns are free text rather than
    # facets, so the inverted index stays small and the UI's filter dropdowns
    # only offer fields worth filtering on.
    text_columns = get_corpus_spec(manifest.corpus).text_columns
    store = FaissStore(dim=manifest.dim, backend=backend, facet_exclude=text_columns)

    for shard in sorted(manifest.completed_shards):
        embeddings = np.load(layout.embeddings_path(shard))
        ids: list[str] = []
        payloads: list[dict[str, Any]] = []
        with layout.records_path(shard).open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                ids.append(str(record["id"]))
                payloads.append(dict(record.get("payload") or {}))
        if len(ids) != embeddings.shape[0]:
            msg = (
                f"shard {shard} has {len(ids)} records but {embeddings.shape[0]} "
                "embeddings; the shard is corrupt"
            )
            raise ValueError(msg)
        store.add(ids, embeddings, payloads)

    store.save(layout.index)
    logger.info("Built %s index with %d vectors at %s", backend, len(store), layout.index)
    return store


def corpus_description(name: str) -> str:
    return get_corpus_spec(name).description
