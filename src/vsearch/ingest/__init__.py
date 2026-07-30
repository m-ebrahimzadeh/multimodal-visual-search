"""Corpus ingestion: load, embed, shard, and index."""

from vsearch.ingest.loaders import (
    CORPORA,
    CorpusSpec,
    ImageRecord,
    available_corpora,
    get_corpus_spec,
    iter_corpus,
)
from vsearch.ingest.pipeline import (
    ArtifactLayout,
    IngestConfig,
    Manifest,
    build_index,
    ingest,
)
from vsearch.ingest.publish import pull_artifacts, push_artifacts

__all__ = [
    "CORPORA",
    "ArtifactLayout",
    "CorpusSpec",
    "ImageRecord",
    "IngestConfig",
    "Manifest",
    "available_corpora",
    "build_index",
    "get_corpus_spec",
    "ingest",
    "iter_corpus",
    "pull_artifacts",
    "push_artifacts",
]
