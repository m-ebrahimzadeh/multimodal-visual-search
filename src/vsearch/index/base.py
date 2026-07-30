"""Vector store abstraction.

Retrieval is expressed against this interface so the FAISS backend can be
swapped for a hosted service (Qdrant) without touching the search layer.

All stores assume L2-normalised vectors and use inner product as the metric,
so scores are cosine similarities in [-1, 1]. See ``vsearch.encoders.base``
for where that invariant is enforced.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vsearch.encoders.base import Embeddings

Payload = Mapping[str, Any]

FilterValue = object | Sequence[object]
Filter = Mapping[str, FilterValue]
"""Facet filter. Fields combine with AND; a sequence within a field is OR.

``{"baseColour": "Red", "gender": ["Men", "Women"]}`` reads as
*red AND (men OR women)*.
"""


@dataclass(frozen=True)
class SearchHit:
    """One ranked result."""

    id: str
    score: float
    payload: Payload = field(default_factory=dict)


class VectorStore(ABC):
    """A searchable collection of embeddings with attached metadata."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimensionality this store accepts."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of indexed vectors."""

    @abstractmethod
    def add(
        self,
        ids: Sequence[str],
        vectors: Embeddings,
        payloads: Sequence[Payload] | None = None,
    ) -> None:
        """Add vectors with stable string ids and optional metadata."""

    @abstractmethod
    def search(
        self,
        queries: Embeddings,
        k: int,
        where: Filter | None = None,
    ) -> list[list[SearchHit]]:
        """Return the top-k hits per query row, best first.

        A result list may be shorter than ``k`` when the store (or the filter)
        holds fewer than ``k`` candidates; it is never padded.
        """

    @abstractmethod
    def save(self, directory: Path) -> None:
        """Persist the index and its metadata sidecar."""

    @property
    @abstractmethod
    def filterable_fields(self) -> list[str]:
        """Payload fields available to ``where``."""
