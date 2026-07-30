"""Search service: the layer the API and UI both call.

Holds one index per encoder over the same corpus. Text queries route to a
multimodal encoder (CLIP/SigLIP2); image queries may use any encoder, or fuse
several. Encoders load lazily, because loading every backbone at startup
would cost far more memory and time than a demo needs.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vsearch.encoders import BaseEncoder, TextNotSupportedError, get_spec, load_encoder
from vsearch.index import FaissStore
from vsearch.index.base import Filter, SearchHit
from vsearch.search.fusion import reciprocal_rank_fusion

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL.Image import Image

logger = logging.getLogger(__name__)


class IndexNotLoadedError(RuntimeError):
    """Raised when a query names an encoder that has no index."""


@dataclass(frozen=True)
class SearchResult:
    """One ranked result, ready for display."""

    rank: int
    id: str
    score: float
    payload: Mapping[str, Any]
    thumbnail: Path | None = None

    @property
    def title(self) -> str:
        """Best available human label for the item."""
        for key in ("productDisplayName", "filename"):
            value = self.payload.get(key)
            if isinstance(value, str) and value:
                return value
        captions = self.payload.get("captions")
        if isinstance(captions, list) and captions:
            return str(captions[0])
        return self.id


@dataclass(frozen=True)
class SearchResponse:
    """Results plus what produced them."""

    results: list[SearchResult]
    encoder: str
    took_ms: float
    total_indexed: int
    fused: bool = False

    def __len__(self) -> int:
        return len(self.results)


@dataclass
class IndexHandle:
    """One built index plus where its thumbnails live."""

    encoder: str
    store: FaissStore
    images_dir: Path | None = None
    _encoder: BaseEncoder | None = field(default=None, repr=False)

    def resolve_encoder(
        self, *, device: str | None = None, batch_size: int | None = None, token: str | None = None
    ) -> BaseEncoder:
        """Load the backbone on first use and cache it."""
        if self._encoder is None:
            logger.info("Loading encoder %r for querying", self.encoder)
            self._encoder = load_encoder(
                self.encoder, device=device, batch_size=batch_size, token=token
            )
        return self._encoder

    def thumbnail_for(self, identifier: str) -> Path | None:
        """Resolve an item's thumbnail, refusing anything outside images_dir.

        The id reaches this method from an HTTP path parameter once the API
        serves thumbnails, so an id like "../../../.env" must not escape the
        directory. Containment is checked here rather than at the route, so a
        future caller cannot reintroduce the hole.
        """
        if self.images_dir is None:
            return None

        root = self.images_dir.resolve()
        candidate = (root / f"{identifier}.jpg").resolve()
        if not candidate.is_relative_to(root):
            logger.warning("Refused thumbnail path escaping the images dir: %r", identifier)
            return None
        return candidate if candidate.is_file() else None


class SearchService:
    """Routes queries to the right index and encoder."""

    def __init__(
        self,
        handles: Mapping[str, IndexHandle],
        *,
        device: str | None = None,
        batch_size: int | None = None,
        token: str | None = None,
        default_text_encoder: str | None = None,
        default_image_encoder: str | None = None,
    ) -> None:
        if not handles:
            msg = "SearchService needs at least one index"
            raise ValueError(msg)
        self._handles = dict(handles)
        self._device = device
        self._batch_size = batch_size
        self._token = token
        self._default_text = default_text_encoder or self._first_multimodal()
        self._default_image = default_image_encoder or next(iter(self._handles))

    # -- Introspection ------------------------------------------------------

    @property
    def encoders(self) -> list[str]:
        return sorted(self._handles)

    @property
    def text_encoders(self) -> list[str]:
        """Encoders that can answer a text query."""
        return sorted(name for name in self._handles if get_spec(name).supports_text)

    def _first_multimodal(self) -> str | None:
        return next(iter(self.text_encoders), None)

    def handle(self, encoder: str) -> IndexHandle:
        try:
            return self._handles[encoder]
        except KeyError:
            available = ", ".join(self.encoders)
            msg = f"no index loaded for encoder {encoder!r}; loaded: {available}"
            raise IndexNotLoadedError(msg) from None

    def warmup(self, encoders: Sequence[str] | None = None) -> list[str]:
        """Load backbones now rather than on a user's first query.

        Encoders load lazily to keep memory and startup honest, but that puts
        a multi-second model load on whoever queries first -- on a public demo
        that is the recruiter. Startup time is invisible; a slow first click
        is not. Returns the encoders actually warmed.
        """
        targets = list(encoders) if encoders is not None else self.encoders
        warmed: list[str] = []
        for name in targets:
            try:
                self.handle(name).resolve_encoder(
                    device=self._device, batch_size=self._batch_size, token=self._token
                )
            except (IndexNotLoadedError, OSError) as exc:
                # A cold demo is better than no demo: log and carry on.
                logger.warning("Could not warm up encoder %r: %s", name, exc)
                continue
            warmed.append(name)
        return warmed

    def facets(self, encoder: str | None = None) -> dict[str, list[Any]]:
        """Facet values for populating filter dropdowns."""
        store = self.handle(encoder or self._default_image).store
        return {name: list(store.facet_values(name)) for name in store.filterable_fields}

    def stats(self) -> dict[str, Any]:
        return {
            "encoders": self.encoders,
            "text_encoders": self.text_encoders,
            "default_text_encoder": self._default_text,
            "default_image_encoder": self._default_image,
            "indexes": {
                name: {
                    "count": len(handle.store),
                    "dim": handle.store.dim,
                    "backend": handle.store.backend,
                    "model_id": get_spec(name).model_id,
                    "has_thumbnails": handle.images_dir is not None,
                }
                for name, handle in sorted(self._handles.items())
            },
        }

    # -- Querying -----------------------------------------------------------

    def search_text(
        self,
        query: str,
        *,
        k: int = 24,
        encoder: str | None = None,
        where: Filter | None = None,
    ) -> SearchResponse:
        """Rank images against a natural-language query."""
        if not query.strip():
            msg = "query must not be empty"
            raise ValueError(msg)

        name = encoder or self._default_text
        if name is None:
            msg = "no multimodal index is loaded, so text search is unavailable"
            raise IndexNotLoadedError(msg)
        if not get_spec(name).supports_text:
            msg = (
                f"encoder {name!r} is vision-only and cannot embed text; "
                f"use one of: {', '.join(self.text_encoders) or '(none)'}"
            )
            raise TextNotSupportedError(msg)

        handle = self.handle(name)
        backbone = handle.resolve_encoder(
            device=self._device, batch_size=self._batch_size, token=self._token
        )
        # Timed after the encoder is resolved: on a cold service that call
        # loads model weights, and folding a one-off multi-second load into
        # "query latency" would misreport it everywhere it is shown.
        started = time.perf_counter()
        vector = backbone.encode_text([query])
        hits = handle.store.search(vector, k=k, where=where)[0]
        return self._respond(hits, handle, started)

    def search_image(
        self,
        image: Image,
        *,
        k: int = 24,
        encoder: str | None = None,
        where: Filter | None = None,
        fuse: Sequence[str] | None = None,
    ) -> SearchResponse:
        """Rank images against an example image.

        ``fuse`` names several encoders to combine with reciprocal rank
        fusion. Fused scores are RRF scores, not cosines, and are only
        comparable within the returned list.
        """
        if fuse:
            return self._search_image_fused(image, k=k, encoders=fuse, where=where)

        handle = self.handle(encoder or self._default_image)
        backbone = handle.resolve_encoder(
            device=self._device, batch_size=self._batch_size, token=self._token
        )
        started = time.perf_counter()
        vector = backbone.encode_image([image])
        hits = handle.store.search(vector, k=k, where=where)[0]
        return self._respond(hits, handle, started)

    def _search_image_fused(
        self,
        image: Image,
        *,
        k: int,
        encoders: Sequence[str],
        where: Filter | None,
    ) -> SearchResponse:
        # Resolve every backbone before timing, for the reason in search_text.
        backbones = [
            (
                self.handle(name),
                self.handle(name).resolve_encoder(
                    device=self._device, batch_size=self._batch_size, token=self._token
                ),
            )
            for name in encoders
        ]

        started = time.perf_counter()
        rankings: list[list[SearchHit]] = []
        for handle, backbone in backbones:
            vector = backbone.encode_image([image])
            # Over-retrieve per encoder: fusion needs depth to find the
            # agreements that make it better than either list alone.
            rankings.append(handle.store.search(vector, k=k * 2, where=where)[0])

        fused = reciprocal_rank_fusion(rankings, k=k)
        primary = self.handle(encoders[0])
        return self._respond(fused, primary, started, encoder="+".join(encoders), fused=True)

    # -- Internals ----------------------------------------------------------

    def _respond(
        self,
        hits: Sequence[SearchHit],
        handle: IndexHandle,
        started: float,
        *,
        encoder: str | None = None,
        fused: bool = False,
    ) -> SearchResponse:
        results = [
            SearchResult(
                rank=position,
                id=hit.id,
                score=hit.score,
                payload=hit.payload,
                thumbnail=handle.thumbnail_for(hit.id),
            )
            for position, hit in enumerate(hits, start=1)
        ]
        return SearchResponse(
            results=results,
            encoder=encoder or handle.encoder,
            took_ms=(time.perf_counter() - started) * 1000,
            total_indexed=len(handle.store),
            fused=fused,
        )


def discover_indexes(artifacts_dir: Path, corpus: str) -> dict[str, IndexHandle]:
    """Find built indexes for a corpus under ``artifacts_dir``.

    Directories are named ``<corpus>__<encoder>``, so the encoder is read back
    from the directory name rather than tracked separately.
    """
    handles: dict[str, IndexHandle] = {}
    if not artifacts_dir.is_dir():
        return handles

    for run_dir in sorted(artifacts_dir.glob(f"{corpus}__*")):
        index_dir = run_dir / "index"
        if not (index_dir / "config.json").exists():
            continue
        encoder = run_dir.name.split("__", 1)[1]
        images = run_dir / "images"
        handles[encoder] = IndexHandle(
            encoder=encoder,
            store=FaissStore.load(index_dir),
            images_dir=images if images.is_dir() else None,
        )
        logger.info("Loaded index %s (%d vectors)", run_dir.name, len(handles[encoder].store))
    return handles


def build_service(
    artifacts_dir: Path,
    corpus: str,
    *,
    device: str | None = None,
    batch_size: int | None = None,
    token: str | None = None,
) -> SearchService:
    """Load every index built for a corpus into one service."""
    handles = discover_indexes(artifacts_dir, corpus)
    if not handles:
        msg = (
            f"no built index for corpus {corpus!r} under {artifacts_dir}. "
            f"Run: vsearch ingest --corpus {corpus} --encoder clip"
        )
        raise IndexNotLoadedError(msg)
    return SearchService(handles, device=device, batch_size=batch_size, token=token)
