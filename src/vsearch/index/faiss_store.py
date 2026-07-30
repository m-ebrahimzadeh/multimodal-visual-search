"""FAISS-backed vector store with pre-filtered search.

Two backends:

``flat``
    ``IndexFlatIP`` -- exhaustive, exact. At corpus sizes in the tens of
    thousands this is genuinely fast and gives a ground-truth baseline to
    measure the approximate index against.

``hnsw``
    ``IndexHNSWFlat`` -- graph-based approximate search. Sub-linear query
    time, at some recall cost, which the benchmark quantifies.

Filtering is applied *inside* the FAISS search via an ``IDSelector`` rather
than by discarding results afterwards. Post-filtering an unfiltered top-k
silently returns fewer (often zero) results whenever a facet is selective;
pre-filtering still returns a full k.
"""

from __future__ import annotations

import json
from collections.abc import Hashable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from vsearch.encoders.base import Embeddings
from vsearch.index.base import Filter, Payload, SearchHit, VectorStore

Backend = Literal["flat", "hnsw"]

INDEX_FILENAME = "index.faiss"
STORE_FILENAME = "store.jsonl"
CONFIG_FILENAME = "config.json"

# FAISS pads short result rows with this id (and -inf score) rather than
# returning a shorter row. Leaking it produces phantom hits.
_MISSING = -1


class FaissStore(VectorStore):
    """Vector store backed by a FAISS index plus a metadata sidecar."""

    def __init__(
        self,
        dim: int,
        backend: Backend = "flat",
        *,
        hnsw_m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
    ) -> None:
        import faiss

        if dim <= 0:
            msg = f"dim must be positive, got {dim}"
            raise ValueError(msg)

        self._faiss = faiss
        self._dim = dim
        self._backend: Backend = backend
        self._hnsw_m = hnsw_m
        self._ef_construction = ef_construction
        self._ef_search = ef_search

        self._index = self._new_index()
        self._ids: list[str] = []
        self._payloads: list[Payload] = []
        self._id_positions: dict[str, int] = {}
        # field -> value -> positions holding that value
        self._facets: dict[str, dict[Hashable, set[int]]] = {}

    def _new_index(self) -> Any:
        faiss = self._faiss
        if self._backend == "flat":
            return faiss.IndexFlatIP(self._dim)
        index = faiss.IndexHNSWFlat(self._dim, self._hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self._ef_construction
        index.hnsw.efSearch = self._ef_search
        return index

    # -- Introspection ------------------------------------------------------

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def filterable_fields(self) -> list[str]:
        return sorted(self._facets)

    def __len__(self) -> int:
        return len(self._ids)

    def __repr__(self) -> str:
        return f"FaissStore(backend={self._backend!r}, dim={self._dim}, n={len(self._ids)})"

    def ids(self) -> list[str]:
        return list(self._ids)

    def facet_values(self, field: str) -> list[Hashable]:
        """Distinct values for a facet, for populating UI dropdowns."""
        if field not in self._facets:
            msg = f"unknown facet {field!r}; available: {', '.join(self.filterable_fields)}"
            raise KeyError(msg)
        return sorted(self._facets[field], key=repr)

    # -- Writing ------------------------------------------------------------

    def add(
        self,
        ids: Sequence[str],
        vectors: Embeddings,
        payloads: Sequence[Payload] | None = None,
    ) -> None:
        if vectors.ndim != 2:
            msg = f"expected a 2-D (n, dim) array, got shape {vectors.shape}"
            raise ValueError(msg)
        if vectors.shape[1] != self._dim:
            msg = f"embedding dim {vectors.shape[1]} does not match index dim {self._dim}"
            raise ValueError(msg)
        if len(ids) != vectors.shape[0]:
            msg = f"got {len(ids)} ids for {vectors.shape[0]} vectors"
            raise ValueError(msg)
        if payloads is not None and len(payloads) != len(ids):
            msg = f"got {len(payloads)} payloads for {len(ids)} ids"
            raise ValueError(msg)

        duplicates = [i for i in ids if i in self._id_positions]
        if duplicates:
            preview = ", ".join(map(repr, duplicates[:3]))
            msg = f"ids must be unique; {len(duplicates)} already indexed (e.g. {preview})"
            raise ValueError(msg)

        # FAISS requires contiguous float32 and will otherwise copy or fail
        # with an opaque error from inside the C++ layer.
        prepared = np.ascontiguousarray(vectors, dtype=np.float32)
        self._index.add(prepared)

        start = len(self._ids)
        for offset, identifier in enumerate(ids):
            position = start + offset
            payload = dict(payloads[offset]) if payloads is not None else {}
            self._ids.append(identifier)
            self._payloads.append(payload)
            self._id_positions[identifier] = position
            self._index_facets(position, payload)

    def _index_facets(self, position: int, payload: Payload) -> None:
        """Add a payload's scalar fields to the inverted index.

        Only hashable scalars are indexed. List-valued fields (Flickr30k's
        five captions per image, say) are stored but not filterable -- they
        are free text, not facets.
        """
        for key, value in payload.items():
            # bool is a subclass of int, so it is covered. None is excluded:
            # "missing" is not a facet worth filtering on.
            if not isinstance(value, str | int | float):
                continue
            # NaN is never equal to itself, so each NaN would become its own
            # dict key and quietly bloat the facet index. The fashion corpus
            # has a float `year` column with gaps, so this is not theoretical.
            if isinstance(value, float) and np.isnan(value):
                continue
            self._facets.setdefault(key, {}).setdefault(value, set()).add(position)

    # -- Reading ------------------------------------------------------------

    def _positions_for(self, where: Filter) -> np.ndarray:
        """Resolve a facet filter to the sorted positions it allows."""
        matched_per_field: list[set[int]] = []
        for field, wanted in where.items():
            if field not in self._facets:
                available = ", ".join(self.filterable_fields) or "(none)"
                msg = f"cannot filter on {field!r}; filterable fields: {available}"
                raise KeyError(msg)

            values: Iterable[object] = (
                wanted if isinstance(wanted, list | tuple | set | frozenset) else [wanted]
            )
            index = self._facets[field]
            matched: set[int] = set()
            for value in values:
                if isinstance(value, Hashable):
                    matched |= index.get(value, set())
            matched_per_field.append(matched)

        if not matched_per_field:
            return np.arange(len(self._ids), dtype=np.int64)
        allowed = set.intersection(*matched_per_field)
        return np.fromiter(sorted(allowed), dtype=np.int64, count=len(allowed))

    def search(
        self,
        queries: Embeddings,
        k: int,
        where: Filter | None = None,
    ) -> list[list[SearchHit]]:
        if k <= 0:
            msg = f"k must be positive, got {k}"
            raise ValueError(msg)
        if queries.ndim != 2:
            msg = f"expected a 2-D (n, dim) query array, got shape {queries.shape}"
            raise ValueError(msg)
        if queries.shape[1] != self._dim:
            msg = f"query dim {queries.shape[1]} does not match index dim {self._dim}"
            raise ValueError(msg)
        if len(self._ids) == 0:
            return [[] for _ in range(queries.shape[0])]

        prepared = np.ascontiguousarray(queries, dtype=np.float32)

        params = None
        if where:
            allowed = self._positions_for(where)
            if allowed.size == 0:
                # Nothing matches the facets; FAISS would happily return a
                # full row of -1 padding, so short-circuit instead.
                return [[] for _ in range(queries.shape[0])]
            # Held in a local so SWIG does not collect it mid-search.
            selector = self._faiss.IDSelectorBatch(allowed)
            params = self._search_params(k)
            params.sel = selector

        scores, positions = self._index.search(prepared, k, params=params)
        return [
            self._to_hits(row_scores, row_positions)
            for row_scores, row_positions in zip(scores, positions, strict=True)
        ]

    def _search_params(self, k: int) -> Any:
        if self._backend != "hnsw":
            return self._faiss.SearchParameters()
        # Present at runtime (faiss-cpu 1.14.3) but absent from its bundled
        # stubs. Exercised by test_hnsw_supports_filtering.
        params = self._faiss.SearchParametersHNSW()  # type: ignore[attr-defined]
        # efSearch below k cannot return k candidates; raise the floor rather
        # than quietly returning a short, low-recall result set.
        params.efSearch = max(self._ef_search, k)
        return params

    def _to_hits(self, scores: np.ndarray, positions: np.ndarray) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for score, position in zip(scores, positions, strict=True):
            index = int(position)
            # FAISS pads short rows with -1 / -inf. Emitting those would
            # produce phantom results and -inf scores downstream.
            if index == _MISSING:
                continue
            hits.append(
                SearchHit(
                    id=self._ids[index],
                    score=float(score),
                    payload=self._payloads[index],
                )
            )
        return hits

    def get(self, identifier: str) -> SearchHit | None:
        """Look up a single indexed item by id."""
        position = self._id_positions.get(identifier)
        if position is None:
            return None
        return SearchHit(id=identifier, score=1.0, payload=self._payloads[position])

    # -- Persistence --------------------------------------------------------

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)

        self._faiss.write_index(self._index, str(directory / INDEX_FILENAME))

        with (directory / STORE_FILENAME).open("w", encoding="utf-8") as handle:
            for identifier, payload in zip(self._ids, self._payloads, strict=True):
                handle.write(json.dumps({"id": identifier, "payload": payload}) + "\n")

        config = {
            "dim": self._dim,
            "backend": self._backend,
            "hnsw_m": self._hnsw_m,
            "ef_construction": self._ef_construction,
            "ef_search": self._ef_search,
            "count": len(self._ids),
        }
        (directory / CONFIG_FILENAME).write_text(json.dumps(config, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> FaissStore:
        import faiss

        config_path = directory / CONFIG_FILENAME
        if not config_path.exists():
            msg = f"no index found at {directory} (missing {CONFIG_FILENAME})"
            raise FileNotFoundError(msg)

        config = json.loads(config_path.read_text(encoding="utf-8"))
        store = cls(
            dim=int(config["dim"]),
            backend=cast("Backend", config["backend"]),
            hnsw_m=int(config.get("hnsw_m", 32)),
            ef_construction=int(config.get("ef_construction", 200)),
            ef_search=int(config.get("ef_search", 64)),
        )
        store._index = faiss.read_index(str(directory / INDEX_FILENAME))

        # Facets are rebuilt from payloads rather than serialised: it is fast,
        # and it removes any chance of the index and its facets drifting apart.
        with (directory / STORE_FILENAME).open(encoding="utf-8") as handle:
            for position, line in enumerate(handle):
                record: Mapping[str, Any] = json.loads(line)
                identifier = str(record["id"])
                payload = dict(record.get("payload") or {})
                store._ids.append(identifier)
                store._payloads.append(payload)
                store._id_positions[identifier] = position
                store._index_facets(position, payload)

        if store._index.ntotal != len(store._ids):
            msg = (
                f"index holds {store._index.ntotal} vectors but the sidecar has "
                f"{len(store._ids)} ids; the artifact is corrupt"
            )
            raise ValueError(msg)
        return store
