"""Export a built index as the static bundle the browser searches.

A browser has no FAISS, no numpy and no Python. What a ``flat`` index actually
*is*, though, is the only thing it needs: a contiguous block of float32
vectors. This module writes that block out verbatim, so the deployed demo
searches the same bytes the evaluation tables were measured on rather than a
re-derived copy that can silently drift from them.

That is also why the bundle carries no vector database. At these corpus sizes
an exhaustive scan is a few tens of thousands of multiply-adds -- microseconds
in JavaScript -- and introducing a hosted index would mean maintaining a second
copy of the embeddings whose agreement with ``index.faiss`` nothing checks.

Four files leave here:

``embeddings.bin``
    ``count x dim`` float32, row-major, little-endian. Not JSON: JSON would
    roughly triple the transfer and still need parsing into a typed array on
    arrival.

``corpus.json``
    Ids, payloads and facet values, **in the row order of embeddings.bin**.
    That order is the only join key -- the binary has no id column.

``examples.json``
    Query vectors for the example searches, encoded here by the same fp32
    PyTorch encoder that built the index. Two jobs: the page answers them
    instantly, before it has fetched 62 MB of in-browser text encoder, and it
    can re-encode the same strings afterwards to measure that encoder against
    a local fp32 ground truth. If the encoder cannot be fetched at all, these
    keep the page working instead of breaking a link on a CV.

``ATTRIBUTION.md``
    Where the thumbnails came from. They are third-party dataset images, and a
    bundle that redistributes them should say so next to them.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from vsearch.encoders.base import BaseEncoder

logger = logging.getLogger(__name__)

EMBEDDINGS_FILENAME = "embeddings.bin"
CORPUS_FILENAME = "corpus.json"
EXAMPLES_FILENAME = "examples.json"
ATTRIBUTION_FILENAME = "ATTRIBUTION.md"
IMAGES_DIRNAME = "images"

# float32 little-endian. JS ``Float32Array`` reads native byte order, which is
# little-endian on every platform Workers runs on -- but "every platform I know
# of" is not a serialisation format, so the width and order are pinned here
# rather than inherited from whatever machine ran the export.
_DTYPE = np.dtype("<f4")

# Unit norm is what makes an inner product a cosine. The encoders normalise, so
# this is an assertion about the artifact rather than a transformation of it;
# the tolerance covers float32 accumulation, not genuine drift.
_NORM_TOLERANCE = 1e-3

# Free-text fields are near-unique per row, so they are terrible filters but
# good labels. The store already excludes them from faceting; the UI needs to
# know which payload key to render as the item's title.
_TITLE_FIELDS = ("productDisplayName", "caption", "filename")

DEFAULT_EXAMPLES: tuple[str, ...] = (
    "a navy blue formal shirt",
    "silver wristwatch",
    "black leather handbag",
    "casual blue jeans",
    "running shoes",
    "something to wear to the beach in summer",
)
"""Seed queries for the demo's one-click examples.

The last one is deliberately not a product name. A corpus of labelled apparel
will match "blue jeans" on the label alone, which demonstrates nothing that a
``LIKE '%jeans%'`` could not; a query with no keyword in common with any stored
field only resolves in embedding space.
"""


@dataclass(frozen=True)
class WebBundle:
    """What an export wrote, for logging and for tests to assert against."""

    destination: Path
    count: int
    dim: int
    corpus: str
    encoder: str
    model_id: str
    images: int
    examples: int

    @property
    def embeddings_bytes(self) -> int:
        return self.count * self.dim * _DTYPE.itemsize


def _title_field(payloads: Sequence[dict[str, Any]]) -> str | None:
    """Pick the payload key to show as an item's caption."""
    for field in _TITLE_FIELDS:
        if any(field in payload for payload in payloads):
            return field
    return None


def _jsonable(value: object) -> object:
    """Replace the float values that Python will emit as invalid JSON.

    Payloads reach here having already round-tripped through ``store.jsonl``,
    so they are JSON-native types -- with one exception. ``json.dumps`` writes
    a bare ``NaN``/``Infinity`` and ``json.loads`` reads it back again, so a
    non-finite float survives the store untouched and lands in the bundle. No
    browser accepts it: ``JSON.parse`` rejects the *whole document*, so one
    missing year takes the entire corpus down.

    The fashion corpus has a float ``year`` column with gaps, so this is a
    live case rather than a defensive one.
    """
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _encode_examples(encoder: BaseEncoder, prompts: Sequence[str]) -> list[dict[str, Any]]:
    """Embed the example queries so the demo can answer without the model."""
    if not prompts:
        return []
    vectors = np.asarray(encoder.encode_text(list(prompts)), dtype=np.float32)
    # Rounded because these ship as JSON text. float32 carries ~7 significant
    # digits, so six decimals on a unit-norm component is lossless in practice
    # while roughly halving the file against full repr().
    return [
        {"text": prompt, "vector": [round(float(component), 6) for component in row]}
        for prompt, row in zip(prompts, vectors, strict=True)
    ]


def export_bundle(
    run_dir: Path,
    destination: Path,
    *,
    encoder: BaseEncoder | None = None,
    examples: Sequence[str] = DEFAULT_EXAMPLES,
    source: str | None = None,
) -> WebBundle:
    """Write ``run_dir``'s index out as static assets under ``destination``.

    ``encoder`` is only needed to embed the example queries. Passing ``None``
    skips them, which keeps the export runnable on a machine with no model
    weights -- at the cost of the offline fallback path.
    """
    from vsearch.index.faiss_store import FaissStore

    index_dir = run_dir / "index"
    store = FaissStore.load(index_dir)
    ids = store.ids()
    if not ids:
        msg = f"{index_dir} holds no vectors; nothing to export"
        raise ValueError(msg)

    vectors = store.vectors_for(ids)
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=_NORM_TOLERANCE):
        # The Worker scores with a plain dot product. Un-normalised rows would
        # not error there -- they would just rank longer vectors higher and
        # quietly stop being a cosine ranking.
        worst = float(np.abs(norms - 1.0).max())
        msg = (
            f"embeddings are not unit-norm (max deviation {worst:.2e}); "
            "the Worker scores by dot product and would not be ranking by cosine"
        )
        raise ValueError(msg)

    manifest_path = run_dir / "manifest.json"
    manifest: dict[str, Any] = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )

    destination.mkdir(parents=True, exist_ok=True)
    (destination / EMBEDDINGS_FILENAME).write_bytes(
        np.ascontiguousarray(vectors, dtype=_DTYPE).tobytes()
    )

    payloads = [dict(hit.payload) if (hit := store.get(identifier)) else {} for identifier in ids]
    title_field = _title_field(payloads)

    images_out = destination / IMAGES_DIRNAME
    images_in = run_dir / IMAGES_DIRNAME
    images_out.mkdir(parents=True, exist_ok=True)
    copied = 0
    items: list[dict[str, Any]] = []
    for identifier, payload in zip(ids, payloads, strict=True):
        thumbnail = images_in / f"{identifier}.jpg"
        has_image = thumbnail.exists()
        if has_image:
            shutil.copy2(thumbnail, images_out / thumbnail.name)
            copied += 1
        items.append(
            {
                "id": identifier,
                "image": f"{IMAGES_DIRNAME}/{thumbnail.name}" if has_image else None,
                "title": str(payload.get(title_field, identifier)) if title_field else identifier,
                "payload": {key: _jsonable(value) for key, value in payload.items()},
            }
        )

    if copied != len(ids):
        # Not fatal: the grid renders a placeholder. Logged because a bundle
        # that is quietly half thumbnails still deploys and still looks broken.
        logger.warning(
            "%d of %d items have no thumbnail in %s", len(ids) - copied, len(ids), images_in
        )

    facets = {
        field: [_jsonable(value) for value in store.facet_values(field)]
        for field in store.filterable_fields
    }
    corpus_name = str(manifest.get("corpus", run_dir.name))
    encoder_name = str(manifest.get("encoder", "unknown"))
    model_id = str(manifest.get("model_id", "unknown"))

    (destination / CORPUS_FILENAME).write_text(
        json.dumps(
            {
                "corpus": corpus_name,
                "encoder": encoder_name,
                "model_id": model_id,
                "dim": store.dim,
                "count": len(ids),
                "title_field": title_field,
                "facets": facets,
                "items": items,
            },
            indent=None,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    encoded_examples = _encode_examples(encoder, examples) if encoder is not None else []
    (destination / EXAMPLES_FILENAME).write_text(
        json.dumps({"model_id": model_id, "examples": encoded_examples}, separators=(",", ":")),
        encoding="utf-8",
    )

    _write_attribution(destination / ATTRIBUTION_FILENAME, corpus_name, source, copied)

    bundle = WebBundle(
        destination=destination,
        count=len(ids),
        dim=store.dim,
        corpus=corpus_name,
        encoder=encoder_name,
        model_id=model_id,
        images=copied,
        examples=len(encoded_examples),
    )
    logger.info(
        "exported %d x %d vectors (%.1f KB) + %d thumbnails -> %s",
        bundle.count,
        bundle.dim,
        bundle.embeddings_bytes / 1024,
        bundle.images,
        destination,
    )
    return bundle


def _write_attribution(path: Path, corpus: str, source: str | None, images: int) -> None:
    origin = source or "see the project README for this corpus's dataset link"
    path.write_text(
        f"""# Attribution

The {images} thumbnails in `{IMAGES_DIRNAME}/` are redistributed from the
`{corpus}` corpus: {origin}

They are included so the demo can render search results, and are the property
of their original creators. No ownership is claimed over them here.
""",
        encoding="utf-8",
    )
