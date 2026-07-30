"""Evaluation protocols: how each corpus defines a query and a correct answer.

Two protocols, deliberately different, because the corpora support different
evidence:

``flickr30k`` (text -> image)
    The canonical benchmark. The corpus carries a train/val/test column, so
    restricting to ``test`` gives the standard 1000 images with five captions
    each -- 5000 queries whose Recall@k is directly comparable to published
    CLIP results. This is real ground truth, written by humans.

``fashion`` (image -> image)
    The corpus has no captions, so there is no honest text->image ground
    truth here. Relevance is a *documented proxy*: two products count as
    matching when they share articleType and baseColour. That is a reasonable
    stand-in for "visually similar product", but it is a proxy, and the
    reported numbers say so rather than implying human judgements.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vsearch.eval.retrieval import Judged
from vsearch.index import FaissStore

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vsearch.encoders.base import BaseEncoder

logger = logging.getLogger(__name__)

FASHION_RELEVANCE_FIELDS = ("articleType", "baseColour")
"""Fields whose exact match defines the image->image relevance proxy."""


@dataclass(frozen=True)
class TextQuery:
    """A caption and the id of the image it was written for."""

    text: str
    target_id: str


def flickr_text_queries(
    store: FaissStore,
    *,
    split: str | None = "test",
    max_queries: int | None = None,
) -> list[TextQuery]:
    """Build text->image queries from captions stored alongside each image.

    Restricting to the ``test`` split is what makes the result comparable to
    published numbers; evaluating over all 31k images would produce a number
    that looks similar but means something else.
    """
    queries: list[TextQuery] = []
    for identifier in store.ids():
        hit = store.get(identifier)
        if hit is None:
            continue
        payload = hit.payload
        if split is not None and payload.get("split") != split:
            continue
        captions = payload.get("captions")
        if not isinstance(captions, list):
            continue
        for caption in captions:
            if isinstance(caption, str) and caption.strip():
                queries.append(TextQuery(text=caption, target_id=identifier))
        if max_queries is not None and len(queries) >= max_queries:
            break
    return queries[:max_queries] if max_queries is not None else queries


def _relevance_key(payload: Any) -> tuple[Any, ...] | None:
    values = tuple(payload.get(field) for field in FASHION_RELEVANCE_FIELDS)
    return None if any(value is None for value in values) else values


def label_relevance_groups(store: FaissStore) -> dict[tuple[Any, ...], set[str]]:
    """Group item ids by their relevance key."""
    groups: dict[tuple[Any, ...], set[str]] = {}
    for identifier in store.ids():
        hit = store.get(identifier)
        if hit is None:
            continue
        key = _relevance_key(hit.payload)
        if key is not None:
            groups.setdefault(key, set()).add(identifier)
    return groups


def evaluate_text_to_image(
    store: FaissStore,
    encoder: BaseEncoder,
    queries: Sequence[TextQuery],
    *,
    k: int,
    batch_size: int = 64,
) -> Iterator[Judged]:
    """Run text queries and yield judgements.

    Queries are encoded in batches; one forward pass per caption would make a
    5000-query run needlessly slow.
    """
    for start in range(0, len(queries), batch_size):
        window = queries[start : start + batch_size]
        vectors = encoder.encode_text([query.text for query in window])
        for query, hits in zip(window, store.search(vectors, k=k), strict=True):
            yield Judged(
                ranked=[hit.id for hit in hits],
                relevant=frozenset({query.target_id}),
            )


def evaluate_image_to_image(
    store: FaissStore,
    query_ids: Sequence[str],
    vectors: Any,
    *,
    k: int,
) -> Iterator[Judged]:
    """Yield judgements for image->image queries under the label proxy.

    The query image is itself in the index, so it would trivially rank first;
    it is removed from both the ranking and the relevant set, otherwise every
    metric is inflated by a guaranteed self-match.
    """
    groups = label_relevance_groups(store)
    results = store.search(vectors, k=k + 1)

    for identifier, hits in zip(query_ids, results, strict=True):
        hit = store.get(identifier)
        if hit is None:
            continue
        key = _relevance_key(hit.payload)
        if key is None:
            continue
        relevant = groups.get(key, set()) - {identifier}
        if not relevant:
            # Nothing else shares its labels, so the query is unscoreable
            # rather than a failure. Counting it as 0 would understate the
            # system; counting it as 1 would overstate it.
            continue
        ranked = [h.id for h in hits if h.id != identifier][:k]
        yield Judged(ranked=ranked, relevant=frozenset(relevant))
