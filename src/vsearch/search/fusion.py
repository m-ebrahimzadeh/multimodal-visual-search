"""Combining ranked lists from different encoders.

CLIP and DINOv3 score on incompatible scales: CLIP image-text cosines cluster
around 0.2-0.35, while DINOv3 image-image cosines run much higher. Averaging
or summing those raw scores lets whichever encoder happens to use a wider
range dominate, regardless of which one actually ranked better.

Reciprocal Rank Fusion sidesteps this by discarding magnitudes and combining
*positions*, so the two encoders contribute comparably by construction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from vsearch.index.base import SearchHit

RRF_K = 60
"""Rank-offset constant. The standard value from the original RRF paper; it
damps the influence of the very top ranks so a single encoder's #1 cannot
automatically win the fused list."""


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchHit]],
    *,
    weights: Sequence[float] | None = None,
    k: int | None = None,
    rrf_k: int = RRF_K,
) -> list[SearchHit]:
    """Fuse ranked hit lists into one, scored by summed reciprocal rank.

    Each list contributes ``weight / (rrf_k + rank)`` per document, with rank
    counted from 1. Documents found by several encoders accumulate score,
    which is the property that makes fusion worth doing.

    The returned ``score`` is an RRF score, not a cosine -- it is only
    meaningful relative to other entries in the same fused list.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        msg = f"got {len(weights)} weights for {len(rankings)} rankings"
        raise ValueError(msg)

    scores: dict[str, float] = {}
    payloads: dict[str, SearchHit] = {}

    for ranking, weight in zip(rankings, weights, strict=True):
        for position, hit in enumerate(ranking, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + weight / (rrf_k + position)
            # Keep the first payload seen; they describe the same item, and
            # every index is built over the same corpus.
            payloads.setdefault(hit.id, hit)

    # Sort by score, breaking ties on id so the ordering is deterministic --
    # otherwise two runs over the same data can disagree.
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if k is not None:
        ordered = ordered[:k]

    return [
        SearchHit(id=identifier, score=score, payload=payloads[identifier].payload)
        for identifier, score in ordered
    ]


def merge_payloads(hits: Sequence[SearchHit]) -> Mapping[str, SearchHit]:
    """Index hits by id, for joining results across encoders."""
    return {hit.id: hit for hit in hits}
