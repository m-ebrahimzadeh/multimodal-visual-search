"""Paired comparison between two evaluated configurations.

A point estimate invites over-reading. On the fashion slice DINOv3 leads CLIP
by +0.086 Recall@10, which reads as a win until you resample the query set and
find the interval straddles zero. With 39 scoreable queries the experiment
simply cannot separate the two encoders, and a table that prints only the delta
does not say so.

So every comparison reports a bootstrap confidence interval next to the delta,
and a verdict derived from it rather than from the sign of the difference.

The test is *paired*: the same queries are scored by both configurations and
the per-query difference is what gets resampled. Comparing two independent
means would throw away the fact that a hard query is hard for both, which is
most of the variance here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from vsearch.eval.retrieval import (
    Judged,
    average_precision,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

logger = logging.getLogger(__name__)

DEFAULT_RESAMPLES = 10_000
"""Enough for a stable 95% interval; the cost is milliseconds at this scale."""

_BLOCK = 2048
"""Resamples drawn per batch. The index array is ``block x queries``, so
drawing all 10k at once would allocate hundreds of MB on a full-corpus run."""

MetricFn = Callable[[Judged], float]

DEFAULT_METRICS: dict[str, MetricFn] = {
    "R@1": lambda judged: recall_at_k(judged, 1),
    "R@5": lambda judged: recall_at_k(judged, 5),
    "R@10": lambda judged: recall_at_k(judged, 10),
    "MRR": reciprocal_rank,
    "mAP": average_precision,
    "nDCG@10": lambda judged: ndcg_at_k(judged, 10),
}


@dataclass(frozen=True)
class Comparison:
    """One metric compared across two configurations."""

    metric: str
    queries: int
    baseline: float
    candidate: float
    ci_low: float
    ci_high: float

    @property
    def delta(self) -> float:
        return self.candidate - self.baseline

    @property
    def significant(self) -> bool:
        """True when the 95% interval excludes zero.

        Not a claim of practical importance -- only that the sign of the
        difference survived resampling the query set.
        """
        return self.ci_low > 0.0 or self.ci_high < 0.0

    @property
    def verdict(self) -> str:
        if not self.significant:
            return "within noise"
        return "candidate wins" if self.delta > 0 else "baseline wins"

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "queries": self.queries,
            "baseline": round(self.baseline, 4),
            "candidate": round(self.candidate, 4),
            "delta": round(self.delta, 4),
            "ci_low": round(self.ci_low, 4),
            "ci_high": round(self.ci_high, 4),
            "significant": self.significant,
        }


def _bootstrap_interval(
    differences: NDArray[np.float64],
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean paired difference.

    Resamples the per-query differences rather than each configuration's
    scores separately: for a shared resampling index the two are algebraically
    identical (``mean(b[i]) - mean(a[i]) == mean((b - a)[i])``) and this form
    halves the memory.
    """
    n = differences.size
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)

    for start in range(0, resamples, _BLOCK):
        size = min(_BLOCK, resamples - start)
        index = rng.integers(0, n, size=(size, n))
        means[start : start + size] = differences[index].mean(axis=1)

    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def paired_bootstrap(
    baseline: Mapping[str, Judged],
    candidate: Mapping[str, Judged],
    *,
    metrics: Mapping[str, MetricFn] | None = None,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> list[Comparison]:
    """Compare two configurations over the queries they both scored.

    ``seed`` is fixed by default so a reported interval is reproducible; a
    confidence interval that moves between runs of the same data is not
    evidence anyone can check.

    Queries are matched by key, never by position: the two runs skip
    unscoreable queries independently, so equal counts do not imply equal
    query sets.
    """
    shared = sorted(baseline.keys() & candidate.keys())
    if not shared:
        msg = "the two runs share no scoreable queries; nothing to compare"
        raise ValueError(msg)

    dropped = (len(baseline) - len(shared)) + (len(candidate) - len(shared))
    if dropped:
        logger.warning(
            "Comparing on %d shared queries; %d were scoreable in only one run.",
            len(shared),
            dropped,
        )

    chosen = dict(DEFAULT_METRICS) if metrics is None else dict(metrics)
    results: list[Comparison] = []
    for name, fn in chosen.items():
        left = np.array([fn(baseline[key]) for key in shared], dtype=np.float64)
        right = np.array([fn(candidate[key]) for key in shared], dtype=np.float64)
        low, high = _bootstrap_interval(right - left, resamples=resamples, seed=seed)
        results.append(
            Comparison(
                metric=name,
                queries=len(shared),
                baseline=float(left.mean()),
                candidate=float(right.mean()),
                ci_low=low,
                ci_high=high,
            )
        )
    return results


def to_markdown_table(
    comparisons: Sequence[Comparison],
    *,
    baseline_label: str,
    candidate_label: str,
) -> str:
    """Render comparisons as a markdown table for the README."""
    if not comparisons:
        return "_no comparison_"

    header = f"| metric | {baseline_label} | {candidate_label} | delta | 95% CI | verdict |"
    lines = [header, "|" + "|".join(["---"] * 6) + "|"]
    for item in comparisons:
        lines.append(
            f"| {item.metric} | {item.baseline:.4f} | {item.candidate:.4f} | "
            f"{item.delta:+.4f} | [{item.ci_low:+.4f}, {item.ci_high:+.4f}] | {item.verdict} |"
        )
    return "\n".join(lines)
