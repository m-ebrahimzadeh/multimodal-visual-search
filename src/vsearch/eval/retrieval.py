"""Retrieval metrics.

All metrics take a ranked list of ids and the set of ids that count as
relevant, so the same code serves both evaluation protocols:

* **text -> image** on Flickr30k, where exactly one image is correct for a
  caption. Recall@k is then a hit rate, which is what published CLIP numbers
  report and what makes ours comparable.
* **image -> image** on the fashion corpus, where relevance is a documented
  label proxy (same articleType and baseColour) and many items qualify. There
  Recall@10 is bounded by 10/|relevant| and looks artificially terrible, so
  Precision@k and mAP are the honest summaries.

Reporting a single "score" across both would be meaningless; the runner picks
the right metric per protocol and the table says which.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field

DEFAULT_KS = (1, 5, 10)


@dataclass(frozen=True)
class Judged:
    """One query's ranked results and the ids that count as relevant."""

    ranked: Sequence[str]
    relevant: frozenset[str]

    def __post_init__(self) -> None:
        if not self.relevant:
            msg = "a query with no relevant items cannot be scored; filter it out first"
            raise ValueError(msg)


def recall_at_k(judged: Judged, k: int) -> float:
    """Fraction of relevant items retrieved in the top k.

    With a single relevant item this is a hit rate (0 or 1), which is the
    standard Flickr30k text->image protocol.
    """
    top = set(judged.ranked[:k])
    return len(top & judged.relevant) / len(judged.relevant)


def precision_at_k(judged: Judged, k: int) -> float:
    """Fraction of the top k that is relevant.

    Divided by k, not by the number retrieved: a query that returns 3 results
    for k=10 should not score as though it returned 10 good ones.
    """
    if k <= 0:
        msg = f"k must be positive, got {k}"
        raise ValueError(msg)
    top = judged.ranked[:k]
    return sum(1 for identifier in top if identifier in judged.relevant) / k


def reciprocal_rank(judged: Judged) -> float:
    """1/rank of the first relevant hit, or 0 if none was retrieved."""
    for position, identifier in enumerate(judged.ranked, start=1):
        if identifier in judged.relevant:
            return 1.0 / position
    return 0.0


def average_precision(judged: Judged) -> float:
    """Mean of the precisions at each relevant hit.

    Normalised by min(|relevant|, |ranked|) rather than |relevant|: when the
    ranked list is shorter than the relevant set, dividing by |relevant|
    reports a low score for a system that retrieved everything it could.
    """
    hits = 0
    total = 0.0
    for position, identifier in enumerate(judged.ranked, start=1):
        if identifier in judged.relevant:
            hits += 1
            total += hits / position
    reachable = min(len(judged.relevant), len(judged.ranked))
    return total / reachable if reachable else 0.0


def ndcg_at_k(judged: Judged, k: int) -> float:
    """Normalised discounted cumulative gain with binary relevance."""
    gain = sum(
        1.0 / math.log2(position + 1)
        for position, identifier in enumerate(judged.ranked[:k], start=1)
        if identifier in judged.relevant
    )
    ideal_hits = min(len(judged.relevant), k)
    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
    return gain / ideal if ideal else 0.0


@dataclass(frozen=True)
class RetrievalMetrics:
    """Aggregated metrics over a query set."""

    queries: int
    recall: dict[int, float] = field(default_factory=dict)
    precision: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    map: float = 0.0
    ndcg: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        # JSON object keys must be strings; int keys would round-trip as "1".
        for name in ("recall", "precision", "ndcg"):
            payload[name] = {str(k): v for k, v in getattr(self, name).items()}
        return payload

    def summary_row(self, ks: Sequence[int] = DEFAULT_KS) -> dict[str, float | int]:
        row: dict[str, float | int] = {"queries": self.queries}
        for k in ks:
            row[f"R@{k}"] = round(self.recall.get(k, 0.0), 4)
        row["MRR"] = round(self.mrr, 4)
        row["mAP"] = round(self.map, 4)
        row[f"nDCG@{max(ks)}"] = round(self.ndcg.get(max(ks), 0.0), 4)
        return row


def evaluate(
    judgements: Iterable[Judged],
    ks: Sequence[int] = DEFAULT_KS,
) -> RetrievalMetrics:
    """Aggregate metrics across a query set (macro-averaged over queries)."""
    items = list(judgements)
    if not items:
        return RetrievalMetrics(queries=0)

    n = len(items)
    return RetrievalMetrics(
        queries=n,
        recall={k: sum(recall_at_k(j, k) for j in items) / n for k in ks},
        precision={k: sum(precision_at_k(j, k) for j in items) / n for k in ks},
        mrr=sum(reciprocal_rank(j) for j in items) / n,
        map=sum(average_precision(j) for j in items) / n,
        ndcg={k: sum(ndcg_at_k(j, k) for j in items) / n for k in ks},
    )


def to_markdown_table(rows: dict[str, dict[str, float | int]], ks: Sequence[int]) -> str:
    """Render named metric rows as a markdown table for the README."""
    if not rows:
        return "_no results_"

    columns = ["config", "queries", *[f"R@{k}" for k in ks], "MRR", "mAP", f"nDCG@{max(ks)}"]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for name, row in rows.items():
        cells = [name] + [str(row.get(column, "")) for column in columns[1:]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
