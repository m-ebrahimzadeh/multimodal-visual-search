"""Retrieval evaluation."""

from vsearch.eval.compare import (
    DEFAULT_METRICS,
    Comparison,
    paired_bootstrap,
)
from vsearch.eval.protocols import (
    FASHION_RELEVANCE_FIELDS,
    TextQuery,
    evaluate_image_to_image,
    evaluate_text_to_image,
    flickr_text_queries,
    image_to_image_pairs,
    label_relevance_groups,
    shuffled_queries,
    text_to_image_pairs,
)
from vsearch.eval.retrieval import (
    DEFAULT_KS,
    Judged,
    RetrievalMetrics,
    average_precision,
    evaluate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    to_markdown_table,
)

__all__ = [
    "DEFAULT_KS",
    "DEFAULT_METRICS",
    "FASHION_RELEVANCE_FIELDS",
    "Comparison",
    "Judged",
    "RetrievalMetrics",
    "TextQuery",
    "average_precision",
    "evaluate",
    "evaluate_image_to_image",
    "evaluate_text_to_image",
    "flickr_text_queries",
    "image_to_image_pairs",
    "label_relevance_groups",
    "ndcg_at_k",
    "paired_bootstrap",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "shuffled_queries",
    "text_to_image_pairs",
    "to_markdown_table",
]
