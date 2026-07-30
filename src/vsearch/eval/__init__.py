"""Retrieval evaluation."""

from vsearch.eval.protocols import (
    FASHION_RELEVANCE_FIELDS,
    TextQuery,
    evaluate_image_to_image,
    evaluate_text_to_image,
    flickr_text_queries,
    label_relevance_groups,
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
    "FASHION_RELEVANCE_FIELDS",
    "Judged",
    "RetrievalMetrics",
    "TextQuery",
    "average_precision",
    "evaluate",
    "evaluate_image_to_image",
    "evaluate_text_to_image",
    "flickr_text_queries",
    "label_relevance_groups",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "to_markdown_table",
]
