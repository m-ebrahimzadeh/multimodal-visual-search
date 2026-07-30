"""Search orchestration over one or more indexes."""

from vsearch.search.fusion import RRF_K, reciprocal_rank_fusion
from vsearch.search.service import (
    IndexHandle,
    IndexNotLoadedError,
    SearchResponse,
    SearchResult,
    SearchService,
    build_service,
    discover_indexes,
)

__all__ = [
    "RRF_K",
    "IndexHandle",
    "IndexNotLoadedError",
    "SearchResponse",
    "SearchResult",
    "SearchService",
    "build_service",
    "discover_indexes",
    "reciprocal_rank_fusion",
]
