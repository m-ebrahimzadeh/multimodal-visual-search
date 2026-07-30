"""Request and response models for the search API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from vsearch.search.service import SearchResponse

FilterSpec = dict[str, str | list[str]]
"""Facet filter. Fields AND together; a list within a field ORs."""


class TextSearchRequest(BaseModel):
    """Search by natural-language query."""

    query: str = Field(min_length=1, max_length=500, examples=["red leather ankle boots"])
    k: int = Field(default=24, ge=1, le=200)
    encoder: str | None = Field(
        default=None, description="Encoder name; defaults to the first multimodal index."
    )
    filters: FilterSpec = Field(default_factory=dict, examples=[{"baseColour": "Red"}])


class SearchResultOut(BaseModel):
    """One ranked result."""

    rank: int
    id: str
    score: float = Field(description="Cosine similarity, or an RRF score when results are fused.")
    title: str
    thumbnail_url: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class SearchResponseOut(BaseModel):
    """A ranked result set plus what produced it."""

    encoder: str
    fused: bool = Field(
        default=False,
        description="When true, scores are RRF scores and comparable only within this list.",
    )
    took_ms: float
    count: int
    total_indexed: int
    results: list[SearchResultOut]

    @classmethod
    def from_response(cls, response: SearchResponse) -> SearchResponseOut:
        return cls(
            encoder=response.encoder,
            fused=response.fused,
            took_ms=round(response.took_ms, 2),
            count=len(response.results),
            total_indexed=response.total_indexed,
            results=[
                SearchResultOut(
                    rank=result.rank,
                    id=result.id,
                    score=round(result.score, 6),
                    title=result.title,
                    thumbnail_url=(
                        f"/thumbnail/{response.encoder}/{result.id}"
                        if result.thumbnail is not None
                        else None
                    ),
                    payload=dict(result.payload),
                )
                for result in response.results
            ],
        )


class IndexInfo(BaseModel):
    count: int
    dim: int
    backend: str
    model_id: str
    has_thumbnails: bool


class HealthResponse(BaseModel):
    """Liveness plus enough detail to diagnose a bad deploy at a glance."""

    status: str = Field(description="ok when an index is loaded, degraded otherwise.")
    version: str
    device: str
    corpus: str
    encoders: list[str] = Field(default_factory=list)
    text_encoders: list[str] = Field(default_factory=list)
    indexes: dict[str, IndexInfo] = Field(default_factory=dict)
    detail: str | None = Field(default=None, description="Why the service is degraded, when it is.")


class FacetsResponse(BaseModel):
    """Filterable fields and their values, for building UI controls."""

    encoder: str
    facets: dict[str, list[Any]]
