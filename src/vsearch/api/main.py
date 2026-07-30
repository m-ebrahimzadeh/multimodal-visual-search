"""FastAPI application exposing the search service.

The app starts even when no index is present: it reports ``degraded`` on
/health and returns 503 from the search routes. A deployed Space that
crash-loops on a missing artifact is far harder to diagnose than one that
comes up and tells you what is wrong.
"""

from __future__ import annotations

import io
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse

from vsearch import __version__
from vsearch.api.schemas import (
    FacetsResponse,
    HealthResponse,
    IndexInfo,
    SearchResponseOut,
    TextSearchRequest,
)
from vsearch.config import get_settings, resolve_device
from vsearch.encoders import TextNotSupportedError
from vsearch.search import IndexNotLoadedError, SearchService, build_service

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
"""Cap on an uploaded query image. Without it, a public demo will eventually
be handed a 500 MB file and fall over decoding it."""


class AppState:
    """Holds the loaded service, or the reason there isn't one."""

    service: SearchService | None = None
    error: str | None = None
    corpus: str = ""


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    state.corpus = settings.default_corpus
    try:
        state.service = build_service(
            settings.artifacts_dir,
            settings.default_corpus,
            device=resolve_device(settings.device),
            batch_size=settings.batch_size,
            token=settings.token,
        )
        state.error = None
        if settings.warmup_on_start:
            warmed = state.service.warmup()
            logger.info("Warmed encoders: %s", ", ".join(warmed) or "(none)")
        logger.info("Search service ready: %s", state.service.encoders)
    except IndexNotLoadedError as exc:
        # Deliberately not fatal -- see the module docstring.
        state.service = None
        state.error = str(exc)
        logger.warning("Starting without an index: %s", exc)
    yield


app = FastAPI(
    title="vsearch",
    version=__version__,
    summary="Multimodal visual search: text-to-image and image-to-image retrieval.",
    lifespan=lifespan,
)


def require_service() -> SearchService:
    """Dependency yielding the service, or 503 with the reason."""
    if state.service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=state.error or "search service is not available",
        )
    return state.service


ServiceDep = Annotated[SearchService, Depends(require_service)]


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Liveness and a summary of what is loaded."""
    settings = get_settings()
    if state.service is None:
        return HealthResponse(
            status="degraded",
            version=__version__,
            device=resolve_device(settings.device),
            corpus=state.corpus,
            detail=state.error,
        )

    stats = state.service.stats()
    return HealthResponse(
        status="ok",
        version=__version__,
        device=resolve_device(settings.device),
        corpus=state.corpus,
        encoders=stats["encoders"],
        text_encoders=stats["text_encoders"],
        indexes={name: IndexInfo(**info) for name, info in stats["indexes"].items()},
    )


@app.get("/facets", response_model=FacetsResponse, tags=["meta"])
def facets(service: ServiceDep, encoder: str | None = Query(default=None)) -> FacetsResponse:
    """Filterable fields and their values."""
    try:
        values = service.facets(encoder)
    except IndexNotLoadedError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return FacetsResponse(encoder=encoder or service.encoders[0], facets=values)


@app.post("/search/text", response_model=SearchResponseOut, tags=["search"])
def search_text(request: TextSearchRequest, service: ServiceDep) -> SearchResponseOut:
    """Rank images against a natural-language query."""
    try:
        response = service.search_text(
            request.query,
            k=request.k,
            encoder=request.encoder,
            where=dict(request.filters) or None,
        )
    except TextNotSupportedError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except IndexNotLoadedError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except KeyError as exc:
        # Raised by the store for an unknown facet field.
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc.args[0])) from exc
    return SearchResponseOut.from_response(response)


@app.post("/search/image", response_model=SearchResponseOut, tags=["search"])
async def search_image(
    service: ServiceDep,
    file: Annotated[UploadFile, File(description="Query image.")],
    k: Annotated[int, Form(ge=1, le=200)] = 24,
    encoder: Annotated[str | None, Form()] = None,
    fuse: Annotated[
        str | None, Form(description="Comma-separated encoders to fuse, e.g. 'clip,dinov3'.")
    ] = None,
) -> SearchResponseOut:
    """Rank images against an uploaded example image."""
    from PIL import Image, UnidentifiedImageError

    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"image exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
        )
    if not payload:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="uploaded file is empty")

    try:
        image = Image.open(io.BytesIO(payload))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, detail=f"could not decode image: {exc}"
        ) from exc

    fuse_list = [name.strip() for name in fuse.split(",") if name.strip()] if fuse else None
    try:
        response = service.search_image(image, k=k, encoder=encoder, fuse=fuse_list)
    except IndexNotLoadedError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return SearchResponseOut.from_response(response)


@app.get("/thumbnail/{encoder}/{item_id}", tags=["media"])
def thumbnail(encoder: str, item_id: str, service: ServiceDep) -> FileResponse:
    """Serve an indexed item's thumbnail.

    The id is only ever resolved through the handle, which refuses paths
    escaping the images directory.
    """
    try:
        handle = service.handle(encoder.split("+", 1)[0])
    except IndexNotLoadedError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    path = handle.thumbnail_for(item_id)
    if path is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"no thumbnail for {item_id!r}")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/", include_in_schema=False)
def root() -> dict[str, Any]:
    return {
        "name": "vsearch",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
    }
