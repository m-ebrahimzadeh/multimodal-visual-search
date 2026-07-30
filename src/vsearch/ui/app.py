"""Gradio interface for the search service.

Mounted onto the FastAPI app so one process serves both the API and the UI --
which is what a Hugging Face Space expects, and keeps the model loaded once
rather than once per surface.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import gradio as gr

from vsearch.encoders import TextNotSupportedError
from vsearch.search import IndexNotLoadedError, SearchResponse, SearchService

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import FastAPI
    from PIL.Image import Image

logger = logging.getLogger(__name__)

MAX_FACET_CARDINALITY = 200
"""Facets with more distinct values than this are left out of the UI: a
dropdown with thousands of entries is not a control anyone can use."""

MAX_FACETS_SHOWN = 4

EXAMPLE_QUERIES = [
    "red leather ankle boots on a white background",
    "silver wrist watch",
    "navy blue formal shirt",
    "black leather handbag",
]

GalleryValue = list[tuple[str, str]]


def _facet_choices(service: SearchService) -> dict[str, list[str]]:
    """Pick the facets worth exposing as dropdowns."""
    try:
        facets = service.facets()
    except (IndexNotLoadedError, KeyError):
        return {}

    usable = {
        name: [str(value) for value in sorted(values, key=repr)]
        for name, values in facets.items()
        if 1 < len(values) <= MAX_FACET_CARDINALITY
    }
    # Fewest values first: the coarsest facets are the most useful filters.
    ranked = sorted(usable.items(), key=lambda item: len(item[1]))
    return dict(ranked[:MAX_FACETS_SHOWN])


def _to_gallery(response: SearchResponse) -> GalleryValue:
    """Render results as (image path, caption) pairs.

    Results without a thumbnail are dropped rather than shown as a broken
    tile -- an empty slot in the grid reads as a bug.
    """
    items: GalleryValue = []
    for result in response.results:
        if result.thumbnail is None:
            continue
        label = "rrf" if response.fused else "cos"
        items.append((str(result.thumbnail), f"{result.score:.3f} {label} · {result.title}"))
    return items


def _status(response: SearchResponse) -> str:
    shown = len(response.results)
    scale = "RRF score" if response.fused else "cosine similarity"
    return (
        f"**{shown}** results in **{response.took_ms:.0f} ms** · "
        f"encoder `{response.encoder}` · {scale} · "
        f"{response.total_indexed:,} images indexed"
    )


def build_ui(service: SearchService) -> gr.Blocks:
    """Build the search interface for a loaded service."""
    facet_choices = _facet_choices(service)
    facet_names = list(facet_choices)
    text_encoders = service.text_encoders or service.encoders
    all_encoders = service.encoders

    def _filters_from(values: list[list[str]]) -> dict[str, Any] | None:
        selected = {
            name: chosen for name, chosen in zip(facet_names, values, strict=True) if chosen
        }
        return selected or None

    def run_text(
        query: str, encoder: str, top_k: int, *facet_values: list[str]
    ) -> tuple[GalleryValue, str]:
        if not query.strip():
            return [], "Enter a query to search."
        try:
            response = service.search_text(
                query,
                k=int(top_k),
                encoder=encoder or None,
                where=_filters_from(list(facet_values)),
            )
        except (TextNotSupportedError, IndexNotLoadedError, ValueError, KeyError) as exc:
            logger.info("Text search rejected: %s", exc)
            return [], f"⚠️ {exc}"
        if not response.results:
            return [], "No matches. Try loosening the filters."
        return _to_gallery(response), _status(response)

    def run_image(
        image: Image | None, encoder: str, top_k: int, fuse: bool, *facet_values: list[str]
    ) -> tuple[GalleryValue, str]:
        if image is None:
            return [], "Upload an image to search."
        try:
            response = service.search_image(
                image,
                k=int(top_k),
                encoder=None if fuse else (encoder or None),
                where=_filters_from(list(facet_values)),
                fuse=all_encoders if fuse and len(all_encoders) > 1 else None,
            )
        except (IndexNotLoadedError, ValueError, KeyError) as exc:
            logger.info("Image search rejected: %s", exc)
            return [], f"⚠️ {exc}"
        if not response.results:
            return [], "No matches. Try loosening the filters."
        return _to_gallery(response), _status(response)

    with gr.Blocks(title="Multimodal Visual Search", fill_width=True) as demo:
        gr.Markdown(
            "# Multimodal Visual Search\n"
            "Search an image collection by **natural language** or **by example image**. "
            "Text queries run through a CLIP-family shared embedding space; image queries "
            "can additionally use a self-supervised vision encoder."
        )

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Text query"):
                        query_box = gr.Textbox(
                            label="Describe what you are looking for",
                            placeholder="red leather ankle boots on a white background",
                            lines=2,
                        )
                        text_encoder = gr.Dropdown(
                            label="Encoder",
                            choices=text_encoders,
                            value=text_encoders[0] if text_encoders else None,
                        )
                        text_button = gr.Button("Search", variant="primary")
                        gr.Examples(examples=EXAMPLE_QUERIES, inputs=query_box)

                    with gr.Tab("Image query"):
                        image_box = gr.Image(
                            label="Upload an example image", type="pil", height=240
                        )
                        image_encoder = gr.Dropdown(
                            label="Encoder",
                            choices=all_encoders,
                            value=all_encoders[0] if all_encoders else None,
                        )
                        fuse_toggle = gr.Checkbox(
                            label="Fuse all encoders (reciprocal rank fusion)",
                            value=False,
                            info="Combines rankings by position, so encoders on "
                            "different score scales contribute comparably.",
                        )
                        image_button = gr.Button("Find similar", variant="primary")

            with gr.Column(scale=1):
                top_k = gr.Slider(label="Results", minimum=4, maximum=60, value=24, step=4)
                facet_inputs: list[gr.Dropdown] = []
                if facet_choices:
                    with gr.Accordion("Filters", open=True):
                        for name, choices in facet_choices.items():
                            facet_inputs.append(
                                gr.Dropdown(label=name, choices=choices, multiselect=True, value=[])
                            )
                else:
                    gr.Markdown("_No filterable facets in this corpus._")

        status = gr.Markdown("Ready.")
        gallery = gr.Gallery(
            label="Results",
            columns=6,
            height=560,
            object_fit="contain",
            show_label=False,
            allow_preview=True,
        )

        text_inputs = [query_box, text_encoder, top_k, *facet_inputs]
        text_button.click(run_text, inputs=text_inputs, outputs=[gallery, status])
        query_box.submit(run_text, inputs=text_inputs, outputs=[gallery, status])
        image_button.click(
            run_image,
            inputs=[image_box, image_encoder, top_k, fuse_toggle, *facet_inputs],
            outputs=[gallery, status],
        )

    return cast("gr.Blocks", demo)


def image_roots(service: SearchService) -> list[str]:
    """Directories Gradio must be allowed to serve thumbnails from.

    Gradio sandboxes filesystem access; without these the results grid renders
    empty with no error, which is a genuinely confusing failure.
    """
    roots: list[str] = []
    for encoder in service.encoders:
        images = service.handle(encoder).images_dir
        if images is not None:
            roots.append(str(Path(images).resolve()))
    return sorted(set(roots))


def mount_ui(app: FastAPI, service: SearchService, path: str = "/") -> FastAPI:
    """Mount the Gradio UI onto an existing FastAPI app."""
    mounted = gr.mount_gradio_app(
        app,
        build_ui(service),
        path=path,
        # Gradio sandboxes filesystem access. Without these the results grid
        # renders empty with no error at all.
        allowed_paths=image_roots(service),
    )
    return cast("FastAPI", mounted)
