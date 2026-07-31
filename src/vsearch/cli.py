"""Command-line entry point for vsearch.

Subcommands are added as each subsystem lands (``ingest`` in the ingestion
phase, ``evaluate`` and ``bench`` later). ``info`` exists from the start
because "which device did it actually pick, and where is it writing?" is the
first question asked when a run behaves unexpectedly on a new machine.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, cast

import typer
from rich.console import Console
from rich.table import Table

from vsearch import __version__
from vsearch.config import get_settings, resolve_device

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vsearch.index.faiss_store import Backend

app = typer.Typer(
    name="vsearch",
    help="Multimodal visual search: text-to-image and image-to-image retrieval.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def version() -> None:
    """Print the installed vsearch version."""
    console.print(__version__)


@app.command()
def info() -> None:
    """Show resolved runtime configuration.

    Resolves the device for real (importing torch), so this doubles as a check
    that the torch install matches the hardware you think you are on.
    """
    settings = get_settings()

    table = Table(title="vsearch runtime", show_header=True, header_style="bold")
    table.add_column("setting")
    table.add_column("value", overflow="fold")

    table.add_row("version", __version__)
    table.add_row("device (requested)", settings.device.value)
    table.add_row("device (resolved)", resolve_device(settings.device))
    table.add_row("batch size", str(settings.batch_size))
    table.add_row("text encoder", settings.text_encoder)
    table.add_row("image encoder", settings.image_encoder)
    table.add_row("index backend", settings.index_backend)
    table.add_row("data dir", str(settings.data_dir))
    table.add_row("artifacts dir", str(settings.artifacts_dir))
    table.add_row("results dir", str(settings.results_dir))
    table.add_row("artifact repo", settings.artifact_repo or "(unset)")
    # Presence only -- never the value.
    table.add_row("HF token", "set" if settings.token else "(unset)")

    console.print(table)


@app.command()
def corpora() -> None:
    """List the image corpora available for ingestion."""
    from vsearch.ingest import CORPORA

    table = Table(title="corpora", show_header=True, header_style="bold")
    table.add_column("name")
    table.add_column("dataset")
    table.add_column("description", overflow="fold")
    for spec in CORPORA.values():
        table.add_row(spec.name, spec.hf_id, spec.description)
    console.print(table)


@app.command()
def encoders() -> None:
    """List the available encoders."""
    from vsearch.encoders import ENCODERS

    table = Table(title="encoders", show_header=True, header_style="bold")
    table.add_column("name")
    table.add_column("model")
    table.add_column("modality")
    table.add_column("dim", justify="right")
    table.add_column("notes", overflow="fold")
    for spec in ENCODERS.values():
        notes = []
        if spec.gated:
            notes.append("gated licence")
        if spec.fallback:
            notes.append(f"falls back to {spec.fallback}")
        table.add_row(
            spec.name, spec.model_id, spec.modality.value, str(spec.dim), "; ".join(notes)
        )
    console.print(table)


@app.command()
def ingest(
    corpus: Annotated[str, typer.Option(help="Corpus name; see `vsearch corpora`.")] = "fashion",
    encoder: Annotated[str, typer.Option(help="Encoder name; see `vsearch encoders`.")] = "clip",
    limit: Annotated[int | None, typer.Option(help="Stop after N images (smoke runs).")] = None,
    split: Annotated[
        str | None, typer.Option(help="Filter on the corpus's internal split column.")
    ] = None,
    shard_size: Annotated[int, typer.Option(help="Records committed per shard.")] = 2048,
    backend: Annotated[str, typer.Option(help="Index backend: flat or hnsw.")] = "flat",
    device: Annotated[str | None, typer.Option(help="Override auto device selection.")] = None,
    batch_size: Annotated[int | None, typer.Option(help="Override encoder batch size.")] = None,
    streaming: Annotated[
        bool, typer.Option(help="Stream the corpus instead of downloading it.")
    ] = False,
) -> None:
    """Embed a corpus into a searchable index.

    Safe to re-run: completed shards are skipped, so an interrupted ingest
    resumes rather than starting over.
    """
    from vsearch.ingest import IngestConfig
    from vsearch.ingest import ingest as run_ingest

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = get_settings()
    settings.ensure_dirs()

    if backend not in {"flat", "hnsw"}:
        console.print(f"[red]Unknown backend {backend!r}.[/] Use 'flat' or 'hnsw'.")
        raise typer.Exit(code=1)

    config = IngestConfig(
        corpus=corpus,
        encoder=encoder,
        shard_size=shard_size,
        limit=limit,
        split_filter=split,
        index_backend=cast("Backend", backend),
        streaming=streaming,
    )

    manifest = run_ingest(
        config,
        artifacts_dir=settings.artifacts_dir,
        device=device,
        batch_size=batch_size,
        token=settings.token,
    )

    console.print(
        f"[bold green]Indexed[/] {manifest.records} images "
        f"({manifest.corpus} / {manifest.encoder}, dim {manifest.dim}) -> "
        f"{settings.artifacts_dir / config.run_name / 'index'}"
    )


@app.command()
def export_web(
    corpus: Annotated[str, typer.Option(help="Corpus of the run to export.")] = "fashion",
    encoder: Annotated[str, typer.Option(help="Encoder of the run to export.")] = "clip",
    destination: Annotated[Path | None, typer.Option(help="Bundle output directory.")] = None,
    device: Annotated[str | None, typer.Option(help="Override auto device selection.")] = None,
    skip_examples: Annotated[
        bool, typer.Option(help="Do not embed example queries (skips loading the encoder).")
    ] = False,
) -> None:
    """Export a built index as static assets for the Cloudflare Worker demo.

    Writes the index's own float32 block rather than rebuilding one, so the
    deployed demo ranks by the same vectors the evaluation tables report on.
    """
    from vsearch.encoders import load_encoder
    from vsearch.web import export_bundle

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = get_settings()

    run_dir = settings.artifacts_dir / f"{corpus}__{encoder}"
    if not (run_dir / "index").exists():
        console.print(f"[red]No index at {run_dir / 'index'}.[/] Run `vsearch ingest` first.")
        raise typer.Exit(code=1)

    text_encoder = None
    if not skip_examples:
        # allow_fallback is left at its default: the example vectors must match
        # the index's space, and the registry only substitutes for gated image
        # encoders, which cannot produce these in the first place.
        text_encoder = load_encoder(encoder, device=device, token=settings.token)

    bundle = export_bundle(
        run_dir,
        destination or Path("web/public/data"),
        encoder=text_encoder,
        source=_corpus_source(corpus),
    )
    console.print(
        f"[bold green]Exported[/] {bundle.count} x {bundle.dim} vectors "
        f"({bundle.embeddings_bytes / 1024:.0f} KB), {bundle.images} thumbnails, "
        f"{bundle.examples} example queries -> {bundle.destination}"
    )


@app.command()
def verify_web(
    url: Annotated[str, typer.Argument(help="Base URL of the deployed Worker.")],
    bundle: Annotated[
        Path | None, typer.Option(help="Bundle directory holding examples.json.")
    ] = None,
) -> None:
    """Measure the deployed Worker's query embeddings against the local ones.

    The demo assumes Workers AI's CLIP is interchangeable with the PyTorch fp32
    encoder the index was built from. This is that assumption, measured.
    """
    from vsearch.web import measure_parity

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parity = measure_parity(url, bundle or Path("web/public/data"))

    table = Table(title=f"query-embedding parity vs {url}", show_header=True)
    table.add_column("query", overflow="fold")
    table.add_column("cosine", justify="right")
    for text, cosine in zip(parity.texts, parity.cosines, strict=True):
        table.add_row(text, f"{cosine:.4f}")
    console.print(table)
    console.print(
        f"[bold]mean[/] {parity.mean:.4f}  [bold]worst[/] {parity.worst:.4f} "
        f"({parity.worst_query()!r})"
    )


def _corpus_source(corpus: str) -> str | None:
    """Resolve a corpus name to its dataset URL for the attribution file."""
    from vsearch.ingest import CORPORA

    spec = CORPORA.get(corpus)
    return f"https://huggingface.co/datasets/{spec.hf_id}" if spec else None


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 7860,
    api_only: Annotated[bool, typer.Option(help="Skip the Gradio UI.")] = False,
    reload: Annotated[bool, typer.Option(help="Reload on code changes.")] = False,
) -> None:
    """Serve the search UI and API."""
    import uvicorn

    target = "vsearch.api.main:app" if api_only else "vsearch.serve:app"
    console.print(f"[bold]Serving[/] {target} on http://{host}:{port}")
    uvicorn.run(target, host=host, port=port, reload=reload)


@app.command()
def publish(
    corpus: Annotated[str, typer.Option(help="Corpus of the run to publish.")] = "fashion",
    encoder: Annotated[str, typer.Option(help="Encoder of the run to publish.")] = "clip",
    repo: Annotated[str | None, typer.Option(help="Hub dataset repo id.")] = None,
    private: Annotated[bool, typer.Option(help="Create the repo as private.")] = True,
) -> None:
    """Upload a built index to the Hub so a Space can serve it."""
    from vsearch.ingest import push_artifacts

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = get_settings()
    target = repo or settings.artifact_repo
    if not target:
        console.print("[red]No repo given.[/] Pass --repo or set VSEARCH_ARTIFACT_REPO.")
        raise typer.Exit(code=1)
    if not settings.token:
        console.print("[red]No HF_TOKEN.[/] Set it in .env to publish.")
        raise typer.Exit(code=1)

    run_dir = settings.artifacts_dir / f"{corpus}__{encoder}"
    url = push_artifacts(run_dir, target, token=settings.token, private=private)
    console.print(f"[bold green]Published[/] -> {url}")


@app.command()
def pull(
    repo: Annotated[str | None, typer.Option(help="Hub dataset repo id.")] = None,
    destination: Annotated[Path | None, typer.Option(help="Where to place artifacts.")] = None,
) -> None:
    """Download a published index."""
    from vsearch.ingest import pull_artifacts

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = get_settings()
    target = repo or settings.artifact_repo
    if not target:
        console.print("[red]No repo given.[/] Pass --repo or set VSEARCH_ARTIFACT_REPO.")
        raise typer.Exit(code=1)

    where = destination or settings.artifacts_dir / "pulled"
    local = pull_artifacts(target, where, token=settings.token)
    console.print(f"[bold green]Pulled[/] -> {local}")


if __name__ == "__main__":  # pragma: no cover
    app()
