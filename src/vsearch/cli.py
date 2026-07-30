"""Command-line entry point for vsearch.

Subcommands are added as each subsystem lands (``ingest`` in the ingestion
phase, ``evaluate`` and ``bench`` later). ``info`` exists from the start
because "which device did it actually pick, and where is it writing?" is the
first question asked when a run behaves unexpectedly on a new machine.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from vsearch import __version__
from vsearch.config import get_settings, resolve_device

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


if __name__ == "__main__":  # pragma: no cover
    app()
