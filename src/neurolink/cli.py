"""Command-line entry point: ``uv run neurolink <command>``.

Commands are stubs until their phase is implemented.
"""

from __future__ import annotations

import typer

from neurolink import __version__

app = typer.Typer(
    name="neurolink",
    help="NeuroLink-lite: EEG motor imagery -> cursor control.",
    no_args_is_help=True,
)


def _not_implemented(phase: str) -> None:
    typer.echo(f"Not implemented yet (planned for {phase}).")
    raise typer.Exit(code=2)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def download() -> None:
    """Download EEGMMIDB motor imagery runs into data/raw (Phase 1)."""
    _not_implemented("Phase 1")


@app.command()
def preprocess() -> None:
    """Filter, epoch and cache the dataset as .npz (Phase 1)."""
    _not_implemented("Phase 1")


@app.command(name="train-csp")
def train_csp() -> None:
    """Run the CSP + LDA baseline under both evaluation protocols (Phase 2)."""
    _not_implemented("Phase 2")


@app.command(name="train-eegnet")
def train_eegnet() -> None:
    """Train and evaluate EEGNet under both evaluation protocols (Phase 3)."""
    _not_implemented("Phase 3")


@app.command()
def stream() -> None:
    """Replay recorded EEG over LSL in real time (Phase 4)."""
    _not_implemented("Phase 4")


@app.command()
def serve() -> None:
    """Run the online classifier and the FastAPI WebSocket server (Phase 4)."""
    _not_implemented("Phase 4")


@app.command()
def demo() -> None:
    """One-command demo: streamer + classifier + server + browser UI (Phase 6)."""
    _not_implemented("Phase 6")


if __name__ == "__main__":
    app()
