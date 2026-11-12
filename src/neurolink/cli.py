"""Command-line entry point: ``uv run neurolink <command>``.

Commands are stubs until their phase is implemented.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import typer

from neurolink import __version__
from neurolink.config import CACHE_DIR, CHANNEL_SETS, CLASS_NAMES, RAW_DIR, PreprocessConfig
from neurolink.utils.subjects import parse_subjects

app = typer.Typer(
    name="neurolink",
    help="NeuroLink-lite: EEG motor imagery -> cursor control.",
    no_args_is_help=True,
)


def _not_implemented(phase: str) -> None:
    typer.echo(f"Not implemented yet (planned for {phase}).")
    raise typer.Exit(code=2)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _subjects_option(value: str) -> tuple[int, ...]:
    try:
        return parse_subjects(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def download(
    subjects: str = typer.Option("dev", help='Subjects: "dev" (1-10), "all", "1-5,8".'),
    raw_dir: Path = typer.Option(RAW_DIR, help="Where EDF files are stored."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Download EEGMMIDB motor imagery runs (4, 8, 12) into data/raw."""
    from neurolink.data.download import download_subjects

    _setup_logging(verbose)
    ids = _subjects_option(subjects)
    typer.echo(f"Downloading {len(ids)} subject(s) into {raw_dir} ...")
    paths = download_subjects(ids, raw_dir=raw_dir)
    n_files = sum(len(p) for p in paths.values())
    typer.echo(f"Done: {n_files} files for subjects {list(paths)}")


@app.command()
def preprocess(
    subjects: str = typer.Option("dev", help='Subjects: "dev" (1-10), "all", "1-5,8".'),
    channels: str = typer.Option("all", help=f"Channel set: {sorted(CHANNEL_SETS)}."),
    car: bool = typer.Option(True, help="Apply common average reference."),
    tmin: float = typer.Option(0.5, help="Epoch start relative to cue (s)."),
    tmax: float = typer.Option(2.5, help="Epoch end relative to cue (s)."),
    low: float = typer.Option(8.0, help="Band-pass low cutoff (Hz)."),
    high: float = typer.Option(30.0, help="Band-pass high cutoff (Hz)."),
    raw_dir: Path = typer.Option(RAW_DIR, help="Where EDF files are stored."),
    cache_dir: Path = typer.Option(CACHE_DIR, help="Where .npz caches are written."),
    force: bool = typer.Option(False, help="Rebuild even if a cache exists."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Filter, epoch and cache the dataset as .npz; reports skipped subjects."""
    from neurolink.data.cache import cache_path
    from neurolink.data.pipeline import build_or_load

    _setup_logging(verbose)
    if channels not in CHANNEL_SETS:
        raise typer.BadParameter(f"unknown channel set {channels!r}")
    config = PreprocessConfig(
        band=(low, high),
        tmin=tmin,
        tmax=tmax,
        car=car,
        channels=CHANNEL_SETS[channels],
        subjects=_subjects_option(subjects),
    )
    try:
        dataset, report = build_or_load(config, cache_dir=cache_dir, force=force, raw_dir=raw_dir)
    except RuntimeError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    typer.echo(f"dataset: {cache_path(config, cache_dir)}")
    classes, counts = np.unique(dataset.y, return_counts=True)
    class_counts = {CLASS_NAMES[int(c)]: int(n) for c, n in zip(classes, counts, strict=True)}
    typer.echo(
        f"X shape {dataset.X.shape}, classes {class_counts}, "
        f"subjects {len(dataset.unique_subjects)}"
    )
    if report is not None:
        typer.echo(report.summary())
    else:
        typer.echo("(loaded from cache; use --force to rebuild)")


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
