"""Command-line entry point: ``uv run neurolink <command>``.

Commands are stubs until their phase is implemented.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import typer

from neurolink import __version__
from neurolink.config import (
    CACHE_DIR,
    CHANNEL_SETS,
    CLASS_NAMES,
    RAW_DIR,
    RESULTS_DIR,
    SEED,
    PreprocessConfig,
)
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
    import mne

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # MNE's CSP logs every covariance estimate; keep only warnings unless --verbose.
    mne.set_log_level("INFO" if verbose else "WARNING")


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
def train_csp(
    subjects: str = typer.Option("all", help='Subjects: "dev" (1-10), "all", "1-5,8".'),
    channels: str = typer.Option("all", help=f"Channel set: {sorted(CHANNEL_SETS)}."),
    car: bool = typer.Option(True, help="Apply common average reference."),
    protocol: str = typer.Option("both", help="within | cross | both"),
    n_components: int = typer.Option(6, help="Number of CSP components."),
    n_splits: int = typer.Option(5, help="Folds per protocol."),
    seed: int = typer.Option(SEED),
    cache_dir: Path = typer.Option(CACHE_DIR),
    results_dir: Path = typer.Option(RESULTS_DIR),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the CSP + LDA baseline under the within- and cross-subject protocols."""
    from neurolink.data.pipeline import build_or_load
    from neurolink.eval.protocols import cross_subject_cv, within_subject_cv
    from neurolink.eval.report import save_result
    from neurolink.models.csp_lda import CSPLDAConfig, make_csp_lda
    from neurolink.utils.seed import set_seed

    _setup_logging(verbose)
    if protocol not in {"within", "cross", "both"}:
        raise typer.BadParameter("protocol must be within, cross or both")
    set_seed(seed)
    config = PreprocessConfig(
        car=car, channels=CHANNEL_SETS[channels], subjects=_subjects_option(subjects)
    )
    dataset, _ = build_or_load(config, cache_dir=cache_dir)
    typer.echo(f"dataset: {dataset.X.shape}, {len(dataset.unique_subjects)} subjects")

    model_cfg = CSPLDAConfig(n_components=n_components)
    model_name = "csp_lda" if channels == "all" else f"csp_lda_{channels}"
    extra = {"channels": channels, "car": car, "n_components": n_components, "seed": seed}

    def factory() -> Any:
        return make_csp_lda(model_cfg)

    protocols = {"within": within_subject_cv, "cross": cross_subject_cv}
    for name, fn in protocols.items():
        if protocol not in {name, "both"}:
            continue
        result = fn(dataset, factory, model_name, n_splits=n_splits, seed=seed)
        path = save_result(result, results_dir, extra)
        acc = result.accuracies
        typer.echo(
            f"{name}-subject: accuracy {100 * acc.mean():.1f} +/- {100 * acc.std(ddof=1):.1f} % "
            f"(pooled {100 * result.pooled_accuracy:.1f} %) -> {path}"
        )


@app.command(name="train-eegnet")
def train_eegnet(
    subjects: str = typer.Option("all", help='Subjects: "dev" (1-10), "all", "1-5,8".'),
    channels: str = typer.Option("motor", help=f"Channel set: {sorted(CHANNEL_SETS)}."),
    car: bool = typer.Option(True, help="Apply common average reference."),
    protocol: str = typer.Option("both", help="within | cross | both"),
    epochs_within: int = typer.Option(100, help="Training epochs per within-subject fold."),
    epochs_cross: int = typer.Option(30, help="Training epochs per cross-subject fold."),
    batch_size: int = typer.Option(32),
    lr: float = typer.Option(1e-3),
    dropout: float = typer.Option(0.5),
    n_splits: int = typer.Option(5, help="Folds per protocol."),
    threads: int = typer.Option(4, help="Torch CPU threads (0 = PyTorch default; 4 was fastest)."),
    seed: int = typer.Option(SEED),
    cache_dir: Path = typer.Option(CACHE_DIR),
    results_dir: Path = typer.Option(RESULTS_DIR),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Train and evaluate EEGNet under the within- and cross-subject protocols."""
    import time

    from neurolink.data.pipeline import build_or_load
    from neurolink.eval.plots import plot_all
    from neurolink.eval.protocols import cross_subject_cv, within_subject_cv
    from neurolink.eval.report import save_result
    from neurolink.models.eegnet import EEGNetConfig
    from neurolink.models.trainer import EEGNetClassifier, TrainConfig
    from neurolink.utils.seed import set_seed

    _setup_logging(verbose)
    if protocol not in {"within", "cross", "both"}:
        raise typer.BadParameter("protocol must be within, cross or both")
    set_seed(seed)
    config = PreprocessConfig(
        car=car, channels=CHANNEL_SETS[channels], subjects=_subjects_option(subjects)
    )
    dataset, _ = build_or_load(config, cache_dir=cache_dir)
    typer.echo(f"dataset: {dataset.X.shape}, {len(dataset.unique_subjects)} subjects")

    model_name = "eegnet" if channels == "motor" else f"eegnet_{channels}"
    model_cfg = EEGNetConfig(dropout=dropout)
    protocols = {
        "within": (within_subject_cv, epochs_within),
        "cross": (cross_subject_cv, epochs_cross),
    }
    for name, (fn, n_epochs) in protocols.items():
        if protocol not in {name, "both"}:
            continue
        train_cfg = TrainConfig(
            epochs=n_epochs, batch_size=batch_size, lr=lr, seed=seed, num_threads=threads
        )
        extra: dict[str, Any] = {
            "channels": channels, "car": car, "seed": seed, "epochs": n_epochs,
            "batch_size": batch_size, "lr": lr, "dropout": dropout,
        }  # fmt: skip

        def factory(train_cfg: TrainConfig = train_cfg) -> Any:
            return EEGNetClassifier(model_cfg, train_cfg)

        t0 = time.perf_counter()
        result = fn(dataset, factory, model_name, n_splits=n_splits, seed=seed)
        extra["wall_seconds"] = time.perf_counter() - t0
        path = save_result(result, results_dir, extra)
        acc = result.accuracies
        typer.echo(
            f"{name}-subject: accuracy {100 * acc.mean():.1f} +/- {100 * acc.std(ddof=1):.1f} % "
            f"(pooled {100 * result.pooled_accuracy:.1f} %, "
            f"{extra['wall_seconds']:.0f} s) -> {path}"
        )
    plot_all(results_dir)


@app.command()
def report(results_dir: Path = typer.Option(RESULTS_DIR)) -> None:
    """Regenerate results/RESULTS.md and the confusion-matrix figures from the JSON files."""
    from neurolink.eval.plots import plot_all
    from neurolink.eval.report import write_markdown_table

    typer.echo(f"wrote {write_markdown_table(results_dir)}")
    for p in plot_all(results_dir):
        typer.echo(f"wrote {p}")


@app.command(name="pseudo-online")
def pseudo_online(
    subjects: str = typer.Option("all", help='Subjects: "dev" (1-10), "all", "1-5,8".'),
    epochs_cross: int = typer.Option(30, help="EEGNet epochs for cross-subject models."),
    epochs_subject: int = typer.Option(100, help="EEGNet epochs for per-subject models."),
    n_splits: int = typer.Option(5, help="Subject-grouped folds."),
    threads: int = typer.Option(4, help="Torch CPU threads."),
    seed: int = typer.Option(SEED),
    raw_dir: Path = typer.Option(RAW_DIR),
    cache_dir: Path = typer.Option(CACHE_DIR),
    results_dir: Path = typer.Option(RESULTS_DIR),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Score every candidate demo model with the online pipeline on held-out recordings."""
    from neurolink.eval.online_eval import default_makers, run_experiment
    from neurolink.eval.report import save_pseudo_online
    from neurolink.utils.seed import set_seed

    _setup_logging(verbose)
    set_seed(seed)
    makers = default_makers(seed, epochs_cross, epochs_subject, threads)
    summary, store = run_experiment(
        _subjects_option(subjects),
        seed,
        makers,
        n_splits=n_splits,
        cache_dir=cache_dir,
        raw_dir=raw_dir,
    )
    windows_path = cache_dir / "pseudo_online_windows.npz"
    np.savez_compressed(str(windows_path), allow_pickle=False, **store.arrays())
    path = save_pseudo_online(summary, results_dir)
    typer.echo(f"window-level outputs: {windows_path}")
    typer.echo(f"wrote {path}")
    for name, row in summary["rows"].items():
        typer.echo(
            f"{name:40s} offline {100 * row['offline_acc']['mean']:5.1f}  "
            f"inside {100 * row['acc_inside_smooth']['mean']:5.1f}  "
            f"all {100 * row['acc_all_smooth']['mean']:5.1f}"
        )


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
