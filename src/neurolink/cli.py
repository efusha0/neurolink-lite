"""Command-line entry point: ``uv run neurolink <command>``.

``demo`` is a stub until Phase 6.
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
    resummarize: bool = typer.Option(
        False,
        "--resummarize",
        help="Do not train: rebuild the summary from the stored window outputs.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Score every candidate demo model with the online pipeline on held-out recordings."""
    import json

    from neurolink.eval import online_eval
    from neurolink.eval.report import PSEUDO_ONLINE_DIR, PSEUDO_ONLINE_JSON, save_pseudo_online
    from neurolink.utils.seed import set_seed

    _setup_logging(verbose)
    set_seed(seed)
    windows_path = cache_dir / "pseudo_online_windows.npz"
    if resummarize:
        previous_path = results_dir / PSEUDO_ONLINE_DIR / PSEUDO_ONLINE_JSON
        if not (windows_path.exists() and previous_path.exists()):
            typer.secho(
                f"--resummarize needs {windows_path} and {previous_path}; "
                "run `neurolink pseudo-online` first.",
                err=True,
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        with np.load(windows_path) as npz:
            arrays = {k: npz[k] for k in npz.files}
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        summary = online_eval.resummarize(previous, arrays)
    else:
        makers = online_eval.default_makers(seed, epochs_cross, epochs_subject, threads)
        summary, store = online_eval.run_experiment(
            _subjects_option(subjects),
            seed,
            makers,
            n_splits=n_splits,
            cache_dir=cache_dir,
            raw_dir=raw_dir,
        )
        np.savez_compressed(str(windows_path), allow_pickle=False, **store.arrays())
        typer.echo(f"window-level outputs: {windows_path}")
    path = save_pseudo_online(summary, results_dir)
    typer.echo(f"wrote {path}")
    for name, row in summary["rows"].items():
        typer.echo(
            f"{name:40s} offline {100 * row['offline_acc']['mean']:5.1f}  "
            f"inside {100 * row['acc_inside_smooth']['mean']:5.1f}  "
            f"all {100 * row['acc_all_smooth']['mean']:5.1f}"
        )


@app.command(name="train-demo")
def train_demo(
    subject: int = typer.Option(
        0, help="Demo subject. 0 = the subject selected by `neurolink pseudo-online`."
    ),
    epochs: int = typer.Option(30, help="EEGNet epochs (as in the cross-subject protocol)."),
    n_splits: int = typer.Option(5, help="Subject-grouped folds."),
    threads: int = typer.Option(4, help="Torch CPU threads."),
    seed: int = typer.Option(SEED),
    raw_dir: Path = typer.Option(RAW_DIR),
    cache_dir: Path = typer.Option(CACHE_DIR),
    results_dir: Path = typer.Option(RESULTS_DIR),
    model_path: Path = typer.Option(None, help="Output file (default data/models/demo_model.pt)."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Train and save the demo model: cross-subject EEGNet that never saw the demo subject."""
    import json

    from neurolink.eval.report import (
        DEMO_MODEL_JSON,
        PSEUDO_ONLINE_DIR,
        PSEUDO_ONLINE_JSON,
        save_realtime,
    )
    from neurolink.realtime import demo_model as dm
    from neurolink.utils.seed import set_seed

    _setup_logging(verbose)
    set_seed(seed)
    experiment_path = results_dir / PSEUDO_ONLINE_DIR / PSEUDO_ONLINE_JSON
    experiment = (
        json.loads(experiment_path.read_text(encoding="utf-8")) if experiment_path.exists() else {}
    )
    selection = experiment.get("demo_selection", {}).get(dm.DEMO_CANDIDATE)
    if subject == 0:
        if selection is None:
            typer.secho(
                "No demo subject selected yet: run `neurolink pseudo-online` or pass --subject.",
                err=True,
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        subject = int(selection["subject"])
    typer.echo(f"demo subject: S{subject:03d}")
    bundle = dm.train_demo_model(
        subject, seed, n_splits=n_splits, epochs=epochs, threads=threads,
        cache_dir=cache_dir, raw_dir=raw_dir,
    )  # fmt: skip
    path = dm.save_bundle(bundle, model_path or dm.DEMO_MODEL_PATH)
    typer.echo(f"wrote {path} (threshold {bundle.threshold:.4f})")

    payload: dict[str, Any] = dm.public_meta(bundle)
    if selection is not None and int(selection["subject"]) == subject:
        # The experiment trained the same fold model; equal numbers show the demo runs the
        # model that was evaluated, not a retuned one.
        row = experiment["rows"][dm.DEMO_CANDIDATE]["per_subject"][str(subject)]
        fold_threshold = experiment["thresholds"][dm.DEMO_CANDIDATE][bundle.meta["fold"]]
        same = bool(
            np.isclose(fold_threshold, bundle.threshold, atol=1e-5)
            and np.isclose(row["offline_acc"], bundle.meta["demo_subject_offline_acc"])
        )
        payload["selection"] = {
            k: selection[k]
            for k in ("selection_rule", "selection_score", "tied_subjects", "tie_break_scores")
            if k in selection
        }
        payload["pseudo_online_run12"] = selection["run12_result"]
        payload["pseudo_online_all_runs"] = row
        payload["experiment_threshold"] = fold_threshold
        payload["matches_pseudo_online_experiment"] = same
        typer.echo(
            f"experiment threshold {fold_threshold:.4f}, offline acc {row['offline_acc']:.3f} "
            f"vs {bundle.meta['demo_subject_offline_acc']:.3f}: "
            + ("matches the evaluated model" if same else "DIFFERS from the evaluated model")
        )
    typer.echo(f"wrote {save_realtime(DEMO_MODEL_JSON, payload, results_dir)}")


def _demo_subject(results_dir: Path) -> int:
    """Demo subject recorded by `train-demo`, else the one selected by `pseudo-online`."""
    import json

    from neurolink.eval.report import (
        DEMO_MODEL_JSON,
        PSEUDO_ONLINE_DIR,
        PSEUDO_ONLINE_JSON,
        load_realtime,
    )
    from neurolink.realtime.demo_model import DEMO_CANDIDATE

    trained = load_realtime(DEMO_MODEL_JSON, results_dir)
    if trained is not None:
        return int(trained["demo_subject"])
    path = results_dir / PSEUDO_ONLINE_DIR / PSEUDO_ONLINE_JSON
    if path.exists():
        selection = json.loads(path.read_text(encoding="utf-8"))["demo_selection"]
        if DEMO_CANDIDATE in selection:
            return int(selection[DEMO_CANDIDATE]["subject"])
    typer.secho(
        "No demo subject known: pass --subject or run `neurolink train-demo`.",
        err=True,
        fg=typer.colors.RED,
    )
    raise typer.Exit(code=1)


def _load_recording(subject: int, run: int, raw_dir: Path) -> Any:
    from neurolink.realtime.pseudo_online import load_recordings

    recordings = load_recordings(subject, (run,), raw_dir)
    if recordings is None:
        typer.secho(
            f"S{subject:03d} run {run} failed validation and cannot be replayed.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    return recordings[0]


def _load_bundle(model_path: Path | None) -> Any:
    from neurolink.realtime.demo_model import DEMO_MODEL_PATH, load_bundle

    try:
        return load_bundle(model_path or DEMO_MODEL_PATH)
    except FileNotFoundError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc


@app.command()
def stream(
    subject: int = typer.Option(0, help="Subject to replay. 0 = the demo subject."),
    run: int = typer.Option(12, help="Run to replay (12 is the held-out run)."),
    loop: bool = typer.Option(True, "--loop/--once", help="Restart when the run ends."),
    chunk: int = typer.Option(8, help="Samples per LSL chunk (8 = 50 ms)."),
    raw_dir: Path = typer.Option(RAW_DIR),
    results_dir: Path = typer.Option(RESULTS_DIR),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Replay a recorded run over LSL at 160 Hz in real time (EEG + true-label markers)."""
    from neurolink.realtime.streamer import EEG_STREAM, MARKER_STREAM, ReplayStreamer

    _setup_logging(verbose)
    if subject == 0:
        subject = _demo_subject(results_dir)
    recording = _load_recording(subject, run, raw_dir)
    streamer = ReplayStreamer(recording, chunk=chunk)
    seconds = len(recording.data) / recording.sfreq
    typer.echo(
        f"streaming S{subject:03d} run {run} ({seconds:.0f} s, {len(recording.ch_names)} channels) "
        f"as LSL streams {EEG_STREAM!r} and {MARKER_STREAM!r}; Ctrl+C to stop"
    )
    try:
        sent = streamer.run(loop=loop)
    except KeyboardInterrupt:
        sent = streamer.n_sent
    typer.echo(f"sent {sent} samples")


@app.command()
def serve(
    model_path: Path = typer.Option(None, help="Demo model (default data/models/demo_model.pt)."),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    timeout: float = typer.Option(10.0, help="Seconds to wait for the LSL EEG stream."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the online classifier on the LSL stream and serve predictions over WebSocket."""
    import uvicorn

    from neurolink.realtime.live import build_live

    _setup_logging(verbose)
    bundle = _load_bundle(model_path)
    try:
        live = build_live(bundle, timeout=timeout)
    except (TimeoutError, ValueError) as exc:
        typer.secho(f"{exc}. Start `neurolink stream` first.", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"model: {bundle.meta.get('candidate', '?')}, threshold {bundle.threshold:.3f}; "
        f"predictions on ws://{host}:{port}/ws"
    )
    try:
        uvicorn.run(live.app, host=host, port=port, log_level="info" if verbose else "warning")
    finally:
        live.close()


@app.command()
def latency(
    duration: float = typer.Option(120.0, help="Seconds of streaming to time."),
    chunk: int = typer.Option(8, help="Samples per LSL chunk (8 = 50 ms)."),
    subject: int = typer.Option(0, help="Subject to replay. 0 = the demo subject."),
    run: int = typer.Option(12),
    model_path: Path = typer.Option(None, help="Demo model (default data/models/demo_model.pt)."),
    name: str = typer.Option("latency.json", help="File name under results/realtime/."),
    raw_dir: Path = typer.Option(RAW_DIR),
    results_dir: Path = typer.Option(RESULTS_DIR),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Measure end-to-end latency: LSL sample -> prediction -> local WebSocket client."""
    from neurolink.eval.report import save_realtime
    from neurolink.realtime.live import measure_latency

    _setup_logging(verbose)
    bundle = _load_bundle(model_path)
    if subject == 0:
        subject = _demo_subject(results_dir)
    recording = _load_recording(subject, run, raw_dir)
    typer.echo(f"timing S{subject:03d} run {run} for {duration:g} s (chunk {chunk} samples) ...")
    result = measure_latency(bundle, recording, duration_s=duration, chunk=chunk)
    for stage, row in result["stages_ms"].items():
        typer.echo(
            f"{stage:15s} p50 {row['p50_ms']:6.1f} ms  p95 {row['p95_ms']:6.1f} ms  "
            f"p99 {row['p99_ms']:6.1f} ms  max {row['max_ms']:6.1f} ms  (n={row['n']:.0f})"
        )
    typer.echo(f"wrote {save_realtime(name, result, results_dir)}")


@app.command(name="behavioural-latency")
def behavioural_latency(
    raw_dir: Path = typer.Option(RAW_DIR),
    cache_dir: Path = typer.Option(CACHE_DIR),
    results_dir: Path = typer.Option(RESULTS_DIR),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Reaction to the cue: time until the demo model's output is right and above threshold."""
    import json

    from neurolink.config import MOTOR_IMAGERY_RUNS
    from neurolink.eval import online_eval
    from neurolink.eval.report import (
        BEHAVIOURAL_JSON,
        DEMO_MODEL_JSON,
        PSEUDO_ONLINE_DIR,
        PSEUDO_ONLINE_JSON,
        load_realtime,
        save_realtime,
    )
    from neurolink.realtime.demo_model import DEMO_CANDIDATE
    from neurolink.realtime.pseudo_online import load_recordings

    _setup_logging(verbose)
    windows_path = cache_dir / "pseudo_online_windows.npz"
    summary_path = results_dir / PSEUDO_ONLINE_DIR / PSEUDO_ONLINE_JSON
    if not (windows_path.exists() and summary_path.exists()):
        typer.secho(
            f"needs {windows_path} and {summary_path}; run `neurolink pseudo-online` first.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    keys = ("subject", "run", "label")
    with np.load(windows_path) as npz:
        arrays = {
            k: npz[k] for k in npz.files if k in keys or k.startswith(f"smooth::{DEMO_CANDIDATE}/")
        }
    trained = load_realtime(DEMO_MODEL_JSON, results_dir)
    demo_subject = int(trained["demo_subject"]) if trained is not None else None
    result = online_eval.behavioural_latency(
        arrays,
        summary,
        lambda s: load_recordings(s, MOTOR_IMAGERY_RUNS, raw_dir),
        DEMO_CANDIDATE,
        demo_subject,
    )
    blocks = {"all subjects": result["all_subjects"]}
    if "demo_subject" in result:
        blocks[f"S{demo_subject:03d} run 12"] = result["demo_subject"]["run_12"]
    for name, block in blocks.items():
        st = block["correct_active"]
        typer.echo(
            f"{name}: reached in {100 * st['reached_share']:.1f} % of {st['n_trials']} trials, "
            f"median of reached {st['quantiles_reached_s']['p50']:.2f} s"
        )
    typer.echo(f"wrote {save_realtime(BEHAVIOURAL_JSON, result, results_dir)}")


@app.command()
def demo() -> None:
    """One-command demo: streamer + classifier + server + browser UI (Phase 6)."""
    _not_implemented("Phase 6")


if __name__ == "__main__":
    app()
