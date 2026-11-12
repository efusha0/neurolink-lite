"""End-to-end offline pipeline: download -> load/validate -> preprocess -> epoch -> cache."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import mne
import numpy as np

from neurolink.config import CACHE_DIR, RAW_DIR, PreprocessConfig
from neurolink.data.cache import Dataset, cache_path, load_dataset, save_dataset
from neurolink.data.download import download_subject
from neurolink.data.epochs import make_epochs
from neurolink.data.loader import ValidationReport, load_subject
from neurolink.data.preprocess import apply_car, bandpass, pick_channels

log = logging.getLogger(__name__)

SubjectLoader = Callable[[int], tuple[list[mne.io.BaseRaw] | None, list[ValidationReport]]]
"""Returns one raw per run of a subject (or ``None`` if invalid) plus per-run reports."""


@dataclass
class BuildReport:
    """What happened to every requested subject while building a dataset."""

    included: list[int] = field(default_factory=list)
    skipped: dict[int, list[str]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"included subjects ({len(self.included)}): {self.included}"]
        if self.skipped:
            lines.append(f"skipped subjects ({len(self.skipped)}):")
            lines.extend(f"  S{s:03d}: {'; '.join(p)}" for s, p in sorted(self.skipped.items()))
        else:
            lines.append("skipped subjects: none")
        return "\n".join(lines)


def default_subject_loader(config: PreprocessConfig, raw_dir: Path = RAW_DIR) -> SubjectLoader:
    """Loader that downloads (if needed) and reads a subject from ``raw_dir``."""

    def _load(subject: int) -> tuple[list[mne.io.BaseRaw] | None, list[ValidationReport]]:
        paths = download_subject(subject, config.runs, raw_dir)
        return load_subject(subject, paths, config.runs)

    return _load


def preprocess_raw(raw: mne.io.BaseRaw, config: PreprocessConfig) -> mne.io.BaseRaw:
    """Band-pass, optional CAR, then channel subset (CAR uses all 64 channels)."""
    bandpass(raw, config.band)
    if config.car:
        apply_car(raw)
    pick_channels(raw, config.channels)
    return raw


def build_dataset(
    config: PreprocessConfig,
    subject_loader: SubjectLoader | None = None,
    raw_dir: Path = RAW_DIR,
) -> tuple[Dataset, BuildReport]:
    """Build an epoched dataset for ``config.subjects``; invalid subjects are skipped."""
    loader = subject_loader or default_subject_loader(config, raw_dir)
    report = BuildReport()
    xs, ys, subs, runs = [], [], [], []
    ch_names: tuple[str, ...] | None = None
    sfreq: float | None = None

    for subject in config.subjects:
        raws, run_reports = loader(subject)
        if raws is None:
            problems = [f"run {r.run}: {'; '.join(r.problems)}" for r in run_reports if not r.ok]
            report.skipped[subject] = problems
            continue

        # Runs are processed independently: filtering and CAR never cross run boundaries,
        # and every epoch keeps its run id (needed for held-out-run splits in the demo).
        n_subject_epochs = 0
        for run_id, raw in zip(config.runs, raws, strict=True):
            preprocess_raw(raw, config)
            x, y = make_epochs(raw, config.tmin, config.tmax)
            xs.append(x)
            ys.append(y)
            subs.append(np.full(len(y), subject, dtype=np.int64))
            runs.append(np.full(len(y), run_id, dtype=np.int64))
            n_subject_epochs += len(y)
            ch_names = tuple(raw.ch_names)
            sfreq = float(raw.info["sfreq"])
        report.included.append(subject)
        log.info("S%03d: %d epochs", subject, n_subject_epochs)

    if not xs or ch_names is None or sfreq is None:
        raise RuntimeError("no valid subjects; nothing to build\n" + report.summary())

    dataset = Dataset(
        X=np.concatenate(xs),
        y=np.concatenate(ys),
        subjects=np.concatenate(subs),
        runs=np.concatenate(runs),
        ch_names=ch_names,
        sfreq=sfreq,
        config=config,
    )
    return dataset, report


def build_or_load(
    config: PreprocessConfig,
    cache_dir: Path = CACHE_DIR,
    force: bool = False,
    subject_loader: SubjectLoader | None = None,
    raw_dir: Path = RAW_DIR,
) -> tuple[Dataset, BuildReport | None]:
    """Return the cached dataset for ``config`` if present, otherwise build and cache it.

    The report is ``None`` on a cache hit.
    """
    path = cache_path(config, cache_dir)
    if path.exists() and not force:
        log.info("cache hit: %s", path)
        return load_dataset(path), None
    dataset, report = build_dataset(config, subject_loader=subject_loader, raw_dir=raw_dir)
    save_dataset(dataset, path)
    log.info("cached dataset: %s", path)
    return dataset, report
