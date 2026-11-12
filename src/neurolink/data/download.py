"""Download EEGMMIDB runs from PhysioNet into ``data/raw``.

Thin wrapper around :func:`mne.datasets.eegbci.load_data` so the rest of the code never
touches MNE's dataset configuration directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from mne.datasets import eegbci

from neurolink.config import MOTOR_IMAGERY_RUNS, RAW_DIR


def download_subject(
    subject: int,
    runs: Sequence[int] = MOTOR_IMAGERY_RUNS,
    raw_dir: Path = RAW_DIR,
) -> list[Path]:
    """Fetch the EDF files of one subject (cached: nothing is re-downloaded).

    Returns the local paths, one per run, in the order of ``runs``.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = eegbci.load_data(
        subjects=[subject], runs=list(runs), path=raw_dir, update_path=False, verbose="ERROR"
    )
    return [Path(p) for p in paths]


def download_subjects(
    subjects: Sequence[int],
    runs: Sequence[int] = MOTOR_IMAGERY_RUNS,
    raw_dir: Path = RAW_DIR,
) -> dict[int, list[Path]]:
    """Fetch several subjects; returns ``{subject: [paths]}``."""
    return {s: download_subject(s, runs, raw_dir) for s in subjects}
