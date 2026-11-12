"""Load raw EEGMMIDB recordings, standardize channel names and validate them.

Known dataset quirks handled here:

* Channel names in the EDF files look like ``Fc5.`` or ``C3..``; they are renamed to the
  10-05 convention (``FC5``, ``C3``) so montages and channel subsets resolve.
* A few subjects (e.g. S088, S092, S100) were recorded at 128 Hz instead of 160 Hz, and
  some runs are truncated or lack annotations. Instead of hard-coding subject IDs we
  measure each recording and report every problem, so new quirks surface loudly.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import mne
import numpy as np

from neurolink.config import EVENT_ID, N_CHANNELS, RUN_DURATION_S, SFREQ

log = logging.getLogger(__name__)

MIN_RUN_DURATION_S = 0.9 * RUN_DURATION_S
MIN_EVENTS_PER_RUN = 10
"""Each motor-imagery run has about 15 T1/T2 cues; far fewer means a truncated recording."""


def standardize_channel_names(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Rename EEGMMIDB channels in place: ``Fc5.`` -> ``FC5``, ``Cz..`` -> ``Cz``.

    Same rules as :func:`mne.datasets.eegbci.standardize`, re-implemented so the behaviour
    is pinned and unit-tested independently of MNE releases.
    """
    rename: dict[str, str] = {}
    for name in raw.ch_names:
        std = name.strip(".").upper()
        if std.endswith("Z"):
            std = std[:-1] + "z"
        if std.startswith("FP"):
            std = "Fp" + std[2:]
        rename[name] = std
    raw.rename_channels(rename)
    return raw


def read_run(path: Path) -> mne.io.BaseRaw:
    """Read one EDF run, standardize names and attach the standard 10-05 montage."""
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    standardize_channel_names(raw)
    raw.set_montage("colin27_1005", on_missing="warn", verbose="ERROR")
    return raw


@dataclass
class ValidationReport:
    """Result of checking one recording against the nominal dataset properties."""

    subject: int
    run: int | None
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def validate_raw(raw: mne.io.BaseRaw, subject: int, run: int | None = None) -> ValidationReport:
    """Check sampling rate, channel count, duration and annotations of one run.

    Every deviation is a human-readable string so the CLI can print a precise report.
    """
    report = ValidationReport(subject=subject, run=run)
    sfreq = float(raw.info["sfreq"])
    if not np.isclose(sfreq, SFREQ):
        report.problems.append(f"sampling rate {sfreq:g} Hz != {SFREQ:g} Hz")

    n_eeg = len(mne.pick_types(raw.info, eeg=True, exclude=[]))
    if n_eeg != N_CHANNELS:
        report.problems.append(f"{n_eeg} EEG channels != {N_CHANNELS}")

    duration = raw.n_times / sfreq
    if duration < MIN_RUN_DURATION_S:
        report.problems.append(f"duration {duration:.1f} s < {MIN_RUN_DURATION_S:.0f} s")

    n_events = sum(d in EVENT_ID for d in raw.annotations.description)
    if n_events < MIN_EVENTS_PER_RUN:
        report.problems.append(f"{n_events} T1/T2 annotations < {MIN_EVENTS_PER_RUN}")
    return report


def load_subject(
    subject: int,
    paths: Sequence[Path],
    runs: Sequence[int],
) -> tuple[list[mne.io.BaseRaw] | None, list[ValidationReport]]:
    """Read and validate the runs of one subject.

    Returns ``(raws, reports)`` with one raw per run, in the order of ``runs``. ``raws`` is
    ``None`` when any run fails validation: a subject with one bad run is skipped entirely
    so that classes and runs stay balanced across subjects.
    """
    raws: list[mne.io.BaseRaw] = []
    reports: list[ValidationReport] = []
    for path, run in zip(paths, runs, strict=True):
        raw = read_run(path)
        report = validate_raw(raw, subject=subject, run=run)
        reports.append(report)
        if report.ok:
            raws.append(raw)
        else:
            log.warning("S%03d run %d skipped: %s", subject, run, "; ".join(report.problems))

    if len(raws) != len(runs):
        return None, reports
    return raws, reports
