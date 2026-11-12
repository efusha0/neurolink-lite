"""Shared pytest fixtures.

Tests never download data. ``make_fake_raw`` builds a synthetic ``mne.io.RawArray`` that
mimics the EEGMMIDB layout: 64 channels with the original dotted names (``Fc5.``, ``C3..``),
160 Hz, and a T0/T1/T2 annotation sequence like a real motor-imagery run.
"""

from __future__ import annotations

from collections.abc import Callable

import mne
import numpy as np
import pytest

from neurolink.config import SFREQ

# Original channel labels as they appear in the EEGMMIDB EDF headers.
EEGMMIDB_RAW_CH_NAMES: tuple[str, ...] = (
    "Fc5.", "Fc3.", "Fc1.", "Fcz.", "Fc2.", "Fc4.", "Fc6.",
    "C5..", "C3..", "C1..", "Cz..", "C2..", "C4..", "C6..",
    "Cp5.", "Cp3.", "Cp1.", "Cpz.", "Cp2.", "Cp4.", "Cp6.",
    "Fp1.", "Fpz.", "Fp2.", "Af7.", "Af3.", "Afz.", "Af4.", "Af8.",
    "F7..", "F5..", "F3..", "F1..", "Fz..", "F2..", "F4..", "F6..", "F8..",
    "Ft7.", "Ft8.", "T7..", "T8..", "T9..", "T10.", "Tp7.", "Tp8.",
    "P7..", "P5..", "P3..", "P1..", "Pz..", "P2..", "P4..", "P6..", "P8..",
    "Po7.", "Po3.", "Poz.", "Po4.", "Po8.", "O1..", "Oz..", "O2..", "Iz..",
)  # fmt: skip

RawFactory = Callable[..., mne.io.RawArray]


def make_fake_raw(
    sfreq: float = SFREQ,
    n_channels: int = 64,
    duration: float = 125.0,
    n_trials: int = 15,
    seed: int = 0,
    labels: str = "alternate",
) -> mne.io.RawArray:
    """Synthetic run: white noise (about 10 uV) plus T0/T1/T2 cues every 8.2 s.

    ``labels="alternate"`` gives T1, T2, T1, ...; ``"T1"`` or ``"T2"`` gives a single class.
    """
    rng = np.random.default_rng(seed)
    n_times = int(duration * sfreq)
    data = rng.normal(0.0, 10e-6, size=(n_channels, n_times))
    names = list(EEGMMIDB_RAW_CH_NAMES[:n_channels])
    info = mne.create_info(names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")

    onsets, durations, descs = [], [], []
    t = 0.0
    for i in range(n_trials):
        onsets += [t, t + 4.1]
        durations += [4.1, 4.1]
        cue = ("T1" if i % 2 == 0 else "T2") if labels == "alternate" else labels
        descs += ["T0", cue]
        t += 8.2
    raw.set_annotations(mne.Annotations(onsets, durations, descs))
    return raw


@pytest.fixture
def fake_raw() -> mne.io.RawArray:
    return make_fake_raw()


@pytest.fixture
def raw_factory() -> RawFactory:
    return make_fake_raw
