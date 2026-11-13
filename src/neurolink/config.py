"""Single source of truth for paths, signal parameters and experiment settings.

Values here are defaults; every CLI command exposes the ones that matter as flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"

SFREQ = 160.0
"""Nominal sampling rate of EEGMMIDB. Subjects deviating from this are skipped."""

N_CHANNELS = 64
"""Number of EEG channels in every EEGMMIDB recording."""

ALL_SUBJECTS: tuple[int, ...] = tuple(range(1, 110))
DEV_SUBJECTS: tuple[int, ...] = tuple(range(1, 11))
"""Small subset used during development; the full set is used from Phase 2 on."""

MOTOR_IMAGERY_RUNS: tuple[int, ...] = (4, 8, 12)
"""Runs where the subject imagines opening/closing the left or right fist."""

RUN_DURATION_S = 120.0
"""Nominal length of one run. Runs much shorter than this are flagged by the loader."""

EVENT_ID: dict[str, int] = {"T1": 1, "T2": 2}
"""MNE event codes. T1 = imagine left fist, T2 = imagine right fist. T0 (rest) is excluded."""

CLASS_NAMES: tuple[str, ...] = ("left", "right")
"""Label ``y`` indexes this tuple: 0 = left (T1), 1 = right (T2)."""

MOTOR_CORTEX_CHANNELS: tuple[str, ...] = (
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
)  # fmt: skip
"""Channel subset over sensorimotor cortex, used by EEGNet and the real-time pipeline."""

CHANNEL_SETS: dict[str, tuple[str, ...] | None] = {
    "all": None,
    "motor": MOTOR_CORTEX_CHANNELS,
}
"""Named channel subsets selectable from the CLI."""


@dataclass(frozen=True)
class PreprocessConfig:
    """Parameters of the offline preprocessing pipeline (Phase 1)."""

    band: tuple[float, float] = (8.0, 30.0)
    tmin: float = 0.5
    tmax: float = 2.5
    car: bool = True
    causal: bool = False
    """False: zero-phase FIR (offline analysis). True: the stateful causal Butterworth used
    by the online classifier, so training data matches what the model sees live."""
    channels: tuple[str, ...] | None = None  # None -> all 64 channels
    subjects: tuple[int, ...] = field(default_factory=lambda: DEV_SUBJECTS)
    runs: tuple[int, ...] = MOTOR_IMAGERY_RUNS

    @property
    def n_times(self) -> int:
        """Number of samples per epoch at the nominal sampling rate."""
        return round((self.tmax - self.tmin) * SFREQ)


@dataclass(frozen=True)
class RealtimeConfig:
    """Parameters of the online classifier (Phase 4)."""

    window_s: float = 2.0
    step_s: float = 0.25
    smoothing_alpha: float = 0.3  # EMA factor applied to class probabilities
    filter_order: int = 4  # Butterworth order of the causal band-pass
    rest_quantile: float = 0.75
    """The confidence threshold is this quantile of the smoothed confidence over *rest*
    windows of the training data; below the threshold the cursor does not move."""
    ws_host: str = "127.0.0.1"
    ws_port: int = 8000

    def window_samples(self, sfreq: float = SFREQ) -> int:
        return round(self.window_s * sfreq)

    def step_samples(self, sfreq: float = SFREQ) -> int:
        return round(self.step_s * sfreq)


SEED = 42
