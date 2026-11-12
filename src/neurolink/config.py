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

MOTOR_IMAGERY_RUNS: tuple[int, ...] = (4, 8, 12)
"""Runs where the subject imagines opening/closing the left or right fist."""

EVENT_ID: dict[str, int] = {"left": 1, "right": 2}
"""Class labels. T1 (left fist) -> 1, T2 (right fist) -> 2. T0 (rest) is excluded."""

MOTOR_CORTEX_CHANNELS: tuple[str, ...] = (
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
)  # fmt: skip
"""Channel subset over sensorimotor cortex, used by EEGNet and the real-time pipeline."""


@dataclass(frozen=True)
class PreprocessConfig:
    """Parameters of the offline preprocessing pipeline (Phase 1)."""

    band: tuple[float, float] = (8.0, 30.0)
    tmin: float = 0.5
    tmax: float = 2.5
    car: bool = True
    channels: tuple[str, ...] | None = None  # None -> all 64 channels
    subjects: tuple[int, ...] = field(default_factory=lambda: tuple(range(1, 110)))
    runs: tuple[int, ...] = MOTOR_IMAGERY_RUNS


@dataclass(frozen=True)
class RealtimeConfig:
    """Parameters of the online classifier (Phase 4)."""

    window_s: float = 2.0
    step_s: float = 0.25
    smoothing_alpha: float = 0.3  # EMA factor applied to class probabilities
    ws_host: str = "127.0.0.1"
    ws_port: int = 8000


SEED = 42
