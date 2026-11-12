"""Save and load preprocessed datasets as ``.npz`` files keyed by their configuration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from neurolink.config import CACHE_DIR, PreprocessConfig

FORMAT_VERSION = 1


@dataclass
class Dataset:
    """Epoched motor-imagery data plus the metadata needed to split and replay it.

    ``X``: ``(n_epochs, n_channels, n_times)`` float32, microvolts.
    ``y``: ``(n_epochs,)`` int64, index into :data:`neurolink.config.CLASS_NAMES`.
    ``subjects`` / ``runs``: ``(n_epochs,)`` int64, provenance of every epoch (used for
    grouped cross-validation and for the held-out-run demo split).
    """

    X: NDArray[np.float32]
    y: NDArray[np.int64]
    subjects: NDArray[np.int64]
    runs: NDArray[np.int64]
    ch_names: tuple[str, ...]
    sfreq: float
    config: PreprocessConfig

    @property
    def n_epochs(self) -> int:
        return int(self.X.shape[0])

    @property
    def unique_subjects(self) -> NDArray[np.int64]:
        return np.unique(self.subjects)

    def __post_init__(self) -> None:
        n = self.X.shape[0]
        if not (self.y.shape == self.subjects.shape == self.runs.shape == (n,)):
            raise ValueError("X, y, subjects and runs must agree on the number of epochs")
        if self.X.ndim != 3 or self.X.shape[1] != len(self.ch_names):
            raise ValueError("X must be (n_epochs, n_channels, n_times) matching ch_names")


def config_to_dict(config: PreprocessConfig) -> dict[str, Any]:
    d = asdict(config)
    d["channels"] = list(d["channels"]) if d["channels"] is not None else None
    d["subjects"] = list(d["subjects"])
    d["runs"] = list(d["runs"])
    d["band"] = list(d["band"])
    return d


def config_from_dict(d: dict[str, Any]) -> PreprocessConfig:
    return PreprocessConfig(
        band=(float(d["band"][0]), float(d["band"][1])),
        tmin=float(d["tmin"]),
        tmax=float(d["tmax"]),
        car=bool(d["car"]),
        channels=tuple(d["channels"]) if d["channels"] is not None else None,
        subjects=tuple(int(s) for s in d["subjects"]),
        runs=tuple(int(r) for r in d["runs"]),
    )


def cache_key(config: PreprocessConfig) -> str:
    """Short, deterministic hash of a configuration (and of the on-disk format)."""
    payload = json.dumps(
        {"format": FORMAT_VERSION, **config_to_dict(config)}, sort_keys=True
    ).encode()
    return hashlib.sha1(payload).hexdigest()[:12]


def cache_path(config: PreprocessConfig, cache_dir: Path = CACHE_DIR) -> Path:
    return cache_dir / f"dataset_{cache_key(config)}.npz"


def save_dataset(dataset: Dataset, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=dataset.X,
        y=dataset.y,
        subjects=dataset.subjects,
        runs=dataset.runs,
        ch_names=np.asarray(dataset.ch_names),
        sfreq=np.float64(dataset.sfreq),
        config=json.dumps(config_to_dict(dataset.config)),
        format_version=np.int64(FORMAT_VERSION),
    )
    return path


def load_dataset(path: Path) -> Dataset:
    with np.load(path, allow_pickle=False) as f:
        version = int(f["format_version"])
        if version != FORMAT_VERSION:
            raise ValueError(f"cache format {version} != {FORMAT_VERSION}; re-run preprocess")
        return Dataset(
            X=np.asarray(f["X"], dtype=np.float32),
            y=np.asarray(f["y"], dtype=np.int64),
            subjects=np.asarray(f["subjects"], dtype=np.int64),
            runs=np.asarray(f["runs"], dtype=np.int64),
            ch_names=tuple(str(c) for c in f["ch_names"]),
            sfreq=float(f["sfreq"]),
            config=config_from_dict(json.loads(str(f["config"]))),
        )
