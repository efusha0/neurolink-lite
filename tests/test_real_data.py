"""Integration checks against real EEGMMIDB files, skipped when they are not downloaded.

Run ``uv run neurolink download --subjects 1`` first; CI does not have the data.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurolink.config import MOTOR_IMAGERY_RUNS, N_CHANNELS, RAW_DIR, SFREQ, PreprocessConfig
from neurolink.data.loader import load_subject, read_run, validate_raw
from neurolink.data.pipeline import build_dataset, default_subject_loader

S001_FILES = [RAW_DIR / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0" / "S001"
              / f"S001R{run:02d}.edf" for run in MOTOR_IMAGERY_RUNS]  # fmt: skip

needs_s001 = pytest.mark.skipif(
    not all(p.exists() for p in S001_FILES), reason="EEGMMIDB subject 1 not downloaded"
)


@needs_s001
@pytest.mark.slow
def test_real_run_has_nominal_properties() -> None:
    raw = read_run(S001_FILES[0])
    assert np.isclose(raw.info["sfreq"], SFREQ)
    assert len(raw.ch_names) == N_CHANNELS
    assert raw.ch_names[:3] == ["FC5", "FC3", "FC1"]
    assert "Cz" in raw.ch_names
    assert validate_raw(raw, subject=1, run=4).ok


@needs_s001
@pytest.mark.slow
def test_real_subject_builds_balanced_epochs() -> None:
    raws, reports = load_subject(1, S001_FILES, MOTOR_IMAGERY_RUNS)
    assert raws is not None and all(r.ok for r in reports)

    cfg = PreprocessConfig(subjects=(1,))
    ds, report = build_dataset(cfg, subject_loader=default_subject_loader(cfg))
    assert report.included == [1]
    assert ds.X.shape == (45, N_CHANNELS, cfg.n_times)
    assert sorted(np.unique(ds.runs).tolist()) == list(MOTOR_IMAGERY_RUNS)
    counts = np.bincount(ds.y, minlength=2)
    assert abs(int(counts[0]) - int(counts[1])) <= 3  # 7 or 8 cues per class per run
    # Band-passed, CAR-referenced EEG in microvolts: sane amplitude range.
    assert 1.0 < float(ds.X.std()) < 100.0
