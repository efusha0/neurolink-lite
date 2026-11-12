from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pytest

from neurolink.config import MOTOR_CORTEX_CHANNELS, PreprocessConfig
from neurolink.data.loader import ValidationReport, standardize_channel_names
from neurolink.data.pipeline import SubjectLoader, build_dataset, build_or_load, preprocess_raw
from tests.conftest import RawFactory


def _fake_loader(raw_factory: RawFactory, bad: frozenset[int] = frozenset()) -> SubjectLoader:
    """Subject loader that fabricates three standardized runs per subject."""

    def _load(subject: int) -> tuple[list[mne.io.BaseRaw] | None, list[ValidationReport]]:
        if subject in bad:
            return None, [ValidationReport(subject, 4, ["sampling rate 128 Hz != 160 Hz"])]
        raws = [standardize_channel_names(raw_factory(seed=subject * 10 + r)) for r in range(3)]
        return list(raws), [ValidationReport(subject, r) for r in (4, 8, 12)]

    return _load


def test_build_dataset_concatenates_subjects_and_runs(raw_factory: RawFactory) -> None:
    cfg = PreprocessConfig(subjects=(1, 2), channels=MOTOR_CORTEX_CHANNELS)
    ds, report = build_dataset(cfg, subject_loader=_fake_loader(raw_factory))
    assert ds.X.shape == (2 * 3 * 15, 21, 320)
    assert ds.ch_names == MOTOR_CORTEX_CHANNELS
    np.testing.assert_array_equal(np.unique(ds.subjects), [1, 2])
    np.testing.assert_array_equal(np.unique(ds.runs), [4, 8, 12])
    assert (ds.subjects == 1).sum() == 45
    assert (ds.runs == 8).sum() == 30
    assert report.included == [1, 2]
    assert report.skipped == {}


def test_build_dataset_skips_and_reports_bad_subjects(raw_factory: RawFactory) -> None:
    cfg = PreprocessConfig(subjects=(1, 88, 3))
    ds, report = build_dataset(cfg, subject_loader=_fake_loader(raw_factory, frozenset({88})))
    assert report.included == [1, 3]
    assert list(report.skipped) == [88]
    assert "128" in report.skipped[88][0]
    assert 88 not in ds.unique_subjects
    assert "S088" in report.summary()


def test_build_dataset_raises_when_nothing_valid(raw_factory: RawFactory) -> None:
    cfg = PreprocessConfig(subjects=(88,))
    with pytest.raises(RuntimeError, match="no valid subjects"):
        build_dataset(cfg, subject_loader=_fake_loader(raw_factory, frozenset({88})))


def test_preprocess_raw_without_car(raw_factory: RawFactory) -> None:
    raw = standardize_channel_names(raw_factory())
    cfg = PreprocessConfig(car=False, channels=("C3", "C4"))
    preprocess_raw(raw, cfg)
    assert raw.ch_names == ["C3", "C4"]


def test_build_or_load_uses_cache(raw_factory: RawFactory, tmp_path: Path) -> None:
    cfg = PreprocessConfig(subjects=(1,), channels=("C3", "Cz", "C4"))
    calls = {"n": 0}
    inner = _fake_loader(raw_factory)

    def counting_loader(
        subject: int,
    ) -> tuple[list[mne.io.BaseRaw] | None, list[ValidationReport]]:
        calls["n"] += 1
        return inner(subject)

    ds1, rep1 = build_or_load(cfg, cache_dir=tmp_path, subject_loader=counting_loader)
    ds2, rep2 = build_or_load(cfg, cache_dir=tmp_path, subject_loader=counting_loader)
    assert calls["n"] == 1
    assert rep1 is not None
    assert rep2 is None
    np.testing.assert_array_equal(ds1.X, ds2.X)

    build_or_load(cfg, cache_dir=tmp_path, force=True, subject_loader=counting_loader)
    assert calls["n"] == 2
