from __future__ import annotations

import mne
import numpy as np

from neurolink.config import MOTOR_CORTEX_CHANNELS, N_CHANNELS
from neurolink.data.loader import (
    MIN_EVENTS_PER_RUN,
    load_subject,
    standardize_channel_names,
    validate_raw,
)
from tests.conftest import RawFactory


def test_standardize_channel_names_matches_1005_convention(fake_raw: mne.io.RawArray) -> None:
    standardize_channel_names(fake_raw)
    names = fake_raw.ch_names
    assert names[:3] == ["FC5", "FC3", "FC1"]
    assert "FCz" in names and "Cz" in names and "CPz" in names
    assert "Fp1" in names and "Fpz" in names
    assert "T10" in names and "Iz" in names
    assert len(set(names)) == N_CHANNELS
    assert all(ch in names for ch in MOTOR_CORTEX_CHANNELS)


def test_standardize_is_idempotent(fake_raw: mne.io.RawArray) -> None:
    standardize_channel_names(fake_raw)
    once = list(fake_raw.ch_names)
    standardize_channel_names(fake_raw)
    assert fake_raw.ch_names == once


def test_standardized_names_resolve_to_montage(fake_raw: mne.io.RawArray) -> None:
    standardize_channel_names(fake_raw)
    fake_raw.set_montage("colin27_1005", on_missing="raise")
    assert fake_raw.get_montage() is not None


def test_validate_ok(fake_raw: mne.io.RawArray) -> None:
    report = validate_raw(fake_raw, subject=1, run=4)
    assert report.ok, report.problems


def test_validate_detects_wrong_sampling_rate(raw_factory: RawFactory) -> None:
    report = validate_raw(raw_factory(sfreq=128.0), subject=88, run=4)
    assert not report.ok
    assert any("128" in p and "160" in p for p in report.problems)


def test_validate_detects_wrong_channel_count(raw_factory: RawFactory) -> None:
    report = validate_raw(raw_factory(n_channels=32), subject=1, run=4)
    assert any("32 EEG channels" in p for p in report.problems)


def test_validate_detects_short_recording(raw_factory: RawFactory) -> None:
    report = validate_raw(raw_factory(duration=30.0, n_trials=3), subject=1, run=4)
    assert any("duration" in p for p in report.problems)


def test_validate_detects_missing_annotations(raw_factory: RawFactory) -> None:
    raw = raw_factory()
    raw.set_annotations(None)
    report = validate_raw(raw, subject=1, run=4)
    assert any(f"0 T1/T2 annotations < {MIN_EVENTS_PER_RUN}" == p for p in report.problems)


def test_load_subject_skips_subject_with_one_bad_run(
    raw_factory: RawFactory, monkeypatch, tmp_path
) -> None:
    raws = {"a": raw_factory(), "b": raw_factory(sfreq=128.0), "c": raw_factory()}
    monkeypatch.setattr("neurolink.data.loader.read_run", lambda p: raws[p.name])
    paths = [tmp_path / k for k in ("a", "b", "c")]

    loaded, reports = load_subject(88, paths, runs=(4, 8, 12))
    assert loaded is None
    assert [r.ok for r in reports] == [True, False, True]
    assert reports[1].run == 8 and reports[1].subject == 88


def test_load_subject_returns_one_raw_per_run(raw_factory: RawFactory, monkeypatch, tmp_path):
    monkeypatch.setattr("neurolink.data.loader.read_run", lambda p: raw_factory())
    paths = [tmp_path / f"r{i}" for i in range(3)]
    loaded, reports = load_subject(1, paths, runs=(4, 8, 12))
    assert loaded is not None and len(loaded) == 3
    assert all(r.ok for r in reports)
    assert all(np.isclose(r.info["sfreq"], 160.0) for r in loaded)
