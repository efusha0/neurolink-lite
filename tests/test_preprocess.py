from __future__ import annotations

import mne
import numpy as np
import pytest

from neurolink.config import MOTOR_CORTEX_CHANNELS, SFREQ
from neurolink.data.loader import standardize_channel_names
from neurolink.data.preprocess import apply_car, bandpass, pick_channels
from tests.conftest import RawFactory


def _band_power(raw: mne.io.BaseRaw, freq: float, ch: int = 0) -> float:
    """Power at ``freq`` via periodogram, skipping filter edge transients."""
    data = raw.get_data()[ch]
    data = data[int(5 * SFREQ) : -int(5 * SFREQ)]
    spectrum = np.abs(np.fft.rfft(data)) ** 2
    freqs = np.fft.rfftfreq(len(data), d=1 / SFREQ)
    return float(spectrum[np.argmin(np.abs(freqs - freq))])


def _tone(t: np.ndarray, f: float) -> np.ndarray:  # type: ignore[type-arg]
    return 50e-6 * np.sin(2 * np.pi * f * t)


def test_bandpass_keeps_passband_and_rejects_stopband(raw_factory: RawFactory) -> None:
    raw = raw_factory(duration=60.0, n_trials=5)
    t = raw.times
    raw._data[:] = _tone(t, 3.0) + _tone(t, 15.0) + _tone(t, 45.0)
    before = {f: _band_power(raw, f) for f in (3.0, 15.0, 45.0)}

    bandpass(raw, (8.0, 30.0))
    after = {f: _band_power(raw, f) for f in (3.0, 15.0, 45.0)}

    assert after[15.0] / before[15.0] > 0.8  # passband nearly untouched
    assert after[3.0] / before[3.0] < 1e-2  # stopbands attenuated by > 20 dB
    assert after[45.0] / before[45.0] < 1e-2


def test_bandpass_preserves_shape_and_annotations(fake_raw: mne.io.RawArray) -> None:
    n_annot = len(fake_raw.annotations)
    shape = fake_raw.get_data().shape
    bandpass(fake_raw, (8.0, 30.0))
    assert fake_raw.get_data().shape == shape
    assert len(fake_raw.annotations) == n_annot


def test_car_removes_common_signal(fake_raw: mne.io.RawArray) -> None:
    common = 100e-6 * np.sin(2 * np.pi * 10 * fake_raw.times)
    fake_raw._data += common
    apply_car(fake_raw)
    mean_across_channels = fake_raw.get_data().mean(axis=0)
    assert np.abs(mean_across_channels).max() < 1e-12


def test_pick_channels_subset_and_order(fake_raw: mne.io.RawArray) -> None:
    standardize_channel_names(fake_raw)
    wanted = ("C4", "Cz", "C3")
    pick_channels(fake_raw, wanted)
    assert tuple(fake_raw.ch_names) == wanted


def test_pick_channels_none_keeps_everything(fake_raw: mne.io.RawArray) -> None:
    n = len(fake_raw.ch_names)
    pick_channels(fake_raw, None)
    assert len(fake_raw.ch_names) == n


def test_pick_channels_motor_subset(fake_raw: mne.io.RawArray) -> None:
    standardize_channel_names(fake_raw)
    pick_channels(fake_raw, MOTOR_CORTEX_CHANNELS)
    assert tuple(fake_raw.ch_names) == MOTOR_CORTEX_CHANNELS


def test_pick_channels_reports_missing(fake_raw: mne.io.RawArray) -> None:
    standardize_channel_names(fake_raw)
    with pytest.raises(ValueError, match="NOPE"):
        pick_channels(fake_raw, ("C3", "NOPE"))
