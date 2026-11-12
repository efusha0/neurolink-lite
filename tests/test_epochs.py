from __future__ import annotations

import mne
import numpy as np
import pytest

from neurolink.config import SFREQ
from neurolink.data.epochs import make_epochs
from tests.conftest import RawFactory


def test_epoch_shape_and_labels(fake_raw: mne.io.RawArray) -> None:
    x, y = make_epochs(fake_raw, tmin=0.5, tmax=2.5)
    assert x.shape == (15, 64, 320)
    assert x.dtype == np.float32
    assert y.dtype == np.int64
    # fixture alternates T1, T2, T1, ... -> 0, 1, 0, ...
    np.testing.assert_array_equal(y, np.arange(15) % 2)


def test_epochs_are_in_microvolts(fake_raw: mne.io.RawArray) -> None:
    # fixture noise has sigma = 10 uV (stored as 10e-6 V)
    x, _ = make_epochs(fake_raw, tmin=0.5, tmax=2.5)
    assert 8.0 < x.std() < 12.0


def test_epoch_window_is_cut_at_the_right_place(raw_factory: RawFactory) -> None:
    raw = raw_factory(n_trials=4, duration=40.0)
    # Stamp a ramp so the sample index is recoverable from the value.
    raw._data[0] = np.arange(raw.n_times) * 1e-6
    x, _ = make_epochs(raw, tmin=0.5, tmax=2.5)
    first_cue_sample = round(4.1 * SFREQ)
    expected_start = first_cue_sample + round(0.5 * SFREQ)
    assert round(x[0, 0, 0]) == expected_start
    assert round(x[0, 0, -1]) == expected_start + 319


def test_single_class_run(raw_factory: RawFactory) -> None:
    raw = raw_factory(labels="T2", n_trials=6, duration=60.0)
    _x, y = make_epochs(raw, tmin=0.5, tmax=2.5)
    assert len(y) == 6 and set(y.tolist()) == {1}


def test_no_events_raises(fake_raw: mne.io.RawArray) -> None:
    fake_raw.set_annotations(None)
    with pytest.raises(ValueError, match="no T1/T2"):
        make_epochs(fake_raw, tmin=0.5, tmax=2.5)
