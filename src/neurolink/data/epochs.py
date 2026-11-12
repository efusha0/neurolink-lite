"""Cut continuous data into labelled motor-imagery epochs."""

from __future__ import annotations

import mne
import numpy as np
from numpy.typing import NDArray

from neurolink.config import EVENT_ID


def make_epochs(
    raw: mne.io.BaseRaw,
    tmin: float,
    tmax: float,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Epoch on T1/T2 annotations.

    Returns ``X`` with shape ``(n_epochs, n_channels, n_times)`` in microvolts and ``y``
    with ``0 = left (T1)``, ``1 = right (T2)``. The window is trimmed to exactly
    ``round((tmax - tmin) * sfreq)`` samples, so a 2 s window at 160 Hz yields 320 samples.
    No baseline correction: the band-pass already removes DC, and a baseline window would
    overlap the preceding rest period.
    """
    events, event_map = mne.events_from_annotations(raw, event_id=EVENT_ID, verbose="ERROR")
    if len(events) == 0:
        raise ValueError("no T1/T2 events found in recording")

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_map,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose="ERROR",
    )
    data = epochs.get_data(units="uV", copy=False)
    n_times = round((tmax - tmin) * raw.info["sfreq"])
    x = np.asarray(data[:, :, :n_times], dtype=np.float32)
    # MNE codes: T1 -> 1, T2 -> 2; shift to 0-based class index.
    y = np.asarray(epochs.events[:, 2] - 1, dtype=np.int64)
    return x, y
