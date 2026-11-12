"""Signal preprocessing: band-pass filter, common average reference, channel subset.

None of these steps learns anything from the data (the filter has fixed coefficients, CAR
is a per-sample mean removal), so they can be applied before train/test splitting without
leaking information across folds. Anything that *is* fitted (normalization, CSP, networks)
lives in ``neurolink.models`` and is fitted on training folds only.
"""

from __future__ import annotations

from collections.abc import Sequence

import mne


def bandpass(raw: mne.io.BaseRaw, band: tuple[float, float]) -> mne.io.BaseRaw:
    """Zero-phase FIR band-pass filter, in place.

    ``skip_by_annotation="edge"`` keeps the filter from smearing across boundaries where
    runs were concatenated.
    """
    low, high = band
    raw.filter(
        l_freq=low,
        h_freq=high,
        method="fir",
        fir_design="firwin",
        skip_by_annotation="edge",
        verbose="ERROR",
    )
    return raw


def apply_car(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Common average reference, applied directly to the data (no projector)."""
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    return raw


def pick_channels(raw: mne.io.BaseRaw, channels: Sequence[str] | None) -> mne.io.BaseRaw:
    """Keep only ``channels`` (in the given order). ``None`` keeps everything.

    Raises ``ValueError`` listing any requested channel that is missing.
    """
    if channels is None:
        return raw
    missing = [ch for ch in channels if ch not in raw.ch_names]
    if missing:
        raise ValueError(f"channels not found in recording: {missing}")
    raw.pick(list(channels))
    raw.reorder_channels(list(channels))
    return raw
