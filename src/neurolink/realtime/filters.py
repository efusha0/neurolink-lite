"""Stateful causal band-pass filter for streaming data.

The online classifier receives EEG in small chunks and cannot use the zero-phase filter of
the offline pipeline (it needs future samples). :class:`CausalBandpass` is a Butterworth
filter in second-order sections that keeps its state between calls, so filtering a
recording chunk by chunk gives the same output as filtering it in one call.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import signal


def design_bandpass(band: tuple[float, float], sfreq: float, order: int = 4) -> NDArray[np.float64]:
    """Butterworth band-pass as second-order sections, shape ``(n_sections, 6)``."""
    low, high = band
    if not 0.0 < low < high < sfreq / 2:
        raise ValueError(f"invalid band {band} for sampling rate {sfreq:g} Hz")
    sos = signal.butter(order, [low, high], btype="bandpass", fs=sfreq, output="sos")
    return np.asarray(sos, dtype=np.float64)


class CausalBandpass:
    """Causal band-pass over ``(n_samples, n_channels)`` chunks, state kept across calls.

    The state is initialised at the first sample as if the input had been constant at that
    value forever, which removes the start-up transient a DC offset would otherwise cause.
    """

    def __init__(
        self,
        band: tuple[float, float],
        sfreq: float,
        n_channels: int,
        order: int = 4,
    ) -> None:
        self.sos = design_bandpass(band, sfreq, order)
        self.n_channels = n_channels
        self._zi_unit = np.asarray(signal.sosfilt_zi(self.sos), dtype=np.float64)
        self._zi: NDArray[np.float64] | None = None

    def reset(self) -> None:
        self._zi = None

    def process(self, chunk: NDArray[np.floating]) -> NDArray[np.float64]:
        x = np.asarray(chunk, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_channels:
            raise ValueError(f"expected (n_samples, {self.n_channels}), got {x.shape}")
        if x.shape[0] == 0:
            return x
        if self._zi is None:
            # (n_sections, 2, 1) * (n_channels,) -> (n_sections, 2, n_channels)
            self._zi = self._zi_unit[:, :, None] * x[0][None, None, :]
        y, self._zi = signal.sosfilt(self.sos, x, axis=0, zi=self._zi)
        return np.asarray(y, dtype=np.float64)
