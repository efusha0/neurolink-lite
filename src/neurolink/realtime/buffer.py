"""Sliding-window buffer: turns a stream of chunks into fixed-length, overlapping windows."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class SlidingWindower:
    """Emit a ``window``-sample window every ``step`` samples.

    The first window ends at sample ``window`` (counted from the start of the stream), the
    next at ``window + step`` and so on, regardless of how the stream is cut into chunks.
    """

    def __init__(self, n_channels: int, window: int, step: int) -> None:
        if window <= 0 or step <= 0:
            raise ValueError("window and step must be positive")
        self.n_channels = n_channels
        self.window = window
        self.step = step
        self._tail = np.zeros((0, n_channels), dtype=np.float32)
        self.n_seen = 0

    def reset(self) -> None:
        self._tail = np.zeros((0, self.n_channels), dtype=np.float32)
        self.n_seen = 0

    def push(self, chunk: NDArray[np.floating]) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Add ``(n_samples, n_channels)``; return the windows completed by this chunk.

        Returns ``(ends, windows)``: ``ends[i]`` is the stream index one past the last sample
        of ``windows[i]``, and ``windows`` has shape ``(k, n_channels, window)`` (the layout
        the classifiers expect). ``k`` is 0 when no window was completed.
        """
        x = np.asarray(chunk, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.n_channels:
            raise ValueError(f"expected (n_samples, {self.n_channels}), got {x.shape}")
        data = np.concatenate([self._tail, x])
        start_index = self.n_seen - len(self._tail)  # stream index of data[0]
        before = self.n_seen
        self.n_seen += len(x)

        # Window ends in (before, n_seen] of the form window + k * step, k >= 0.
        first_k = max(0, -(-(before + 1 - self.window) // self.step))
        ends = np.arange(self.window + first_k * self.step, self.n_seen + 1, self.step)
        ends = ends.astype(np.int64)
        windows = np.empty((len(ends), self.n_channels, self.window), dtype=np.float32)
        for i, end in enumerate(ends.tolist()):
            lo = end - self.window - start_index
            windows[i] = data[lo : lo + self.window].T

        # Only the last window - 1 samples can be part of a future window.
        self._tail = data[max(0, len(data) - (self.window - 1)) :]
        return ends, windows
