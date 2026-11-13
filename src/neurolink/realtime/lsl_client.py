"""Read the EEG and marker streams from LSL for the online classifier."""

from __future__ import annotations

import bisect

import numpy as np
from numpy.typing import NDArray

from neurolink.realtime.streamer import EEG_STREAM, MARKER_STREAM


class LabelTimeline:
    """True state over time, built from marker events: ``label_at(t)`` = last marker <= t."""

    def __init__(self) -> None:
        self._times: list[float] = []
        self._labels: list[str] = []

    def add(self, timestamp: float, label: str) -> None:
        i = bisect.bisect_right(self._times, timestamp)
        self._times.insert(i, timestamp)
        self._labels.insert(i, label)

    def label_at(self, timestamp: float) -> str | None:
        i = bisect.bisect_right(self._times, timestamp)
        return self._labels[i - 1] if i else None


def _resolve(name: str, timeout: float):  # type: ignore[no-untyped-def]
    import pylsl

    streams = pylsl.resolve_byprop("name", name, 1, timeout)
    if not streams:
        raise TimeoutError(f"LSL stream {name!r} not found within {timeout:g} s")
    return streams[0]


class LslEegReader:
    """Pull ``(n_samples, n_channels)`` chunks and their LSL timestamps from the EEG stream."""

    def __init__(self, name: str = EEG_STREAM, timeout: float = 10.0) -> None:
        import pylsl

        self._inlet = pylsl.StreamInlet(_resolve(name, timeout), max_buflen=30)
        info = self._inlet.info()
        self.sfreq = float(info.nominal_srate())
        self.n_channels = int(info.channel_count())
        names = []
        ch = info.desc().child("channels").child("channel")
        for _ in range(self.n_channels):
            names.append(ch.child_value("label"))
            ch = ch.next_sibling()
        self.ch_names = tuple(names)

    def pull(self, timeout: float = 0.2) -> tuple[NDArray[np.float32], NDArray[np.float64]]:
        """Block up to ``timeout`` for data; returns empty arrays if nothing arrived."""
        samples, stamps = self._inlet.pull_chunk(timeout=timeout, max_samples=4096)
        if not stamps:
            return (
                np.zeros((0, self.n_channels), dtype=np.float32),
                np.zeros(0, dtype=np.float64),
            )
        return np.asarray(samples, dtype=np.float32), np.asarray(stamps, dtype=np.float64)

    def close(self) -> None:
        self._inlet.close_stream()


class LslMarkerReader:
    """Collect marker events into a :class:`LabelTimeline` without blocking."""

    def __init__(self, name: str = MARKER_STREAM, timeout: float = 10.0) -> None:
        import pylsl

        self._inlet = pylsl.StreamInlet(_resolve(name, timeout))
        self.timeline = LabelTimeline()

    def poll(self) -> None:
        while True:
            sample, stamp = self._inlet.pull_sample(timeout=0.0)
            if sample is None:
                return
            self.timeline.add(float(stamp), str(sample[0]))

    def close(self) -> None:
        self._inlet.close_stream()
