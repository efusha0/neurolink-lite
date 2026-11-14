"""Read the EEG and marker streams from LSL for the online classifier."""

from __future__ import annotations

import bisect

import numpy as np
from numpy.typing import NDArray

from neurolink.realtime.streamer import EEG_STREAM, MARKER_STREAM, parse_marker


class LabelTimeline:
    """True state over time, built from marker events: ``label_at(t)`` = last marker <= t.

    Each marker may also carry the replay pass it belongs to (``pass_at``).
    """

    def __init__(self) -> None:
        self._times: list[float] = []
        self._labels: list[str] = []
        self._passes: list[int | None] = []

    def add(self, timestamp: float, label: str, replay_pass: int | None = None) -> None:
        i = bisect.bisect_right(self._times, timestamp)
        self._times.insert(i, timestamp)
        self._labels.insert(i, label)
        self._passes.insert(i, replay_pass)

    def label_at(self, timestamp: float) -> str | None:
        i = bisect.bisect_right(self._times, timestamp)
        return self._labels[i - 1] if i else None

    def pass_at(self, timestamp: float) -> int | None:
        i = bisect.bisect_right(self._times, timestamp)
        return self._passes[i - 1] if i else None


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
        """Return whatever has arrived, waiting up to ``timeout`` for the first sample.

        ``pull_chunk`` with a timeout would wait the *whole* timeout trying to fill its
        buffer, which adds up to ``timeout`` of latency to every prediction (measured:
        p50 107 ms, p95 202 ms with a 0.2 s timeout). So: take what is there without
        waiting; only if nothing is there, block for a single sample and then drain.
        """
        samples, stamps = self._inlet.pull_chunk(timeout=0.0, max_samples=4096)
        if not stamps:
            sample, stamp = self._inlet.pull_sample(timeout=timeout)
            if sample is None:
                return (
                    np.zeros((0, self.n_channels), dtype=np.float32),
                    np.zeros(0, dtype=np.float64),
                )
            more, more_stamps = self._inlet.pull_chunk(timeout=0.0, max_samples=4096)
            samples, stamps = [sample, *more], [stamp, *more_stamps]
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
            state, replay_pass = parse_marker(str(sample[0]))
            self.timeline.add(float(stamp), state, replay_pass)

    def close(self) -> None:
        self._inlet.close_stream()
