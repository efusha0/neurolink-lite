"""Replay a recorded run over Lab Streaming Layer as if it came from a live amplifier.

Two outlets are created:

* an EEG stream with the raw, unfiltered 64-channel signal in microvolts at 160 Hz, sent
  in small chunks at real-time pace, each sample stamped with the time it "was acquired";
* a marker stream that announces the true state (``rest`` / ``left`` / ``right``) whenever
  it changes. The classifier never reads it; it only feeds the "true label vs prediction"
  indicator in the UI.
"""

from __future__ import annotations

import threading
import time

import numpy as np
from numpy.typing import NDArray

from neurolink.config import CLASS_NAMES
from neurolink.realtime.pseudo_online import REST, Recording

EEG_STREAM = "NeuroLinkEEG"
MARKER_STREAM = "NeuroLinkMarkers"
DEFAULT_CHUNK = 8
"""Samples per LSL chunk (50 ms at 160 Hz), typical for EEG amplifiers."""


def label_name(label: int) -> str:
    """Name of a sample label: ``left`` / ``right`` / ``rest`` / ``none``."""
    if label >= 0:
        return CLASS_NAMES[label]
    return "rest" if label == REST else "none"


def marker_events(sample_labels: NDArray[np.int8]) -> list[tuple[int, str]]:
    """``(sample_index, label_name)`` at the start and at every change of the true state."""
    if len(sample_labels) == 0:
        return []
    change = np.flatnonzero(np.diff(sample_labels)) + 1
    idx = np.concatenate([[0], change])
    return [(int(i), label_name(int(sample_labels[i]))) for i in idx]


class ReplayStreamer:
    """Push a :class:`Recording` to LSL at real-time pace (``speed`` > 1 replays faster)."""

    def __init__(
        self,
        recording: Recording,
        chunk: int = DEFAULT_CHUNK,
        speed: float = 1.0,
        eeg_name: str = EEG_STREAM,
        marker_name: str = MARKER_STREAM,
    ) -> None:
        import pylsl

        self.recording = recording
        self.chunk = chunk
        self.rate = recording.sfreq * speed
        source = f"neurolink-S{recording.subject:03d}R{recording.run:02d}"
        info = pylsl.StreamInfo(
            eeg_name, "EEG", len(recording.ch_names), self.rate, "float32", source
        )
        channels = info.desc().append_child("channels")
        for name in recording.ch_names:
            ch = channels.append_child("channel")
            ch.append_child_value("label", name)
            ch.append_child_value("unit", "microvolts")
            ch.append_child_value("type", "EEG")
        self._eeg = pylsl.StreamOutlet(info, chunk_size=chunk)
        marker_info = pylsl.StreamInfo(marker_name, "Markers", 1, 0.0, "string", source + "-m")
        self._markers = pylsl.StreamOutlet(marker_info)
        self._clock = pylsl.local_clock
        self.n_sent = 0

    def run(self, stop: threading.Event | None = None, loop: bool = False) -> int:
        """Stream the recording once (or forever with ``loop``); returns samples sent."""
        data = self.recording.data
        events = marker_events(self.recording.sample_labels)
        while True:
            t0 = self._clock()  # LSL time at which sample 0 starts being acquired
            next_event = 0
            for start in range(0, len(data), self.chunk):
                if stop is not None and stop.is_set():
                    return self.n_sent
                end = min(start + self.chunk, len(data))
                # A chunk leaves the "amplifier" once its last sample has been acquired.
                due = t0 + end / self.rate
                delay = due - self._clock()
                if delay > 0:
                    time.sleep(delay)
                while next_event < len(events) and events[next_event][0] < end:
                    index, name = events[next_event]
                    # Sample ``index`` is stamped t0 + (index + 1) / rate; the marker goes
                    # half a sample earlier so "last marker <= sample time" is unambiguous.
                    self._markers.push_sample([name], t0 + (index + 0.5) / self.rate)
                    next_event += 1
                self._eeg.push_chunk(data[start:end].tolist(), due)
                self.n_sent += end - start
            if not loop:
                return self.n_sent
