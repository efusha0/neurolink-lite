"""Latency bookkeeping for the real-time path.

Stages (all on the LSL clock of one machine):

* ``stream_to_pred``: from the acquisition time of the newest sample in the window to the
  moment the smoothed prediction exists (LSL transport + chunking + filter + model).
* ``pred_to_ws``: from there to the moment the message is handed to the WebSocket.
* ``end_to_end``: acquisition time of the newest sample to reception by a WebSocket
  client (measured by ``neurolink latency`` with a local client; no browser rendering).

The 2 s analysis window is *not* latency in this sense, but it does delay how fast a
change of intent shows up: a window ending shortly after a cue still contains mostly
rest. That cost is visible in the pseudo-online results as the gap between the accuracy
of windows entirely inside a trial and of all windows ending during a trial.
"""

from __future__ import annotations

import threading
from collections import defaultdict

import numpy as np


class LatencyTracker:
    """Thread-safe collection of latency samples (milliseconds) per stage."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._samples: dict[str, list[float]] = defaultdict(list)

    def add(self, stage: str, milliseconds: float) -> None:
        with self._lock:
            self._samples[stage].append(float(milliseconds))

    def summary(self) -> dict[str, dict[str, float]]:
        with self._lock:
            data = {k: np.asarray(v, dtype=np.float64) for k, v in self._samples.items()}
        return {
            stage: {
                "n": float(len(v)),
                "mean_ms": float(v.mean()),
                "p50_ms": float(np.percentile(v, 50)),
                "p95_ms": float(np.percentile(v, 95)),
                "p99_ms": float(np.percentile(v, 99)),
                "max_ms": float(v.max()),
            }
            for stage, v in sorted(data.items())
            if len(v)
        }
