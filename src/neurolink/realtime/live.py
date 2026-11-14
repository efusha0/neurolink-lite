"""Assemble the live system (LSL -> online decoder -> WebSocket) and measure its latency."""

from __future__ import annotations

import json
import logging
import os
import platform
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from fastapi import FastAPI

from neurolink.config import CLASS_NAMES, SFREQ
from neurolink.realtime.demo_model import DemoBundle, public_meta
from neurolink.realtime.latency import LatencyTracker
from neurolink.realtime.lsl_client import LslEegReader, LslMarkerReader
from neurolink.realtime.online import OnlineDecoder
from neurolink.realtime.pseudo_online import Recording
from neurolink.realtime.server import DecodeWorker, Emit, Hub, create_app
from neurolink.realtime.streamer import DEFAULT_CHUNK, EEG_STREAM, MARKER_STREAM, ReplayStreamer

log = logging.getLogger(__name__)

WARMUP_S = 2.0
"""Predictions of the first seconds are left out of latency statistics (model warm-up and
the backlog that accumulates while the server starts)."""


def make_decoder(bundle: DemoBundle, in_ch_names: tuple[str, ...]) -> OnlineDecoder:
    return OnlineDecoder(
        bundle.model,
        in_ch_names,
        bundle.ch_names,
        bundle.band,
        config=bundle.realtime,
        sfreq=SFREQ,
        car=bundle.car,
        threshold=bundle.threshold,
    )


@dataclass
class LiveSystem:
    app: FastAPI
    hub: Hub
    reader: LslEegReader
    markers: LslMarkerReader | None

    def close(self) -> None:
        """Close the inlets (after the app has shut down and stopped its worker)."""
        worker = getattr(self.app.state, "worker", None)
        if worker is not None:
            worker.stop()
            worker.join(2.0)  # never close an inlet while the worker is pulling from it
        self.reader.close()
        if self.markers is not None:
            self.markers.close()


def build_live(
    bundle: DemoBundle,
    eeg_name: str = EEG_STREAM,
    marker_name: str = MARKER_STREAM,
    timeout: float = 10.0,
    marker_timeout: float = 3.0,
) -> LiveSystem:
    """Resolve the LSL streams and build the FastAPI app around a decode worker.

    Raises ``TimeoutError`` if the EEG stream is not there. The marker stream is optional:
    without it the ``true`` field of the prediction messages is ``null``.
    """
    import pylsl

    reader = LslEegReader(eeg_name, timeout=timeout)
    if abs(reader.sfreq - SFREQ) > 1e-6:
        reader.close()
        raise ValueError(f"stream rate is {reader.sfreq:g} Hz, the model needs {SFREQ:g} Hz")
    decoder = make_decoder(bundle, reader.ch_names)  # raises if a model channel is missing
    markers: LslMarkerReader | None
    try:
        markers = LslMarkerReader(marker_name, timeout=marker_timeout)
    except TimeoutError:
        log.warning("marker stream %r not found: true labels will not be shown", marker_name)
        markers = None

    hub = Hub(clock=pylsl.local_clock)

    def start(emit: Emit) -> DecodeWorker:
        worker = DecodeWorker(
            decoder, reader, emit, pylsl.local_clock, bundle.ch_names,
            labels=markers, tracker=hub.tracker,
        )  # fmt: skip
        worker.start()
        return worker

    info = {**public_meta(bundle), "eeg_stream": eeg_name, "sfreq": reader.sfreq}
    return LiveSystem(create_app(hub, start, info=info), hub, reader, markers)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def session_stats(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """What a viewer of the session saw: accuracy and cursor activity per true state."""
    trial = [m for m in messages if m["true"] in CLASS_NAMES]
    rest = [m for m in messages if m["true"] == "rest"]
    active = [m for m in trial if m["active"]]

    def share(hits: int, total: int) -> float:
        return hits / total if total else float("nan")

    return {
        "n_predictions": len(messages),
        "n_trial_windows": len(trial),
        "n_rest_windows": len(rest),
        "acc_all_smooth": share(sum(m["pred"] == m["true"] for m in trial), len(trial)),
        "imagery_active_rate": share(len(active), len(trial)),
        "acc_when_active": share(sum(m["pred"] == m["true"] for m in active), len(active)),
        "rest_active_rate": share(sum(m["active"] for m in rest), len(rest)),
    }


def measure_latency(
    bundle: DemoBundle,
    recording: Recording,
    duration_s: float = 120.0,
    chunk: int = DEFAULT_CHUNK,
    warmup_s: float = WARMUP_S,
) -> dict[str, Any]:
    """Run streamer, server and a local WebSocket client; time every prediction.

    ``end_to_end`` is the LSL-clock time from the acquisition of the newest sample of a
    window to the arrival of its prediction at a WebSocket client on the same machine.
    Browser rendering is not included.
    """
    import pylsl
    import uvicorn
    from websockets.sync.client import connect

    tag = f"{os.getpid()}"
    eeg_name, marker_name = f"{EEG_STREAM}-latency-{tag}", f"{MARKER_STREAM}-latency-{tag}"
    streamer = ReplayStreamer(recording, chunk=chunk, eeg_name=eeg_name, marker_name=marker_name)
    stop = threading.Event()
    stream_thread = threading.Thread(
        target=streamer.run, kwargs={"stop": stop, "loop": True}, daemon=True
    )
    stream_thread.start()
    live: LiveSystem | None = None
    server = None
    server_thread = None
    received: list[tuple[float, dict[str, Any]]] = []
    try:
        live = build_live(bundle, eeg_name, marker_name)
        port = _free_port()
        server = uvicorn.Server(
            uvicorn.Config(live.app, host="127.0.0.1", port=port, log_level="warning")
        )
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()
        t_wait = time.perf_counter()
        while not server.started:
            if time.perf_counter() - t_wait > 20.0:
                raise TimeoutError("the WebSocket server did not start")
            time.sleep(0.01)

        with connect(f"ws://127.0.0.1:{port}/ws") as ws:
            deadline = time.perf_counter() + duration_s
            while time.perf_counter() < deadline:
                try:
                    text = ws.recv(timeout=1.0)
                except TimeoutError:
                    continue
                arrived = pylsl.local_clock()
                message = json.loads(text)
                if message["type"] == "prediction":
                    received.append((arrived, message))
        worker_error = getattr(live.app.state.worker, "error", None)
        if worker_error is not None:
            raise RuntimeError(f"decode worker failed: {worker_error!r}")
    finally:
        if server is not None and server_thread is not None:
            server.should_exit = True
            server_thread.join(10.0)
        if live is not None:
            live.close()  # inlets first, so liblsl does not report a broken stream
        stop.set()
        stream_thread.join(5.0)

    if not received:
        raise RuntimeError("no prediction arrived; is the recording shorter than one window?")
    first = received[0][1]["sample_time"]
    kept = [(a, m) for a, m in received if m["sample_time"] - first >= warmup_s]
    tracker = LatencyTracker()
    for arrived, m in kept:
        tracker.add("end_to_end", (arrived - m["sample_time"]) * 1000.0)
        tracker.add("stream_to_pred", m["latency_ms"]["stream_to_pred"])
        tracker.add("pred_to_ws", m["latency_ms"]["pred_to_ws"])
    step = bundle.realtime.step_s
    gaps = np.diff([m["sample_time"] for _a, m in received])
    return {
        "stages_ms": tracker.summary(),
        "duration_s": duration_s,
        "warmup_discarded_s": warmup_s,
        "chunk_samples": chunk,
        "chunk_ms": 1000.0 * chunk / recording.sfreq,
        "window_s": bundle.realtime.window_s,
        "step_s": step,
        "n_predictions_received": len(received),
        "n_predictions_timed": len(kept),
        # Every step must produce exactly one prediction: larger gaps = lost samples.
        "max_gap_between_predictions_s": float(gaps.max()) if len(gaps) else float("nan"),
        "subject": recording.subject,
        "run": recording.run,
        "session": session_stats([m for _a, m in received]),
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
        },
    }
