"""FastAPI WebSocket server that broadcasts online predictions to the browser.

A background thread (:class:`DecodeWorker`) reads EEG chunks, runs the
:class:`~neurolink.realtime.online.OnlineDecoder` and hands JSON-ready messages to the
:class:`Hub`, which forwards them, in order, to every connected WebSocket client.
The message schema is mirrored in ``frontend/src/types.ts``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
import time
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from numpy.typing import NDArray

from neurolink.config import CLASS_NAMES
from neurolink.realtime.latency import LatencyTracker
from neurolink.realtime.online import OnlineDecoder, Prediction

log = logging.getLogger(__name__)

Clock = Callable[[], float]
Emit = Callable[[dict[str, Any]], None]

DISPLAY_CHANNELS = ("C3", "Cz", "C4")
DISPLAY_DECIMATION = 4
"""Waveform messages carry every 4th filtered sample (40 Hz), plenty for an 8-30 Hz view."""


def prediction_message(
    pred: Prediction,
    t: float,
    true_label: str | None,
    sample_time: float,
    stream_to_pred_ms: float,
    replay_pass: int | None = None,
) -> dict[str, Any]:
    """JSON-ready ``PredictionMessage`` (``pred_to_ws`` is filled in by the hub).

    ``replay_pass`` counts the passes of the looped recording (1, 2, ...); the UI scores
    each pass separately.
    """
    return {
        "type": "prediction",
        "t": t,
        "probs": {name: float(pred.probs[i]) for i, name in enumerate(CLASS_NAMES)},
        "pred": CLASS_NAMES[pred.pred],
        "true": true_label if true_label in (*CLASS_NAMES, "rest") else None,
        "confidence": pred.confidence,
        "active": pred.active,
        "sample_time": sample_time,
        "replay_pass": replay_pass,
        "latency_ms": {"stream_to_pred": stream_to_pred_ms, "pred_to_ws": 0.0},
    }


def eeg_message(t: float, channels: Sequence[str], samples: NDArray[np.floating]) -> dict[str, Any]:
    """JSON-ready ``EegMessage``; ``samples`` is ``(n_samples, n_channels)``."""
    return {
        "type": "eeg",
        "t": t,
        "channels": list(channels),
        "samples": [[round(float(v), 3) for v in samples[:, i]] for i in range(samples.shape[1])],
    }


class ChunkSource(Protocol):
    """Where the worker gets EEG from (an LSL inlet in production, a fake in tests)."""

    def pull(self, timeout: float = 0.2) -> tuple[NDArray[np.float32], NDArray[np.float64]]: ...


class LabelSource(Protocol):
    def poll(self) -> None: ...

    @property
    def timeline(self) -> Any: ...


class DecodeWorker(threading.Thread):
    """Pull chunks, decode, emit prediction and waveform messages until stopped."""

    def __init__(
        self,
        decoder: OnlineDecoder,
        source: ChunkSource,
        emit: Emit,
        clock: Clock,
        model_ch_names: Sequence[str],
        labels: LabelSource | None = None,
        tracker: LatencyTracker | None = None,
        display_channels: Sequence[str] = DISPLAY_CHANNELS,
    ) -> None:
        super().__init__(name="neurolink-decode", daemon=True)
        self.decoder = decoder
        self.source = source
        self.emit = emit
        self.clock = clock
        self.labels = labels
        self.tracker = tracker or LatencyTracker()
        self.display_channels = [ch for ch in display_channels if ch in model_ch_names]
        self._display_idx = [list(model_ch_names).index(ch) for ch in self.display_channels]
        self._stop_event = threading.Event()
        self._t0: float | None = None
        self._n_display = 0
        self.error: BaseException | None = None

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        try:
            while not self._stop_event.is_set():
                chunk, stamps = self.source.pull(timeout=0.2)
                if len(stamps):
                    self.step(chunk, stamps)
        except BaseException as exc:  # surfaced through /health instead of dying silently
            self.error = exc
            log.exception("decode worker stopped")

    def step(self, chunk: NDArray[np.float32], stamps: NDArray[np.float64]) -> None:
        """Process one chunk (public so tests can drive the worker without a thread)."""
        if self._t0 is None:
            self._t0 = float(stamps[0])
        offset = self.decoder.windower.n_seen  # stream index of chunk[0]
        preds = self.decoder.push(chunk)
        done = self.clock()
        if self.labels is not None:
            self.labels.poll()

        filtered = self.decoder.last_filtered
        if self._display_idx and len(filtered):
            # Keep a fixed decimation phase across chunks.
            first = (-self._n_display) % DISPLAY_DECIMATION
            self._n_display += len(filtered)
            shown = filtered[first::DISPLAY_DECIMATION][:, self._display_idx]
            if len(shown):
                self.emit(eeg_message(float(stamps[-1]) - self._t0, self.display_channels, shown))

        for p in preds:
            sample_time = float(stamps[p.end_sample - 1 - offset])
            latency_ms = (done - sample_time) * 1000.0
            self.tracker.add("stream_to_pred", latency_ms)
            timeline = self.labels.timeline if self.labels else None
            true = timeline.label_at(sample_time) if timeline else None
            replay_pass = timeline.pass_at(sample_time) if timeline else None
            msg = prediction_message(
                p, sample_time - self._t0, true, sample_time, latency_ms, replay_pass
            )
            msg["_created"] = done
            self.emit(msg)


class Hub:
    """Fan-out of messages to WebSocket clients, fed from another thread via :meth:`emit`."""

    def __init__(self, clock: Clock = time.perf_counter, tracker: LatencyTracker | None = None):
        self.clock = clock
        self.tracker = tracker or LatencyTracker()
        self.clients: set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[dict[str, Any]] | None = None

    def bind(self) -> None:
        """Attach to the running event loop (called from the app's lifespan)."""
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()

    def emit(self, message: dict[str, Any]) -> None:
        """Thread-safe: queue a message for broadcast. Dropped if the app is not running."""
        if self._loop is None or self._queue is None or self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(self._queue.put_nowait, message)

    async def pump(self) -> None:
        assert self._queue is not None
        while True:
            message = await self._queue.get()
            created = message.pop("_created", None)
            if created is not None:
                ms = (self.clock() - created) * 1000.0
                message["latency_ms"]["pred_to_ws"] = ms
                self.tracker.add("pred_to_ws", ms)
            text = json.dumps(message)
            for ws in list(self.clients):
                try:
                    await ws.send_text(text)
                except Exception:
                    self.clients.discard(ws)


def create_app(
    hub: Hub,
    start_worker: Callable[[Emit], DecodeWorker | None] | None = None,
    info: dict[str, Any] | None = None,
    static_dir: Path | None = None,
) -> FastAPI:
    """Build the app. ``start_worker(emit)`` is called on startup and stopped on shutdown.

    With ``static_dir`` (the built browser UI) the page is served from ``/`` on the same
    port as the WebSocket and the JSON endpoints.
    """

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        hub.bind()
        pump = asyncio.create_task(hub.pump())
        worker = start_worker(hub.emit) if start_worker else None
        app.state.worker = worker
        try:
            yield
        finally:
            if worker is not None:
                worker.stop()
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump

    app = FastAPI(title="NeuroLink-lite", lifespan=lifespan)
    app.state.hub = hub

    @app.get("/health")
    def health() -> dict[str, Any]:
        worker = getattr(app.state, "worker", None)
        error = getattr(worker, "error", None)
        return {
            "status": "error" if error else "ok",
            "error": repr(error) if error else None,
            "clients": len(hub.clients),
        }

    @app.get("/info")
    def model_info() -> dict[str, Any]:
        return {"class_names": list(CLASS_NAMES), **(info or {})}

    @app.get("/latency")
    def latency() -> dict[str, Any]:
        return hub.tracker.summary()

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        hub.clients.add(ws)
        try:
            while True:
                await ws.receive_text()  # clients only listen; this detects disconnects
        except WebSocketDisconnect:
            pass
        finally:
            hub.clients.discard(ws)

    if static_dir is not None:
        # Mounted last: the routes above win, everything else is a file of the UI.
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="ui")
    return app
