from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from neurolink.realtime.latency import LatencyTracker
from neurolink.realtime.lsl_client import LabelTimeline
from neurolink.realtime.online import OnlineDecoder, Prediction
from neurolink.realtime.pseudo_online import OUTSIDE, REST, Recording
from neurolink.realtime.server import (
    DecodeWorker,
    Hub,
    create_app,
    eeg_message,
    prediction_message,
)
from neurolink.realtime.streamer import (
    format_marker,
    label_name,
    marker_events,
    parse_marker,
)
from tests.test_realtime_core import PowerModel

SFREQ = 160.0
IN_CH = ["C3", "Cz", "C4", "Pz"]
MODEL_CH = ["C3", "C4"]


def make_decoder(threshold: float = 0.5) -> OnlineDecoder:
    return OnlineDecoder(
        PowerModel(), IN_CH, MODEL_CH, (8.0, 30.0), sfreq=SFREQ, car=False, threshold=threshold
    )


def stream_chunks(n: int = 800, chunk: int = 8, t0: float = 100.0):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, 4)).astype(np.float32)
    x[:, 2] += 20 * np.sin(2 * np.pi * 12 * np.arange(n) / SFREQ)  # "right": power on C4
    for i in range(0, n, chunk):
        stamps = t0 + (np.arange(i, min(i + chunk, n)) + 1) / SFREQ
        yield x[i : i + chunk], stamps


# ----------------------------------------------------------------------------- helpers
def test_marker_events_and_label_names() -> None:
    labels = np.array([REST, REST, 0, 0, REST, 1, OUTSIDE], dtype=np.int8)
    assert marker_events(labels) == [
        (0, "rest"), (2, "left"), (4, "rest"), (5, "right"), (6, "none"),
    ]  # fmt: skip
    assert marker_events(np.zeros(0, dtype=np.int8)) == []
    # The heartbeat repeats the current state at a fixed interval, for late joiners.
    assert marker_events(labels, heartbeat=3) == [
        (0, "rest"), (2, "left"), (3, "left"), (4, "rest"), (5, "right"), (6, "none"),
    ]  # fmt: skip
    assert format_marker("left", 3) == "left|3"
    assert parse_marker("left|3") == ("left", 3)
    assert parse_marker("rest") == ("rest", None) and parse_marker("rest|x") == ("rest", None)
    assert [label_name(v) for v in (0, 1, REST, OUTSIDE)] == ["left", "right", "rest", "none"]


def test_label_timeline_returns_last_marker_before_time() -> None:
    tl = LabelTimeline()
    assert tl.label_at(5.0) is None
    tl.add(10.0, "rest")
    tl.add(14.1, "left")
    tl.add(12.0, "right")  # out-of-order arrival is tolerated
    assert tl.label_at(9.9) is None
    assert tl.label_at(10.0) == "rest"
    assert tl.label_at(13.0) == "right"
    assert tl.label_at(99.0) == "left"
    # Markers may carry the replay pass; it is looked up the same way.
    assert tl.pass_at(13.0) is None
    tl.add(20.0, "rest", replay_pass=2)
    assert tl.pass_at(19.9) is None and tl.pass_at(20.0) == 2 and tl.label_at(21.0) == "rest"


def test_latency_tracker_summary() -> None:
    tr = LatencyTracker()
    assert tr.summary() == {}
    for v in range(1, 101):
        tr.add("stream_to_pred", float(v))
    s = tr.summary()["stream_to_pred"]
    assert s["n"] == 100 and s["max_ms"] == 100.0
    assert s["p50_ms"] == pytest.approx(50.5) and s["p95_ms"] == pytest.approx(95.05)


def test_message_builders_match_frontend_schema() -> None:
    pred = Prediction(
        end_sample=320, raw_probs=np.array([0.1, 0.9]), probs=np.array([0.25, 0.75]),
        pred=1, confidence=0.75, active=True,
    )  # fmt: skip
    msg = prediction_message(pred, t=2.0, true_label="right", sample_time=102.0,
                             stream_to_pred_ms=12.5)  # fmt: skip
    assert msg == {
        "type": "prediction", "t": 2.0, "probs": {"left": 0.25, "right": 0.75},
        "pred": "right", "true": "right", "confidence": 0.75, "active": True,
        "sample_time": 102.0, "replay_pass": None,
        "latency_ms": {"stream_to_pred": 12.5, "pred_to_ws": 0.0},
    }  # fmt: skip
    assert prediction_message(pred, 0.0, "left", 0.0, 0.0, replay_pass=4)["replay_pass"] == 4
    assert prediction_message(pred, 0.0, "none", 0.0, 0.0)["true"] is None
    eeg = eeg_message(1.0, ["C3", "C4"], np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    assert eeg == {"type": "eeg", "t": 1.0, "channels": ["C3", "C4"],
                   "samples": [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]}  # fmt: skip


# ------------------------------------------------------------------------------ worker
class FakeLabels:
    def __init__(self) -> None:
        self.timeline = LabelTimeline()
        self.timeline.add(0.0, "right", replay_pass=1)
        self.timeline.add(103.0, "right", replay_pass=2)

    def poll(self) -> None:
        pass


def test_worker_emits_predictions_with_timestamps_and_latency() -> None:
    out: list[dict[str, Any]] = []
    now = {"t": 0.0}
    worker = DecodeWorker(
        make_decoder(), source=None, emit=out.append, clock=lambda: now["t"],  # type: ignore[arg-type]
        model_ch_names=MODEL_CH, labels=FakeLabels(),
    )  # fmt: skip
    for chunk, stamps in stream_chunks():
        now["t"] = float(stamps[-1]) + 0.004  # decoding finishes 4 ms after the last sample
        worker.step(chunk, stamps)

    preds = [m for m in out if m["type"] == "prediction"]
    assert len(preds) == (800 - 320) // 40 + 1
    # Window k ends at sample 320 + 40 k; its newest sample was acquired at t0 + end / fs.
    for k, m in enumerate(preds):
        assert m["sample_time"] == pytest.approx(100.0 + (320 + 40 * k) / SFREQ)
        assert m["t"] == pytest.approx((320 + 40 * k - 1) / SFREQ)
        assert m["latency_ms"]["stream_to_pred"] == pytest.approx(4.0, abs=1e-6)
        assert m["true"] == "right"
        assert m["replay_pass"] == (1 if m["sample_time"] < 103.0 else 2)
    assert all(m["pred"] == "right" for m in preds[4:])
    assert worker.tracker.summary()["stream_to_pred"]["n"] == len(preds)

    eeg = [m for m in out if m["type"] == "eeg"]
    assert eeg[0]["channels"] == ["C3", "C4"]  # Cz is not a model channel here
    assert sum(len(m["samples"][0]) for m in eeg) == 800 // 4  # decimated by 4, no gaps


def test_worker_window_timestamp_inside_a_large_chunk() -> None:
    """With irregular chunking the window end may fall in the middle of a chunk."""
    out: list[dict[str, Any]] = []
    worker = DecodeWorker(make_decoder(), None, out.append, lambda: 0.0, MODEL_CH)  # type: ignore[arg-type]
    chunks = list(stream_chunks(n=400, chunk=400))
    worker.step(*chunks[0])
    preds = [m for m in out if m["type"] == "prediction"]
    assert [m["sample_time"] for m in preds] == pytest.approx(
        [100.0 + e / SFREQ for e in (320, 360, 400)]
    )


# --------------------------------------------------------------------------------- app
class FakeSource:
    """Replays chunks at full speed, then idles."""

    def __init__(self) -> None:
        self._chunks = list(stream_chunks())
        self.started = threading.Event()

    def pull(self, timeout: float = 0.2):  # type: ignore[no-untyped-def]
        self.started.wait()
        if self._chunks:
            return self._chunks.pop(0)
        time.sleep(0.01)
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0)


def test_websocket_receives_ordered_predictions() -> None:
    source = FakeSource()
    hub = Hub()

    def start(emit):  # type: ignore[no-untyped-def]
        worker = DecodeWorker(make_decoder(), source, emit, time.perf_counter, MODEL_CH,
                              tracker=hub.tracker)  # fmt: skip
        worker.start()
        return worker

    app = create_app(hub, start, info={"model": "fake", "threshold": 0.5})
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/info").json() == {
            "class_names": ["left", "right"], "model": "fake", "threshold": 0.5,
        }  # fmt: skip
        with client.websocket_connect("/ws") as ws:
            source.started.set()  # start the stream only once the client listens
            preds: list[dict[str, Any]] = []
            while len(preds) < 13:
                msg = ws.receive_json()
                if msg["type"] == "prediction":
                    preds.append(msg)
        times = [m["t"] for m in preds]
        assert times == sorted(times) and len(set(times)) == 13
        assert all("_created" not in m for m in preds)
        assert all(m["latency_ms"]["pred_to_ws"] >= 0.0 for m in preds)
        summary = client.get("/latency").json()
        assert summary["pred_to_ws"]["n"] >= 13 and summary["stream_to_pred"]["n"] == 13
    assert not app.state.worker.is_alive() or app.state.worker.join(2.0) is None


def test_health_reports_worker_errors() -> None:
    class Boom:
        def pull(self, timeout: float = 0.2):  # type: ignore[no-untyped-def]
            raise RuntimeError("stream lost")

    hub = Hub()

    def start(emit):  # type: ignore[no-untyped-def]
        worker = DecodeWorker(make_decoder(), Boom(), emit, time.perf_counter, MODEL_CH)
        worker.start()
        worker.join(2.0)
        return worker

    with TestClient(create_app(hub, start)) as client:
        body = client.get("/health").json()
        assert body["status"] == "error" and "stream lost" in body["error"]


# ----------------------------------------------------------------------- LSL end to end
@pytest.mark.lsl
def test_lsl_replay_reaches_decoder_in_real_time() -> None:
    pylsl = pytest.importorskip("pylsl")
    from neurolink.realtime.lsl_client import LslEegReader, LslMarkerReader
    from neurolink.realtime.streamer import ReplayStreamer

    n = int(4.0 * SFREQ)
    rng = np.random.default_rng(1)
    data = rng.normal(size=(n, 4)).astype(np.float32)
    labels = np.array([REST] * 320 + [1] * (n - 320), dtype=np.int8)
    rec = Recording(1, 12, data, tuple(IN_CH), SFREQ, labels)
    eeg_name, marker_name = "NeuroLinkTestEEG", "NeuroLinkTestMarkers"
    try:
        streamer = ReplayStreamer(rec, eeg_name=eeg_name, marker_name=marker_name)
    except Exception as exc:  # no usable liblsl / network stack
        pytest.skip(f"LSL unavailable: {exc}")

    stop = threading.Event()
    reader = markers = None
    thread = threading.Thread(target=streamer.run, kwargs={"stop": stop, "loop": True})
    try:
        thread.start()
        reader = LslEegReader(eeg_name, timeout=10.0)
        markers = LslMarkerReader(marker_name, timeout=10.0)
        assert reader.ch_names == tuple(IN_CH) and reader.sfreq == SFREQ

        out: list[dict[str, Any]] = []
        worker = DecodeWorker(make_decoder(), reader, out.append, pylsl.local_clock, MODEL_CH,
                              labels=markers)  # fmt: skip
        deadline = time.perf_counter() + 6.0
        while time.perf_counter() < deadline:
            chunk, stamps = reader.pull(timeout=0.2)
            if len(stamps):
                worker.step(chunk, stamps)
        preds = [m for m in out if m["type"] == "prediction"]
    finally:
        stop.set()
        thread.join(5.0)
        for inlet in (reader, markers):
            if inlet is not None:
                inlet.close()

    # ~6 s of real-time data minus the first 2 s window, one prediction per 0.25 s.
    assert 8 <= len(preds) <= 20
    gaps = np.diff([m["sample_time"] for m in preds])
    np.testing.assert_allclose(gaps, 0.25, atol=0.02)  # no samples lost, real-time pace
    # The window ends on a chunk boundary, so a prediction must not wait for more data:
    # a reader that blocks for its whole timeout would show a median near 100 ms.
    latency = [m["latency_ms"]["stream_to_pred"] for m in preds]
    assert 0.0 <= float(np.median(latency)) < 60.0
    assert {m["true"] for m in preds} <= {"rest", "right", None}
    # The 4 s recording is looped: late predictions belong to the second pass, and thanks
    # to the one-second heartbeat the state is known from the first prediction on.
    passes = [m["replay_pass"] for m in preds]
    assert passes == sorted(passes) and passes[0] == 1 and passes[-1] >= 2
    assert all(m["true"] is not None for m in preds)
