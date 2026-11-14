from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from neurolink.cli import app
from neurolink.config import RealtimeConfig
from neurolink.models.csp_lda import make_csp_lda
from neurolink.models.eegnet import EEGNetConfig
from neurolink.models.trainer import EEGNetClassifier, TrainConfig
from neurolink.realtime.demo_model import DemoBundle, load_bundle, public_meta, save_bundle
from neurolink.realtime.live import make_decoder, measure_latency, session_stats
from neurolink.realtime.pseudo_online import REST, Recording
from tests.test_realtime_core import PowerModel

SFREQ = 160.0
runner = CliRunner()


def tiny_eegnet() -> EEGNetClassifier:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(16, 4, 320)).astype(np.float32)
    y = np.arange(16, dtype=np.int64) % 2
    clf = EEGNetClassifier(EEGNetConfig(), TrainConfig(epochs=1, seed=0, num_threads=1))
    return clf.fit(x, y)


# ------------------------------------------------------------------------------ bundle
def test_bundle_round_trip_preserves_predictions_and_settings(tmp_path: Path) -> None:
    clf = tiny_eegnet()
    bundle = DemoBundle(
        model=clf, ch_names=("C3", "Cz", "C4", "Pz"), band=(8.0, 30.0), car=True,
        threshold=0.66, realtime=RealtimeConfig(smoothing_alpha=0.5),
        meta={"demo_subject": 7, "fold": 4, "train_subjects": [1, 2, 3]},
    )  # fmt: skip
    path = save_bundle(bundle, tmp_path / "models" / "demo.pt")
    loaded = load_bundle(path)

    x = np.random.default_rng(1).normal(size=(5, 4, 320)).astype(np.float32)
    np.testing.assert_array_equal(loaded.model.predict_proba(x), clf.predict_proba(x))
    assert loaded.ch_names == bundle.ch_names and loaded.band == (8.0, 30.0) and loaded.car
    assert loaded.threshold == 0.66
    assert loaded.realtime.smoothing_alpha == 0.5 and loaded.realtime.window_s == 2.0
    assert loaded.meta == bundle.meta
    info = public_meta(loaded)
    assert "train_subjects" not in info and info["demo_subject"] == 7
    assert info["channels"] == ["C3", "Cz", "C4", "Pz"] and info["step_s"] == 0.25


def test_bundle_rejects_unsupported_models_and_missing_files(tmp_path: Path) -> None:
    bundle = DemoBundle(make_csp_lda(), ("C3", "C4"), (8.0, 30.0), True, 0.6)
    with pytest.raises(TypeError, match="EEGNetClassifier"):
        save_bundle(bundle, tmp_path / "x.pt")
    with pytest.raises(FileNotFoundError, match="train-demo"):
        load_bundle(tmp_path / "missing.pt")


def test_make_decoder_uses_the_bundle_settings() -> None:
    bundle = DemoBundle(
        PowerModel(), ("C3", "C4"), (8.0, 30.0), False, 0.8, RealtimeConfig(step_s=0.5)
    )
    decoder = make_decoder(bundle, ("C3", "Cz", "C4", "Pz"))
    assert decoder.threshold == 0.8
    assert decoder.windower.step == 80 and decoder.windower.window == 320
    with pytest.raises(ValueError, match="channels not found"):
        make_decoder(bundle, ("Cz", "Pz"))


# ------------------------------------------------------------------------------- stats
def test_session_stats_on_a_hand_made_session() -> None:
    def msg(true: str | None, pred: str, active: bool) -> dict[str, Any]:
        return {"true": true, "pred": pred, "active": active}

    stats = session_stats([
        msg("left", "left", True), msg("left", "right", True), msg("right", "right", False),
        msg("right", "right", True), msg("rest", "left", True), msg("rest", "left", False),
        msg(None, "left", True),
    ])  # fmt: skip
    assert stats["n_predictions"] == 7
    assert stats["n_trial_windows"] == 4 and stats["n_rest_windows"] == 2
    assert stats["acc_all_smooth"] == 0.75
    assert stats["imagery_active_rate"] == 0.75
    assert stats["acc_when_active"] == pytest.approx(2 / 3)
    assert stats["rest_active_rate"] == 0.5
    assert np.isnan(session_stats([])["acc_all_smooth"])


# --------------------------------------------------------------------------------- CLI
def test_demo_is_still_a_stub() -> None:
    assert runner.invoke(app, ["demo"]).exit_code == 2


def test_commands_fail_cleanly_without_a_model_or_selection(tmp_path: Path) -> None:
    missing = str(tmp_path / "missing.pt")
    empty = str(tmp_path)
    for args in (
        ["serve", "--model-path", missing],
        ["latency", "--model-path", missing],
        ["stream", "--results-dir", empty],
        ["train-demo", "--results-dir", empty],
        ["pseudo-online", "--resummarize", "--results-dir", empty, "--cache-dir", empty],
    ):
        result = runner.invoke(app, args)
        assert result.exit_code == 1, (args, result.output)


# ----------------------------------------------------------------------- LSL end to end
@pytest.mark.lsl
def test_measure_latency_end_to_end_over_lsl_and_websocket() -> None:
    pytest.importorskip("pylsl")
    n = int(8.0 * SFREQ)
    rng = np.random.default_rng(2)
    data = rng.normal(size=(n, 4)).astype(np.float32)
    data[:, 2] += 20 * np.sin(2 * np.pi * 12 * np.arange(n) / SFREQ)  # "right": power on C4
    labels = np.array([REST] * 320 + [1] * (n - 320), dtype=np.int8)
    recording = Recording(1, 12, data, ("C3", "Cz", "C4", "Pz"), SFREQ, labels)
    bundle = DemoBundle(PowerModel(), ("C3", "C4"), (8.0, 30.0), False, 0.5)

    outcome: dict[str, Any] = {}

    def run() -> None:
        try:
            outcome["result"] = measure_latency(bundle, recording, duration_s=6.0, warmup_s=1.0)
        except Exception as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(60.0)
    assert not thread.is_alive(), "latency measurement hung"
    if isinstance(outcome.get("error"), TimeoutError):
        pytest.skip(f"LSL unavailable: {outcome['error']}")
    assert "error" not in outcome, outcome.get("error")

    result = outcome["result"]
    stages = result["stages_ms"]
    assert set(stages) == {"end_to_end", "pred_to_ws", "stream_to_pred"}
    # ~6 s of listening minus the first 2 s window: about 16 predictions, one per 0.25 s.
    assert 8 <= result["n_predictions_received"] <= 26
    assert result["n_predictions_timed"] == stages["end_to_end"]["n"] > 0
    assert result["max_gap_between_predictions_s"] == pytest.approx(0.25, abs=0.02)
    assert 0.0 <= stages["stream_to_pred"]["p50_ms"] <= stages["end_to_end"]["p50_ms"] < 60.0
    session = result["session"]
    assert session["n_trial_windows"] > 0 and session["acc_all_smooth"] > 0.9
