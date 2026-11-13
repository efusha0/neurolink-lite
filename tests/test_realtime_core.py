from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from neurolink.config import RealtimeConfig
from neurolink.realtime.buffer import SlidingWindower
from neurolink.realtime.filters import CausalBandpass, design_bandpass
from neurolink.realtime.online import OnlineDecoder, OnlinePreprocessor, ProbabilitySmoother

SFREQ = 160.0
BAND = (8.0, 30.0)


def _chunks(x: np.ndarray, sizes: list[int]) -> list[np.ndarray]:  # type: ignore[type-arg]
    out, i, k = [], 0, 0
    while i < len(x):
        n = sizes[k % len(sizes)]
        out.append(x[i : i + n])
        i += n
        k += 1
    return out


# ------------------------------------------------------------------------------ filter
def test_filter_chunked_equals_one_shot() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 5)) + 50.0  # with a DC offset
    whole = CausalBandpass(BAND, SFREQ, 5).process(x)
    filt = CausalBandpass(BAND, SFREQ, 5)
    parts = [filt.process(c) for c in _chunks(x, [1, 7, 40, 333])]
    np.testing.assert_allclose(np.concatenate(parts), whole, atol=1e-9)


def test_filter_is_causal() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1000, 2))
    y = CausalBandpass(BAND, SFREQ, 2).process(x)
    x2 = x.copy()
    x2[600:] += rng.normal(size=(400, 2)) * 10  # change the future only
    y2 = CausalBandpass(BAND, SFREQ, 2).process(x2)
    np.testing.assert_array_equal(y[:600], y2[:600])
    assert not np.allclose(y[600:], y2[600:])


def test_filter_passes_band_and_rejects_outside() -> None:
    t = np.arange(int(8 * SFREQ)) / SFREQ

    def gain(freq: float) -> float:
        x = np.sin(2 * np.pi * freq * t)[:, None]
        y = CausalBandpass(BAND, SFREQ, 1).process(x)
        return float(y[len(y) // 2 :].std() / x.std())

    assert gain(15.0) > 0.95
    assert gain(2.0) < 0.05
    assert gain(60.0) < 0.05


def test_filter_has_no_dc_startup_transient() -> None:
    x = np.full((400, 3), 1000.0)
    y = CausalBandpass(BAND, SFREQ, 3).process(x)
    assert np.abs(y).max() < 1e-6


def test_filter_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="invalid band"):
        design_bandpass((30.0, 8.0), SFREQ)
    with pytest.raises(ValueError, match="expected"):
        CausalBandpass(BAND, SFREQ, 4).process(np.zeros((10, 3)))


def test_filter_matches_scipy_reference() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(500, 1))
    sos = signal.butter(4, BAND, btype="bandpass", fs=SFREQ, output="sos")
    ref = signal.sosfilt(sos, x, axis=0, zi=signal.sosfilt_zi(sos)[:, :, None] * x[0])[0]
    np.testing.assert_allclose(CausalBandpass(BAND, SFREQ, 1).process(x), ref, atol=1e-12)


# ---------------------------------------------------------------------------- windower
def test_windower_ends_and_content() -> None:
    x = np.arange(1000, dtype=np.float32)[:, None] * np.ones((1, 2), dtype=np.float32)
    ends, windows = SlidingWindower(2, window=320, step=40).push(x)
    assert ends.tolist() == list(range(320, 1001, 40))
    assert windows.shape == (len(ends), 2, 320)
    for end, w in zip(ends.tolist(), windows, strict=True):
        np.testing.assert_array_equal(w[0], np.arange(end - 320, end))


@pytest.mark.parametrize("sizes", [[1], [13], [40], [39, 41], [500], [320, 1, 1000]])
def test_windower_is_chunk_invariant(sizes: list[int]) -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(1500, 3)).astype(np.float32)
    ref_ends, ref_windows = SlidingWindower(3, 320, 40).push(x)
    w = SlidingWindower(3, 320, 40)
    parts = [w.push(c) for c in _chunks(x, sizes)]
    np.testing.assert_array_equal(np.concatenate([p[0] for p in parts]), ref_ends)
    np.testing.assert_array_equal(np.concatenate([p[1] for p in parts]), ref_windows)


def test_windower_emits_nothing_before_first_full_window() -> None:
    w = SlidingWindower(1, 320, 40)
    ends, windows = w.push(np.zeros((319, 1)))
    assert len(ends) == 0 and windows.shape == (0, 1, 320)
    ends, _ = w.push(np.zeros((1, 1)))
    assert ends.tolist() == [320]


# ------------------------------------------------------------------------ preprocessor
def test_preprocessor_car_uses_all_channels_then_picks() -> None:
    rng = np.random.default_rng(4)
    names = ["A", "B", "C", "D"]
    x = rng.normal(size=(800, 4)) * 20
    out = OnlinePreprocessor(names, ["C", "A"], BAND, SFREQ, car=True).process(x)
    car = x - x.mean(axis=1, keepdims=True)
    ref = CausalBandpass(BAND, SFREQ, 2).process(car[:, [2, 0]])
    np.testing.assert_allclose(out, ref, atol=1e-4)
    # Equivalent to the offline order: filter all channels, then CAR, then pick.
    filtered = CausalBandpass(BAND, SFREQ, 4).process(x)
    offline_order = (filtered - filtered.mean(axis=1, keepdims=True))[:, [2, 0]]
    np.testing.assert_allclose(out, offline_order, atol=1e-4)


def test_preprocessor_reports_missing_channels() -> None:
    with pytest.raises(ValueError, match="not found"):
        OnlinePreprocessor(["A", "B"], ["C3"], BAND, SFREQ)


# ---------------------------------------------------------------------------- smoother
def test_smoother_is_an_ema() -> None:
    s = ProbabilitySmoother(alpha=0.5)
    np.testing.assert_allclose(s.update(np.array([1.0, 0.0])), [1.0, 0.0])
    np.testing.assert_allclose(s.update(np.array([0.0, 1.0])), [0.5, 0.5])
    np.testing.assert_allclose(s.update(np.array([0.0, 1.0])), [0.25, 0.75])
    s.reset()
    np.testing.assert_allclose(s.update(np.array([0.2, 0.8])), [0.2, 0.8])
    with pytest.raises(ValueError, match="alpha"):
        ProbabilitySmoother(0.0)


# ----------------------------------------------------------------------------- decoder
class PowerModel:
    """Says "right" with probability given by the share of power on the second channel."""

    def fit(self, X, y):  # type: ignore[no-untyped-def]
        return self

    def predict_proba(self, X):  # type: ignore[no-untyped-def]
        power = X.astype(np.float64).var(axis=2)
        p_right = power[:, 1] / power.sum(axis=1)
        return np.stack([1 - p_right, p_right], axis=1)

    def predict(self, X):  # type: ignore[no-untyped-def]
        return self.predict_proba(X).argmax(axis=1)


def _two_state_stream(n_each: int = 1600) -> np.ndarray:  # type: ignore[type-arg]
    """4 channels; first half strong 12 Hz on channel 0, second half on channel 1."""
    rng = np.random.default_rng(5)
    t = np.arange(2 * n_each) / SFREQ
    x = rng.normal(size=(2 * n_each, 4))
    x[:n_each, 0] += 20 * np.sin(2 * np.pi * 12 * t[:n_each])
    x[n_each:, 1] += 20 * np.sin(2 * np.pi * 12 * t[n_each:])
    return x


def test_decoder_follows_the_signal_and_gates_low_confidence() -> None:
    x = _two_state_stream()
    dec = OnlineDecoder(
        PowerModel(), ["a", "b", "c", "d"], ["a", "b"], BAND, RealtimeConfig(), SFREQ,
        car=False, threshold=0.9,
    )  # fmt: skip
    preds = [p for c in _chunks(x, [40]) for p in dec.push(c)]
    assert [p.end_sample for p in preds] == list(range(320, len(x) + 1, 40))
    early = [p for p in preds if p.end_sample <= 1600]
    late = [p for p in preds if p.end_sample >= 1600 + 320 + 400]  # past window + smoothing lag
    assert all(p.pred == 0 and p.active for p in early)
    assert all(p.pred == 1 and p.active for p in late)
    mixed = [p for p in preds if 1600 < p.end_sample < 1600 + 320]
    assert any(not p.active for p in mixed)  # ambiguous windows fall below the threshold
    for p in preds:
        assert p.confidence == pytest.approx(p.probs.max())
        assert p.probs.sum() == pytest.approx(1.0)


def test_decoder_output_does_not_depend_on_chunking() -> None:
    x = _two_state_stream(800)

    def run(sizes: list[int]) -> np.ndarray:  # type: ignore[type-arg]
        dec = OnlineDecoder(PowerModel(), ["a", "b", "c", "d"], ["a", "b"], BAND, sfreq=SFREQ)
        return np.array([p.probs for c in _chunks(x, sizes) for p in dec.push(c)])

    ref = run([len(x)])
    np.testing.assert_allclose(run([1]), ref, atol=1e-5)
    np.testing.assert_allclose(run([17, 64]), ref, atol=1e-5)


def test_decoder_reset_restarts_the_stream() -> None:
    x = _two_state_stream(400)
    dec = OnlineDecoder(PowerModel(), ["a", "b", "c", "d"], ["a", "b"], BAND, sfreq=SFREQ)
    first = [p.probs for p in dec.push(x)]
    dec.reset()
    second = [p.probs for p in dec.push(x)]
    np.testing.assert_allclose(first, second)
