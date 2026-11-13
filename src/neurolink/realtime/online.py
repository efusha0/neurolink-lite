"""Transport-independent core of the online classifier.

``raw chunk -> CAR -> channel subset -> causal band-pass -> sliding window -> model ->
probability smoothing -> confidence gate``. Nothing here knows about LSL or WebSockets:
the live server and the pseudo-online evaluation both drive these same classes, so the
accuracy measured offline is the accuracy of the code that runs in the demo.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neurolink.config import SFREQ, RealtimeConfig
from neurolink.models.base import Classifier
from neurolink.realtime.buffer import SlidingWindower
from neurolink.realtime.filters import CausalBandpass

PREDICT_BATCH = 512


class OnlinePreprocessor:
    """Raw multi-channel chunk (microvolts) -> filtered chunk of the model's channels.

    CAR is computed over *all* incoming channels before the subset is taken, as in the
    offline pipeline. CAR, channel selection and the linear filter commute, so filtering
    only the selected channels is equivalent to the offline order and cheaper.
    """

    def __init__(
        self,
        in_ch_names: Sequence[str],
        out_ch_names: Sequence[str],
        band: tuple[float, float],
        sfreq: float = SFREQ,
        car: bool = True,
        order: int = 4,
    ) -> None:
        missing = [ch for ch in out_ch_names if ch not in in_ch_names]
        if missing:
            raise ValueError(f"channels not found in stream: {missing}")
        self.n_in = len(in_ch_names)
        self.picks = np.asarray([list(in_ch_names).index(ch) for ch in out_ch_names])
        self.car = car
        self.filter = CausalBandpass(band, sfreq, n_channels=len(self.picks), order=order)

    def reset(self) -> None:
        self.filter.reset()

    def process(self, chunk: NDArray[np.floating]) -> NDArray[np.float32]:
        x = np.asarray(chunk, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_in:
            raise ValueError(f"expected (n_samples, {self.n_in}), got {x.shape}")
        if self.car:
            x = x - x.mean(axis=1, keepdims=True)
        return self.filter.process(x[:, self.picks]).astype(np.float32)


class ProbabilitySmoother:
    """Exponential moving average over successive class-probability vectors."""

    def __init__(self, alpha: float) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._state: NDArray[np.float64] | None = None

    def reset(self) -> None:
        self._state = None

    def update(self, probs: NDArray[np.floating]) -> NDArray[np.float64]:
        p = np.asarray(probs, dtype=np.float64)
        if self._state is None:
            self._state = p
        else:
            self._state = self.alpha * p + (1.0 - self.alpha) * self._state
        return self._state


def predict_proba_batched(
    model: Classifier, windows: NDArray[np.float32], batch: int = PREDICT_BATCH
) -> NDArray[np.float64]:
    """``model.predict_proba`` in bounded-memory batches; ``(0, 2)`` for no windows."""
    if len(windows) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    parts = [model.predict_proba(windows[i : i + batch]) for i in range(0, len(windows), batch)]
    return np.concatenate(parts).astype(np.float64)


@dataclass(frozen=True)
class Prediction:
    end_sample: int
    """Stream index one past the last sample of the window."""
    raw_probs: NDArray[np.float64]
    probs: NDArray[np.float64]
    """Smoothed class probabilities, ordered as :data:`neurolink.config.CLASS_NAMES`."""
    pred: int
    confidence: float
    active: bool
    """False when confidence is below the threshold: the cursor should not move."""


class OnlineDecoder:
    """Feed raw chunks with :meth:`push`; get zero or more :class:`Prediction` back."""

    def __init__(
        self,
        model: Classifier,
        in_ch_names: Sequence[str],
        model_ch_names: Sequence[str],
        band: tuple[float, float],
        config: RealtimeConfig | None = None,
        sfreq: float = SFREQ,
        car: bool = True,
        threshold: float = 0.5,
    ) -> None:
        cfg = config or RealtimeConfig()
        self.model = model
        self.threshold = threshold
        self.preprocessor = OnlinePreprocessor(
            in_ch_names, model_ch_names, band, sfreq, car=car, order=cfg.filter_order
        )
        self.windower = SlidingWindower(
            len(model_ch_names), cfg.window_samples(sfreq), cfg.step_samples(sfreq)
        )
        self.smoother = ProbabilitySmoother(cfg.smoothing_alpha)

    def reset(self) -> None:
        self.preprocessor.reset()
        self.windower.reset()
        self.smoother.reset()

    def push(self, chunk: NDArray[np.floating]) -> list[Prediction]:
        ends, windows = self.windower.push(self.preprocessor.process(chunk))
        raw = predict_proba_batched(self.model, windows)
        out: list[Prediction] = []
        for end, p in zip(ends.tolist(), raw, strict=True):
            smooth = self.smoother.update(p)
            confidence = float(smooth.max())
            out.append(
                Prediction(
                    end_sample=int(end),
                    raw_probs=p,
                    probs=smooth.copy(),
                    pred=int(smooth.argmax()),
                    confidence=confidence,
                    active=confidence >= self.threshold,
                )
            )
        return out
