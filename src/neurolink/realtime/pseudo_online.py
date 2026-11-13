"""Pseudo-online evaluation: run the online pipeline over recorded data without LSL.

A continuous recording is pushed through the same preprocessing and windowing classes as
the live system (causal filter, windows every 0.25 s that ignore where the cues are), and
every window is scored against the annotation that is active when the window ends.
This is the honest counterpart of the offline, cue-aligned accuracy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
from numpy.typing import NDArray

from neurolink.config import MOTOR_IMAGERY_RUNS, RAW_DIR, RealtimeConfig
from neurolink.data.download import download_subject
from neurolink.data.loader import load_subject
from neurolink.realtime.buffer import SlidingWindower
from neurolink.realtime.online import OnlinePreprocessor, ProbabilitySmoother

REST = -1
"""Window/sample label for the rest annotation (T0)."""
OUTSIDE = -2
"""Sample not covered by any annotation."""

_ANNOTATION_LABEL = {"T0": REST, "T1": 0, "T2": 1}


@dataclass
class Recording:
    """One continuous run: raw microvolts plus a label for every sample."""

    subject: int
    run: int
    data: NDArray[np.float32]
    """``(n_samples, n_channels)``, unfiltered, microvolts."""
    ch_names: tuple[str, ...]
    sfreq: float
    sample_labels: NDArray[np.int8]
    """``(n_samples,)``: 0 = left, 1 = right, :data:`REST`, or :data:`OUTSIDE`."""


def sample_labels_from_annotations(raw: mne.io.BaseRaw) -> NDArray[np.int8]:
    """Label every sample with the annotation covering it (T0/T1/T2)."""
    sfreq = float(raw.info["sfreq"])
    labels = np.full(raw.n_times, OUTSIDE, dtype=np.int8)
    ann = raw.annotations
    for onset, duration, desc in zip(ann.onset, ann.duration, ann.description, strict=True):
        if desc not in _ANNOTATION_LABEL:
            continue
        start = round(float(onset) * sfreq)
        stop = min(raw.n_times, start + round(float(duration) * sfreq))
        labels[start:stop] = _ANNOTATION_LABEL[str(desc)]
    return labels


def recording_from_raw(raw: mne.io.BaseRaw, subject: int, run: int) -> Recording:
    return Recording(
        subject=subject,
        run=run,
        data=np.asarray(raw.get_data(units="uV").T, dtype=np.float32),
        ch_names=tuple(raw.ch_names),
        sfreq=float(raw.info["sfreq"]),
        sample_labels=sample_labels_from_annotations(raw),
    )


def load_recordings(
    subject: int,
    runs: Sequence[int] = MOTOR_IMAGERY_RUNS,
    raw_dir: Path = RAW_DIR,
) -> list[Recording] | None:
    """Continuous, unfiltered runs of one subject; ``None`` if the subject fails validation."""
    paths = download_subject(subject, runs, raw_dir)
    raws, _reports = load_subject(subject, paths, runs)
    if raws is None:
        return None
    return [recording_from_raw(raw, subject, run) for raw, run in zip(raws, runs, strict=True)]


@dataclass
class RunWindows:
    """All sliding windows of one run, as the online classifier would see them."""

    subject: int
    run: int
    ends: NDArray[np.int64]
    windows: NDArray[np.float32]
    """``(n_windows, n_channels, window)``, preprocessed."""
    label: NDArray[np.int8]
    """Label of the last sample of each window."""
    inside: NDArray[np.bool_]
    """True when the whole window lies inside one left/right trial."""


def window_labels(
    sample_labels: NDArray[np.int8], ends: NDArray[np.int64], window: int
) -> tuple[NDArray[np.int8], NDArray[np.bool_]]:
    """Label of each window (its last sample) and whether it is entirely one trial."""
    label = sample_labels[ends - 1]
    inside = np.zeros(len(ends), dtype=np.bool_)
    for i, end in enumerate(ends.tolist()):
        seg = sample_labels[end - window : end]
        inside[i] = seg[-1] >= 0 and bool(np.all(seg == seg[-1]))
    return label, inside


def make_windows(
    rec: Recording,
    model_ch_names: Sequence[str],
    band: tuple[float, float],
    config: RealtimeConfig | None = None,
    car: bool = True,
    chunk: int | None = None,
) -> RunWindows:
    """Push a recording through the online preprocessor and windower.

    ``chunk`` is the number of samples fed per call (``None`` = the whole run at once);
    the result does not depend on it, which the tests verify.
    """
    cfg = config or RealtimeConfig()
    window = cfg.window_samples(rec.sfreq)
    pre = OnlinePreprocessor(
        rec.ch_names, model_ch_names, band, rec.sfreq, car=car, order=cfg.filter_order
    )
    windower = SlidingWindower(len(model_ch_names), window, cfg.step_samples(rec.sfreq))
    size = chunk or len(rec.data)
    ends_parts, win_parts = [], []
    for start in range(0, len(rec.data), size):
        ends, wins = windower.push(pre.process(rec.data[start : start + size]))
        ends_parts.append(ends)
        win_parts.append(wins)
    all_ends = np.concatenate(ends_parts)
    label, inside = window_labels(rec.sample_labels, all_ends, window)
    return RunWindows(rec.subject, rec.run, all_ends, np.concatenate(win_parts), label, inside)


def smooth_sequence(raw_probs: NDArray[np.float64], alpha: float) -> NDArray[np.float64]:
    """Apply the online EMA smoother to a run's successive probability vectors."""
    smoother = ProbabilitySmoother(alpha)
    out = np.empty_like(raw_probs, dtype=np.float64)
    for i, p in enumerate(raw_probs):
        out[i] = smoother.update(p)
    return out


def confidence(p_right: NDArray[np.floating]) -> NDArray[np.float64]:
    p = np.asarray(p_right, dtype=np.float64)
    return np.maximum(p, 1.0 - p)


def rest_threshold(
    p_smooth_right: NDArray[np.floating], label: NDArray[np.int8], quantile: float
) -> float:
    """Confidence below which the cursor stays still: a quantile of *rest* confidence.

    Must be computed on training data only. Falls back to 0.5 (never gate) without rest.
    """
    rest = label == REST
    if not rest.any():
        return 0.5
    return float(np.quantile(confidence(p_smooth_right[rest]), quantile))


def _mean(mask: NDArray[np.bool_], values: NDArray[np.bool_]) -> float:
    return float(values[mask].mean()) if mask.any() else float("nan")


def window_metrics(
    p_raw_right: NDArray[np.floating],
    p_smooth_right: NDArray[np.floating],
    label: NDArray[np.int8],
    inside: NDArray[np.bool_],
    threshold: float,
) -> dict[str, float]:
    """Pseudo-online scores of one subject (or any set of windows).

    * ``acc_inside_*``: windows entirely inside a left/right trial.
    * ``acc_all_*``: every window ending during a trial, including those that still
      contain the preceding rest.
    * ``rest_active_rate`` / ``imagery_active_rate``: share of rest / trial windows whose
      smoothed confidence passes ``threshold`` (the cursor moves).
    * ``acc_when_active``: accuracy over trial windows that pass the threshold.
    * ``rest_right_rate``: share of rest windows decoded as "right" (bias check).
    """
    imagery = label >= 0
    rest = label == REST
    ok_raw = (np.asarray(p_raw_right) >= 0.5).astype(np.int8) == label
    pred_smooth = (np.asarray(p_smooth_right) >= 0.5).astype(np.int8)
    ok_smooth = pred_smooth == label
    active = confidence(p_smooth_right) >= threshold
    return {
        "acc_inside_raw": _mean(imagery & inside, ok_raw),
        "acc_inside_smooth": _mean(imagery & inside, ok_smooth),
        "acc_all_raw": _mean(imagery, ok_raw),
        "acc_all_smooth": _mean(imagery, ok_smooth),
        "rest_active_rate": _mean(rest, active),
        "imagery_active_rate": _mean(imagery, active),
        "acc_when_active": _mean(imagery & active, ok_smooth),
        "rest_right_rate": _mean(rest, pred_smooth == 1),
        "n_inside": float((imagery & inside).sum()),
        "n_all": float(imagery.sum()),
        "n_rest": float(rest.sum()),
    }
