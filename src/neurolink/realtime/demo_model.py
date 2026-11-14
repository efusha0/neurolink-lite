"""Train, save and load the model that drives the live demo.

The demo model is the cross-subject EEGNet trained on causally filtered epochs: exactly
the model of the pseudo-online experiment's fold in which the demo subject is held out
(same subject-grouped folds, same seed). The demo subject therefore contributes nothing
to it: not to the weights, not to the normalization statistics, not to the confidence
threshold, which is a quantile of the smoothed confidence over *rest* windows of the
training subjects' continuous recordings.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neurolink.config import (
    ALL_SUBJECTS,
    CACHE_DIR,
    DATA_DIR,
    MOTOR_CORTEX_CHANNELS,
    RAW_DIR,
    PreprocessConfig,
    RealtimeConfig,
)
from neurolink.data.cache import Dataset
from neurolink.data.pipeline import build_or_load
from neurolink.eval.online_eval import RecordingLoader, subject_folds
from neurolink.eval.protocols import grouped_subject_folds
from neurolink.models.base import Classifier
from neurolink.models.eegnet import EEGNet, EEGNetConfig
from neurolink.models.trainer import EEGNetClassifier, TrainConfig
from neurolink.realtime.online import predict_proba_batched
from neurolink.realtime.pseudo_online import (
    load_recordings,
    make_windows,
    rest_threshold,
    smooth_sequence,
)

log = logging.getLogger(__name__)

MODELS_DIR = DATA_DIR / "models"
DEMO_MODEL_PATH = MODELS_DIR / "demo_model.pt"
DEMO_CANDIDATE = "cross/eegnet/causal"
"""Name of the demo model's row in the pseudo-online results."""


@dataclass
class DemoBundle:
    """Everything the online classifier needs, plus where the model came from."""

    model: Classifier
    ch_names: tuple[str, ...]
    band: tuple[float, float]
    car: bool
    threshold: float
    realtime: RealtimeConfig = field(default_factory=RealtimeConfig)
    meta: dict[str, Any] = field(default_factory=dict)
    """JSON-serializable provenance: demo subject, fold, training subjects, seed, ..."""


def train_demo_model(
    subject: int,
    seed: int,
    n_splits: int = 5,
    epochs: int = 30,
    threads: int = 4,
    cache_dir: Path = CACHE_DIR,
    raw_dir: Path = RAW_DIR,
    realtime: RealtimeConfig | None = None,
    subjects: tuple[int, ...] | None = None,
    dataset: Dataset | None = None,
    recording_loader: RecordingLoader | None = None,
    maker: Callable[[], Classifier] | None = None,
) -> DemoBundle:
    """Fit the fold model that holds ``subject`` out and derive its confidence threshold.

    ``dataset``, ``recording_loader`` and ``maker`` exist for tests; by default the
    causally filtered motor-channel dataset of all subjects and EEGNet are used.
    """
    rt = realtime or RealtimeConfig()
    if dataset is None:
        cfg = PreprocessConfig(
            channels=MOTOR_CORTEX_CHANNELS, subjects=subjects or ALL_SUBJECTS, causal=True
        )
        dataset, _ = build_or_load(cfg, cache_dir=cache_dir, raw_dir=raw_dir)
    if not dataset.config.causal:
        raise ValueError("the demo model must be trained on causally filtered epochs")
    fold_of = subject_folds(dataset.subjects, n_splits, seed)
    if subject not in fold_of:
        raise ValueError(f"S{subject:03d} is not a valid subject of this dataset")
    fold = fold_of[subject]
    train_idx, test_idx = list(grouped_subject_folds(dataset.subjects, n_splits, seed))[fold]
    train_subjects = np.unique(dataset.subjects[train_idx]).tolist()
    assert subject not in train_subjects and subject in dataset.subjects[test_idx]

    if maker is None:
        train_cfg = TrainConfig(epochs=epochs, seed=seed, num_threads=threads)
        model: Classifier = EEGNetClassifier(EEGNetConfig(), train_cfg)
    else:
        model = maker()
    log.info(
        "training on %d subjects (%d epochs); S%03d and %d others held out",
        len(train_subjects), len(train_idx), subject,
        len(np.unique(dataset.subjects[test_idx])) - 1,
    )  # fmt: skip
    model.fit(dataset.X[train_idx], dataset.y[train_idx])

    # Confidence threshold: rest windows of the *training* subjects, run through the
    # online pipeline, exactly as in the pseudo-online experiment.
    load = recording_loader or (lambda s: load_recordings(s, dataset.config.runs, raw_dir))
    p_parts, label_parts = [], []
    for s in train_subjects:
        recs = load(s)
        if recs is None:
            raise RuntimeError(f"S{s:03d} is in the dataset but has no valid recording")
        for rec in recs:
            win = make_windows(rec, dataset.ch_names, dataset.config.band, rt, dataset.config.car)
            raw = predict_proba_batched(model, win.windows)
            p_parts.append(smooth_sequence(raw, rt.smoothing_alpha)[:, 1])
            label_parts.append(win.label)
    threshold = rest_threshold(
        np.concatenate(p_parts), np.concatenate(label_parts), rt.rest_quantile
    )

    held_out = dataset.subjects == subject
    meta = {
        "candidate": DEMO_CANDIDATE,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "demo_subject": subject,
        "fold": fold,
        "n_splits": n_splits,
        "seed": seed,
        "epochs": epochs,
        "n_train_subjects": len(train_subjects),
        "n_train_epochs": len(train_idx),
        "train_subjects": train_subjects,
        "threshold": threshold,
        # Cue-aligned accuracy on all 45 epochs of the held-out demo subject (a check that
        # this is the model of the pseudo-online experiment, not a selection criterion).
        "demo_subject_offline_acc": float(
            (np.asarray(model.predict(dataset.X[held_out])) == dataset.y[held_out]).mean()
        ),
    }
    return DemoBundle(
        model=model,
        ch_names=tuple(dataset.ch_names),
        band=(float(dataset.config.band[0]), float(dataset.config.band[1])),
        car=dataset.config.car,
        threshold=threshold,
        realtime=rt,
        meta=meta,
    )


_REALTIME_FIELDS = ("window_s", "step_s", "smoothing_alpha", "filter_order", "rest_quantile")


def save_bundle(bundle: DemoBundle, path: Path = DEMO_MODEL_PATH) -> Path:
    """Write the bundle as a tensors-and-primitives file (loadable with ``weights_only``)."""
    model = bundle.model
    if not isinstance(model, EEGNetClassifier) or model.model is None:
        raise TypeError("only a fitted EEGNetClassifier can be saved")
    assert model.mean_ is not None and model.std_ is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": 1,
        "weights": model.model.state_dict(),
        "model_config": asdict(model.model.config),
        "mean": torch.from_numpy(model.mean_),
        "std": torch.from_numpy(model.std_),
        "ch_names": list(bundle.ch_names),
        "band": list(bundle.band),
        "car": bundle.car,
        "threshold": bundle.threshold,
        "realtime": {k: getattr(bundle.realtime, k) for k in _REALTIME_FIELDS},
        "meta": bundle.meta,
    }
    torch.save(payload, path)
    return path


def load_bundle(path: Path = DEMO_MODEL_PATH) -> DemoBundle:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `neurolink train-demo` first")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != 1:
        raise ValueError(f"unsupported demo model format in {path}")
    config = EEGNetConfig(**payload["model_config"])
    net = EEGNet(config)
    net.load_state_dict(payload["weights"])
    net.eval()
    clf = EEGNetClassifier(model_config=config)
    clf.model = net
    clf.mean_ = payload["mean"].numpy().astype(np.float32)
    clf.std_ = payload["std"].numpy().astype(np.float32)
    defaults = RealtimeConfig()
    realtime = RealtimeConfig(
        **payload["realtime"], ws_host=defaults.ws_host, ws_port=defaults.ws_port
    )
    return DemoBundle(
        model=clf,
        ch_names=tuple(payload["ch_names"]),
        band=(float(payload["band"][0]), float(payload["band"][1])),
        car=bool(payload["car"]),
        threshold=float(payload["threshold"]),
        realtime=realtime,
        meta=dict(payload["meta"]),
    )


def public_meta(bundle: DemoBundle) -> dict[str, Any]:
    """Provenance without the long subject list (for ``/info`` and result files)."""
    out = {k: v for k, v in bundle.meta.items() if k != "train_subjects"}
    out.update(
        channels=list(bundle.ch_names),
        band=list(bundle.band),
        car=bundle.car,
        **{k: getattr(bundle.realtime, k) for k in _REALTIME_FIELDS},
    )
    return out


__all__ = [
    "DEMO_CANDIDATE",
    "DEMO_MODEL_PATH",
    "DemoBundle",
    "load_bundle",
    "public_meta",
    "save_bundle",
    "train_demo_model",
]
