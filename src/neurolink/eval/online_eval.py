"""Pseudo-online experiment: how well do the models work the way the demo uses them?

For every candidate model the same question is asked twice: offline (cue-aligned epochs,
as in Phases 2-3) and pseudo-online (causal filter, free-running sliding windows).

Candidates = scope x model x training filter:

* scope ``cross``: one model per subject-grouped fold, tested on the fold's held-out
  subjects (all three runs). Same folds and seed as the offline cross-subject protocol.
* scope ``subject``: one model per subject, trained on runs 4 and 8, tested on run 12.
* model: EEGNet or CSP+LDA, 21 motor channels.
* training filter ``zero_phase`` (the offline pipeline, mismatched with the causal online
  filter) or ``causal`` (training epochs filtered exactly like the online stream).

Nothing is fitted on test data: models, normalization and the confidence threshold come
from training subjects (cross) or training runs (subject) only. The demo-subject selection
scores use runs 4 and 8 only, never the replayed run 12.

Demo subject: highest selection score; ties between cross-subject candidates are broken
by the pseudo-online accuracy on runs 4 and 8 (many more windows than the 30 epochs
behind the selection score), then by the lowest subject id.
"""

from __future__ import annotations

import itertools
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedKFold

from neurolink.config import (
    CACHE_DIR,
    MOTOR_CORTEX_CHANNELS,
    RAW_DIR,
    PreprocessConfig,
    RealtimeConfig,
)
from neurolink.data.cache import Dataset
from neurolink.data.pipeline import build_or_load
from neurolink.eval.protocols import grouped_subject_folds
from neurolink.models.base import Classifier
from neurolink.realtime.online import predict_proba_batched
from neurolink.realtime.pseudo_online import (
    Recording,
    load_recordings,
    make_windows,
    rest_threshold,
    smooth_sequence,
    window_metrics,
)

log = logging.getLogger(__name__)

FILTERS = ("zero_phase", "causal")
KINDS = ("eegnet", "csp_lda")
TRAIN_RUNS = (4, 8)
TEST_RUN = 12
METRICS = (
    "offline_acc",
    "acc_inside_raw",
    "acc_inside_smooth",
    "acc_all_raw",
    "acc_all_smooth",
    "rest_active_rate",
    "imagery_active_rate",
    "acc_when_active",
    "rest_right_rate",
)

Maker = Callable[[str], Classifier]
"""``maker(scope)`` returns an unfitted model for scope ``"cross"`` or ``"subject"``."""
RecordingLoader = Callable[[int], list[Recording] | None]


@dataclass
class WindowStore:
    """Window-level outputs of every model, enough to recompute all metrics."""

    subject: list[NDArray[np.int64]] = field(default_factory=list)
    run: list[NDArray[np.int64]] = field(default_factory=list)
    label: list[NDArray[np.int8]] = field(default_factory=list)
    inside: list[NDArray[np.bool_]] = field(default_factory=list)
    p_raw: dict[str, list[NDArray[np.float32]]] = field(default_factory=dict)
    p_smooth: dict[str, list[NDArray[np.float32]]] = field(default_factory=dict)

    def arrays(self) -> dict[str, NDArray[Any]]:
        out: dict[str, NDArray[Any]] = {
            "subject": np.concatenate(self.subject),
            "run": np.concatenate(self.run),
            "label": np.concatenate(self.label),
            "inside": np.concatenate(self.inside),
        }
        for key, parts in self.p_raw.items():
            out[f"raw::{key}"] = np.concatenate(parts)
        for key, parts in self.p_smooth.items():
            out[f"smooth::{key}"] = np.concatenate(parts)
        return out


def default_makers(
    seed: int, epochs_cross: int, epochs_subject: int, threads: int
) -> dict[str, Maker]:
    """EEGNet and CSP+LDA with the Phase 2-3 settings."""
    from neurolink.models.csp_lda import make_csp_lda
    from neurolink.models.eegnet import EEGNetConfig
    from neurolink.models.trainer import EEGNetClassifier, TrainConfig

    def eegnet(scope: str) -> Classifier:
        epochs = epochs_cross if scope == "cross" else epochs_subject
        return EEGNetClassifier(
            EEGNetConfig(), TrainConfig(epochs=epochs, seed=seed, num_threads=threads)
        )

    def csp_lda(scope: str) -> Classifier:
        model: Classifier = make_csp_lda()
        return model

    return {"eegnet": eegnet, "csp_lda": csp_lda}


def subject_folds(subjects: NDArray[np.int64], n_splits: int, seed: int) -> dict[int, int]:
    """Subject id -> index of the cross-subject fold in which that subject is held out."""
    fold_of: dict[int, int] = {}
    for k, (_tr, te) in enumerate(grouped_subject_folds(subjects, n_splits, seed)):
        for s in np.unique(subjects[te]).tolist():
            fold_of[int(s)] = k
    return fold_of


def _accuracy(model: Classifier, x: NDArray[np.float32], y: NDArray[np.int64]) -> float:
    if len(y) == 0:
        return float("nan")
    return float((np.asarray(model.predict(x)) == y).mean())


def _cv_accuracy(
    make: Callable[[], Classifier], x: NDArray[np.float32], y: NDArray[np.int64], seed: int
) -> float:
    """Pooled 5-fold accuracy on training-run epochs (used only to pick a demo subject)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    hits = 0
    for tr, te in skf.split(x, y):
        model = make().fit(x[tr], y[tr])
        hits += int((np.asarray(model.predict(x[te])) == y[te]).sum())
    return hits / len(y)


def run_experiment(
    subjects: Sequence[int],
    seed: int,
    makers: dict[str, Maker],
    n_splits: int = 5,
    cache_dir: Path = CACHE_DIR,
    raw_dir: Path = RAW_DIR,
    realtime: RealtimeConfig | None = None,
    datasets: dict[str, Dataset] | None = None,
    recording_loader: RecordingLoader | None = None,
    selection_cv_kinds: Sequence[str] = KINDS,
) -> tuple[dict[str, Any], WindowStore]:
    """Train every candidate, score it offline and pseudo-online; see module docstring.

    ``datasets`` and ``recording_loader`` exist for tests; by default the motor-channel
    caches are built/loaded and recordings come from ``raw_dir``.
    """
    rt = realtime or RealtimeConfig()
    if datasets is None:
        datasets = {}
        for filt in FILTERS:
            cfg = PreprocessConfig(
                channels=MOTOR_CORTEX_CHANNELS, subjects=tuple(subjects), causal=filt == "causal"
            )
            datasets[filt], _ = build_or_load(cfg, cache_dir=cache_dir, raw_dir=raw_dir)
    ref = datasets["causal"]
    for ds in datasets.values():
        # Both filters must describe the same epochs so results are comparable row by row.
        if not (
            np.array_equal(ds.subjects, ref.subjects)
            and np.array_equal(ds.y, ref.y)
            and np.array_equal(ds.runs, ref.runs)
        ):
            raise RuntimeError("zero-phase and causal datasets disagree on epochs")
    load = recording_loader or (lambda s: load_recordings(s, ref.config.runs, raw_dir))
    valid = ref.unique_subjects.tolist()
    kinds = [k for k in KINDS if k in makers]

    # ---------------------------------------------------------------- cross-subject models
    folds = list(grouped_subject_folds(ref.subjects, n_splits, seed))
    fold_of = subject_folds(ref.subjects, n_splits, seed)
    cross_models: dict[tuple[str, str], list[Classifier]] = {}
    for kind in kinds:
        for filt in FILTERS:
            ds = datasets[filt]
            t0 = time.perf_counter()
            cross_models[(kind, filt)] = [
                makers[kind]("cross").fit(ds.X[tr], ds.y[tr]) for tr, _te in folds
            ]
            log.info(
                "cross %s/%s: %d folds in %.0f s", kind, filt, n_splits, time.perf_counter() - t0
            )

    # ------------------------------------------------------------------- per-subject pass
    store = WindowStore()
    offline: dict[str, dict[int, float]] = {}
    offline_test_run: dict[str, dict[int, float]] = {}
    selection: dict[str, dict[int, float]] = {}

    def put(table: dict[str, dict[int, float]], key: str, subject: int, value: float) -> None:
        table.setdefault(key, {})[subject] = value

    for n_done, subject in enumerate(valid, start=1):
        t0 = time.perf_counter()
        recs = load(subject)
        if recs is None:
            raise RuntimeError(f"S{subject:03d} is in the dataset but has no valid recording")
        run_windows = [
            make_windows(r, ref.ch_names, ref.config.band, rt, car=ref.config.car) for r in recs
        ]
        windows = np.concatenate([w.windows for w in run_windows])
        bounds = np.cumsum([0] + [len(w.ends) for w in run_windows])
        store.subject.append(np.full(len(windows), subject, dtype=np.int64))
        store.run.append(np.concatenate([np.full(len(w.ends), w.run) for w in run_windows]))
        store.label.append(np.concatenate([w.label for w in run_windows]))
        store.inside.append(np.concatenate([w.inside for w in run_windows]))

        def record(
            key: str,
            model: Classifier,
            windows: NDArray[np.float32] = windows,
            bounds: NDArray[np.int64] = bounds,
        ) -> None:
            raw = predict_proba_batched(model, windows)
            smooth = np.concatenate(
                [
                    smooth_sequence(raw[a:b], rt.smoothing_alpha)
                    for a, b in itertools.pairwise(bounds.tolist())
                ]
            )
            store.p_raw.setdefault(key, []).append(raw[:, 1].astype(np.float32))
            store.p_smooth.setdefault(key, []).append(smooth[:, 1].astype(np.float32))

        for (kind, filt), models in cross_models.items():
            ds = datasets[filt]
            mine = ds.subjects == subject
            train_runs = mine & np.isin(ds.runs, TRAIN_RUNS)
            test_run = mine & (ds.runs == TEST_RUN)
            held_out = models[fold_of[subject]]
            key = f"cross/{kind}/{filt}"
            put(offline, key, subject, _accuracy(held_out, ds.X[mine], ds.y[mine]))
            put(offline_test_run, key, subject, _accuracy(held_out, ds.X[test_run], ds.y[test_run]))
            put(selection, key, subject, _accuracy(held_out, ds.X[train_runs], ds.y[train_runs]))
            for k, model in enumerate(models):
                record(f"{key}/fold{k}", model)

        for kind in kinds:
            for filt in FILTERS:
                ds = datasets[filt]
                mine = ds.subjects == subject
                train_runs = mine & np.isin(ds.runs, TRAIN_RUNS)
                test_run = mine & (ds.runs == TEST_RUN)
                x_train, y_train = ds.X[train_runs], ds.y[train_runs]
                key = f"subject/{kind}/{filt}"
                model = makers[kind]("subject").fit(x_train, y_train)
                acc = _accuracy(model, ds.X[test_run], ds.y[test_run])
                put(offline, key, subject, acc)
                put(offline_test_run, key, subject, acc)
                record(key, model)
                if filt == "causal" and kind in selection_cv_kinds:
                    make = lambda kind=kind: makers[kind]("subject")  # noqa: E731
                    put(selection, key, subject, _cv_accuracy(make, x_train, y_train, seed))
        log.info(
            "S%03d (%d/%d) done in %.1f s", subject, n_done, len(valid), time.perf_counter() - t0
        )

    summary = summarize(
        store.arrays(), fold_of, offline, offline_test_run, selection, rt.rest_quantile
    )
    summary["config"] = {
        "seed": seed,
        "n_splits": n_splits,
        "n_subjects": len(valid),
        "channels": "motor",
        "band": list(ref.config.band),
        "window_s": rt.window_s,
        "step_s": rt.step_s,
        "smoothing_alpha": rt.smoothing_alpha,
        "filter_order": rt.filter_order,
        "rest_quantile": rt.rest_quantile,
        "train_runs": list(TRAIN_RUNS),
        "test_run": TEST_RUN,
    }
    return summary, store


def _aggregate(per_subject: dict[int, dict[str, float]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_subjects": len(per_subject)}
    for m in METRICS:
        v = np.asarray([row[m] for row in per_subject.values()], dtype=np.float64)
        v = v[~np.isnan(v)]
        out[m] = {
            "mean": float(v.mean()) if len(v) else float("nan"),
            "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        }
    out["per_subject"] = {str(s): row for s, row in sorted(per_subject.items())}
    return out


def summarize(
    arrays: dict[str, NDArray[Any]],
    fold_of: dict[int, int],
    offline: dict[str, dict[int, float]],
    offline_test_run: dict[str, dict[int, float]],
    selection: dict[str, dict[int, float]],
    rest_quantile: float,
) -> dict[str, Any]:
    """Thresholds from training data, metrics on test data, aggregated over subjects."""
    subject, run = arrays["subject"], arrays["run"]
    label, inside = arrays["label"], arrays["inside"]
    subjects = sorted(fold_of)
    n_folds = max(fold_of.values()) + 1
    candidates = sorted(
        {k.split("::")[1].split("/fold")[0] for k in arrays if k.startswith("raw::")}
    )
    is_train_run = np.isin(run, TRAIN_RUNS)
    is_test_run = run == TEST_RUN

    rows: dict[str, Any] = {}
    thresholds: dict[str, Any] = {}
    tie_break: dict[str, dict[int, float]] = {}
    for cand in candidates:
        scope = cand.split("/")[0]
        if scope == "cross":
            all_runs: dict[int, dict[str, float]] = {}
            test_run: dict[int, dict[str, float]] = {}
            thresholds[cand] = []
            for k in range(n_folds):
                raw, smooth = arrays[f"raw::{cand}/fold{k}"], arrays[f"smooth::{cand}/fold{k}"]
                test_subjects = [s for s in subjects if fold_of[s] == k]
                train = ~np.isin(subject, test_subjects)
                thr = rest_threshold(smooth[train], label[train], rest_quantile)
                thresholds[cand].append(thr)
                for s in test_subjects:
                    for table, mask, off in (
                        (all_runs, subject == s, offline),
                        (test_run, (subject == s) & is_test_run, offline_test_run),
                    ):
                        row = window_metrics(
                            raw[mask], smooth[mask], label[mask], inside[mask], thr
                        )
                        row["offline_acc"] = off[cand][s]
                        row["threshold"] = thr
                        table[s] = row
                    mask = (subject == s) & is_train_run
                    tie_break.setdefault(cand, {})[s] = window_metrics(
                        raw[mask], smooth[mask], label[mask], inside[mask], thr
                    )["acc_inside_smooth"]
            rows[cand] = _aggregate(all_runs)
            rows[f"{cand} (run {TEST_RUN} only)"] = _aggregate(test_run)
        else:
            raw, smooth = arrays[f"raw::{cand}"], arrays[f"smooth::{cand}"]
            per: dict[int, dict[str, float]] = {}
            for s in subjects:
                train = (subject == s) & is_train_run
                test = (subject == s) & is_test_run
                thr = rest_threshold(smooth[train], label[train], rest_quantile)
                row = window_metrics(raw[test], smooth[test], label[test], inside[test], thr)
                row["offline_acc"] = offline[cand][s]
                row["threshold"] = thr
                per[s] = row
            rows[cand] = _aggregate(per)

    demo: dict[str, Any] = {}
    for cand, scores in sorted(selection.items()):
        is_cross = cand.startswith("cross")
        top = max(scores.values())
        tied = sorted(s for s, v in scores.items() if np.isclose(v, top))
        # The pseudo-online tie-break needs a model that has not seen runs 4+8 of the
        # subject, which only holds for cross-subject models; per-subject models were
        # trained on those runs, so their ties fall through to the lowest subject id.
        extra = tie_break.get(cand, {}) if is_cross else {}
        best = max(tied, key=lambda s: (extra.get(s, 0.0), -s))
        outcome_key = f"{cand} (run {TEST_RUN} only)" if is_cross else cand
        demo[cand] = {
            "selection_rule": (
                "held-out fold model, offline accuracy on runs 4+8; ties: pseudo-online "
                "accuracy (inside, smoothed) on runs 4+8, then lowest subject id"
                if is_cross
                else "5-fold CV on runs 4+8; ties: lowest subject id"
            ),
            "subject": best,
            "selection_score": scores[best],
            "tied_subjects": tied,
            "tie_break_scores": {str(s): extra[s] for s in tied if s in extra},
            "run12_result": rows[outcome_key]["per_subject"][str(best)],
            "selection_scores": {str(s): v for s, v in sorted(scores.items())},
        }
    return {
        "rows": rows,
        "thresholds": thresholds,
        "demo_selection": demo,
        "fold_of": {str(s): k for s, k in sorted(fold_of.items())},
    }


def resummarize(previous: dict[str, Any], arrays: dict[str, NDArray[Any]]) -> dict[str, Any]:
    """Recompute the summary from stored window outputs without retraining any model.

    ``previous`` is an earlier summary (it carries the offline accuracies and selection
    scores, which need the models) and ``arrays`` the matching :class:`WindowStore` dump.
    Used when only the aggregation changes, e.g. the demo-subject tie-break rule.
    """
    rows, config = previous["rows"], previous["config"]
    test_suffix = f" (run {TEST_RUN} only)"

    def column(row_key: str) -> dict[int, float]:
        per = rows[row_key]["per_subject"]
        return {int(s): float("nan") if r["offline_acc"] is None else r["offline_acc"]
                for s, r in per.items()}  # fmt: skip

    cands = [k for k in rows if not k.endswith(test_suffix)]
    offline = {c: column(c) for c in cands}
    offline_test_run = {c: column(c + test_suffix if c.startswith("cross") else c) for c in cands}
    selection = {
        c: {int(s): float(v) for s, v in d["selection_scores"].items()}
        for c, d in previous["demo_selection"].items()
    }
    if "fold_of" in previous:
        fold_of = {int(s): int(k) for s, k in previous["fold_of"].items()}
    else:
        valid = np.unique(arrays["subject"])
        fold_of = subject_folds(valid, config["n_splits"], config["seed"])
    summary = summarize(
        arrays, fold_of, offline, offline_test_run, selection, config["rest_quantile"]
    )
    summary["config"] = config
    return summary
