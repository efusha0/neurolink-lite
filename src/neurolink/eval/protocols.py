"""Evaluation protocols.

Both protocols share one contract: a fresh model is created per fold and fitted on the
training indices only. Nothing is estimated from test data, and the dataset given here
has only had fixed, data-independent preprocessing (see ``neurolink.data.preprocess``).

* :func:`within_subject_cv`: stratified 5-fold over the epochs of each subject separately.
  Measures how well a model works for a user it was calibrated on.
* :func:`cross_subject_cv`: subject-grouped K-fold. Every subject's epochs are either all in
  the training folds or all in the test fold; this is enforced, not assumed.
  Measures generalization to unseen users.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedKFold

from neurolink.data.cache import Dataset
from neurolink.eval.metrics import Scores, score
from neurolink.models.base import ModelFactory

log = logging.getLogger(__name__)

Index = NDArray[np.int64]


@dataclass
class FoldResult:
    name: str
    scores: Scores
    train_subjects: list[int]
    test_subjects: list[int]
    fit_seconds: float
    subject_correct: dict[int, int] = field(default_factory=dict)
    """Correct test predictions per test subject."""
    subject_total: dict[int, int] = field(default_factory=dict)
    """Test epochs per test subject."""


@dataclass
class ProtocolResult:
    protocol: str
    model: str
    folds: list[FoldResult] = field(default_factory=list)

    @property
    def accuracies(self) -> NDArray[np.float64]:
        return np.asarray([f.scores.accuracy for f in self.folds], dtype=np.float64)

    @property
    def confusion(self) -> NDArray[np.int64]:
        total = np.sum([f.scores.confusion for f in self.folds], axis=0)
        return np.asarray(total, dtype=np.int64)

    @property
    def pooled_accuracy(self) -> float:
        """Accuracy over all test predictions pooled (weights folds by size)."""
        cm = self.confusion
        return float(np.trace(cm) / cm.sum())

    def per_subject_accuracy(self) -> dict[int, float]:
        """Accuracy of every subject over its own test epochs, pooled across folds.

        Computed from per-subject prediction counts, never from fold accuracies: in the
        cross-subject protocol a fold holds many subjects whose accuracies differ widely.
        """
        correct: dict[int, int] = {}
        total: dict[int, int] = {}
        for f in self.folds:
            for s, n in f.subject_total.items():
                correct[s] = correct.get(s, 0) + f.subject_correct[s]
                total[s] = total.get(s, 0) + n
        return {s: correct[s] / total[s] for s in sorted(total)}


def grouped_subject_folds(
    subjects: NDArray[np.int64], n_splits: int, seed: int
) -> Iterator[tuple[Index, Index]]:
    """Yield ``(train_idx, test_idx)`` with disjoint subject sets in every split.

    Subjects are shuffled with ``seed`` and dealt into ``n_splits`` groups of (almost)
    equal size. Implemented by hand so the shuffling is reproducible across scikit-learn
    versions.
    """
    unique = np.unique(subjects)
    if len(unique) < n_splits:
        raise ValueError(f"{len(unique)} subjects cannot be split into {n_splits} folds")
    rng = np.random.default_rng(seed)
    order = rng.permutation(unique)
    for test_subjects in np.array_split(order, n_splits):
        test_mask = np.isin(subjects, test_subjects)
        train_idx = np.flatnonzero(~test_mask).astype(np.int64)
        test_idx = np.flatnonzero(test_mask).astype(np.int64)
        _assert_disjoint_subjects(subjects, train_idx, test_idx)
        yield train_idx, test_idx


def _assert_disjoint_subjects(subjects: NDArray[np.int64], train: Index, test: Index) -> None:
    overlap = set(subjects[train].tolist()) & set(subjects[test].tolist())
    if overlap:
        raise RuntimeError(f"subject leakage between train and test: {sorted(overlap)}")


def _run_fold(
    name: str,
    dataset: Dataset,
    make_model: ModelFactory,
    train_idx: Index,
    test_idx: Index,
) -> FoldResult:
    if np.intersect1d(train_idx, test_idx).size:
        raise RuntimeError(f"{name}: train and test indices overlap")
    model = make_model()
    t0 = time.perf_counter()
    model.fit(dataset.X[train_idx], dataset.y[train_idx])
    fit_s = time.perf_counter() - t0
    y_pred = np.asarray(model.predict(dataset.X[test_idx]), dtype=np.int64)
    y_true = dataset.y[test_idx]
    test_subjects = dataset.subjects[test_idx]
    hits = y_pred == y_true
    unique_test = sorted(set(test_subjects.tolist()))
    return FoldResult(
        name=name,
        scores=score(y_true, y_pred),
        train_subjects=sorted(set(dataset.subjects[train_idx].tolist())),
        test_subjects=unique_test,
        fit_seconds=fit_s,
        subject_correct={s: int(hits[test_subjects == s].sum()) for s in unique_test},
        subject_total={s: int((test_subjects == s).sum()) for s in unique_test},
    )


def within_subject_cv(
    dataset: Dataset,
    make_model: ModelFactory,
    model_name: str,
    n_splits: int = 5,
    seed: int = 0,
) -> ProtocolResult:
    """Stratified K-fold inside each subject; one :class:`FoldResult` per subject and fold."""
    result = ProtocolResult(protocol="within_subject", model=model_name)
    for subject in dataset.unique_subjects.tolist():
        idx = np.flatnonzero(dataset.subjects == subject).astype(np.int64)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for k, (tr, te) in enumerate(skf.split(idx, dataset.y[idx])):
            fold = _run_fold(f"S{subject:03d}/fold{k}", dataset, make_model, idx[tr], idx[te])
            result.folds.append(fold)
        acc = np.mean([f.scores.accuracy for f in result.folds[-n_splits:]])
        log.info("S%03d within-subject accuracy %.3f", subject, acc)
    return result


def cross_subject_cv(
    dataset: Dataset,
    make_model: ModelFactory,
    model_name: str,
    n_splits: int = 5,
    seed: int = 0,
) -> ProtocolResult:
    """Subject-grouped K-fold; one :class:`FoldResult` per fold."""
    result = ProtocolResult(protocol="cross_subject", model=model_name)
    splits = grouped_subject_folds(dataset.subjects, n_splits, seed)
    for k, (tr, te) in enumerate(splits):
        fold = _run_fold(f"fold{k}", dataset, make_model, tr, te)
        result.folds.append(fold)
        log.info(
            "fold %d: %d train / %d test subjects, accuracy %.3f (fit %.1f s)",
            k, len(fold.train_subjects), len(fold.test_subjects),
            fold.scores.accuracy, fold.fit_seconds,
        )  # fmt: skip
    return result
