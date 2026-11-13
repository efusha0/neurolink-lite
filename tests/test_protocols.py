from __future__ import annotations

import numpy as np
import pytest

from neurolink.data.cache import Dataset
from neurolink.eval.protocols import cross_subject_cv, grouped_subject_folds, within_subject_cv
from neurolink.models.csp_lda import make_csp_lda
from tests.conftest import make_synthetic_dataset


class RecordingModel:
    """Fake classifier that records which subjects it was fitted on."""

    fitted_on: list[np.ndarray] = []  # type: ignore[type-arg]

    def __init__(self) -> None:
        self.train_hash: int | None = None

    def fit(self, X, y):  # type: ignore[no-untyped-def]
        self.train_hash = hash(X.tobytes())
        RecordingModel.fitted_on.append(X)
        return self

    def predict(self, X):  # type: ignore[no-untyped-def]
        return np.zeros(len(X), dtype=np.int64)

    def predict_proba(self, X):  # type: ignore[no-untyped-def]
        return np.tile([1.0, 0.0], (len(X), 1))


def test_grouped_folds_keep_subjects_disjoint() -> None:
    subjects = np.repeat(np.arange(1, 11), 7).astype(np.int64)
    seen_test: list[int] = []
    for train, test in grouped_subject_folds(subjects, n_splits=5, seed=1):
        assert not set(subjects[train]) & set(subjects[test])
        assert len(train) + len(test) == len(subjects)
        assert np.intersect1d(train, test).size == 0
        seen_test += sorted(set(subjects[test].tolist()))
    assert sorted(seen_test) == list(range(1, 11))  # every subject tested exactly once


def test_grouped_folds_are_seeded() -> None:
    subjects = np.repeat(np.arange(1, 11), 3).astype(np.int64)
    a = [t.tolist() for _, t in grouped_subject_folds(subjects, 5, seed=3)]
    b = [t.tolist() for _, t in grouped_subject_folds(subjects, 5, seed=3)]
    c = [t.tolist() for _, t in grouped_subject_folds(subjects, 5, seed=4)]
    assert a == b
    assert a != c


def test_grouped_folds_reject_too_few_subjects() -> None:
    with pytest.raises(ValueError, match="cannot be split"):
        list(grouped_subject_folds(np.array([1, 1, 2, 2]), n_splits=5, seed=0))


def test_within_subject_fits_only_on_that_subject(synthetic_dataset: Dataset) -> None:
    ds = synthetic_dataset
    RecordingModel.fitted_on = []
    result = within_subject_cv(ds, RecordingModel, "fake", n_splits=5, seed=0)
    assert len(result.folds) == 4 * 5
    for fold in result.folds:
        assert len(fold.test_subjects) == 1
        assert fold.train_subjects == fold.test_subjects  # same single subject
        assert fold.scores.n == 6  # 30 epochs / 5 folds
    assert sorted(result.per_subject_accuracy()) == [1, 2, 3, 4]
    assert result.confusion.sum() == ds.n_epochs  # every epoch tested exactly once


def test_within_subject_accuracy_on_separable_data(synthetic_dataset: Dataset) -> None:
    result = within_subject_cv(synthetic_dataset, make_csp_lda, "csp_lda", seed=0)
    assert result.accuracies.mean() > 0.85
    assert result.confusion.shape == (2, 2)


def test_cross_subject_never_mixes_subjects(synthetic_dataset: Dataset) -> None:
    ds = synthetic_dataset
    result = cross_subject_cv(ds, RecordingModel, "fake", n_splits=4, seed=0)
    assert len(result.folds) == 4
    tested: list[int] = []
    for fold in result.folds:
        assert not set(fold.train_subjects) & set(fold.test_subjects)
        assert len(fold.train_subjects) == 3
        tested += fold.test_subjects
    assert sorted(tested) == [1, 2, 3, 4]
    assert result.confusion.sum() == ds.n_epochs


def test_cross_subject_accuracy_on_separable_data() -> None:
    ds = make_synthetic_dataset(n_subjects=6, epochs_per_subject=40)
    result = cross_subject_cv(ds, make_csp_lda, "csp_lda", n_splits=3, seed=0)
    assert result.accuracies.mean() > 0.8
    assert result.pooled_accuracy > 0.8


def test_fold_results_are_reproducible(synthetic_dataset: Dataset) -> None:
    a = within_subject_cv(synthetic_dataset, make_csp_lda, "m", seed=7)
    b = within_subject_cv(synthetic_dataset, make_csp_lda, "m", seed=7)
    np.testing.assert_array_equal(a.accuracies, b.accuracies)


class FirstSubjectOracle:
    """Predicts the true label for epochs tagged 1.0 and the wrong one for the rest."""

    def fit(self, X, y):  # type: ignore[no-untyped-def]
        return self

    def predict(self, X):  # type: ignore[no-untyped-def]
        truth = X[:, 0, 0].astype(np.int64)  # label smuggled into the signal
        right = X[:, 1, 0] > 0.5  # 1.0 only for the subject that should be decoded
        return np.where(right, truth, 1 - truth)

    def predict_proba(self, X):  # type: ignore[no-untyped-def]
        raise NotImplementedError


def test_cross_subject_per_subject_accuracy_is_not_the_fold_accuracy() -> None:
    """Regression: subjects sharing a test fold must keep their own accuracies."""
    ds = make_synthetic_dataset(n_subjects=4, epochs_per_subject=20)
    ds.X[:, 0, 0] = ds.y
    result = cross_subject_cv(ds, RecordingModel, "fake", n_splits=2, seed=0)
    # Decode exactly one subject of every test fold, so fold accuracy is 0.5 everywhere.
    good = [f.test_subjects[0] for f in result.folds]
    ds.X[:, 1, 0] = np.isin(ds.subjects, good).astype(np.float32)
    result = cross_subject_cv(ds, FirstSubjectOracle, "fake", n_splits=2, seed=0)
    assert all(len(f.test_subjects) == 2 for f in result.folds)
    assert all(f.scores.accuracy == 0.5 for f in result.folds)
    expected = {s: (1.0 if s in good else 0.0) for s in (1, 2, 3, 4)}
    assert result.per_subject_accuracy() == expected
    for fold in result.folds:
        assert sum(fold.subject_total.values()) == fold.scores.n
        assert sum(fold.subject_correct.values()) == round(fold.scores.accuracy * fold.scores.n)


def test_within_subject_per_subject_accuracy_pools_folds(synthetic_dataset: Dataset) -> None:
    result = within_subject_cv(synthetic_dataset, RecordingModel, "fake", n_splits=5, seed=0)
    # RecordingModel always predicts class 0 and classes are balanced.
    assert result.per_subject_accuracy() == {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5}
