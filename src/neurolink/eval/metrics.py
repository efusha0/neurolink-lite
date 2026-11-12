"""Classification metrics with a fixed label set so confusion matrices always align."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from neurolink.config import CLASS_NAMES

LABELS = list(range(len(CLASS_NAMES)))


@dataclass(frozen=True)
class Scores:
    accuracy: float
    balanced_accuracy: float
    n: int
    confusion: NDArray[np.int64]
    """Rows = true class, columns = predicted class, ordered as :data:`CLASS_NAMES`."""


def score(y_true: NDArray[np.int64], y_pred: NDArray[np.int64]) -> Scores:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred length mismatch")
    return Scores(
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        n=len(y_true),
        confusion=np.asarray(confusion_matrix(y_true, y_pred, labels=LABELS), dtype=np.int64),
    )
