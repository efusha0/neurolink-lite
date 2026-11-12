"""Minimal classifier interface shared by the CSP+LDA baseline and EEGNet.

Every model is created *fresh* for each cross-validation fold through a
:data:`ModelFactory`, so nothing fitted on one fold can leak into another.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class Classifier(Protocol):
    """Anything with scikit-learn style ``fit`` / ``predict`` / ``predict_proba``."""

    def fit(self, X: NDArray[np.float32], y: NDArray[np.int64]) -> Classifier: ...

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.int64]: ...

    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float64]: ...


ModelFactory = Callable[[], Classifier]
"""Returns an unfitted classifier. Called once per fold."""
