"""Common Spatial Patterns + Linear Discriminant Analysis baseline.

The whole chain is a scikit-learn ``Pipeline`` so that ``fit`` on a training fold is the
only place where data-dependent parameters (CSP filters, feature scaling, LDA weights)
are estimated. ``predict`` on the test fold only applies them.
"""

from __future__ import annotations

from dataclasses import dataclass

from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class CSPLDAConfig:
    n_components: int = 6
    """Number of spatial filters (the top n/2 and bottom n/2 eigenvectors)."""
    reg: str | None = "ledoit_wolf"
    """Covariance shrinkage; helps with 64 channels and ~36 training trials per fold."""
    shrinkage: str | float | None = "auto"
    """LDA shrinkage (Ledoit-Wolf) via the ``lsqr`` solver."""


def make_csp_lda(config: CSPLDAConfig | None = None) -> Pipeline:
    """Build an unfitted CSP -> StandardScaler -> LDA pipeline."""
    cfg = config or CSPLDAConfig()
    csp = CSP(
        n_components=cfg.n_components,
        reg=cfg.reg,
        log=True,
        norm_trace=False,
        transform_into="average_power",
    )
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=cfg.shrinkage)
    return Pipeline([("csp", csp), ("scaler", StandardScaler()), ("lda", lda)])
