from __future__ import annotations

import numpy as np

from neurolink.data.cache import Dataset
from neurolink.models.csp_lda import CSPLDAConfig, make_csp_lda
from tests.conftest import make_synthetic_dataset


def test_pipeline_steps_and_unfitted() -> None:
    model = make_csp_lda(CSPLDAConfig(n_components=4))
    assert [name for name, _ in model.steps] == ["csp", "scaler", "lda"]
    assert model.named_steps["csp"].n_components == 4
    assert not hasattr(model.named_steps["lda"], "coef_")


def test_learns_separable_classes(synthetic_dataset: Dataset) -> None:
    ds = synthetic_dataset
    train = ds.subjects != 4
    model = make_csp_lda().fit(ds.X[train], ds.y[train])
    acc = (model.predict(ds.X[~train]) == ds.y[~train]).mean()
    assert acc > 0.9
    proba = model.predict_proba(ds.X[~train])
    assert proba.shape == ((~train).sum(), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_chance_level_on_unseparable_data() -> None:
    ds = make_synthetic_dataset(n_subjects=2, epochs_per_subject=60, separable=False)
    train = ds.subjects == 1
    model = make_csp_lda().fit(ds.X[train], ds.y[train])
    acc = (model.predict(ds.X[~train]) == ds.y[~train]).mean()
    assert 0.25 < acc < 0.75


def test_fit_does_not_touch_test_data(synthetic_dataset: Dataset) -> None:
    """Predictions for a test epoch must not depend on what else is in the test set."""
    ds = synthetic_dataset
    train = ds.subjects != 4
    model = make_csp_lda().fit(ds.X[train], ds.y[train])
    test_x = ds.X[~train]
    single = model.predict(test_x[:1])
    batch = model.predict(test_x)
    assert single[0] == batch[0]
