from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from neurolink.data.cache import Dataset
from neurolink.eval.plots import plot_all, plot_confusion
from neurolink.eval.protocols import cross_subject_cv, within_subject_cv
from neurolink.eval.report import save_result
from neurolink.models.eegnet import EEGNet, EEGNetConfig
from neurolink.models.trainer import EEGNetClassifier, TrainConfig
from tests.conftest import make_synthetic_dataset

FAST = TrainConfig(epochs=30, batch_size=16, lr=5e-3, seed=0)


def test_eegnet_forward_shape_and_size() -> None:
    cfg = EEGNetConfig(n_channels=21, n_times=320)
    net = EEGNet(cfg)
    out = net(torch.zeros(4, 21, 320))
    assert out.shape == (4, 2)
    assert 1_000 < net.n_parameters() < 5_000  # EEGNet-8,2 is tiny on purpose


def test_eegnet_handles_other_shapes() -> None:
    net = EEGNet(EEGNetConfig(n_channels=8, n_times=256))
    assert net(torch.zeros(2, 8, 256)).shape == (2, 2)


def test_max_norm_constraint() -> None:
    net = EEGNet(EEGNetConfig(n_channels=8, n_times=128))
    with torch.no_grad():
        net.block1[2].weight.mul_(100.0)
        net.classifier.weight.mul_(100.0)
    net.apply_max_norm()
    spatial_norms = net.block1[2].weight.flatten(1).norm(dim=1)
    assert torch.all(spatial_norms <= 1.0 + 1e-5)
    assert torch.all(net.classifier.weight.norm(dim=1) <= 0.25 + 1e-5)


def test_classifier_learns_separable_data(synthetic_dataset: Dataset) -> None:
    ds = synthetic_dataset
    train = ds.subjects != 4
    clf = EEGNetClassifier(train_config=FAST).fit(ds.X[train], ds.y[train])
    acc = (clf.predict(ds.X[~train]) == ds.y[~train]).mean()
    assert acc > 0.85
    proba = clf.predict_proba(ds.X[~train])
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert clf.history_[-1] < clf.history_[0]  # loss went down


def test_normalization_uses_train_statistics_only(synthetic_dataset: Dataset) -> None:
    ds = synthetic_dataset
    train = ds.subjects != 4
    clf = EEGNetClassifier(train_config=TrainConfig(epochs=1)).fit(ds.X[train], ds.y[train])
    assert clf.mean_ is not None and clf.std_ is not None
    np.testing.assert_allclose(
        clf.mean_.ravel(), ds.X[train].mean(axis=(0, 2)), rtol=1e-5, atol=1e-5
    )
    all_mean = ds.X.mean(axis=(0, 2))
    assert not np.allclose(clf.mean_.ravel(), all_mean, atol=1e-4)


def test_training_is_deterministic(synthetic_dataset: Dataset) -> None:
    ds = synthetic_dataset
    cfg = TrainConfig(epochs=5, seed=3)
    a = EEGNetClassifier(train_config=cfg).fit(ds.X, ds.y).predict_proba(ds.X[:8])
    b = EEGNetClassifier(train_config=cfg).fit(ds.X, ds.y).predict_proba(ds.X[:8])
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_state_dict_roundtrip(synthetic_dataset: Dataset) -> None:
    ds = synthetic_dataset
    clf = EEGNetClassifier(train_config=TrainConfig(epochs=3)).fit(ds.X, ds.y)
    restored = EEGNetClassifier.from_state_dict(clf.state_dict())
    np.testing.assert_allclose(restored.predict_proba(ds.X[:5]), clf.predict_proba(ds.X[:5]))


def test_protocols_with_eegnet(tmp_path: Path) -> None:
    ds = make_synthetic_dataset(n_subjects=4, epochs_per_subject=20)

    def factory() -> EEGNetClassifier:
        return EEGNetClassifier(train_config=TrainConfig(epochs=3, seed=0))

    w = within_subject_cv(ds, factory, "eegnet", n_splits=2, seed=0)
    c = cross_subject_cv(ds, factory, "eegnet", n_splits=2, seed=0)
    assert len(w.folds) == 8 and len(c.folds) == 2
    assert w.confusion.sum() == ds.n_epochs
    save_result(w, tmp_path)
    save_result(c, tmp_path)
    pngs = plot_all(tmp_path)
    assert len(pngs) == 2 and all(p.exists() and p.stat().st_size > 1000 for p in pngs)


def test_plot_confusion_writes_png(tmp_path: Path) -> None:
    out = plot_confusion(np.array([[30, 10], [5, 35]]), ["left", "right"], "t", tmp_path / "cm.png")
    assert out.exists()
