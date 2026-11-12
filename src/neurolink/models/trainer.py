"""Scikit-learn style wrapper that trains an :class:`EEGNet` on a training fold.

Everything data-dependent happens inside :meth:`EEGNetClassifier.fit`:

* per-channel z-score statistics are computed from the training epochs only,
* the network is initialised from ``seed`` and trained for a fixed number of epochs
  (no early stopping, so no validation split is carved out of tiny within-subject folds),
* max-norm weight constraints are applied after every optimizer step.

``predict`` / ``predict_proba`` only apply the stored statistics and weights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from neurolink.models.eegnet import EEGNet, EEGNetConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 0
    num_threads: int = 0
    """0 = leave PyTorch's default thread count."""


class EEGNetClassifier:
    """Implements :class:`neurolink.models.base.Classifier` around :class:`EEGNet`."""

    def __init__(
        self,
        model_config: EEGNetConfig | None = None,
        train_config: TrainConfig | None = None,
    ) -> None:
        self.model_config = model_config or EEGNetConfig()
        self.train_config = train_config or TrainConfig()
        self.model: EEGNet | None = None
        self.mean_: NDArray[np.float32] | None = None
        self.std_: NDArray[np.float32] | None = None
        self.history_: list[float] = []

    # ------------------------------------------------------------------ fitting
    def fit(self, X: NDArray[np.float32], y: NDArray[np.int64]) -> EEGNetClassifier:
        cfg = self.train_config
        if cfg.num_threads > 0:
            torch.set_num_threads(cfg.num_threads)
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)

        # Normalization statistics from the training fold only.
        self.mean_ = X.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        self.std_ = (X.std(axis=(0, 2), keepdims=True) + 1e-6).astype(np.float32)
        xt = torch.from_numpy(self._normalize(X))
        yt = torch.from_numpy(np.asarray(y, dtype=np.int64))

        n_ch, n_t = X.shape[1], X.shape[2]
        model_cfg = replace(self.model_config, n_channels=n_ch, n_times=n_t)
        model = EEGNet(model_cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        loss_fn = nn.CrossEntropyLoss()

        n = len(xt)
        batch = min(cfg.batch_size, n)
        self.history_ = []
        model.train()
        for _epoch in range(cfg.epochs):
            order = rng.permutation(n)
            total = 0.0
            for start in range(0, n, batch):
                idx = torch.from_numpy(order[start : start + batch])
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(xt[idx]), yt[idx])
                loss.backward()
                optimizer.step()
                model.apply_max_norm()
                total += float(loss.detach()) * len(idx)
            self.history_.append(total / n)
        model.eval()
        self.model = model
        return self

    # --------------------------------------------------------------- inference
    def _normalize(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        assert self.mean_ is not None and self.std_ is not None, "call fit first"
        return ((X - self.mean_) / self.std_).astype(np.float32)

    @torch.no_grad()
    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float64]:
        assert self.model is not None, "call fit first"
        self.model.eval()
        logits = self.model(torch.from_numpy(self._normalize(X)))
        return torch.softmax(logits, dim=1).numpy().astype(np.float64)

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.int64]:
        return np.asarray(self.predict_proba(X).argmax(axis=1), dtype=np.int64)

    # ------------------------------------------------------------- persistence
    def state_dict(self) -> dict[str, object]:
        assert self.model is not None and self.mean_ is not None and self.std_ is not None
        return {
            "model": self.model.state_dict(),
            "model_config": self.model.config,
            "mean": self.mean_,
            "std": self.std_,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> EEGNetClassifier:
        model_config = state["model_config"]
        assert isinstance(model_config, EEGNetConfig)
        clf = cls(model_config=model_config)
        model = EEGNet(model_config)
        model.load_state_dict(state["model"])  # type: ignore[arg-type]
        model.eval()
        clf.model = model
        clf.mean_ = np.asarray(state["mean"], dtype=np.float32)
        clf.std_ = np.asarray(state["std"], dtype=np.float32)
        return clf
