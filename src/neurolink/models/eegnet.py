"""EEGNet (Lawhern et al., 2018) in PyTorch.

Compact convolutional network for EEG: a temporal convolution learns frequency filters,
a depthwise spatial convolution learns one spatial filter per temporal filter (the CSP
analogue), and a separable convolution mixes the resulting feature maps. The default
``EEGNet-8,2`` has about 2k parameters, which is why it trains in seconds on a CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class EEGNetConfig:
    n_channels: int = 21
    n_times: int = 320
    n_classes: int = 2
    f1: int = 8
    """Temporal filters."""
    d: int = 2
    """Spatial filters per temporal filter (depth multiplier)."""
    f2: int = 16
    """Pointwise filters in the separable block (F1 * D in the paper)."""
    kernel_length: int = 80
    """Temporal kernel = half the sampling rate (160 Hz -> 80 samples = 0.5 s)."""
    dropout: float = 0.5
    pool1: int = 4
    pool2: int = 8


class EEGNet(nn.Module):
    """Input ``(batch, n_channels, n_times)`` -> logits ``(batch, n_classes)``."""

    def __init__(self, config: EEGNetConfig | None = None) -> None:
        super().__init__()
        cfg = config or EEGNetConfig()
        self.config = cfg
        f1d = cfg.f1 * cfg.d

        self.block1 = nn.Sequential(
            nn.Conv2d(1, cfg.f1, (1, cfg.kernel_length), padding=(0, cfg.kernel_length // 2),
                      bias=False),
            nn.BatchNorm2d(cfg.f1),
            # Depthwise spatial convolution over all channels.
            nn.Conv2d(cfg.f1, f1d, (cfg.n_channels, 1), groups=cfg.f1, bias=False),
            nn.BatchNorm2d(f1d),
            nn.ELU(),
            nn.AvgPool2d((1, cfg.pool1)),
            nn.Dropout(cfg.dropout),
        )  # fmt: skip
        self.block2 = nn.Sequential(
            # Separable convolution = depthwise temporal + pointwise.
            nn.Conv2d(f1d, f1d, (1, 16), padding=(0, 8), groups=f1d, bias=False),
            nn.Conv2d(f1d, cfg.f2, 1, bias=False),
            nn.BatchNorm2d(cfg.f2),
            nn.ELU(),
            nn.AvgPool2d((1, cfg.pool2)),
            nn.Dropout(cfg.dropout),
        )
        with torch.no_grad():
            n_features = self._features(torch.zeros(1, cfg.n_channels, cfg.n_times)).shape[1]
        self.classifier = nn.Linear(n_features, cfg.n_classes)

        # Max-norm constraints from the paper are applied after each optimizer step
        # by the trainer (see ``apply_max_norm``).

    def _features(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)  # (batch, 1, channels, times)
        x = self.block1(x)
        x = self.block2(x)
        return x.flatten(start_dim=1)

    def forward(self, x: Tensor) -> Tensor:
        logits: Tensor = self.classifier(self._features(x))
        return logits

    @torch.no_grad()
    def apply_max_norm(self, spatial: float = 1.0, dense: float = 0.25) -> None:
        """Clip the spatial-filter and dense weights as in the original EEGNet."""
        spatial_conv = self.block1[2]
        assert isinstance(spatial_conv, nn.Conv2d)
        w = spatial_conv.weight
        norm = w.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp(min=1e-8)
        w.mul_(torch.clamp(norm, max=spatial) / norm)
        wd = self.classifier.weight
        norm_d = wd.norm(dim=1, keepdim=True).clamp(min=1e-8)
        wd.mul_(torch.clamp(norm_d, max=dense) / norm_d)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
