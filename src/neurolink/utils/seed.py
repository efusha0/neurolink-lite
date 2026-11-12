"""Deterministic seeding across random, NumPy and PyTorch."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch RNGs.

    Torch is imported lazily so the data pipeline does not pay its import cost.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover
        return
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
