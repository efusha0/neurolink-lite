"""Confusion-matrix figures for every saved result (``results/figures/*.png``)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from neurolink.config import RESULTS_DIR
from neurolink.eval.report import load_results


def plot_confusion(
    cm: NDArray[np.int64] | list[list[int]],
    class_names: list[str],
    title: str,
    path: Path,
) -> Path:
    counts = np.asarray(cm, dtype=np.int64)
    norm = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(3.6, 3.2), dpi=150)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            color = "white" if norm[i, j] > 0.5 else "black"
            label = f"{counts[i, j]}" + chr(10) + f"{100 * norm[i, j]:.0f} %"
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=9)
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_all(results_dir: Path = RESULTS_DIR) -> list[Path]:
    """One PNG per ``results/*.json``; returns the written paths."""
    paths = []
    for r in load_results(results_dir):
        acc = 100 * r["accuracy_mean"]
        title = f"{r['model']} / {r['protocol'].replace('_', '-')}: {acc:.1f} %"
        out = results_dir / "figures" / f"{r['model']}_{r['protocol']}_cm.png"
        paths.append(plot_confusion(r["confusion"], r["class_names"], title, out))
    return paths
