"""Persist protocol results as JSON and regenerate the markdown summary table."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from neurolink import __version__
from neurolink.config import CLASS_NAMES, RESULTS_DIR
from neurolink.eval.protocols import ProtocolResult

RESULTS_MD = "RESULTS.md"


def result_to_dict(result: ProtocolResult, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    acc = result.accuracies
    per_subject = result.per_subject_accuracy()
    subj = np.asarray(list(per_subject.values()), dtype=np.float64)
    cm = result.confusion
    return {
        "model": result.model,
        "protocol": result.protocol,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "version": __version__,
        "n_folds": len(result.folds),
        "n_test_epochs": int(cm.sum()),
        "n_subjects": len(per_subject),
        "accuracy_mean": float(acc.mean()),
        "accuracy_std": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
        "accuracy_median": float(np.median(acc)),
        "accuracy_min": float(acc.min()),
        "accuracy_max": float(acc.max()),
        "pooled_accuracy": result.pooled_accuracy,
        "subject_accuracy_mean": float(subj.mean()),
        "subject_accuracy_std": float(subj.std(ddof=1)) if len(subj) > 1 else 0.0,
        "subject_accuracy_median": float(np.median(subj)),
        "subject_accuracy_min": float(subj.min()),
        "subject_accuracy_max": float(subj.max()),
        "balanced_accuracy_mean": float(
            np.mean([f.scores.balanced_accuracy for f in result.folds])
        ),
        "subjects_above_70": int(sum(a >= 0.70 for a in per_subject.values())),
        "class_names": list(CLASS_NAMES),
        "confusion": cm.tolist(),
        "per_subject_accuracy": {str(s): a for s, a in per_subject.items()},
        "fit_seconds_total": float(sum(f.fit_seconds for f in result.folds)),
        "folds": [
            {
                "name": f.name,
                "accuracy": f.scores.accuracy,
                "balanced_accuracy": f.scores.balanced_accuracy,
                "n": f.scores.n,
                "test_subjects": f.test_subjects,
            }
            for f in result.folds
        ],
        **(extra or {}),
    }


def save_result(
    result: ProtocolResult,
    results_dir: Path = RESULTS_DIR,
    extra: dict[str, Any] | None = None,
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{result.model}_{result.protocol}.json"
    path.write_text(
        json.dumps(result_to_dict(result, extra), indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    write_markdown_table(results_dir)
    return path


def load_results(results_dir: Path = RESULTS_DIR) -> list[dict[str, Any]]:
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(results_dir.glob("*.json"))]
    return sorted(rows, key=lambda r: (r["protocol"], r["model"]))


def write_markdown_table(results_dir: Path = RESULTS_DIR) -> Path:
    """Rebuild ``results/RESULTS.md`` from every JSON file in the directory."""
    rows = load_results(results_dir)
    lines = [
        "# Results",
        "",
        "Binary left/right motor imagery, chance = 50 %. Every statistic except the last",
        "column is over per-subject accuracies (each subject's own test epochs, pooled",
        'across folds), so the spread shows inter-subject variability. "Fold std" is the',
        "std over folds (within-subject: subject x fold, 9 test epochs each, hence large;",
        "cross-subject: the 5 subject-grouped folds).",
        "Regenerated automatically; do not edit by hand.",
        "",
        "| Model | Protocol | Subjects | Test epochs | Accuracy (%), mean +/- subject std "
        "| Median | Min-Max | Pooled | Subjects >= 70 % | Fold std |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        head = (
            f"| {r['model']} | {r['protocol'].replace('_', '-')} | {r['n_subjects']} | "
            f"{r['n_test_epochs']} | "
        )
        if "subject_accuracy_mean" not in r:
            # Written before per-subject statistics existed; its per-subject numbers are
            # unreliable, so show nothing rather than something wrong.
            lines.append(
                head + f"re-run needed | - | - | {100 * r['pooled_accuracy']:.1f} | - | - |"
            )
            continue
        lines.append(
            head
            + f"{100 * r['subject_accuracy_mean']:.1f} +/- {100 * r['subject_accuracy_std']:.1f} | "
            f"{100 * r['subject_accuracy_median']:.1f} | "
            f"{100 * r['subject_accuracy_min']:.0f}-{100 * r['subject_accuracy_max']:.0f} | "
            f"{100 * r['pooled_accuracy']:.1f} | {r['subjects_above_70']}/{r['n_subjects']} | "
            f"{100 * r['accuracy_std']:.1f} |"
        )
    lines.append("")
    for r in rows:
        cm = r["confusion"]
        names = r["class_names"]
        lines += [
            f"## {r['model']} / {r['protocol'].replace('_', '-')}: confusion matrix",
            "",
            rf"| true \ pred | {names[0]} | {names[1]} |",
            "|---|---:|---:|",
            f"| {names[0]} | {cm[0][0]} | {cm[0][1]} |",
            f"| {names[1]} | {cm[1][0]} | {cm[1][1]} |",
            "",
        ]
    path = results_dir / RESULTS_MD
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path
