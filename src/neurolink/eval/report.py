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
    lines += pseudo_online_markdown(results_dir)
    lines += realtime_markdown(results_dir)
    lines += behavioural_markdown(results_dir)
    path = results_dir / RESULTS_MD
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path


PSEUDO_ONLINE_DIR = "pseudo_online"
PSEUDO_ONLINE_JSON = "pseudo_online.json"


def _nan_to_none(obj: Any) -> Any:
    """JSON has no NaN; metrics without data (e.g. no active window) become ``null``."""
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def save_pseudo_online(summary: dict[str, Any], results_dir: Path = RESULTS_DIR) -> Path:
    """Write ``results/pseudo_online/pseudo_online.json`` and refresh ``RESULTS.md``."""
    path = results_dir / PSEUDO_ONLINE_DIR / PSEUDO_ONLINE_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "version": __version__,
        **_nan_to_none(summary),
    }
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8", newline="\n")
    write_markdown_table(results_dir)
    return path


def _pct(cell: dict[str, Any]) -> str:
    if cell["mean"] is None:
        return "-"
    return f"{100 * cell['mean']:.1f} +/- {100 * cell['std']:.1f}"


def pseudo_online_markdown(results_dir: Path = RESULTS_DIR) -> list[str]:
    """Markdown section for the pseudo-online results; empty if they do not exist."""
    path = results_dir / PSEUDO_ONLINE_DIR / PSEUDO_ONLINE_JSON
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    cfg = data["config"]
    lines = [
        "## Pseudo-online accuracy",
        "",
        "The online pipeline run over held-out recordings: causal band-pass,"
        f" {cfg['window_s']:g} s windows every {cfg['step_s']:g} s that ignore the cue timing,",
        f"EMA smoothing (alpha = {cfg['smoothing_alpha']:g}). Mean +/- std across subjects, in %.",
        "",
        "* Candidate = scope / model / filter used for the *training* epochs. `cross`: model",
        "  of the fold that held the subject out, all three runs. `subject`: trained on runs",
        f"  4 and 8, tested on run {cfg['test_run']}. `zero_phase` training is mismatched with",
        "  the causal online filter; `causal` training matches it.",
        "* Offline = cue-aligned 0.5-2.5 s epochs of the same test data, filtered like the",
        "  training data (the Phase 2-3 style number).",
        "* Inside = windows entirely inside a left/right trial. All = every window ending",
        "  during a trial, including windows that still contain rest.",
        f"* Threshold = {cfg['rest_quantile']:g} quantile of smoothed confidence over rest",
        "  windows of the *training* data. Rest active / Trial active = share of rest / trial",
        "  windows above it (cursor moves); Acc. active = accuracy of the trial windows above it.",
        "",
        "| Candidate | Subjects | Offline | Inside, raw | Inside, smoothed | All, smoothed "
        "| Rest active | Trial active | Acc. active |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in data["rows"].items():
        lines.append(
            f"| {name} | {row['n_subjects']} | {_pct(row['offline_acc'])} | "
            f"{_pct(row['acc_inside_raw'])} | {_pct(row['acc_inside_smooth'])} | "
            f"{_pct(row['acc_all_smooth'])} | {_pct(row['rest_active_rate'])} | "
            f"{_pct(row['imagery_active_rate'])} | {_pct(row['acc_when_active'])} |"
        )
    lines.append("")
    return lines


REALTIME_DIR = "realtime"
DEMO_MODEL_JSON = "demo_model.json"
LATENCY_JSON = "latency.json"
BEHAVIOURAL_JSON = "behavioural_latency.json"


def save_realtime(name: str, payload: dict[str, Any], results_dir: Path = RESULTS_DIR) -> Path:
    """Write ``results/realtime/<name>`` (demo model, latency) and refresh ``RESULTS.md``."""
    path = results_dir / REALTIME_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"version": __version__, **_nan_to_none(payload)}
    path.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8", newline="\n")
    write_markdown_table(results_dir)
    return path


def load_realtime(name: str, results_dir: Path = RESULTS_DIR) -> dict[str, Any] | None:
    path = results_dir / REALTIME_DIR / name
    if not path.exists():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _pct1(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.1f}"


def realtime_markdown(results_dir: Path = RESULTS_DIR) -> list[str]:
    """Markdown sections for the demo model and the latency measurements, if present."""
    lines: list[str] = []
    demo = load_realtime(DEMO_MODEL_JSON, results_dir)
    if demo is not None:
        subject = f"S{demo['demo_subject']:03d}"
        lines += [
            "## Demo model",
            "",
            f"`{demo['candidate']}`, the fold model that holds {subject} out: trained on"
            f" {demo['n_train_subjects']} other subjects ({demo['n_train_epochs']} epochs),"
            f" confidence threshold {demo['threshold']:.3f} from their rest windows.",
        ]
        selection = demo.get("selection")
        if selection is not None:
            ties = ""
            if len(selection.get("tied_subjects", [])) > 1:
                scores = ", ".join(
                    f"S{int(s):03d} {100 * v:.1f} %"
                    for s, v in selection["tie_break_scores"].items()
                )
                ties = f" Tie-break among equal scores: {scores}."
            lines += [
                f"Demo subject rule: {selection['selection_rule']}."
                f" Selection score {100 * selection['selection_score']:.1f} %.{ties}",
                "Run 12 was not used to select the model or the subject.",
            ]
        rows = [
            (f"{subject}, run 12 (replayed in the demo)", demo.get("pseudo_online_run12")),
            (f"{subject}, all three runs", demo.get("pseudo_online_all_runs")),
        ]
        if any(row is not None for _name, row in rows):
            lines += [
                "",
                "| Data | Offline | Inside, smoothed | All, smoothed | Rest active "
                "| Trial active | Acc. active |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
            for name, row in rows:
                if row is None:
                    continue
                lines.append(
                    f"| {name} | {_pct1(row['offline_acc'])} | {_pct1(row['acc_inside_smooth'])} "
                    f"| {_pct1(row['acc_all_smooth'])} | {_pct1(row['rest_active_rate'])} "
                    f"| {_pct1(row['imagery_active_rate'])} | {_pct1(row['acc_when_active'])} |"
                )
            lines += [
                "",
                "This subject is a best case; the average over all subjects is the"
                f" `{demo['candidate']}` row of the pseudo-online table above.",
            ]
        lines.append("")

    latency_files = sorted((results_dir / REALTIME_DIR).glob("latency*.json"))
    if latency_files:
        lines += [
            "## Latency",
            "",
            "Time from the acquisition (LSL timestamp) of the newest sample of a window to:",
            "the smoothed prediction (`stream_to_pred`), the hand-over to the WebSocket",
            "(`pred_to_ws`, measured from the prediction) and the arrival at a WebSocket",
            "client on the same machine (`end_to_end`). Browser rendering is not included.",
            "The 2 s analysis window is not part of these numbers.",
            "",
            "| Measurement | Stage | Predictions | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) "
            "| Max (ms) |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for path in latency_files:
            data = json.loads(path.read_text(encoding="utf-8"))
            name = (
                f"S{data['subject']:03d} run {data['run']}, {data['duration_s']:g} s,"
                f" {data['chunk_ms']:g} ms chunks"
            )
            for stage in ("stream_to_pred", "pred_to_ws", "end_to_end"):
                row = data["stages_ms"].get(stage)
                if row is None:
                    continue
                cells = " | ".join(
                    "-" if row[k] is None else f"{row[k]:.1f}"
                    for k in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms")
                )
                lines.append(f"| {name} | {stage} | {row['n']:.0f} | {cells} |")
        lines.append("")
    return lines


def _sec(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _reaction_rows(name: str, block: dict[str, Any]) -> list[str]:
    labels = {
        "correct_active": "right class, above threshold",
        "correct_active_change": "same, output not already right at the cue",
        "correct": "right class, threshold ignored",
    }
    rows = []
    for key, label in labels.items():
        st = block[key]
        q, by = st["quantiles_reached_s"], st["reached_by"]
        rows.append(
            f"| {name} | {label} | {st['n_trials']} | {_pct1(st['reached_share'])} "
            f"| {_sec(st['median_all_trials_s'])} | {_sec(q['p25'])} | {_sec(q['p50'])} "
            f"| {_sec(q['p75'])} | {_sec(q['p90'])} | {_pct1(by['1'])} | {_pct1(by['2'])} "
            f"| {_pct1(by['3'])} |"
        )
    return rows


def behavioural_markdown(results_dir: Path = RESULTS_DIR) -> list[str]:
    """Markdown section for the reaction-to-cue measurement, if present."""
    data = load_realtime(BEHAVIOURAL_JSON, results_dir)
    if data is None:
        return []
    everyone = data["all_subjects"]
    lines = [
        "## Behavioural latency (reaction to the cue)",
        "",
        f"`{data['candidate']}`, pseudo-online outputs of {data['n_subjects']} held-out subjects.",
        "Time from the onset of a left/right trial (4.1 s long) to the end of the first",
        f"{data['window_s']:g} s window whose smoothed prediction is the trial's class.",
        "Reached = share of trials in which that happens before the trial ends.",
        "Median (all) counts trials",
        "never reached as infinitely late (- when at least half are never reached); p25 to",
        "p90 are over the reached trials only. Within 1/2/3 s = share of all trials reached",
        "by then. In"
        f" {_pct1(everyone['already_correct_active_share'])} % of all trials the last window"
        " before the cue already showed",
        "the coming class above the threshold; that is carry-over from rest, not a reaction,",
        'and the "not already right" rows leave those trials out.',
        "",
        "| Data | Condition | Trials | Reached (%) | Median, all (s) | p25 (s) "
        "| Median, reached (s) | p75 (s) | p90 (s) | Within 1 s (%) | Within 2 s (%) "
        "| Within 3 s (%) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *_reaction_rows(f"All {data['n_subjects']} subjects", everyone),
    ]
    demo = data.get("demo_subject")
    if demo is not None:
        subject = f"S{demo['subject']:03d}"
        lines += _reaction_rows(f"{subject}, all runs", demo["all_runs"])
        run_keys = [k for k in demo if k.startswith("run_")]
        for key in run_keys:
            lines += _reaction_rows(f"{subject}, run {key.removeprefix('run_')}", demo[key])
    lines += [
        "",
        f"First above-threshold output of a trial, all subjects: right class in"
        f" {_pct1(everyone['first_command_right_share'])} % of trials, wrong class in"
        f" {_pct1(everyone['first_command_wrong_share'])} %, none at all in"
        f" {_pct1(everyone['never_active_share'])} %.",
        "",
    ]
    return lines
