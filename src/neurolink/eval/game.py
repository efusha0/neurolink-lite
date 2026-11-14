"""The "reach the target" game of the UI, simulated on pseudo-online outputs.

The browser game and this module implement the same rules (``frontend/src/game.ts``; a
shared fixture keeps them in step):

* The cursor lives on a line from -1 (left wall) to +1 (right wall).
* When a left/right trial starts, the cursor is put back to the centre and a target appears
  at the wall of the cued side.
* Every prediction sets the velocity: ``(p_right - p_left) / travel_time`` when the
  prediction is above the confidence threshold, zero otherwise. ``travel_time`` is the time
  from the centre to a wall at full confidence.
* Reaching the cued wall during the trial is a hit, reaching the other wall is a wrong-side
  hit; either freezes the cursor until the trial ends. A trial that ends first is a timeout.
* During rest there is no target and the cursor keeps following the predictions.

The simulation is used to choose ``travel_time`` on training data only and to report what
the game does on held-out data, without anyone playing with the knob while watching.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from neurolink.realtime.pseudo_online import confidence

TRAVEL_TIME_GRID_S: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)
"""Candidate cursor speeds, fixed before any outcome was computed."""


@dataclass
class GameOutcome:
    trials: int = 0
    hits: int = 0
    wrong: int = 0
    timeouts: int = 0
    hit_times_s: list[float] = field(default_factory=list)
    """Seconds from the first prediction of a trial to the hit."""

    @property
    def net(self) -> int:
        """Hits minus wrong-side hits: the selection criterion."""
        return self.hits - self.wrong

    def add(self, other: GameOutcome) -> None:
        self.trials += other.trials
        self.hits += other.hits
        self.wrong += other.wrong
        self.timeouts += other.timeouts
        self.hit_times_s += other.hit_times_s

    def to_dict(self) -> dict[str, Any]:
        n = self.trials
        return {
            "trials": n,
            "hits": self.hits,
            "wrong": self.wrong,
            "timeouts": self.timeouts,
            "hit_rate": self.hits / n if n else float("nan"),
            "wrong_rate": self.wrong / n if n else float("nan"),
            "timeout_rate": self.timeouts / n if n else float("nan"),
            "median_hit_time_s": (
                float(np.median(self.hit_times_s)) if self.hit_times_s else float("nan")
            ),
        }


def simulate_game(
    p_smooth_right: NDArray[np.floating],
    label: NDArray[np.integer],
    threshold: float,
    travel_time_s: float,
    step_s: float,
) -> GameOutcome:
    """Play one run: ``p_smooth_right[k]`` and ``label[k]`` belong to the k-th prediction.

    ``label`` is 0 (left) / 1 (right) during trials and negative otherwise. Between two
    predictions the velocity is constant for ``step_s`` seconds.
    """
    if travel_time_s <= 0:
        raise ValueError("travel_time_s must be positive")
    p = np.asarray(p_smooth_right, dtype=np.float64)
    active = confidence(p) >= threshold
    out = GameOutcome()
    x = 0.0
    target: int | None = None
    resolved = False
    elapsed = 0.0
    for k in range(len(p)):
        cue = int(label[k]) if label[k] >= 0 else None
        if cue is None and target is not None:  # the trial is over
            if not resolved:
                out.timeouts += 1
            target = None
        if cue is not None and cue != target:  # a new trial starts
            if target is not None and not resolved:
                out.timeouts += 1
            target, resolved, x, elapsed = cue, False, 0.0, 0.0
            out.trials += 1
        frozen = target is not None and resolved
        # p_right - p_left, written like the browser computes it from the two probabilities.
        v = (p[k] - (1.0 - p[k])) / travel_time_s if active[k] and not frozen else 0.0
        moved = x + v * step_s
        if target is not None and not resolved and abs(moved) >= 1.0:
            # Same arithmetic as the browser: the wall is reached within this step.
            wall = 1.0 if moved > 0 else -1.0
            resolved = True
            if (wall > 0) == (target == 1):
                out.hits += 1
                out.hit_times_s.append(elapsed + (wall - x) / v)
            else:
                out.wrong += 1
        x = min(1.0, max(-1.0, moved))
        elapsed += step_s
    if target is not None and not resolved:
        out.timeouts += 1
    return out


def play_runs(
    runs: Sequence[tuple[NDArray[np.floating], NDArray[np.integer]]],
    threshold: float,
    travel_time_s: float,
    step_s: float,
) -> GameOutcome:
    total = GameOutcome()
    for p, label in runs:
        total.add(simulate_game(p, label, threshold, travel_time_s, step_s))
    return total


def select_travel_time(
    runs: Sequence[tuple[NDArray[np.floating], NDArray[np.integer]]],
    threshold: float,
    step_s: float,
    grid_s: Sequence[float] = TRAVEL_TIME_GRID_S,
) -> tuple[float, dict[str, dict[str, Any]]]:
    """Travel time with the most hits minus wrong-side hits; ties go to the slower cursor."""
    table = {f"{t:g}": play_runs(runs, threshold, t, step_s) for t in grid_s}
    best = max(grid_s, key=lambda t: (table[f"{t:g}"].net, t))
    return float(best), {k: {**v.to_dict(), "net": v.net} for k, v in table.items()}


def game_report(
    arrays: dict[str, NDArray[Any]],
    summary: dict[str, Any],
    candidate: str,
    demo_subject: int,
    train_runs: Sequence[int],
    test_run: int,
) -> dict[str, Any]:
    """Choose the cursor speed on the demo subject's training runs, then report everything.

    ``arrays`` / ``summary`` are the stored pseudo-online outputs and their summary. The
    demo subject is decoded by the fold model that held it out, with that fold's
    threshold; the held-out ``test_run`` plays no part in the choice.
    """
    step_s = float(summary["config"]["step_s"])
    fold_of = {int(s): int(k) for s, k in summary["fold_of"].items()}
    thresholds = summary["thresholds"][candidate]
    subject, run, label = arrays["subject"], arrays["run"], arrays["label"]

    def runs_of(s: int, which: Sequence[int]) -> list[tuple[NDArray[Any], NDArray[Any]]]:
        smooth = arrays[f"smooth::{candidate}/fold{fold_of[s]}"]
        out = []
        for r in which:
            mask = (subject == s) & (run == r)
            out.append((smooth[mask], label[mask]))
        return out

    demo_threshold = thresholds[fold_of[demo_subject]]
    travel_time, table = select_travel_time(
        runs_of(demo_subject, train_runs), demo_threshold, step_s
    )
    all_runs = sorted(np.unique(run).tolist())
    per_subject = {
        s: play_runs(runs_of(s, all_runs), thresholds[fold_of[s]], travel_time, step_s).to_dict()
        for s in sorted(fold_of)
    }

    def mean(key: str) -> float:
        return float(np.mean([row[key] for row in per_subject.values()]))

    return {
        "candidate": candidate,
        "rule": (
            "travel time (centre to wall at full confidence) from a fixed grid that "
            "maximises hits minus wrong-side hits on the demo subject's training runs; "
            "ties go to the slower cursor"
        ),
        "grid_s": list(TRAVEL_TIME_GRID_S),
        "demo_subject": demo_subject,
        "train_runs": list(train_runs),
        "test_run": test_run,
        "threshold": demo_threshold,
        "step_s": step_s,
        "selection_table": table,
        "travel_time_s": travel_time,
        "demo_test_run": play_runs(
            runs_of(demo_subject, [test_run]), demo_threshold, travel_time, step_s
        ).to_dict(),
        "all_subjects": {
            "n_subjects": len(per_subject),
            "hit_rate_mean": mean("hit_rate"),
            "wrong_rate_mean": mean("wrong_rate"),
            "timeout_rate_mean": mean("timeout_rate"),
            "subjects_hit_rate_above_70": int(
                sum(row["hit_rate"] >= 0.70 for row in per_subject.values())
            ),
        },
    }
