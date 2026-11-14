from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neurolink.eval.game import (
    TRAVEL_TIME_GRID_S,
    GameOutcome,
    game_report,
    play_runs,
    select_travel_time,
    simulate_game,
)

REST = -1
STEP = 0.25
FIXTURE = Path(__file__).resolve().parents[1] / "frontend" / "src" / "fixtures" / "game_cases.json"


def run(p: list[float], label: list[int], threshold: float = 0.6, travel: float = 1.0):  # type: ignore[no-untyped-def]
    return simulate_game(np.array(p), np.array(label), threshold, travel, STEP)


def test_confident_correct_output_hits_after_the_travel_time() -> None:
    # p_right = 1 -> velocity 1 / travel_time: the wall is reached after exactly 1 s.
    out = run([0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5], [REST, 1, 1, 1, 1, 1, REST])
    assert (out.trials, out.hits, out.wrong, out.timeouts) == (1, 1, 0, 0)
    assert out.hit_times_s == [pytest.approx(1.0)]


def test_confident_wrong_output_is_a_wrong_side_hit() -> None:
    out = run([0.0] * 6, [1] * 6)
    assert (out.trials, out.hits, out.wrong, out.timeouts) == (1, 0, 1, 0)
    assert out.net == -1


def test_below_threshold_output_does_not_move_the_cursor() -> None:
    out = run([0.59] * 8 + [0.5], [1] * 8 + [REST])  # confidence 0.59 < 0.6
    assert (out.hits, out.wrong, out.timeouts) == (0, 0, 1)


def test_slow_cursor_times_out_and_velocity_scales_with_confidence() -> None:
    # p = 0.8 -> velocity 0.6 / travel. 8 steps of 0.25 s cover 1.2 at travel 1 s, 0.6 at 2 s.
    p, label = [0.8] * 8 + [0.5], [0] * 8 + [REST]
    assert run([1 - v for v in p], label, travel=1.0).hits == 1
    slow = run([1 - v for v in p], label, travel=2.0)
    assert (slow.hits, slow.timeouts) == (0, 1)


def test_cursor_returns_to_the_centre_at_every_cue_and_freezes_after_a_hit() -> None:
    # Rest drives the cursor to the left wall; the "right" trial must still start from 0
    # and need a full second, not benefit or suffer from where the cursor was.
    p = [0.0] * 8 + [1.0] * 4 + [0.0] * 4 + [0.5]
    label = [REST] * 8 + [1] * 8 + [REST]
    out = run(p, label)
    # Hit after 1 s; the later confident "left" outputs of the same trial change nothing.
    assert (out.trials, out.hits, out.wrong) == (1, 1, 0)
    assert out.hit_times_s == [pytest.approx(1.0)]


def test_back_to_back_trials_and_unfinished_last_trial_are_counted() -> None:
    out = run([0.5] * 6, [0, 0, 1, 1, REST, 0])
    assert (out.trials, out.timeouts) == (3, 3)
    with pytest.raises(ValueError, match="positive"):
        run([0.5], [0], travel=0.0)


def test_selection_maximises_hits_minus_wrong_and_prefers_the_slower_cursor() -> None:
    # Trial 1 ("right"): a wrong blip, then steadily right. Trial 2 ("left"): steadily left.
    p = [0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    label = [1] * 8 + [REST] + [0] * 8
    runs = [(np.array(p), np.array(label))]
    best, table = select_travel_time(runs, threshold=0.6, step_s=STEP, grid_s=(0.1, 0.5, 1.0, 5.0))
    # Too fast: the blip ends trial 1 on the wrong side. Too slow: nothing is reached.
    assert table["0.1"]["wrong"] == 1 and table["0.1"]["net"] == 0
    assert table["5"]["timeouts"] == 2 and table["5"]["net"] == 0
    assert table["0.5"]["net"] == table["1"]["net"] == 2
    assert best == 1.0  # tie between 0.5 and 1.0 -> slower cursor
    assert play_runs(runs, 0.6, best, STEP).hits == 2


def test_game_report_chooses_on_training_runs_only() -> None:
    n = 12
    good = np.r_[np.full(n, 0.9), 0.5].astype(np.float32)  # steadily right
    bad = np.r_[np.full(n, 0.1), 0.5].astype(np.float32)  # steadily wrong
    label = np.r_[np.ones(n), REST].astype(np.int8)
    parts = {"subject": [], "run": [], "label": [], "p": []}  # type: ignore[var-annotated]
    for s in (1, 2):
        for r in (4, 8, 12):
            parts["subject"].append(np.full(n + 1, s))
            parts["run"].append(np.full(n + 1, r))
            parts["label"].append(label)
            # Subject 1 decodes its training runs and fails run 12; subject 2 always fails.
            parts["p"].append(good if (s == 1 and r != 12) else bad)
    arrays = {k: np.concatenate(v) for k, v in parts.items()}
    arrays["smooth::cross/m/causal/fold0"] = arrays["smooth::cross/m/causal/fold1"] = arrays.pop(
        "p"
    )
    summary = {
        "config": {"step_s": STEP},
        "fold_of": {"1": 0, "2": 1},
        "thresholds": {"cross/m/causal": [0.6, 0.6]},
    }
    report = game_report(arrays, summary, "cross/m/causal", 1, (4, 8), 12)

    assert report["travel_time_s"] in TRAVEL_TIME_GRID_S
    chosen = report["selection_table"][f"{report['travel_time_s']:g}"]
    assert chosen["trials"] == 2 and chosen["hits"] == 2  # runs 4 and 8 only
    # Run 12 is reported as it is, however bad: it had no say in the choice.
    assert report["demo_test_run"]["hits"] == 0 and report["demo_test_run"]["wrong"] == 1
    assert report["all_subjects"]["n_subjects"] == 2
    assert report["all_subjects"]["hit_rate_mean"] == pytest.approx((2 / 3 + 0) / 2)
    json.dumps(report)


def test_outcome_dict_handles_no_trials() -> None:
    d = GameOutcome().to_dict()
    assert d["trials"] == 0 and np.isnan(d["hit_rate"]) and np.isnan(d["median_hit_time_s"])


def test_shared_fixture_matches_the_simulation() -> None:
    """The browser game (frontend/src/game.test.ts) replays the same cases."""
    cases = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]
    assert len(cases) >= 4
    for case in cases:
        label = [{"left": 0, "right": 1}.get(t, REST) for t in case["true"]]
        out = simulate_game(
            np.array(case["p_right"]), np.array(label), case["threshold"],
            case["travel_time_s"], case["step_s"],
        )  # fmt: skip
        got = {"trials": out.trials, "hits": out.hits, "wrong": out.wrong,
               "timeouts": out.timeouts}  # fmt: skip
        assert got == case["expected"], case["name"]
        assert out.hit_times_s == pytest.approx(case["hit_times_s"]), case["name"]
