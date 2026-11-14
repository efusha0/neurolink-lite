from __future__ import annotations

import json
from pathlib import Path

from neurolink.data.cache import Dataset
from neurolink.eval.protocols import cross_subject_cv, within_subject_cv
from neurolink.eval.report import (
    BEHAVIOURAL_JSON,
    DEMO_MODEL_JSON,
    GAME_JSON,
    LATENCY_JSON,
    behavioural_markdown,
    game_markdown,
    load_realtime,
    load_results,
    realtime_markdown,
    result_to_dict,
    save_realtime,
    save_result,
    ui_info,
    write_markdown_table,
)
from neurolink.models.csp_lda import make_csp_lda


def test_result_dict_fields(synthetic_dataset: Dataset) -> None:
    result = within_subject_cv(synthetic_dataset, make_csp_lda, "csp_lda", seed=0)
    d = result_to_dict(result, extra={"channels": "all"})
    assert d["model"] == "csp_lda" and d["protocol"] == "within_subject"
    assert d["n_folds"] == 20 and d["n_subjects"] == 4
    assert 0.0 <= d["subject_accuracy_min"] <= d["subject_accuracy_mean"] <= 1.0
    assert d["subject_accuracy_std"] >= 0.0
    assert d["n_test_epochs"] == synthetic_dataset.n_epochs
    assert 0.0 <= d["accuracy_mean"] <= 1.0
    assert len(d["confusion"]) == 2 and len(d["confusion"][0]) == 2
    assert d["channels"] == "all"
    assert set(d["per_subject_accuracy"]) == {"1", "2", "3", "4"}
    json.dumps(d)  # must be serializable


def test_save_and_markdown(synthetic_dataset: Dataset, tmp_path: Path) -> None:
    ws = within_subject_cv(synthetic_dataset, make_csp_lda, "csp_lda", seed=0)
    cs = cross_subject_cv(synthetic_dataset, make_csp_lda, "csp_lda", n_splits=4, seed=0)
    p1 = save_result(ws, tmp_path)
    p2 = save_result(cs, tmp_path)
    assert p1.name == "csp_lda_within_subject.json"
    assert p2.name == "csp_lda_cross_subject.json"

    rows = load_results(tmp_path)
    assert [r["protocol"] for r in rows] == ["cross_subject", "within_subject"]

    md = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert "| csp_lda | within-subject |" in md
    assert "| csp_lda | cross-subject |" in md
    assert md.count("confusion matrix") == 2

    # Re-saving overwrites instead of duplicating rows.
    save_result(ws, tmp_path)
    assert len(load_results(tmp_path)) == 2
    assert write_markdown_table(tmp_path).exists()


def test_markdown_tolerates_results_from_before_subject_stats(
    synthetic_dataset: Dataset, tmp_path: Path
) -> None:
    ws = within_subject_cv(synthetic_dataset, make_csp_lda, "csp_lda", seed=0)
    old = {k: v for k, v in result_to_dict(ws).items() if not k.startswith("subject_accuracy")}
    (tmp_path / "old_within_subject.json").write_text(json.dumps(old), encoding="utf-8")
    save_result(ws, tmp_path)  # must not crash on the legacy file
    md = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert "re-run needed" in md


def test_realtime_sections_appear_in_the_markdown_table(tmp_path: Path) -> None:
    assert realtime_markdown(tmp_path) == []
    row = {
        "offline_acc": 1.0, "acc_inside_smooth": 0.9, "acc_all_smooth": 0.8,
        "rest_active_rate": 0.5, "imagery_active_rate": 0.6, "acc_when_active": None,
    }  # fmt: skip
    save_realtime(
        DEMO_MODEL_JSON,
        {
            "candidate": "cross/eegnet/causal", "demo_subject": 7, "n_train_subjects": 84,
            "n_train_epochs": 3780, "threshold": 0.6882,
            "selection": {
                "selection_rule": "rule", "selection_score": 29 / 30,
                "tied_subjects": [2, 7], "tie_break_scores": {"2": 0.88, "7": 0.94},
            },
            "pseudo_online_run12": row, "pseudo_online_all_runs": row,
        },
        tmp_path,
    )  # fmt: skip
    stage = {"n": 464.0, "mean_ms": 4.0, "p50_ms": 3.5, "p95_ms": 6.25, "p99_ms": 8.0,
             "max_ms": float("nan")}  # fmt: skip
    save_realtime(
        LATENCY_JSON,
        {"subject": 7, "run": 12, "duration_s": 120.0, "chunk_ms": 50.0,
         "stages_ms": {"end_to_end": stage, "stream_to_pred": stage}},
        tmp_path,
    )  # fmt: skip
    assert load_realtime(LATENCY_JSON, tmp_path)["stages_ms"]["end_to_end"]["max_ms"] is None  # type: ignore[index]

    md = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert "## Demo model" in md and "holds S007 out" in md and "threshold 0.688" in md
    assert "Tie-break among equal scores: S002 88.0 %, S007 94.0 %." in md
    assert "| S007, run 12 (replayed in the demo) | 100.0 | 90.0 | 80.0 | 50.0 | 60.0 | - |" in md
    assert "## Latency" in md
    line = "| S007 run 12, 120 s, 50 ms chunks | end_to_end | 464 | 4.0 | 3.5 | 6.2 | 8.0 | - |"
    assert line in md
    assert md.index("stream_to_pred | 464") < md.index("end_to_end | 464")


def test_behavioural_latency_section(tmp_path: Path) -> None:
    assert behavioural_markdown(tmp_path) == []

    def stats(n: int, median_all: float) -> dict[str, object]:
        return {
            "n_trials": n, "reached_share": 0.5, "median_all_trials_s": median_all,
            "quantiles_reached_s": {"p10": 0.1, "p25": 0.8, "p50": 1.8, "p75": 2.7, "p90": 3.5},
            "reached_by": {"1": 0.1, "2": 0.25, "3": 0.4},
        }  # fmt: skip

    def block(n: int) -> dict[str, object]:
        return {
            "correct_active": stats(n, float("nan")), "correct_active_change": stats(n - 2, 1.65),
            "correct": stats(n, 0.25), "already_correct_active_share": 0.185,
            "first_command_right_share": 0.345, "first_command_wrong_share": 0.203,
            "never_active_share": 0.452,
        }  # fmt: skip

    save_realtime(
        BEHAVIOURAL_JSON,
        {
            "candidate": "cross/eegnet/causal", "n_subjects": 105, "window_s": 2.0,
            "all_subjects": block(4725),
            "demo_subject": {"subject": 7, "all_runs": block(45), "run_12": block(15)},
        },
        tmp_path,
    )  # fmt: skip
    md = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert "## Behavioural latency (reaction to the cue)" in md
    assert "In 18.5 % of all trials" in md
    row = "| S007, run 12 | same, output not already right at the cue | 13 | 50.0 | 1.65 |"
    assert row in md
    assert (
        "| All 105 subjects | right class, above threshold | 4725 | 50.0 | - | 0.80 | 1.80 |" in md
    )
    assert "right class in 34.5 % of trials, wrong class in 20.3 %, none at all in 45.2 %" in md


def _game_payload() -> dict[str, object]:
    def row(hits: int, wrong: int, trials: int = 30) -> dict[str, object]:
        return {"trials": trials, "hits": hits, "wrong": wrong,
                "timeouts": trials - hits - wrong, "net": hits - wrong}  # fmt: skip

    return {
        "rule": "the rule", "demo_subject": 7, "train_runs": [4, 8], "test_run": 12,
        "travel_time_s": 1.0,
        "selection_table": {"0.75": row(15, 3), "1": row(14, 2), "1.5": row(11, 0)},
        "demo_test_run": {"trials": 15, "hit_rate": 0.8, "wrong_rate": 0.0,
                          "timeout_rate": 0.2, "median_hit_time_s": 3.27},
        "all_subjects": {"n_subjects": 105, "hit_rate_mean": 0.196, "wrong_rate_mean": 0.07,
                         "timeout_rate_mean": 0.734, "subjects_hit_rate_above_70": 0},
    }  # fmt: skip


def test_game_section_and_ui_info(tmp_path: Path) -> None:
    assert game_markdown(tmp_path) == []
    assert ui_info(tmp_path, "cross/eegnet/causal") == {}  # UI falls back to its defaults

    save_realtime(GAME_JSON, _game_payload(), tmp_path)
    md = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert "## Target game" in md and "Rule for the travel time: the rule." in md
    assert "| 1 **(selected)** | 14/30 | 2 | 14 | 12 |" in md
    assert "| 0.75 | 15/30 | 3 | 12 | 12 |" in md
    assert "| S007, run 12 (the demo) | 15 | 80.0 | 0.0 | 20.0 | 3.27 |" in md
    assert "| Mean of 105 subjects, all runs | - | 19.6 | 7.0 | 73.4 | - |" in md
    assert "0/105 subjects hit at least 70 % of their targets." in md

    info = ui_info(tmp_path, "cross/eegnet/causal")
    assert info["cursor_travel_time_s"] == 1.0
    assert info["reference"] == {
        "demo_run_hit_rate": 0.8, "demo_run_trials": 15, "mean_hit_rate": 0.196,
        "n_subjects": 105,
    }  # fmt: skip
    # With the pseudo-online summary present, the population accuracy is added.
    pseudo = tmp_path / "pseudo_online" / "pseudo_online.json"
    pseudo.parent.mkdir()
    rows = {"cross/eegnet/causal": {"acc_all_smooth": {"mean": 0.61, "std": 0.09}}}
    pseudo.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    assert ui_info(tmp_path, "cross/eegnet/causal")["reference"]["mean_acc_all_smooth"] == 0.61
    assert "mean_acc_all_smooth" not in ui_info(tmp_path, "other")["reference"]
