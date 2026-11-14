from __future__ import annotations

import json
from pathlib import Path

from neurolink.data.cache import Dataset
from neurolink.eval.protocols import cross_subject_cv, within_subject_cv
from neurolink.eval.report import (
    DEMO_MODEL_JSON,
    LATENCY_JSON,
    load_realtime,
    load_results,
    realtime_markdown,
    result_to_dict,
    save_realtime,
    save_result,
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
