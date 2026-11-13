from __future__ import annotations

import json
from pathlib import Path

from neurolink.data.cache import Dataset
from neurolink.eval.protocols import cross_subject_cv, within_subject_cv
from neurolink.eval.report import load_results, result_to_dict, save_result, write_markdown_table
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
