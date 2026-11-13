from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np
import pytest

from neurolink.config import MOTOR_CORTEX_CHANNELS, SFREQ, PreprocessConfig, RealtimeConfig
from neurolink.data.cache import cache_key
from neurolink.data.loader import ValidationReport, standardize_channel_names
from neurolink.data.pipeline import build_dataset
from neurolink.eval.online_eval import run_experiment, summarize
from neurolink.eval.report import save_pseudo_online
from neurolink.models.base import Classifier
from neurolink.models.csp_lda import make_csp_lda
from neurolink.realtime.pseudo_online import (
    OUTSIDE,
    REST,
    Recording,
    make_windows,
    recording_from_raw,
    rest_threshold,
    sample_labels_from_annotations,
    smooth_sequence,
    window_labels,
    window_metrics,
)
from tests.conftest import EEGMMIDB_RAW_CH_NAMES, RawFactory

RUNS = (4, 8, 12)


def make_imagery_raw(subject: int, run: int, n_trials: int = 10) -> mne.io.RawArray:
    """Synthetic run where left trials add a 12 Hz rhythm on C3 and right trials on C4."""
    rng = np.random.default_rng(1000 * subject + run)
    n_times = int((n_trials * 8.2 + 1.0) * SFREQ)
    data = rng.normal(0.0, 5e-6, size=(64, n_times))
    info = mne.create_info(list(EEGMMIDB_RAW_CH_NAMES), sfreq=SFREQ, ch_types="eeg")
    onsets, durations, descs = [], [], []
    t = np.arange(n_times) / SFREQ
    order = rng.permutation(n_trials) % 2
    for i in range(n_trials):
        rest_on, cue_on = i * 8.2, i * 8.2 + 4.1
        cue = "T1" if order[i] == 0 else "T2"
        onsets += [rest_on, cue_on]
        durations += [4.1, 4.1]
        descs += ["T0", cue]
        ch = EEGMMIDB_RAW_CH_NAMES.index("C3.." if cue == "T1" else "C4..")
        seg = slice(round(cue_on * SFREQ), round((cue_on + 4.1) * SFREQ))
        data[ch, seg] += 30e-6 * np.sin(2 * np.pi * 12 * t[seg])
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.set_annotations(mne.Annotations(onsets, durations, descs))
    standardize_channel_names(raw)
    return raw


def fake_subject_loader(subject: int):  # type: ignore[no-untyped-def]
    raws = [make_imagery_raw(subject, run) for run in RUNS]
    return raws, [ValidationReport(subject, run) for run in RUNS]


def fake_recordings(subject: int) -> list[Recording]:
    return [recording_from_raw(make_imagery_raw(subject, run), subject, run) for run in RUNS]


# --------------------------------------------------------------------------- labelling
def test_sample_labels_follow_annotations(raw_factory: RawFactory) -> None:
    raw = raw_factory(n_trials=3, duration=30.0)  # T0, T1, T0, T2, T0, T1 every 4.1 s
    labels = sample_labels_from_annotations(raw)
    n = round(4.1 * SFREQ)
    assert labels[0] == REST and labels[n - 1] == REST
    assert labels[n] == 0 and labels[2 * n - 1] == 0
    assert labels[3 * n] == 1
    assert labels[-1] == OUTSIDE  # after the last annotation


def test_window_labels_and_inside_flag() -> None:
    labels = np.array([REST] * 400 + [1] * 656 + [REST] * 400, dtype=np.int8)
    ends = np.array([400, 401, 719, 720, 1056, 1057], dtype=np.int64)
    label, inside = window_labels(labels, ends, window=320)
    assert label.tolist() == [REST, 1, 1, 1, 1, REST]
    # Only windows whose 320 samples are all inside the trial count as "inside".
    assert inside.tolist() == [False, False, False, True, True, False]


def test_make_windows_is_chunk_invariant_and_matches_recording() -> None:
    rec = fake_recordings(1)[0]
    whole = make_windows(rec, MOTOR_CORTEX_CHANNELS, (8.0, 30.0))
    chunked = make_windows(rec, MOTOR_CORTEX_CHANNELS, (8.0, 30.0), chunk=37)
    np.testing.assert_array_equal(whole.ends, chunked.ends)
    np.testing.assert_allclose(whole.windows, chunked.windows, atol=1e-4)
    assert whole.windows.shape[1:] == (21, 320)
    assert whole.ends[0] == 320 and np.all(np.diff(whole.ends) == 40)
    assert {REST, 0, 1} <= set(whole.label.tolist())
    assert 0 < whole.inside.sum() < (whole.label >= 0).sum()


def test_causal_training_epochs_equal_online_windows() -> None:
    """A cue-aligned window of the online stream is exactly a causal training epoch."""
    cfg = PreprocessConfig(channels=MOTOR_CORTEX_CHANNELS, subjects=(1,), causal=True)
    ds, _ = build_dataset(cfg, subject_loader=fake_subject_loader)
    rec = fake_recordings(1)[0]
    # Step 1 sample so that a window ends exactly at cue + 2.5 s.
    rt = RealtimeConfig(step_s=1 / SFREQ)
    win = make_windows(rec, MOTOR_CORTEX_CHANNELS, (8.0, 30.0), rt)
    first_cue = round(4.1 * SFREQ)
    i = int(np.flatnonzero(win.ends == first_cue + round(2.5 * SFREQ))[0])
    np.testing.assert_allclose(win.windows[i], ds.X[0], atol=1e-3)


def test_causal_flag_changes_data_but_default_cache_key_is_stable() -> None:
    base = PreprocessConfig(subjects=tuple(range(1, 110)))
    # Key of the 105-subject, 64-channel cache built in Phase 2 must not move.
    assert cache_key(base) == "218d5878d5eb"
    causal = PreprocessConfig(subjects=tuple(range(1, 110)), causal=True)
    assert cache_key(causal) != cache_key(base)
    a, _ = build_dataset(PreprocessConfig(subjects=(1,)), subject_loader=fake_subject_loader)
    b, _ = build_dataset(
        PreprocessConfig(subjects=(1,), causal=True), subject_loader=fake_subject_loader
    )
    assert a.X.shape == b.X.shape and not np.allclose(a.X, b.X)


# ----------------------------------------------------------------------------- metrics
def test_smooth_sequence_matches_manual_ema() -> None:
    raw = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    out = smooth_sequence(raw, alpha=0.5)
    np.testing.assert_allclose(out[:, 1], [0.0, 0.5, 0.75])


def test_rest_threshold_is_a_quantile_of_rest_confidence() -> None:
    p = np.array([0.5, 0.9, 0.1, 0.6, 0.99])
    label = np.array([REST, REST, REST, REST, 1], dtype=np.int8)
    # rest confidences: 0.5, 0.9, 0.9, 0.6 -> median 0.75; the trial window is ignored
    assert rest_threshold(p, label, 0.5) == pytest.approx(0.75)
    assert rest_threshold(p[4:], label[4:], 0.5) == 0.5  # no rest -> never gate


def test_window_metrics_on_a_hand_made_case() -> None:
    label = np.array([REST, REST, 0, 0, 1, 1], dtype=np.int8)
    inside = np.array([False, False, False, True, True, True])
    p_raw = np.array([0.6, 0.4, 0.2, 0.7, 0.9, 0.3])
    p_smooth = np.array([0.95, 0.4, 0.3, 0.2, 0.9, 0.6])
    m = window_metrics(p_raw, p_smooth, label, inside, threshold=0.8)
    assert m["acc_inside_raw"] == pytest.approx(1 / 3)
    assert m["acc_inside_smooth"] == pytest.approx(1.0)
    assert m["acc_all_raw"] == pytest.approx(0.5)
    assert m["acc_all_smooth"] == pytest.approx(1.0)
    assert m["rest_active_rate"] == pytest.approx(0.5)
    assert m["imagery_active_rate"] == pytest.approx(0.5)  # 0.2 -> conf 0.8, 0.9
    assert m["acc_when_active"] == pytest.approx(1.0)
    assert m["rest_right_rate"] == pytest.approx(0.5)
    assert (m["n_inside"], m["n_all"], m["n_rest"]) == (3, 4, 2)


def test_window_metrics_without_trials_is_nan() -> None:
    label = np.array([REST, REST], dtype=np.int8)
    m = window_metrics(np.array([0.5, 0.5]), np.array([0.5, 0.5]), label, np.zeros(2, bool), 0.6)
    assert np.isnan(m["acc_all_smooth"]) and m["rest_active_rate"] == 0.0


# -------------------------------------------------------------------------- experiment
@pytest.fixture(scope="module")
def experiment():  # type: ignore[no-untyped-def]
    subjects = (1, 2, 3, 4)
    datasets = {}
    for filt in ("zero_phase", "causal"):
        cfg = PreprocessConfig(
            channels=MOTOR_CORTEX_CHANNELS, subjects=subjects, causal=filt == "causal"
        )
        datasets[filt], _ = build_dataset(cfg, subject_loader=fake_subject_loader)

    def csp(scope: str) -> Classifier:
        model: Classifier = make_csp_lda()
        return model

    return run_experiment(
        subjects, seed=0, makers={"csp_lda": csp}, n_splits=2,
        datasets=datasets, recording_loader=fake_recordings,
    )  # fmt: skip


def test_experiment_rows_and_scores(experiment) -> None:  # type: ignore[no-untyped-def]
    summary, _store = experiment
    rows = summary["rows"]
    assert set(rows) == {
        "cross/csp_lda/zero_phase", "cross/csp_lda/causal",
        "cross/csp_lda/zero_phase (run 12 only)", "cross/csp_lda/causal (run 12 only)",
        "subject/csp_lda/zero_phase", "subject/csp_lda/causal",
    }  # fmt: skip
    for name, row in rows.items():
        assert row["n_subjects"] == 4, name
        assert row["offline_acc"]["mean"] > 0.9, name
    # Matched (causal) training decodes the clean synthetic rhythm online as well.
    for name in ("cross/csp_lda/causal", "subject/csp_lda/causal"):
        assert rows[name]["acc_inside_smooth"]["mean"] > 0.9
        # Windows that still contain rest are harder than windows inside the trial.
        assert rows[name]["acc_all_smooth"]["mean"] <= rows[name]["acc_inside_smooth"]["mean"]
        assert 0.0 <= rows[name]["rest_active_rate"]["mean"] <= 1.0


def test_experiment_thresholds_come_from_training_data(experiment) -> None:  # type: ignore[no-untyped-def]
    summary, store = experiment
    arrays = store.arrays()
    key = "cross/csp_lda/causal"
    # Recompute fold 0's threshold by hand from the *training* subjects' rest windows.
    per_subject = summary["rows"][key]["per_subject"]
    thr0 = summary["thresholds"][key][0]
    test_subjects = [int(s) for s, row in per_subject.items() if row["threshold"] == thr0]
    train = ~np.isin(arrays["subject"], test_subjects)
    expected = rest_threshold(arrays[f"smooth::{key}/fold0"][train], arrays["label"][train], 0.75)
    assert thr0 == pytest.approx(expected)
    assert 0 < len(test_subjects) < 4


def test_experiment_demo_selection_ignores_the_test_run(experiment) -> None:  # type: ignore[no-untyped-def]
    summary, store = experiment
    demo = summary["demo_selection"]
    assert set(demo) == {"cross/csp_lda/zero_phase", "cross/csp_lda/causal",
                         "subject/csp_lda/causal"}  # fmt: skip
    pick = demo["subject/csp_lda/causal"]
    scores = pick["selection_scores"]
    assert pick["selection_score"] == max(scores.values())
    assert (
        pick["run12_result"]
        == summary["rows"]["subject/csp_lda/causal"]["per_subject"][str(pick["subject"])]
    )
    # Corrupting run-12 outputs must not change which subject is picked.
    arrays = store.arrays()
    fold_of = {1: 0, 2: 0, 3: 1, 4: 1}
    for k in arrays:
        if k.startswith(("raw::", "smooth::")):
            arrays[k] = np.where(arrays["run"] == 12, 0.5, arrays[k]).astype(np.float32)
    offline = {c: {s: 0.0 for s in fold_of} for c in summary["rows"]}
    selection = {c: {int(s): v for s, v in d["selection_scores"].items()} for c, d in demo.items()}
    again = summarize(arrays, fold_of, offline, offline, selection, 0.75)
    assert again["demo_selection"]["subject/csp_lda/causal"]["subject"] == pick["subject"]


def test_save_pseudo_online_writes_json_and_markdown(experiment, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    summary, _ = experiment
    path = save_pseudo_online(summary, tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))  # strict JSON: no NaN
    assert data["config"]["test_run"] == 12
    md = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert "## Pseudo-online accuracy" in md
    assert "| subject/csp_lda/causal | 4 |" in md
