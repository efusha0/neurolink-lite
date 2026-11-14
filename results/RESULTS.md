# Results

Binary left/right motor imagery, chance = 50 %. Every statistic except the last
column is over per-subject accuracies (each subject's own test epochs, pooled
across folds), so the spread shows inter-subject variability. "Fold std" is the
std over folds (within-subject: subject x fold, 9 test epochs each, hence large;
cross-subject: the 5 subject-grouped folds).
Regenerated automatically; do not edit by hand.

| Model | Protocol | Subjects | Test epochs | Accuracy (%), mean +/- subject std | Median | Min-Max | Pooled | Subjects >= 70 % | Fold std |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| csp_lda | cross-subject | 105 | 4725 | 62.1 +/- 12.6 | 57.8 | 38-93 | 62.1 | 31/105 | 1.8 |
| csp_lda_motor | cross-subject | 105 | 4725 | 62.3 +/- 12.8 | 60.0 | 36-93 | 62.3 | 29/105 | 1.3 |
| eegnet | cross-subject | 105 | 4725 | 65.0 +/- 13.2 | 62.2 | 29-96 | 65.0 | 35/105 | 1.8 |
| csp_lda | within-subject | 105 | 4725 | 59.9 +/- 15.5 | 57.8 | 33-98 | 59.9 | 24/105 | 20.4 |
| csp_lda_motor | within-subject | 105 | 4725 | 61.3 +/- 16.5 | 60.0 | 27-100 | 61.3 | 31/105 | 21.7 |
| eegnet | within-subject | 105 | 4725 | 59.0 +/- 14.1 | 57.8 | 31-98 | 59.0 | 22/105 | 18.8 |

## csp_lda / cross-subject: confusion matrix

| true \ pred | left | right |
|---|---:|---:|
| left | 1467 | 917 |
| right | 873 | 1468 |

## csp_lda_motor / cross-subject: confusion matrix

| true \ pred | left | right |
|---|---:|---:|
| left | 1431 | 953 |
| right | 829 | 1512 |

## eegnet / cross-subject: confusion matrix

| true \ pred | left | right |
|---|---:|---:|
| left | 1670 | 714 |
| right | 939 | 1402 |

## csp_lda / within-subject: confusion matrix

| true \ pred | left | right |
|---|---:|---:|
| left | 1479 | 905 |
| right | 991 | 1350 |

## csp_lda_motor / within-subject: confusion matrix

| true \ pred | left | right |
|---|---:|---:|
| left | 1495 | 889 |
| right | 940 | 1401 |

## eegnet / within-subject: confusion matrix

| true \ pred | left | right |
|---|---:|---:|
| left | 1388 | 996 |
| right | 941 | 1400 |

## Pseudo-online accuracy

The online pipeline run over held-out recordings: causal band-pass, 2 s windows every 0.25 s that ignore the cue timing,
EMA smoothing (alpha = 0.3). Mean +/- std across subjects, in %.

* Candidate = scope / model / filter used for the *training* epochs. `cross`: model
  of the fold that held the subject out, all three runs. `subject`: trained on runs
  4 and 8, tested on run 12. `zero_phase` training is mismatched with
  the causal online filter; `causal` training matches it.
* Offline = cue-aligned 0.5-2.5 s epochs of the same test data, filtered like the
  training data (the Phase 2-3 style number).
* Inside = windows entirely inside a left/right trial. All = every window ending
  during a trial, including windows that still contain rest.
* Threshold = 0.75 quantile of smoothed confidence over rest
  windows of the *training* data. Rest active / Trial active = share of rest / trial
  windows above it (cursor moves); Acc. active = accuracy of the trial windows above it.

| Candidate | Subjects | Offline | Inside, raw | Inside, smoothed | All, smoothed | Rest active | Trial active | Acc. active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cross/csp_lda/causal | 105 | 62.6 +/- 13.6 | 60.6 +/- 12.1 | 61.6 +/- 12.5 | 59.6 +/- 9.2 | 28.3 +/- 24.6 | 29.1 +/- 24.5 | 71.6 +/- 16.7 |
| cross/csp_lda/causal (run 12 only) | 105 | 63.9 +/- 15.3 | 61.2 +/- 13.5 | 62.0 +/- 13.5 | 59.1 +/- 10.3 | 28.2 +/- 25.5 | 28.9 +/- 25.4 | 69.7 +/- 24.4 |
| cross/csp_lda/zero_phase | 105 | 62.3 +/- 12.8 | 60.0 +/- 11.7 | 61.3 +/- 12.1 | 59.5 +/- 8.9 | 27.8 +/- 23.8 | 28.7 +/- 24.0 | 69.4 +/- 17.1 |
| cross/csp_lda/zero_phase (run 12 only) | 105 | 63.3 +/- 15.3 | 60.5 +/- 13.6 | 61.7 +/- 13.9 | 59.2 +/- 10.1 | 27.8 +/- 24.7 | 28.3 +/- 25.0 | 68.8 +/- 23.7 |
| cross/eegnet/causal | 105 | 66.0 +/- 13.7 | 62.9 +/- 11.6 | 64.1 +/- 12.1 | 61.0 +/- 8.7 | 28.0 +/- 23.2 | 28.7 +/- 23.8 | 71.2 +/- 14.8 |
| cross/eegnet/causal (run 12 only) | 105 | 66.4 +/- 15.9 | 63.1 +/- 13.1 | 64.0 +/- 14.1 | 60.5 +/- 11.1 | 29.8 +/- 24.7 | 29.7 +/- 25.4 | 69.3 +/- 20.8 |
| cross/eegnet/zero_phase | 105 | 65.0 +/- 13.2 | 62.9 +/- 11.1 | 64.2 +/- 11.6 | 61.2 +/- 8.2 | 28.1 +/- 23.2 | 29.2 +/- 23.9 | 68.9 +/- 17.6 |
| cross/eegnet/zero_phase (run 12 only) | 105 | 64.4 +/- 14.7 | 63.4 +/- 12.7 | 64.5 +/- 13.7 | 61.0 +/- 10.8 | 30.0 +/- 25.1 | 30.5 +/- 25.7 | 68.0 +/- 23.4 |
| subject/csp_lda/causal | 105 | 62.7 +/- 18.0 | 60.4 +/- 15.1 | 61.4 +/- 16.1 | 58.6 +/- 11.7 | 36.2 +/- 20.7 | 36.1 +/- 20.3 | 63.1 +/- 19.0 |
| subject/csp_lda/zero_phase | 105 | 62.5 +/- 16.4 | 61.0 +/- 14.8 | 61.5 +/- 15.2 | 58.6 +/- 11.6 | 35.4 +/- 20.3 | 35.2 +/- 19.8 | 63.7 +/- 18.3 |
| subject/eegnet/causal | 105 | 60.6 +/- 16.7 | 58.0 +/- 13.8 | 58.9 +/- 14.6 | 56.4 +/- 10.5 | 33.5 +/- 13.6 | 31.0 +/- 12.9 | 60.0 +/- 16.8 |
| subject/eegnet/zero_phase | 105 | 59.4 +/- 17.6 | 58.4 +/- 13.2 | 59.0 +/- 14.4 | 55.7 +/- 11.2 | 33.0 +/- 14.4 | 32.0 +/- 14.3 | 60.0 +/- 17.2 |

## Demo model

`cross/eegnet/causal`, the fold model that holds S007 out: trained on 84 other subjects (3780 epochs), confidence threshold 0.688 from their rest windows.
Demo subject rule: held-out fold model, offline accuracy on runs 4+8; ties: pseudo-online accuracy (inside, smoothed) on runs 4+8, then lowest subject id. Selection score 96.7 %. Tie-break among equal scores: S002 88.4 %, S007 94.2 %, S085 86.8 %.
Run 12 was not used to select the model or the subject.

| Data | Offline | Inside, smoothed | All, smoothed | Rest active | Trial active | Acc. active |
|---|---:|---:|---:|---:|---:|---:|
| S007, run 12 (replayed in the demo) | 100.0 | 100.0 | 87.0 | 53.5 | 64.2 | 94.3 |
| S007, all three runs | 97.8 | 96.1 | 83.5 | 51.0 | 58.9 | 88.3 |

This subject is a best case; the average over all subjects is the `cross/eegnet/causal` row of the pseudo-online table above.

## Latency

Time from the acquisition (LSL timestamp) of the newest sample of a window to:
the smoothed prediction (`stream_to_pred`), the hand-over to the WebSocket
(`pred_to_ws`, measured from the prediction) and the arrival at a WebSocket
client on the same machine (`end_to_end`). Browser rendering is not included.
The 2 s analysis window is not part of these numbers.

| Measurement | Stage | Predictions | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Max (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| S007 run 12, 120 s, 50 ms chunks | stream_to_pred | 465 | 6.2 | 6.4 | 10.4 | 13.7 | 17.9 |
| S007 run 12, 120 s, 50 ms chunks | pred_to_ws | 465 | 1.6 | 1.3 | 3.6 | 4.2 | 4.8 |
| S007 run 12, 120 s, 50 ms chunks | end_to_end | 465 | 8.8 | 9.3 | 15.3 | 18.0 | 23.5 |
| S007 run 12, 60 s, 200 ms chunks | stream_to_pred | 225 | 79.5 | 53.2 | 163.6 | 165.7 | 168.5 |
| S007 run 12, 60 s, 200 ms chunks | pred_to_ws | 225 | 0.5 | 0.4 | 0.7 | 2.1 | 3.1 |
| S007 run 12, 60 s, 200 ms chunks | end_to_end | 225 | 80.3 | 54.1 | 164.9 | 167.1 | 169.8 |
