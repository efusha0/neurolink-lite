# Architecture notes

See the Mermaid diagram in the README for the high-level picture. This file records
decisions that are not obvious from the code.

## Decisions (Phase 0)

- **Epoch window 0.5 to 2.5 s after the cue.** Two seconds, the same length as the online
  sliding window. Offline accuracy is nevertheless an *optimistic* estimate of online
  behaviour, for two reasons found in the Phase 3 review: offline epochs are filtered with a
  zero-phase (non-causal) filter while the online classifier must filter causally, and
  offline epochs are aligned to the cue while online windows slide freely and often straddle
  rest and imagery. Phase 4 therefore measures a separate "pseudo-online" accuracy (see
  below) and reports it next to the offline numbers.
- **Band-pass before splitting is not leakage.** The 8-30 Hz filter is a zero-phase FIR
  (MNE ``firwin`` design, about 1.65 s long at 160 Hz) with fixed coefficients, applied per
  run; it learns nothing from the data. Its support stays inside the 4.1 s trial, so no
  epoch contains samples of another trial. Everything that *is* fitted (z-score statistics,
  CSP filters, EEGNet weights) is fitted on the training fold only.
- **Cross-subject protocol = subject-grouped 5-fold.** Pure leave-one-subject-out would be
  about 105 EEGNet trainings on CPU; grouped 5-fold keeps subjects disjoint at a fraction of the cost.
- **The demo replays run 12** of the chosen subject, and the demo model never sees that
  data. The original plan (a per-subject model trained on runs 4 and 8) was replaced in
  Phase 4 by a cross-subject model that has not seen the demo subject at all; see
  "Decisions (Phase 4)".
- **Streamer sends raw (unfiltered) EEG.** The online classifier filters incoming chunks
  with a stateful causal filter, as a real headset pipeline would.

## Decisions (Phase 2)

- **No hyper-parameter search on the evaluation set.** Before running the baseline on all
  subjects, a small sanity sweep (epoch window, CAR on/off, 64 vs 21 channels, 4 vs 6 CSP
  components) was run on the 10 development subjects only. All variants landed within
  noise of each other (within-subject 57-66 %, cross-subject 46-56 %), two subjects were
  clearly decodable (S002, S007 at 87-98 %) and most others sat near chance. The planned
  defaults (0.5-2.5 s, CAR on, 6 components) were kept; both the 64-channel and the
  21-channel motor subset are reported.
- **Leakage guards are executable.** ``grouped_subject_folds`` raises if any subject
  appears on both sides of a split, ``_run_fold`` raises on overlapping indices, and the
  model is re-created per fold through a factory. Tests cover all three.
- **Why cross-subject CSP+LDA is near chance.** CSP filters are spatial patterns of
  covariance differences; electrode placement, skull conductivity and the individual
  mu/beta rhythm frequencies vary between people, so filters learned on one group rarely
  transfer. This is a well-known limitation and the motivation for EEGNet in Phase 3.
- **CSP covariance shrinkage.** Checked on the 10 dev subjects after Phase 2 review:
  ``reg`` in {None, ledoit_wolf, oas, 0.1} changed within-subject accuracy by at most
  2 points (std ~21) and cross-subject by at most 2 points. Within noise; the default
  ``ledoit_wolf`` was kept.

## Decisions (Phase 3)

- **EEGNet-8,2 follows Lawhern et al. (2018)** in structure: temporal convolution with a
  kernel of half the sampling rate (80 samples), depthwise spatial convolution (D = 2,
  max-norm 1), separable convolution (F2 = 16), dense layer with max-norm 0.25.
  Deliberate or inherited deviations from the paper:
  - *Dropout 0.5 in both protocols.* The paper uses 0.5 within-subject and 0.25
    cross-subject. We did not re-run with 0.25: choosing between the two after seeing the
    cross-subject test folds would be tuning on the evaluation set.
  - *Pooling (4, 8) and the separable kernel (16) are the paper's 128 Hz values* used
    unchanged at 160 Hz, so the separable kernel spans 0.4 s instead of 0.5 s and the final
    feature rate is 5 Hz instead of 4 Hz.
  - *BatchNorm uses PyTorch defaults* (momentum 0.1, eps 1e-5), not the Keras defaults of
    the reference implementation (0.99 running-average decay, eps 1e-3).
  - *Input is 8-30 Hz at 160 Hz*, as fixed for the whole project; the paper band-passes
    4-40 Hz and resamples to 128 Hz.
  - *No early stopping and no validation split.* The paper selects the best epoch on a
    validation set; with 36 training epochs per within-subject fold we train for a fixed
    budget instead.
- **Small last batch in within-subject training.** 36 training epochs with batch size 32
  give batches of 32 and 4. BatchNorm statistics on 4 samples are noisy, which may cost
  some within-subject accuracy. It is not leakage and was left as is; fixing it after the
  fact would again be tuning on the evaluation folds.
- **Where the training hyper-parameters come from.** Adam 1e-3 and batch 32 are common
  EEGNet defaults. How the epoch budgets (100 within-subject, 30 cross-subject) were
  chosen was not logged. A 10-subject motor-channel cache was built shortly before the full
  runs, which suggests trial runs on the development subjects, but the extent of that
  search is unknown. The development subjects (1-10)
  are also part of the 105-subject evaluation, so these choices (and the Phase 2 sanity
  sweep) carry a small optimistic bias. This goes into the README "Limitations" section.
- **Per-subject accuracy is computed from each subject's own predictions.** Until the
  Phase 3 review, cross-subject results assigned the fold accuracy to every test subject,
  which hid the inter-subject spread (0/105 subjects >= 70 % was reported; the real counts
  are in ``results/RESULTS.md``). Means and pooled accuracies were unaffected. Tables now
  report mean +/- std across subjects; the fold std is a separate column.
- **Reproducibility.** Re-running CSP+LDA (motor) and EEGNet cross-subject from the cache
  reproduced every fold accuracy exactly on the same machine (seed 42, 4 CPU threads).
  Bit-identical results across machines or PyTorch versions are not guaranteed.

## Requirements carried into Phase 4

- **Pseudo-online accuracy.** Evaluate the online pipeline offline on held-out data:
  causal stateful band-pass on the continuous recording, CAR, 2 s windows every 0.25 s
  that are *not* aligned to the cue, the same smoothing as the live system. Windows are
  scored against the annotation active at the window end. The model is trained without
  access to the replayed data (held-out run or held-out subjects). Report this number
  beside the offline accuracy, for the demo subject and as an average over subjects.

## Decisions (Phase 4)

- **One implementation for live and for evaluation.** `OnlinePreprocessor`,
  `SlidingWindower`, the EMA smoother and the confidence gate in `realtime/online.py` know
  nothing about LSL or WebSockets. The live server and the pseudo-online evaluation drive
  the same classes, so the pseudo-online accuracy is the accuracy of the demo code.
- **Online filter.** 4th-order Butterworth band-pass 8-30 Hz as second-order sections,
  stateful across chunks, state initialised to the steady state of the first sample. The
  result does not depend on how the stream is chunked (tested).
- **Pseudo-online protocol.** 2 s windows every 0.25 s over the continuous recording,
  scored against the annotation of the window's last sample. "Inside" windows lie entirely
  inside one left/right trial; "all" windows are every window ending during a trial,
  including those that still contain part of the preceding rest. Two scopes:
  `cross` (the fold model that held the subject out, all three runs) and `subject`
  (trained on runs 4 and 8, tested on run 12). Results: `results/RESULTS.md`.
- **Training filter: causal.** The online model is trained on epochs filtered with the
  online filter (`PreprocessConfig.causal`), decided before any pseudo-online number
  existed. The zero-phase-trained models were evaluated once as well to price the mismatch:
  the difference in pseudo-online accuracy is -0.1 to +0.3 points across the four model
  families (paired over subjects, p = 0.41-0.83). The mismatch costs nothing measurable,
  presumably because both models rely on band power rather than phase.
- **Where accuracy is lost online.** From cue-aligned epochs to windows inside the trial:
  about 1-2 points. Counting every window that ends during a trial: 3-5 points (cross-subject
  EEGNet, the demo model: 66.0 % offline, 64.1 % inside, 61.0 % all, i.e. 5 points). The
  loss comes from window alignment, not from the causal filter.
- **Demo model: cross-subject EEGNet trained on causally filtered data.** The criterion was
  fixed before choosing: the highest mean pseudo-online accuracy across the 105 subjects.
  Cross-subject EEGNet wins it (64.1 % inside the trial) over cross-subject CSP+LDA
  (61.6 %; +2.5 points paired over subjects, t = 3.05, p = 0.003) and both per-subject
  models (61.4 %, 58.9 %).
  Between the two EEGNet training filters the criterion does not discriminate (64.1 % vs
  64.2 %, not significant); causal was taken because it had been decided beforehand and
  matches the online pipeline.
  The model was **not** chosen by how any candidate's demo subject did on run 12. For
  transparency: those run-12 numbers had already been printed next to the population
  table when the choice was made, and they favoured CSP+LDA (its rule-selected subject
  scored 91 % against 70 % for EEGNet's at that point). The choice went with the
  population criterion.
  Caveat: picking the best of eight candidates on the same held-out data makes the winner's
  own number slightly optimistic. The ranking EEGNet > CSP+LDA agrees with the offline
  Phase 3 comparison, which was not part of this choice.
- **Demo subject: S007, selected without looking at run 12.** Rule: the subject with the
  highest accuracy of the held-out fold model on the cue-aligned epochs of runs 4 and 8
  (30 epochs). S002, S007 and S085 tied at 96.7 % (29/30). The tie is broken by the
  pseudo-online accuracy of the same model on runs 4 and 8 (windows inside the trial,
  smoothed; about 260 windows per subject): S007 94.2 %, S002 88.4 %, S085 86.8 %.
  Run-12 results of other subjects were not consulted. Two things to keep in mind:
  the rule selects for a strong subject, so the demo is a best case (the README reports
  the average next to it); and S007 was already known as an easily decodable subject from
  the Phase 2 development sweep, although the rule is mechanical and did not use that.
  Ties between per-subject candidates fall back to the lowest subject id, because a
  per-subject model has seen runs 4 and 8 in training.
- **The demo runs the evaluated model, not a retrained one.** `train-demo` refits the fold
  model that holds S007 out (fold 4: 84 training subjects, 3780 epochs, seed 42) instead of
  training on all 104 other subjects. It reproduces the experiment exactly: same confidence
  threshold (0.688) and the same offline accuracy for S007 (44/45), recorded in
  `results/realtime/demo_model.json`. The numbers reported for S007 are therefore numbers
  of the model in the demo.
- **Confidence threshold, and what it does not do.** The cursor moves only when the
  smoothed confidence reaches the threshold, the 0.75 quantile of the smoothed confidence
  over *rest* windows of the training subjects (by construction 25 % of training rest
  windows pass). On held-out subjects 28 % of rest windows and 29 % of trial windows pass:
  the model is as confident during rest as during imagery, so the threshold does **not**
  detect rest. This is the "Midas touch" problem of BCIs: a two-class left/right decoder
  has no notion of "no intent" and turns everything into a command. What the threshold
  does do is mute low-confidence predictions: accuracy over the trial windows that pass is
  71 % against 61 % over all trial windows. It was kept for that reason and is described
  as such.
  A third "rest" class was not added. Per-subject models make it worse: thresholds
  computed on their own training runs are inflated by in-sample over-confidence (up to
  1.00), which nearly freezes the cursor.
- **True labels travel separately.** The streamer publishes a marker stream with the true
  state. The classifier never reads it; the server only attaches it to each prediction for
  the "true label vs prediction" display.
- **Latency.** Defined on the LSL clock from the acquisition time of the newest sample
  of a window. Measured by `neurolink latency` over 120 s of the demo run with
  streamer, classifier, server and a WebSocket client in one process on one machine
  (465 predictions, first 2 s discarded): end to end p50 9.3 ms,
  p95 15.3 ms, max 23.5 ms; of that, p50 6.4 ms up to the prediction.
  Not included: the 2 s analysis window (a property of the decoder, not a delay of the
  pipeline) and browser rendering. With 50 ms chunks every window ends on a chunk
  boundary; with 200 ms chunks a prediction also waits for its chunk to complete
  and the end-to-end latency becomes p50 54 ms, p95 165 ms.
- **A bug the latency measurement found.** The first measurement showed p50 107 ms and
  p95 202 ms: `pull_chunk(timeout=0.2)` waits the whole timeout trying to fill its buffer.
  The reader now takes what is available and blocks only for the first sample. The LSL
  tests assert a median below 60 ms so this cannot come back unnoticed.
- **The live session agrees with the pseudo-online numbers.** During the latency run
  (229 trial windows of S007 run 12 received over the WebSocket) the live system was right
  on 86.5 % of all trial windows; the pseudo-online evaluation of the same run gives
  87.0 %. The two differ in where the 0.25 s window grid starts and in the last
  seconds of the 125 s run, which the 120 s session does not reach.
