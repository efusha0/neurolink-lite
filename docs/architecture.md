# Architecture notes

See the Mermaid diagram in the README for the high-level picture. This file records
decisions that are not obvious from the code.

## Decisions (Phase 0)

- **Epoch window 0.5 to 2.5 s after the cue.** Two seconds, identical to the online sliding
  window, so offline accuracy is a fair estimate of online behaviour.
- **Band-pass before splitting is not leakage.** The 8-30 Hz Butterworth filter has fixed
  coefficients and learns nothing from the data. Everything that *is* fitted (z-score
  statistics, CSP filters, EEGNet weights) is fitted on the training fold only.
- **Cross-subject protocol = subject-grouped 5-fold.** Pure leave-one-subject-out would be
  about 105 EEGNet trainings on CPU; grouped 5-fold keeps subjects disjoint at a fraction of the cost.
- **Demo model trains on runs 4 and 8, replays run 12** of the chosen subject, so the
  replayed data is never seen in training. A flag will allow a cross-subject model instead.
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
