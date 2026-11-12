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
