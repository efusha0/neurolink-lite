# NeuroLink-lite — Thought-Controlled Cursor (EEG + Deep Learning)

## Goal
We are building a portfolio-quality, open-source brain-computer interface (BCI) project.
The system classifies EEG signals recorded while a person IMAGINES moving their left or right hand,
and uses that prediction to move a cursor/ball in the browser in real time.
No hardware: we replay recorded data as if it were a live stream.

## Data
- PhysioNet "EEG Motor Movement/Imagery Dataset" (EEGMMIDB): 109 subjects, 64 channels, 160 Hz.
- Motor imagery runs (imagine opening/closing left or right fist): runs 4, 8, 12.
  Annotations: T0 = rest, T1 = left, T2 = right.
- Download with `mne.datasets.eegbci.load_data`. Data never goes into the repo (`data/` is gitignored).
- Some subjects are known to have sampling-rate / recording-length issues. The loader must check
  for these, report problematic subjects, and skip them.

## Tech stack
- Python 3.11+, uv (package management), MNE-Python, NumPy, SciPy, scikit-learn, PyTorch, pylsl
- Server: FastAPI + WebSocket
- Frontend: Vite + TypeScript + Canvas (no framework needed)
- Quality: ruff, pytest, mypy, GitHub Actions CI

## Phases
**Phase 0 — Plan and skeleton:** Before writing any code, show me the architecture, folder structure,
and data flow between modules, then wait for my approval. Then create the repo skeleton: pyproject,
ruff, pytest, CI, README draft.

**Phase 1 — Data pipeline:** Loading, channel name standardization, 8–30 Hz band-pass filter,
epoching on T1/T2 events, optional common average reference, caching results as .npz.
Tests for every step.

**Phase 2 — Baseline:** CSP + LDA with two evaluation protocols:
(a) within-subject 5-fold cross-validation, (b) cross-subject leave-subjects-out.
Save results under `results/` as JSON and a markdown table.

**Phase 3 — EEGNet:** EEGNet in PyTorch, same two protocols, fixed seeds. Training must finish in
reasonable time on a CPU (use a channel subset around the motor cortex if needed).
Produce a comparison table and confusion matrices.

**Phase 4 — Real-time system:**
- Replay streamer: streams recorded test data over LSL at 160 Hz in real time.
- Online classifier: reads from LSL, predicts with a sliding window (e.g. 2 s window, 0.25 s step),
  smooths probabilities.
- Sends prediction + confidence to the browser via FastAPI WebSocket.
- Measure and report end-to-end latency.

**Phase 5 — UI:** Moving ball/cursor on canvas, live EEG waveform view (a few channels),
class probability bar, "true label vs prediction" indicator (for transparency),
a simple "reach the target" mini-game.

**Phase 6 — Polish:** README (project overview, Mermaid architecture diagram, results table,
"Limitations" section, how to run), instructions for recording a demo GIF,
one-command demo (`uv run neurolink demo`).

## Rules
- STOP at the end of every phase: summarize what you did, test results, and open questions,
  then wait for my approval.
- No data leakage: filter parameters, CSP, and normalization are fit on the training fold only.
- Never inflate results. If accuracy is low, report it honestly and analyze why.
- No paid APIs or services.
- Code, comments, and README in English. Communicate with me in Turkish.
- Small, meaningful commits (conventional commits).
- When unsure, don't assume — ask.

## Progress
Update this checklist when I approve a phase.
- [ ] Phase 0 — Plan and skeleton
- [ ] Phase 1 — Data pipeline
- [ ] Phase 2 — Baseline (CSP + LDA)
- [ ] Phase 3 — EEGNet
- [ ] Phase 4 — Real-time system
- [ ] Phase 5 — UI
- [ ] Phase 6 — Polish
