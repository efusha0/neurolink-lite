/**
 * The "reach the target" game. Pure logic, no drawing.
 *
 * Same rules as the Python simulation in `src/neurolink/eval/game.py` (which chose the
 * cursor speed); `fixtures/game_cases.json` is replayed by the tests of both.
 *
 * - The cursor lives on a line from -1 (left wall) to +1 (right wall).
 * - When a left/right trial starts, the cursor returns to the centre and the target is the
 *   wall of the cued side. The cue comes from the replayed recording, not from the user.
 * - Each prediction sets the velocity: (p_right - p_left) / travelTime while the
 *   prediction is active (above the confidence threshold), zero otherwise.
 * - Reaching the cued wall is a hit, the other wall a wrong-side hit; either freezes the
 *   cursor until the trial ends. A trial that ends first is a timeout.
 * - During rest there is no target and the cursor keeps following the predictions.
 *
 * Time is stream time, not wall-clock time. Each prediction lets its velocity act for
 * exactly one decoder step (`stepS`): `advance` spends that budget frame by frame, and the
 * next prediction first spends whatever is left. The outcome therefore does not depend on
 * how often, or whether, the page is drawn. (A hidden browser tab gets no animation
 * frames while predictions keep arriving; a cursor moved only by frames would stand still
 * there and every trial would time out.)
 *
 * The browser has one rule the simulation does not need: a trial that was already running
 * when the page connected, or when the stream was interrupted, is not scored, because its
 * start was not seen. The simulation always sees a run from its first prediction.
 */

import type { ClassName, PredictionMessage } from "./types";

export type Outcome = "hit" | "wrong" | "timeout";

export interface GameState {
  /** Cursor position in [-1, 1]. */
  x: number;
  /** Velocity in units per second. */
  v: number;
  target: ClassName | null;
  /** True once the current trial ended in a hit or a wrong-side hit. */
  resolved: boolean;
  /** Stream seconds since the first prediction of the current trial. */
  elapsed: number;
  /** Stream seconds the current velocity may still act for. */
  budget: number;
  /** Cue of the previous prediction; `undefined` while there is none to compare with. */
  cue: ClassName | null | undefined;
  trials: number;
  hits: number;
  wrong: number;
  timeouts: number;
  hitTimes: number[];
  lastOutcome: Outcome | null;
  /** Increases with every outcome, so the UI can flash on change. */
  outcomeCount: number;
}

export type GameInput = Pick<PredictionMessage, "probs" | "active" | "true">;

export function createGame(): GameState {
  return {
    x: 0,
    v: 0,
    target: null,
    resolved: false,
    elapsed: 0,
    budget: 0,
    cue: undefined,
    trials: 0,
    hits: 0,
    wrong: 0,
    timeouts: 0,
    hitTimes: [],
    lastOutcome: null,
    outcomeCount: 0,
  };
}

function record(state: GameState, outcome: Outcome): void {
  if (outcome === "hit") state.hits += 1;
  else if (outcome === "wrong") state.wrong += 1;
  else state.timeouts += 1;
  state.lastOutcome = outcome;
  state.outcomeCount += 1;
}

/**
 * Apply one prediction: finish the previous step, start/end trials and set the velocity,
 * which may then act for `stepS` seconds of stream time.
 */
export function onPrediction(
  state: GameState,
  msg: GameInput,
  travelTimeS: number,
  stepS: number,
): void {
  settle(state);
  const cue: ClassName | null = msg.true === "left" || msg.true === "right" ? msg.true : null;
  if (state.target !== null && cue !== state.target) {
    // The scored trial is over (rest, or the other cue without a rest in between).
    if (!state.resolved) record(state, "timeout");
    state.target = null;
  }
  if (cue !== null && state.cue !== undefined && cue !== state.cue) {
    // A trial starts here, and its start was seen.
    state.target = cue;
    state.resolved = false;
    state.x = 0;
    state.elapsed = 0;
    state.trials += 1;
  }
  state.cue = cue;
  const frozen = state.target !== null && state.resolved;
  state.v = msg.active && !frozen ? (msg.probs.right - msg.probs.left) / travelTimeS : 0;
  state.budget = stepS;
}

/**
 * Draw-time update: move the cursor for `dt` seconds of wall-clock time, but never beyond
 * the step the last prediction paid for. Calling it rarely, or never, changes nothing in
 * the outcome: the next prediction settles the rest.
 */
export function advance(state: GameState, dt: number): void {
  move(state, Math.min(dt, state.budget));
}

/** Spend what is left of the current step. */
export function settle(state: GameState): void {
  move(state, state.budget);
}

/**
 * The stream was interrupted (reconnect, restart, lost predictions): what happened in
 * between is unknown. A trial that was still open is dropped without an outcome, and the
 * trial that may be running when the stream comes back is not scored.
 */
export function interrupt(state: GameState): void {
  if (state.target !== null && !state.resolved) state.trials -= 1;
  state.target = null;
  state.resolved = false;
  state.v = 0;
  state.budget = 0;
  state.cue = undefined;
}

/** Zero the score; a trial still in progress keeps running and counts when it ends. */
export function resetCounters(state: GameState): void {
  state.trials = state.target !== null && !state.resolved ? 1 : 0;
  state.hits = 0;
  state.wrong = 0;
  state.timeouts = 0;
  state.hitTimes = [];
  state.lastOutcome = null;
  state.outcomeCount = 0;
}

/** Move the cursor for `dt` seconds at the current velocity and detect wall contact. */
function move(state: GameState, dt: number): void {
  if (dt <= 0) return;
  state.budget = Math.max(0, state.budget - dt);
  const moved = state.x + state.v * dt;
  if (state.target !== null && !state.resolved && Math.abs(moved) >= 1) {
    const wall = moved > 0 ? 1 : -1;
    state.resolved = true;
    if (wall > 0 === (state.target === "right")) {
      state.hitTimes.push(state.elapsed + (wall - state.x) / state.v);
      record(state, "hit");
    } else {
      record(state, "wrong");
    }
    state.v = 0;
  }
  state.x = Math.min(1, Math.max(-1, moved));
  state.elapsed += dt;
}

/** Count a trial that is still open (used when a session is summarised mid-trial). */
export function finish(state: GameState): void {
  if (state.target !== null && !state.resolved) record(state, "timeout");
  state.target = null;
}

export function median(values: readonly number[]): number {
  if (values.length === 0) return Number.NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  const hi = sorted[mid] ?? Number.NaN;
  if (sorted.length % 2 === 1) return hi;
  return ((sorted[mid - 1] ?? Number.NaN) + hi) / 2;
}
