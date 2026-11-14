import { describe, expect, it } from "vitest";
import fixture from "./fixtures/game_cases.json";
import {
  advance,
  createGame,
  finish,
  interrupt,
  median,
  onPrediction,
  resetCounters,
  settle,
  type GameInput,
  type GameState,
} from "./game";
import type { TrueLabel } from "./types";

type Case = (typeof fixture.cases)[number];
const STEP = 0.25;

function input(pRight: number, truth: TrueLabel, threshold = 0.6): GameInput {
  return {
    probs: { left: 1 - pRight, right: pRight },
    active: Math.max(pRight, 1 - pRight) >= threshold,
    true: truth,
  };
}

/**
 * Replay a case. `frames(k)` lists the animation frames (seconds each) drawn after the
 * k-th prediction: none for a hidden tab, many short ones for a fast display.
 */
function play(c: Case, frames: (k: number) => number[]): GameState {
  const game = createGame();
  c.p_right.forEach((p, k) => {
    onPrediction(game, input(p, c.true[k] as TrueLabel, c.threshold), c.travel_time_s, c.step_s);
    for (const dt of frames(k)) advance(game, dt);
  });
  finish(game);
  return game;
}

function score(game: GameState) {
  return { trials: game.trials, hits: game.hits, wrong: game.wrong, timeouts: game.timeouts };
}

function expectCase(game: GameState, c: Case, digits: number): void {
  expect(score(game), c.name).toEqual(c.expected);
  expect(game.hitTimes.length, c.name).toBe(c.hit_times_s.length);
  game.hitTimes.forEach((t, i) => expect(t, c.name).toBeCloseTo(c.hit_times_s[i] ?? NaN, digits));
}

/** Deterministic pseudo-random numbers in [0, 1). */
function lcg(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) % 4294967296;
    return s / 4294967296;
  };
}

describe("target game", () => {
  it.each(fixture.cases.map((c) => [c.name, c] as const))(
    "matches the Python simulation: %s",
    (_name, c) => expectCase(play(c, () => [c.step_s]), c, 9),
  );

  it("scores the same without a single animation frame (hidden browser tab)", () => {
    // Regression: the cursor used to move only in requestAnimationFrame, which a hidden
    // tab does not get, so every trial played in the background ended as a timeout.
    for (const c of fixture.cases) expectCase(play(c, () => []), c, 9);
  });

  it("scores the same whatever the frame timing", () => {
    const random = lcg(7);
    const timings: Array<(c: Case) => (k: number) => number[]> = [
      (c) => () => Array.from({ length: 15 }, () => c.step_s / 15), // 60 fps
      (c) => () => [c.step_s / 3], // too few frames to cover a step
      () => () => [5], // one huge frame after a stall: must not run ahead of the stream
      (c) => (k) => (k % 3 === 0 ? [] : [c.step_s * 0.4, c.step_s * 2]), // frames come and go
      (c) => () => Array.from({ length: Math.floor(random() * 6) }, () => random() * c.step_s),
    ];
    for (const timing of timings) {
      for (const c of fixture.cases) expectCase(play(c, timing(c)), c, 6);
    }
  });

  it("never moves further than the step a prediction paid for", () => {
    const game = createGame();
    onPrediction(game, input(1, "rest"), 1, STEP);
    advance(game, 0.1);
    expect(game.x).toBeCloseTo(0.1);
    advance(game, 60); // a late frame after a long stall
    expect(game.x).toBeCloseTo(0.25);
    advance(game, 60);
    expect(game.x).toBeCloseTo(0.25);
    onPrediction(game, input(1, "rest"), 1, STEP);
    settle(game);
    expect(game.x).toBeCloseTo(0.5);
  });

  it("returns the cursor to the centre at the cue and aims at the cued wall", () => {
    const game = createGame();
    onPrediction(game, input(0.0, "rest"), 1, STEP);
    onPrediction(game, input(0.0, "rest"), 1, STEP);
    advance(game, STEP);
    expect(game.x).toBeCloseTo(-0.5);
    expect(game.target).toBeNull();
    onPrediction(game, input(0.9, "right"), 1, STEP);
    expect(game.x).toBe(0);
    expect(game.target).toBe("right");
    expect(game.v).toBeCloseTo(0.8);
  });

  it("holds still below the threshold and stays inside the walls during rest", () => {
    const game = createGame();
    onPrediction(game, input(0.5, "rest"), 1, STEP);
    onPrediction(game, input(0.55, "left"), 1, STEP);
    settle(game);
    expect(game.x).toBe(0);
    for (let k = 0; k < 8; k += 1) onPrediction(game, input(1, null), 1, STEP);
    settle(game);
    expect(game.x).toBe(1);
    expect(game.hits + game.wrong).toBe(0); // no target, so the wall is not a hit
    expect(score(game)).toEqual({ trials: 1, hits: 0, wrong: 0, timeouts: 1 });
  });

  it("freezes after a hit until the trial ends, then reports the outcome once", () => {
    const game = createGame();
    onPrediction(game, input(0.5, "rest"), 0.25, STEP);
    onPrediction(game, input(1, "right"), 0.25, STEP);
    advance(game, STEP);
    expect(game.lastOutcome).toBe("hit");
    expect(game.outcomeCount).toBe(1);
    onPrediction(game, input(0, "right"), 0.25, STEP);
    advance(game, STEP);
    expect(game.x).toBe(1);
    onPrediction(game, input(0.5, "rest"), 0.25, STEP);
    expect(game.outcomeCount).toBe(1);
    expect(game.target).toBeNull();
  });

  it("does not score a trial that was already running when the page connected", () => {
    const game = createGame();
    // First message ever: a "right" trial is in progress, its start was not seen.
    for (let k = 0; k < 6; k += 1) onPrediction(game, input(1, "right"), 1, STEP);
    expect(game.target).toBeNull();
    expect(game.x).toBe(1); // the cursor still follows the decoder
    onPrediction(game, input(0.5, "rest"), 1, STEP);
    expect(score(game)).toEqual({ trials: 0, hits: 0, wrong: 0, timeouts: 0 });
    // The next trial is seen from its start and counts.
    for (let k = 0; k < 5; k += 1) onPrediction(game, input(0, "left"), 1, STEP);
    onPrediction(game, input(0.5, "rest"), 1, STEP);
    expect(score(game)).toEqual({ trials: 1, hits: 1, wrong: 0, timeouts: 0 });
    expect(game.hitTimes[0]).toBeCloseTo(1.0);
  });

  it("drops an open trial when the stream is interrupted and skips the one it returns into", () => {
    const game = createGame();
    onPrediction(game, input(0.5, "rest"), 1, STEP);
    onPrediction(game, input(0.9, "right"), 1, STEP);
    expect(game.trials).toBe(1);
    interrupt(game);
    expect(score(game)).toEqual({ trials: 0, hits: 0, wrong: 0, timeouts: 0 });
    expect(game.v).toBe(0);
    // Back in the middle of some trial: unknown how long it has been running.
    for (let k = 0; k < 6; k += 1) onPrediction(game, input(1, "right"), 1, STEP);
    onPrediction(game, input(0.5, "rest"), 1, STEP);
    expect(score(game)).toEqual({ trials: 0, hits: 0, wrong: 0, timeouts: 0 });
  });

  it("keeps an outcome that was already decided when the stream is interrupted", () => {
    const game = createGame();
    onPrediction(game, input(0.5, "rest"), 0.25, STEP);
    onPrediction(game, input(1, "right"), 0.25, STEP);
    settle(game);
    interrupt(game);
    expect(score(game)).toEqual({ trials: 1, hits: 1, wrong: 0, timeouts: 0 });
  });

  it("resets the score but lets a running trial finish and count", () => {
    const game = createGame();
    onPrediction(game, input(0.5, "rest"), 1, STEP);
    for (let k = 0; k < 5; k += 1) onPrediction(game, input(1, "right"), 1, STEP);
    onPrediction(game, input(0.5, "rest"), 1, STEP);
    onPrediction(game, input(0, "left"), 1, STEP);
    expect(score(game)).toEqual({ trials: 2, hits: 1, wrong: 0, timeouts: 0 });
    resetCounters(game);
    expect(score(game)).toEqual({ trials: 1, hits: 0, wrong: 0, timeouts: 0 });
    expect(game.hitTimes).toEqual([]);
    expect(game.target).toBe("left");
    for (let k = 0; k < 4; k += 1) onPrediction(game, input(0, "left"), 1, STEP);
    expect(score(game)).toEqual({ trials: 1, hits: 1, wrong: 0, timeouts: 0 });
    resetCounters(game); // the trial is decided: nothing is carried over
    expect(game.trials).toBe(0);
  });

  it("computes medians", () => {
    expect(median([])).toBeNaN();
    expect(median([3])).toBe(3);
    expect(median([4, 1, 3])).toBe(3);
    expect(median([4, 1, 3, 2])).toBe(2.5);
  });
});
