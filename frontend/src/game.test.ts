import { describe, expect, it } from "vitest";
import fixture from "./fixtures/game_cases.json";
import { advance, createGame, finish, median, onPrediction, type GameInput } from "./game";
import type { TrueLabel } from "./types";

function input(pRight: number, truth: TrueLabel, threshold: number): GameInput {
  return {
    probs: { left: 1 - pRight, right: pRight },
    active: Math.max(pRight, 1 - pRight) >= threshold,
    true: truth,
  };
}

/** Replay a case; `framesPerStep` splits each 0.25 s step into animation frames. */
function play(c: (typeof fixture.cases)[number], framesPerStep: number) {
  const game = createGame();
  c.p_right.forEach((p, k) => {
    onPrediction(game, input(p, c.true[k] as TrueLabel, c.threshold), c.travel_time_s);
    for (let f = 0; f < framesPerStep; f += 1) advance(game, c.step_s / framesPerStep);
  });
  finish(game);
  return game;
}

describe("target game", () => {
  it.each(fixture.cases.map((c) => [c.name, c] as const))(
    "matches the Python simulation: %s",
    (_name, c) => {
      const game = play(c, 1);
      expect({
        trials: game.trials,
        hits: game.hits,
        wrong: game.wrong,
        timeouts: game.timeouts,
      }).toEqual(c.expected);
      expect(game.hitTimes.length).toBe(c.hit_times_s.length);
      game.hitTimes.forEach((t, i) => expect(t).toBeCloseTo(c.hit_times_s[i] ?? NaN, 9));
    },
  );

  it("gives the same outcomes when a step is drawn in many frames", () => {
    for (const c of fixture.cases) {
      const game = play(c, 15); // 60 fps
      expect([game.trials, game.hits, game.wrong, game.timeouts], c.name).toEqual([
        c.expected.trials,
        c.expected.hits,
        c.expected.wrong,
        c.expected.timeouts,
      ]);
      game.hitTimes.forEach((t, i) => expect(t).toBeCloseTo(c.hit_times_s[i] ?? NaN, 6));
    }
  });

  it("returns the cursor to the centre at the cue and aims at the cued wall", () => {
    const game = createGame();
    onPrediction(game, input(0.0, "rest", 0.6), 1);
    advance(game, 0.5);
    expect(game.x).toBeCloseTo(-0.5);
    expect(game.target).toBeNull();
    onPrediction(game, input(0.9, "right", 0.6), 1);
    expect(game.x).toBe(0);
    expect(game.target).toBe("right");
    expect(game.v).toBeCloseTo(0.8);
  });

  it("holds still below the threshold and stays inside the walls during rest", () => {
    const game = createGame();
    onPrediction(game, input(0.55, "left", 0.6), 1);
    advance(game, 1);
    expect(game.x).toBe(0);
    onPrediction(game, input(1, null, 0.6), 1);
    advance(game, 5);
    expect(game.x).toBe(1);
    expect(game.hits + game.wrong).toBe(0); // no target, so the wall is not a hit
  });

  it("freezes after a hit until the trial ends, then reports the outcome once", () => {
    const game = createGame();
    onPrediction(game, input(1, "right", 0.6), 0.5);
    advance(game, 0.5);
    expect(game.lastOutcome).toBe("hit");
    expect(game.outcomeCount).toBe(1);
    onPrediction(game, input(0, "right", 0.6), 0.5);
    advance(game, 1);
    expect(game.x).toBe(1);
    onPrediction(game, input(0.5, "rest", 0.6), 0.5);
    expect(game.outcomeCount).toBe(1);
    expect(game.target).toBeNull();
  });

  it("computes medians", () => {
    expect(median([])).toBeNaN();
    expect(median([3])).toBe(3);
    expect(median([4, 1, 3])).toBe(3);
    expect(median([4, 1, 3, 2])).toBe(2.5);
  });
});
