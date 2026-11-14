import { describe, expect, it } from "vitest";
import { createGame, interrupt, onPrediction, resetCounters, settle, type GameInput } from "./game";
import {
  HISTORY_LENGTH,
  createSession,
  describeHistory,
  describePass,
  markIncomplete,
  observe,
  type Score,
} from "./session";
import type { TrueLabel } from "./types";

const STEP = 0.25;
const NONE: Score = { trials: 0, hits: 0, wrong: 0, timeouts: 0 };

describe("replay session", () => {
  it("flags the pass the page joined as incomplete and the next one as complete", () => {
    const s = createSession();
    expect(observe(s, { t: 40, replay_pass: 1 }, STEP, NONE)).toEqual({ gap: false, newPass: false });
    expect(s.pass).toBe(1);
    expect(s.complete).toBe(false);
    expect(describePass(s)).toContain("seen only in part");

    const first: Score = { trials: 9, hits: 7, wrong: 0, timeouts: 2 };
    expect(observe(s, { t: 40.25, replay_pass: 2 }, STEP, first)).toEqual({
      gap: false,
      newPass: true,
    });
    expect(s.pass).toBe(2);
    expect(s.complete).toBe(true);
    expect(s.history).toEqual([{ pass: 1, complete: false, ...first }]);
    expect(describePass(s)).toBe("Replay pass 2 of the same recording.");
    expect(describeHistory(s)).toBe("Earlier passes: #1 7/9 (partial)");
  });

  it("detects lost predictions, a restarted stream and a pass begun unseen", () => {
    const s = createSession();
    observe(s, { t: 10, replay_pass: 1 }, STEP, NONE);
    observe(s, { t: 10.25, replay_pass: 2 }, STEP, NONE);
    expect(s.complete).toBe(true);
    // One late prediction is fine, more than a second of silence is not.
    expect(observe(s, { t: 11.0, replay_pass: 2 }, STEP, NONE).gap).toBe(false);
    expect(observe(s, { t: 12.5, replay_pass: 2 }, STEP, NONE).gap).toBe(true);
    expect(s.complete).toBe(false);

    // The server restarted: its clock starts again.
    const t = observe(s, { t: 0.5, replay_pass: 3 }, STEP, { ...NONE, trials: 4, hits: 4 });
    expect(t).toEqual({ gap: true, newPass: true });
    expect(s.complete).toBe(false); // pass 3 was not seen from its start
    expect(s.history.at(-1)).toMatchObject({ pass: 2, complete: false, hits: 4 });
  });

  it("works without pass numbers and adopts them when they appear", () => {
    const s = createSession();
    for (let k = 0; k < 5; k += 1) {
      expect(observe(s, { t: k * STEP, replay_pass: null }, STEP, NONE).newPass).toBe(false);
    }
    expect(s.pass).toBeNull();
    expect(describePass(s)).toBe("");
    expect(describeHistory(s)).toBe("");
    expect(observe(s, { t: 5 * STEP, replay_pass: 4 }, STEP, NONE).newPass).toBe(false);
    expect(s.pass).toBe(4);
    // A message without the number in between does not end the pass.
    expect(observe(s, { t: 6 * STEP, replay_pass: null }, STEP, NONE).newPass).toBe(false);
    expect(s.pass).toBe(4);
  });

  it("keeps a short history, skips passes without trials, and notes a manual reset", () => {
    const s = createSession();
    let t = 0;
    for (let pass = 1; pass <= HISTORY_LENGTH + 3; pass += 1) {
      const trials = pass === 2 ? 0 : 15;
      observe(s, { t: (t += STEP), replay_pass: pass }, STEP, { ...NONE, trials, hits: pass });
    }
    expect(s.history.length).toBe(HISTORY_LENGTH);
    expect(s.history.map((p) => p.pass)).not.toContain(2);
    expect(s.history.at(-1)?.pass).toBe(HISTORY_LENGTH + 2);
    expect(s.complete).toBe(true);
    markIncomplete(s);
    expect(s.complete).toBe(false);
  });

  it("scores every pass of a looped recording alike, however long the page stays open", () => {
    // One pass: rest, a "right" trial the decoder gets, rest, a "left" trial it misses.
    const pass: Array<[number, TrueLabel]> = [
      [0.5, "rest"], [1, "right"], [1, "right"], [1, "right"], [1, "right"], [1, "right"],
      [0.5, "rest"], [0.5, "left"], [0.5, "left"], [0.5, "rest"],
    ]; // prettier-ignore
    const game = createGame();
    const s = createSession();
    let k = 0;
    const feed = (replayPass: number, [p, truth]: [number, TrueLabel]): void => {
      const msg: GameInput = {
        probs: { left: 1 - p, right: p },
        active: Math.max(p, 1 - p) >= 0.6,
        true: truth,
      };
      // Same order as main.ts.
      settle(game);
      const { gap, newPass } = observe(s, { t: (k += 1) * STEP, replay_pass: replayPass }, STEP, game);
      if (gap) interrupt(game);
      if (newPass) resetCounters(game);
      onPrediction(game, msg, 1, STEP);
    };

    // The page joins pass 1 in the middle of its first trial.
    for (const step of pass.slice(3)) feed(1, step);
    for (let n = 2; n <= 4; n += 1) for (const step of pass) feed(n, step);
    feed(5, pass[0] as [number, TrueLabel]);

    expect(s.history).toEqual([
      { pass: 1, complete: false, trials: 1, hits: 0, wrong: 0, timeouts: 1 },
      { pass: 2, complete: true, trials: 2, hits: 1, wrong: 0, timeouts: 1 },
      { pass: 3, complete: true, trials: 2, hits: 1, wrong: 0, timeouts: 1 },
      { pass: 4, complete: true, trials: 2, hits: 1, wrong: 0, timeouts: 1 },
    ]);
    expect(game.trials).toBe(0); // pass 5 starts from zero
    expect(describeHistory(s)).toBe("Earlier passes: #1 0/1 (partial) · #2 1/2 · #3 1/2 · #4 1/2");
  });
});
