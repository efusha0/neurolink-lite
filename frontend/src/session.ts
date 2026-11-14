/**
 * Which part of the replay do the counters on the page cover?
 *
 * The streamer replays one recorded run in a loop and numbers the passes. Adding the
 * passes up would show a score ("24 / 52 hits") that cannot be compared with anything:
 * the offline result is for one pass of the run. So the counters cover one pass at a
 * time, the scores of earlier passes are kept in a short history, and a pass that was not
 * seen from its start, or was interrupted, is flagged as incomplete.
 *
 * Pure bookkeeping: `main.ts` resets the game and the counters when told to.
 */

import type { PredictionMessage } from "./types";

/** More stream time than this many decoder steps between two predictions is a gap. */
export const GAP_STEPS = 4;
export const HISTORY_LENGTH = 5;

export interface Score {
  trials: number;
  hits: number;
  wrong: number;
  timeouts: number;
}

export interface PassScore extends Score {
  pass: number;
  /** False when the page saw only part of this pass. */
  complete: boolean;
}

export interface Session {
  /** Replay pass of the last prediction; null when the server does not number passes. */
  pass: number | null;
  /** True when the counters cover the current pass from its start, without a gap. */
  complete: boolean;
  /** Stream time (s) of the last prediction. */
  lastT: number | null;
  /** Scores of earlier passes, oldest first. */
  history: PassScore[];
}

export interface Transition {
  /** Predictions were lost or the stream restarted: open trials cannot be scored. */
  gap: boolean;
  /** A new pass began: its counters must start from zero. */
  newPass: boolean;
}

export type SessionInput = Pick<PredictionMessage, "t" | "replay_pass">;

export function createSession(): Session {
  return { pass: null, complete: false, lastT: null, history: [] };
}

/**
 * Register one prediction, before the game sees it. `score` is the score so far: when a
 * new pass begins it is filed under the pass that just ended.
 */
export function observe(
  session: Session,
  msg: SessionInput,
  stepS: number,
  score: Score,
): Transition {
  const first = session.lastT === null;
  const dt = first ? 0 : msg.t - (session.lastT as number);
  const gap = !first && (dt <= 0 || dt > GAP_STEPS * stepS);
  session.lastT = msg.t;

  const newPass =
    msg.replay_pass !== null && session.pass !== null && msg.replay_pass !== session.pass;
  if (newPass) {
    if (score.trials > 0) {
      const { trials, hits, wrong, timeouts } = score;
      session.history.push({
        pass: session.pass as number,
        complete: session.complete,
        trials,
        hits,
        wrong,
        timeouts,
      });
      if (session.history.length > HISTORY_LENGTH) session.history.shift();
    }
    // Seen from its first prediction on, unless we arrived through a gap.
    session.complete = !gap;
  } else if (gap) {
    session.complete = false;
  }
  if (msg.replay_pass !== null) session.pass = msg.replay_pass;
  return { gap, newPass };
}

/** The counters were zeroed by hand: they no longer cover the whole pass. */
export function markIncomplete(session: Session): void {
  session.complete = false;
}

export function describePass(session: Session): string {
  if (session.pass === null) return "";
  return (
    `Replay pass ${session.pass} of the same recording` +
    (session.complete ? "." : ", seen only in part: fewer trials than a full pass.")
  );
}

export function describeHistory(session: Session): string {
  if (session.history.length === 0) return "";
  const items = session.history.map(
    (p) => `#${p.pass} ${p.hits}/${p.trials}${p.complete ? "" : " (partial)"}`,
  );
  return `Earlier passes: ${items.join(" · ")}`;
}
