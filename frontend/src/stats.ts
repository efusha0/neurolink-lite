/** Running honesty counters for the live session: how often is the decoder right? */

import type { PredictionMessage } from "./types";

export interface SessionStats {
  /** Predictions whose window ended during a left/right trial. */
  trialWindows: number;
  trialCorrect: number;
  /** Trial windows above the confidence threshold (cursor moving). */
  trialActive: number;
  trialActiveCorrect: number;
  /** Predictions whose window ended during rest. */
  restWindows: number;
  /** Rest windows above the threshold: the cursor moved although nothing was imagined. */
  restActive: number;
  /** Predictions without a known true state. */
  unlabelled: number;
}

export type StatsInput = Pick<PredictionMessage, "pred" | "true" | "active">;

export function createStats(): SessionStats {
  return {
    trialWindows: 0,
    trialCorrect: 0,
    trialActive: 0,
    trialActiveCorrect: 0,
    restWindows: 0,
    restActive: 0,
    unlabelled: 0,
  };
}

export function updateStats(stats: SessionStats, msg: StatsInput): void {
  if (msg.true === "left" || msg.true === "right") {
    const correct = msg.pred === msg.true;
    stats.trialWindows += 1;
    if (correct) stats.trialCorrect += 1;
    if (msg.active) {
      stats.trialActive += 1;
      if (correct) stats.trialActiveCorrect += 1;
    }
  } else if (msg.true === "rest") {
    stats.restWindows += 1;
    if (msg.active) stats.restActive += 1;
  } else {
    stats.unlabelled += 1;
  }
}

/** `part / whole`, or NaN when there is nothing to divide by. */
export function share(part: number, whole: number): number {
  return whole > 0 ? part / whole : Number.NaN;
}

export function formatPercent(value: number, digits = 0): string {
  return Number.isFinite(value) ? `${(100 * value).toFixed(digits)} %` : "–";
}
