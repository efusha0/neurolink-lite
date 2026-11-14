/** Small time-indexed buffers behind the EEG view. */

import type { EegMessage, TrueLabel } from "./types";

/** Fixed-capacity ring of (time, value) samples; the oldest samples are overwritten. */
export class TraceBuffer {
  private readonly times: Float64Array;
  private readonly values: Float32Array;
  private start = 0;
  private count = 0;

  constructor(readonly capacity: number) {
    if (capacity <= 0) throw new Error("capacity must be positive");
    this.times = new Float64Array(capacity);
    this.values = new Float32Array(capacity);
  }

  get length(): number {
    return this.count;
  }

  push(time: number, value: number): void {
    const index = (this.start + this.count) % this.capacity;
    this.times[index] = time;
    this.values[index] = value;
    if (this.count < this.capacity) this.count += 1;
    else this.start = (this.start + 1) % this.capacity;
  }

  /** Visit samples from oldest to newest. */
  forEach(visit: (time: number, value: number) => void): void {
    for (let i = 0; i < this.count; i += 1) {
      const index = (this.start + i) % this.capacity;
      visit(this.times[index] ?? 0, this.values[index] ?? 0);
    }
  }

  /** Standard deviation of the stored values (0 when empty). */
  std(): number {
    if (this.count === 0) return 0;
    let sum = 0;
    let sumSq = 0;
    this.forEach((_t, v) => {
      sum += v;
      sumSq += v * v;
    });
    const mean = sum / this.count;
    return Math.sqrt(Math.max(0, sumSq / this.count - mean * mean));
  }

  clear(): void {
    this.start = 0;
    this.count = 0;
  }
}

/**
 * Spread the samples of one EEG message over time: the message carries the time of its
 * last sample, the others are `1 / rate` apart before it.
 */
export function pushEeg(
  buffers: Map<string, TraceBuffer>,
  msg: EegMessage,
  rateHz: number,
  capacity: number,
): void {
  msg.channels.forEach((name, c) => {
    let buffer = buffers.get(name);
    if (buffer === undefined) {
      buffer = new TraceBuffer(capacity);
      buffers.set(name, buffer);
    }
    const row = msg.samples[c] ?? [];
    row.forEach((value, i) => {
      buffer.push(msg.t - (row.length - 1 - i) / rateHz, value);
    });
  });
}

export interface StateSegment {
  /** Stream time at which this state was first seen. */
  t: number;
  state: TrueLabel;
}

/** The true state over time, as a list of change points (for the EEG background). */
export class StateTimeline {
  readonly segments: StateSegment[] = [];

  push(t: number, state: TrueLabel): void {
    const last = this.segments[this.segments.length - 1];
    if (last === undefined || last.state !== state) this.segments.push({ t, state });
  }

  /** Drop change points that ended before `t` (keeps the one still in effect at `t`). */
  prune(t: number): void {
    while (this.segments.length > 1 && (this.segments[1]?.t ?? Infinity) <= t) {
      this.segments.shift();
    }
  }

  stateAt(t: number): TrueLabel {
    let state: TrueLabel = null;
    for (const segment of this.segments) {
      if (segment.t > t) break;
      state = segment.state;
    }
    return state;
  }
}
