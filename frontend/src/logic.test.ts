import { describe, expect, it } from "vitest";
import { StateTimeline, TraceBuffer, pushEeg } from "./buffers";
import { parseMessage } from "./protocol";
import { createStats, formatPercent, share, updateStats } from "./stats";

const prediction = {
  type: "prediction",
  t: 2.0,
  probs: { left: 0.25, right: 0.75 },
  pred: "right",
  true: "right",
  confidence: 0.75,
  active: true,
  sample_time: 102.0,
  replay_pass: null,
  latency_ms: { stream_to_pred: 12.5, pred_to_ws: 0.4 },
};

describe("protocol", () => {
  it("parses the prediction message built by the server", () => {
    // Same payload as tests/test_realtime_server.py::test_message_builders_match_frontend_schema.
    expect(parseMessage(JSON.stringify(prediction))).toEqual(prediction);
    const unlabelled = parseMessage(JSON.stringify({ ...prediction, true: null }));
    expect(unlabelled?.type === "prediction" && unlabelled.true).toBeNull();
  });

  it("reads the replay pass and tolerates a server that does not send it", () => {
    const third = parseMessage(JSON.stringify({ ...prediction, replay_pass: 3 }));
    expect(third?.type === "prediction" && third.replay_pass).toBe(3);
    const { replay_pass: _omitted, ...old } = prediction;
    const parsed = parseMessage(JSON.stringify(old));
    expect(parsed?.type === "prediction" && parsed.replay_pass).toBeNull();
    expect(parseMessage(JSON.stringify({ ...prediction, replay_pass: "3" }))).toBeNull();
    expect(parseMessage(JSON.stringify({ ...prediction, replay_pass: 1.5 }))).toBeNull();
  });

  it("parses the EEG message built by the server", () => {
    const eeg = {
      type: "eeg",
      t: 1.0,
      channels: ["C3", "C4"],
      samples: [
        [1.0, 3.0, 5.0],
        [2.0, 4.0, 6.0],
      ],
    };
    expect(parseMessage(JSON.stringify(eeg))).toEqual(eeg);
  });

  it("drops malformed messages instead of throwing", () => {
    const bad = [
      "not json",
      "[1, 2]",
      JSON.stringify({ type: "other" }),
      JSON.stringify({ ...prediction, pred: "up" }),
      JSON.stringify({ ...prediction, true: "foot" }),
      JSON.stringify({ ...prediction, active: "yes" }),
      JSON.stringify({ ...prediction, probs: { left: 0.5 } }),
      JSON.stringify({ ...prediction, latency_ms: null }),
      JSON.stringify({ type: "eeg", t: 1, channels: ["C3"], samples: [[1], [2]] }),
      JSON.stringify({ type: "eeg", t: 1, channels: ["C3", "C4"], samples: [[1, 2], [3]] }),
      JSON.stringify({ type: "eeg", t: 1, channels: ["C3"], samples: [["a"]] }),
    ];
    for (const text of bad) expect(parseMessage(text), text).toBeNull();
  });
});

describe("session stats", () => {
  it("counts accuracy on trial windows and cursor movement during rest", () => {
    const stats = createStats();
    const feed = [
      { pred: "left", true: "left", active: true },
      { pred: "right", true: "left", active: true },
      { pred: "right", true: "right", active: false },
      { pred: "right", true: "right", active: true },
      { pred: "left", true: "rest", active: true },
      { pred: "left", true: "rest", active: false },
      { pred: "left", true: null, active: true },
    ] as const;
    for (const m of feed) updateStats(stats, m);
    expect(stats).toEqual({
      trialWindows: 4,
      trialCorrect: 3,
      trialActive: 3,
      trialActiveCorrect: 2,
      restWindows: 2,
      restActive: 1,
      unlabelled: 1,
    });
    expect(share(stats.trialCorrect, stats.trialWindows)).toBe(0.75);
    expect(share(1, 0)).toBeNaN();
    expect(formatPercent(0.754)).toBe("75 %");
    expect(formatPercent(0.754, 1)).toBe("75.4 %");
    expect(formatPercent(Number.NaN)).toBe("–");
  });
});

describe("buffers", () => {
  it("keeps the newest samples in order", () => {
    const buffer = new TraceBuffer(3);
    for (let i = 0; i < 5; i += 1) buffer.push(i, 10 * i);
    const seen: Array<[number, number]> = [];
    buffer.forEach((t, v) => seen.push([t, v]));
    expect(seen).toEqual([
      [2, 20],
      [3, 30],
      [4, 40],
    ]);
    expect(buffer.length).toBe(3);
    expect(buffer.std()).toBeCloseTo(Math.sqrt(200 / 3));
    buffer.clear();
    expect(buffer.length).toBe(0);
    expect(buffer.std()).toBe(0);
    expect(() => new TraceBuffer(0)).toThrow();
  });

  it("spreads the samples of an EEG message backwards from its timestamp", () => {
    const buffers = new Map<string, TraceBuffer>();
    pushEeg(buffers, { type: "eeg", t: 1.0, channels: ["C3", "C4"], samples: [[1, 2], [3, 4]] }, 40, 8);
    const c3: Array<[number, number]> = [];
    buffers.get("C3")?.forEach((t, v) => c3.push([t, v]));
    expect(c3).toEqual([
      [0.975, 1],
      [1.0, 2],
    ]);
    expect(buffers.get("C4")?.length).toBe(2);
  });

  it("tracks the true state as change points", () => {
    const timeline = new StateTimeline();
    timeline.push(0, "rest");
    timeline.push(0.25, "rest");
    timeline.push(4.25, "left");
    timeline.push(8.5, "rest");
    expect(timeline.segments).toHaveLength(3);
    expect(timeline.stateAt(-1)).toBeNull();
    expect(timeline.stateAt(4.0)).toBe("rest");
    expect(timeline.stateAt(5.0)).toBe("left");
    timeline.prune(5.0);
    expect(timeline.segments[0]).toEqual({ t: 4.25, state: "left" });
    expect(timeline.stateAt(5.0)).toBe("left");
  });
});
