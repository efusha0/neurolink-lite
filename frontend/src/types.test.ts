import { describe, expect, it } from "vitest";
import type { PredictionMessage } from "./types";

describe("message schema", () => {
  it("accepts a well-formed prediction message", () => {
    const msg: PredictionMessage = {
      type: "prediction",
      t: 1.25,
      probs: { left: 0.3, right: 0.7 },
      pred: "right",
      true: "right",
      confidence: 0.7,
      active: true,
      sample_time: 1234.5,
      latency_ms: { stream_to_pred: 10, pred_to_ws: 1 },
    };
    expect(msg.probs.left + msg.probs.right).toBeCloseTo(1);
  });
});
