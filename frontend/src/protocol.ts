/** Parse and validate what arrives on the WebSocket; anything malformed is dropped. */

import type { ClassName, EegMessage, PredictionMessage, ServerMessage, TrueLabel } from "./types";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isClassName(value: unknown): value is ClassName {
  return value === "left" || value === "right";
}

function isTrueLabel(value: unknown): value is TrueLabel {
  return value === null || value === "rest" || isClassName(value);
}

function parsePrediction(m: Record<string, unknown>): PredictionMessage | null {
  const { probs, latency_ms: latency } = m;
  if (!isRecord(probs) || !isFiniteNumber(probs.left) || !isFiniteNumber(probs.right)) return null;
  if (!isRecord(latency)) return null;
  if (!isFiniteNumber(latency.stream_to_pred) || !isFiniteNumber(latency.pred_to_ws)) return null;
  if (!isFiniteNumber(m.t) || !isFiniteNumber(m.confidence) || !isFiniteNumber(m.sample_time)) {
    return null;
  }
  if (!isClassName(m.pred) || !isTrueLabel(m.true) || typeof m.active !== "boolean") return null;
  // Optional, so that a page still works against a server that does not send it.
  const replayPass = m.replay_pass ?? null;
  if (replayPass !== null && !(isFiniteNumber(replayPass) && Number.isInteger(replayPass))) {
    return null;
  }
  return {
    type: "prediction",
    t: m.t,
    probs: { left: probs.left, right: probs.right },
    pred: m.pred,
    true: m.true,
    confidence: m.confidence,
    active: m.active,
    sample_time: m.sample_time,
    replay_pass: replayPass,
    latency_ms: { stream_to_pred: latency.stream_to_pred, pred_to_ws: latency.pred_to_ws },
  };
}

function parseEeg(m: Record<string, unknown>): EegMessage | null {
  const { channels, samples } = m;
  if (!isFiniteNumber(m.t) || !Array.isArray(channels) || !Array.isArray(samples)) return null;
  if (channels.length !== samples.length) return null;
  if (!channels.every((c): c is string => typeof c === "string")) return null;
  const n = Array.isArray(samples[0]) ? samples[0].length : -1;
  for (const row of samples) {
    if (!Array.isArray(row) || row.length !== n || !row.every(isFiniteNumber)) return null;
  }
  return { type: "eeg", t: m.t, channels, samples: samples as number[][] };
}

export function parseMessage(text: string): ServerMessage | null {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    return null;
  }
  if (!isRecord(raw)) return null;
  if (raw.type === "prediction") return parsePrediction(raw);
  if (raw.type === "eeg") return parseEeg(raw);
  return null;
}
