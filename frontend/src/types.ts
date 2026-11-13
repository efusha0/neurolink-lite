/** WebSocket message schema shared with the Python server (see realtime/server.py). */

export type ClassName = "left" | "right";
export type TrueLabel = ClassName | "rest" | null;

export interface PredictionMessage {
  type: "prediction";
  /** Stream time in seconds since replay start. */
  t: number;
  probs: Record<ClassName, number>;
  pred: ClassName;
  true: TrueLabel;
  confidence: number;
  /** False when confidence is below the threshold: the cursor must not move. */
  active: boolean;
  /** LSL timestamp (s) of the newest sample in the analysis window. */
  sample_time: number;
  latency_ms: { stream_to_pred: number; pred_to_ws: number };
}

export interface EegMessage {
  type: "eeg";
  t: number;
  channels: string[];
  /** One array of samples per channel, already decimated for display. */
  samples: number[][];
}

export type ServerMessage = PredictionMessage | EegMessage;
