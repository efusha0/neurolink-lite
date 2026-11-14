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
  /** Stream time (s) of the last sample in this message. */
  t: number;
  channels: string[];
  /** One array of samples per channel, already decimated for display. */
  samples: number[][];
}

export type ServerMessage = PredictionMessage | EegMessage;

/** Reference numbers measured offline, shown next to the live session for honesty. */
export interface ReferenceNumbers {
  demo_run_hit_rate?: number;
  demo_run_trials?: number;
  mean_hit_rate?: number;
  mean_acc_all_smooth?: number;
  n_subjects?: number;
}

/** `GET /info`: where the model came from and how the UI should be set up. */
export interface InfoResponse {
  class_names: string[];
  candidate?: string;
  demo_subject?: number;
  threshold?: number;
  window_s?: number;
  step_s?: number;
  n_train_subjects?: number;
  /** Seconds from the centre to a wall at full confidence (chosen on runs 4 and 8). */
  cursor_travel_time_s?: number;
  /** Sampling rate (Hz) of the waveform samples in `eeg` messages. */
  display_rate_hz?: number;
  reference?: ReferenceNumbers;
}
