/** Wires the WebSocket stream to the game, the counters and the two canvases. */

import { StateTimeline, TraceBuffer, pushEeg } from "./buffers";
import { connect, websocketUrl, type ConnectionStatus } from "./connection";
import {
  advance,
  createGame,
  interrupt,
  median,
  onPrediction,
  resetCounters,
  settle,
} from "./game";
import { COLORS, classColor, drawArena, drawEeg, fitCanvas } from "./render";
import {
  createSession,
  describeHistory,
  describePass,
  markIncomplete,
  observe,
} from "./session";
import { createStats, formatPercent, share, updateStats } from "./stats";
import type { InfoResponse, PredictionMessage, ServerMessage } from "./types";

/** Used only until `/info` answers; the server's values are authoritative. */
const DEFAULTS = { travelTimeS: 1.0, stepS: 0.25, displayRateHz: 40 };
const EEG_SPAN_S = 8;
const STALL_AFTER_MS = 3000;
const LATENCY_WINDOW = 40;

function byId<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (element === null) throw new Error(`missing element #${id}`);
  return element as T;
}

const el = {
  status: byId("status"),
  provenance: byId("provenance"),
  arena: byId<HTMLCanvasElement>("arena"),
  eeg: byId<HTMLCanvasElement>("eeg"),
  probLeft: byId("prob-left"),
  probRight: byId("prob-right"),
  probFill: byId("prob-fill"),
  thresholdLeft: byId("prob-threshold-left"),
  thresholdRight: byId("prob-threshold-right"),
  thresholdValue: byId("threshold-value"),
  trueLabel: byId("true-label"),
  predLabel: byId("pred-label"),
  cursorLabel: byId("cursor-label"),
  verdict: byId("verdict"),
  gameHits: byId("game-hits"),
  gameTrials: byId("game-trials"),
  gameDetail: byId("game-detail"),
  gamePass: byId("game-pass"),
  gameHistory: byId("game-history"),
  accTitle: byId("acc-title"),
  accTrial: byId("acc-trial"),
  accDetail: byId("acc-detail"),
  restActive: byId("rest-active"),
  latency: byId("latency"),
  reference: byId("reference"),
  reset: byId<HTMLButtonElement>("reset"),
};

const game = createGame();
const session = createSession();
let stats = createStats();
const buffers = new Map<string, TraceBuffer>();
const timeline = new StateTimeline();
let channels: string[] = [];
let lastPrediction: PredictionMessage | null = null;
let lastPredictionAt = -Infinity;
let lastEegT = 0;
let lastEegAt = -Infinity;
let lastOutcomeCount = 0;
let lastOutcomeAt = -Infinity;
let latencies: number[] = [];
let connection: ConnectionStatus = "connecting";
let travelTimeS = DEFAULTS.travelTimeS;
let stepS = DEFAULTS.stepS;
let displayRateHz = DEFAULTS.displayRateHz;
let infoLoaded = false;

// ---------------------------------------------------------------------------- /info
function showInfo(info: InfoResponse): void {
  if (typeof info.cursor_travel_time_s === "number") travelTimeS = info.cursor_travel_time_s;
  if (typeof info.step_s === "number" && info.step_s > 0) stepS = info.step_s;
  if (typeof info.display_rate_hz === "number") displayRateHz = info.display_rate_hz;
  if (typeof info.threshold === "number") {
    const offset = 50 * (2 * info.threshold - 1); // |p_right - p_left| at the threshold
    el.thresholdLeft.style.left = `${50 - offset}%`;
    el.thresholdRight.style.left = `${50 + offset}%`;
    el.thresholdLeft.style.display = el.thresholdRight.style.display = "block";
    el.thresholdValue.textContent = info.threshold.toFixed(2);
  }
  const parts: string[] = [];
  if (typeof info.demo_subject === "number") {
    parts.push(`Subject S${String(info.demo_subject).padStart(3, "0")}.`);
  }
  if (typeof info.n_train_subjects === "number") {
    parts.push(
      `The model was trained on ${info.n_train_subjects} other people and has never seen this subject.`,
    );
  }
  el.provenance.textContent = parts.join(" ");

  const ref = info.reference;
  const notes: string[] = [];
  if (ref !== undefined) {
    if (typeof ref.demo_run_hit_rate === "number") {
      const rate = formatPercent(ref.demo_run_hit_rate);
      const trials = ref.demo_run_trials;
      notes.push(
        "This subject is a best case: in the offline simulation one full pass of this run gives " +
          (typeof trials === "number"
            ? `${Math.round(ref.demo_run_hit_rate * trials)} of ${trials} hits (${rate}).`
            : `${rate} hits.`),
      );
      notes.push(
        "The run is replayed in a loop; the counters above cover one pass and restart with the next.",
      );
    }
    if (typeof ref.mean_hit_rate === "number" && typeof ref.n_subjects === "number") {
      let text = `Average of ${ref.n_subjects} subjects: ${formatPercent(ref.mean_hit_rate)} hits`;
      if (typeof ref.mean_acc_all_smooth === "number") {
        text += `, ${formatPercent(ref.mean_acc_all_smooth)} of cue windows decoded correctly (chance 50 %)`;
      }
      notes.push(`${text}.`);
    }
  }
  el.reference.textContent = notes.join(" ");
}

async function loadInfo(): Promise<void> {
  if (infoLoaded) return;
  try {
    const response = await fetch("/info");
    if (!response.ok) return;
    showInfo((await response.json()) as InfoResponse);
    infoLoaded = true;
  } catch {
    // Server not up yet: the next (re)connection tries again.
  }
}

// ------------------------------------------------------------------------- messages
function showPrediction(msg: PredictionMessage): void {
  const d = msg.probs.right - msg.probs.left;
  el.probLeft.textContent = `Left ${formatPercent(msg.probs.left)}`;
  el.probRight.textContent = `Right ${formatPercent(msg.probs.right)}`;
  el.probFill.style.left = `${d >= 0 ? 50 : 50 + 50 * d}%`;
  el.probFill.style.width = `${50 * Math.abs(d)}%`;
  el.probFill.style.backgroundColor = msg.active ? classColor(msg.pred) : COLORS.rest;

  const truth = msg.true;
  el.trueLabel.textContent = truth === null ? "–" : truth.toUpperCase();
  el.trueLabel.style.color =
    truth === "left" || truth === "right" ? classColor(truth) : COLORS.muted;
  el.predLabel.textContent = `${msg.pred.toUpperCase()} ${formatPercent(msg.confidence)}`;
  el.predLabel.style.color = classColor(msg.pred);
  el.cursorLabel.textContent = msg.active ? "MOVING" : "HELD";
  el.cursorLabel.style.color = msg.active ? COLORS.text : COLORS.muted;

  let verdict = "True state unknown.";
  let tone = "neutral";
  if (truth === "left" || truth === "right") {
    const correct = msg.pred === truth;
    if (!msg.active) verdict = correct ? "Right class, but below the threshold." : "Below the threshold.";
    else verdict = correct ? "Correct: moving towards the target." : "Wrong: moving away from the target.";
    tone = msg.active ? (correct ? "good" : "bad") : "neutral";
  } else if (truth === "rest") {
    verdict = msg.active ? "False command: the cursor moves during rest." : "Rest, cursor held.";
    tone = msg.active ? "bad" : "good";
  }
  el.verdict.textContent = verdict;
  el.verdict.className = `verdict verdict-${tone}`;
}

function showCounters(): void {
  el.gameHits.textContent = String(game.hits);
  el.gameTrials.textContent = String(game.trials);
  const hitTime = median(game.hitTimes);
  el.gameDetail.textContent =
    `wrong side ${game.wrong} · timeouts ${game.timeouts}` +
    (Number.isFinite(hitTime) ? ` · median time to hit ${hitTime.toFixed(1)} s` : "");
  el.gamePass.textContent = describePass(session);
  el.gameHistory.textContent = describeHistory(session);
  el.accTitle.textContent = session.pass === null ? "Decoder, this session" : "Decoder, this pass";
  el.accTrial.textContent = formatPercent(share(stats.trialCorrect, stats.trialWindows));
  el.accDetail.textContent =
    `of ${stats.trialWindows} windows during a cue are correct` +
    (stats.trialActive > 0
      ? ` (${formatPercent(share(stats.trialActiveCorrect, stats.trialActive))} while moving)`
      : "");
  el.restActive.textContent = formatPercent(share(stats.restActive, stats.restWindows));
  const latency = median(latencies);
  el.latency.textContent = Number.isFinite(latency) ? `${latency.toFixed(0)} ms` : "–";
}

function handle(message: ServerMessage): void {
  const now = performance.now();
  if (message.type === "eeg") {
    channels = message.channels;
    pushEeg(buffers, message, displayRateHz, Math.ceil((EEG_SPAN_S + 2) * displayRateHz));
    lastEegT = message.t;
    lastEegAt = now;
    return;
  }
  lastPrediction = message;
  lastPredictionAt = now;
  // The game runs on stream time: each prediction moves the cursor by one decoder step,
  // whether or not the page was drawn in between (see game.ts).
  settle(game); // the last step of the previous prediction still belongs to its pass
  const { gap, newPass } = observe(session, message, stepS, game);
  if (gap) interrupt(game);
  if (newPass) {
    resetCounters(game);
    lastOutcomeCount = 0;
    stats = createStats();
  }
  onPrediction(game, message, travelTimeS, stepS);
  updateStats(stats, message);
  timeline.push(message.t, message.true);
  timeline.prune(message.t - EEG_SPAN_S - 2);
  latencies.push(message.latency_ms.stream_to_pred + message.latency_ms.pred_to_ws);
  if (latencies.length > LATENCY_WINDOW) latencies = latencies.slice(-LATENCY_WINDOW);
  showPrediction(message);
  showCounters();
}

// --------------------------------------------------------------------------- status
function showStatus(now: number): void {
  let text = "disconnected, retrying…";
  let tone = "closed";
  if (connection === "open") {
    const live = now - lastPredictionAt < STALL_AFTER_MS;
    text = live ? "live" : "connected, waiting for data…";
    tone = live ? "open" : "stalled";
  } else if (connection === "connecting") {
    text = "connecting…";
  }
  if (el.status.textContent !== text) el.status.textContent = text;
  el.status.className = `status status-${tone}`;
}

// ---------------------------------------------------------------------------- frame
let previousFrame = performance.now();

function frame(now: number): void {
  // Only for smooth drawing: `advance` never moves past the step the last prediction
  // paid for, so frame timing (or a hidden tab without frames) cannot change the score.
  const dt = (now - previousFrame) / 1000;
  previousFrame = now;
  advance(game, dt);
  if (game.outcomeCount !== lastOutcomeCount) {
    lastOutcomeCount = game.outcomeCount;
    lastOutcomeAt = now;
    showCounters();
  }

  const arena = el.arena.getContext("2d");
  if (arena !== null) {
    const { width, height } = fitCanvas(el.arena);
    drawArena(arena, width, height, {
      game,
      pred: lastPrediction?.pred ?? null,
      active: lastPrediction?.active ?? false,
      outcomeAge: (now - lastOutcomeAt) / 1000,
    });
  }
  const eeg = el.eeg.getContext("2d");
  if (eeg !== null) {
    const { width, height } = fitCanvas(el.eeg);
    // Scroll smoothly between messages, but do not run ahead of a stalled stream.
    const streamNow = lastEegT + Math.min(0.5, Math.max(0, (now - lastEegAt) / 1000));
    drawEeg(eeg, width, height, { channels, buffers, timeline, now: streamNow, span: EEG_SPAN_S });
  }
  showStatus(now);
  requestAnimationFrame(frame);
}

// ----------------------------------------------------------------------------- start
el.reset.addEventListener("click", () => {
  resetCounters(game); // a trial still in progress keeps running and counts when it ends
  markIncomplete(session);
  lastOutcomeCount = 0;
  stats = createStats();
  latencies = [];
  showCounters();
});

connect(websocketUrl(window.location), {
  onMessage: handle,
  onStatus: (status) => {
    connection = status;
    if (status === "open") void loadInfo();
  },
});
void loadInfo();
requestAnimationFrame(frame);
