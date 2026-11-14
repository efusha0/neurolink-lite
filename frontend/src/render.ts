/** Canvas drawing. No state lives here: every function draws what it is given. */

import type { StateTimeline, TraceBuffer } from "./buffers";
import type { GameState, Outcome } from "./game";
import type { ClassName, TrueLabel } from "./types";

export const COLORS = {
  background: "#0e1116",
  panel: "#161b22",
  grid: "#262d38",
  text: "#e6edf3",
  muted: "#8b949e",
  left: "#58a6ff",
  right: "#f0883e",
  rest: "#6e7681",
  hit: "#3fb950",
  wrong: "#f85149",
} as const;

export function classColor(name: ClassName): string {
  return name === "left" ? COLORS.left : COLORS.right;
}

/** Match the canvas bitmap to its CSS size and the device pixel ratio. */
export function fitCanvas(canvas: HTMLCanvasElement): { width: number; height: number } {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
  }
  canvas.getContext("2d")?.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { width, height };
}

export interface ArenaView {
  game: GameState;
  /** Class the decoder currently outputs, or null before the first prediction. */
  pred: ClassName | null;
  /** True when the prediction is above the threshold (cursor allowed to move). */
  active: boolean;
  /** Seconds since the last outcome, for the fading banner. */
  outcomeAge: number;
}

const OUTCOME_TEXT: Record<Outcome, string> = {
  hit: "HIT",
  wrong: "WRONG SIDE",
  timeout: "TIMEOUT",
};

export function drawArena(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  view: ArenaView,
): void {
  const { game } = view;
  ctx.clearRect(0, 0, width, height);
  const wall = 26; // width of the target zones
  const margin = 18;
  const radius = 16;
  const y = height / 2;
  const left = margin + wall + radius;
  const right = width - margin - wall - radius;
  const toPx = (x: number): number => left + ((x + 1) / 2) * (right - left);

  // Target zones at both walls; the cued one lights up.
  (["left", "right"] as const).forEach((side) => {
    const x0 = side === "left" ? margin : width - margin - wall;
    const cued = game.target === side;
    ctx.fillStyle = cued ? classColor(side) : COLORS.grid;
    ctx.globalAlpha = cued ? (game.resolved ? 0.45 : 1) : 0.6;
    ctx.beginPath();
    ctx.roundRect(x0, y - 60, wall, 120, 6);
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.fillStyle = cued ? classColor(side) : COLORS.muted;
    ctx.font = "600 12px system-ui, sans-serif";
    ctx.textAlign = side === "left" ? "left" : "right";
    ctx.fillText(side.toUpperCase(), side === "left" ? margin : width - margin, y - 72);
  });

  // Track and centre mark.
  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(left - radius, y);
  ctx.lineTo(right + radius, y);
  ctx.moveTo(toPx(0), y - 14);
  ctx.lineTo(toPx(0), y + 14);
  ctx.stroke();

  // Cursor: coloured by the decoded class while it moves, grey while held.
  const moving = view.active && view.pred !== null && !(game.target !== null && game.resolved);
  ctx.fillStyle = moving && view.pred !== null ? classColor(view.pred) : COLORS.rest;
  ctx.beginPath();
  ctx.arc(toPx(game.x), y, radius, 0, 2 * Math.PI);
  ctx.fill();
  ctx.strokeStyle = COLORS.text;
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Caption: what the recording asks for right now.
  ctx.textAlign = "center";
  ctx.font = "600 14px system-ui, sans-serif";
  if (game.target === null) {
    ctx.fillStyle = COLORS.muted;
    ctx.fillText("REST: no target. Any movement now is a false command.", width / 2, height - 14);
  } else {
    ctx.fillStyle = classColor(game.target);
    ctx.fillText(`CUE: imagine the ${game.target.toUpperCase()} hand`, width / 2, height - 14);
  }

  // Outcome banner fades out over 1.5 s.
  if (game.lastOutcome !== null && view.outcomeAge < 1.5) {
    ctx.globalAlpha = 1 - view.outcomeAge / 1.5;
    ctx.fillStyle =
      game.lastOutcome === "hit"
        ? COLORS.hit
        : game.lastOutcome === "wrong"
          ? COLORS.wrong
          : COLORS.muted;
    ctx.font = "700 26px system-ui, sans-serif";
    ctx.fillText(OUTCOME_TEXT[game.lastOutcome], width / 2, 34);
    ctx.globalAlpha = 1;
  }
}

function stateColor(state: TrueLabel): string | null {
  if (state === "left") return COLORS.left;
  if (state === "right") return COLORS.right;
  return null;
}

export interface EegView {
  channels: string[];
  buffers: Map<string, TraceBuffer>;
  timeline: StateTimeline;
  /** Stream time at the right edge of the plot. */
  now: number;
  /** Seconds of signal shown. */
  span: number;
}

export function drawEeg(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  view: EegView,
): void {
  ctx.clearRect(0, 0, width, height);
  const labelWidth = 34;
  const axisHeight = 18;
  const plotWidth = width - labelWidth;
  const plotHeight = height - axisHeight;
  const t0 = view.now - view.span;
  const toX = (t: number): number => labelWidth + ((t - t0) / view.span) * plotWidth;

  // Background: the true state of the recording (blue = left cue, orange = right cue).
  const segments = view.timeline.segments;
  segments.forEach((segment, i) => {
    const color = stateColor(segment.state);
    const end = segments[i + 1]?.t ?? view.now;
    if (color === null || end <= t0) return;
    const x0 = Math.max(labelWidth, toX(segment.t));
    const x1 = Math.min(width, toX(end));
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.14;
    ctx.fillRect(x0, 0, Math.max(0, x1 - x0), plotHeight);
    ctx.globalAlpha = 1;
  });

  // Time axis: one tick per second, labelled as seconds ago.
  ctx.font = "11px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.strokeStyle = COLORS.grid;
  ctx.fillStyle = COLORS.muted;
  ctx.lineWidth = 1;
  for (let ago = 0; ago <= view.span; ago += 1) {
    const x = toX(view.now - ago);
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, plotHeight);
    ctx.stroke();
    if (ago % 2 === 0 && ago > 0) ctx.fillText(`-${ago} s`, x, height - 4);
  }

  // One row per channel, each scaled to its own spread so the rhythm is visible.
  const rowHeight = plotHeight / Math.max(1, view.channels.length);
  view.channels.forEach((name, row) => {
    const mid = rowHeight * (row + 0.5);
    ctx.fillStyle = COLORS.muted;
    ctx.textAlign = "left";
    ctx.font = "600 12px system-ui, sans-serif";
    ctx.fillText(name, 4, mid + 4);
    const buffer = view.buffers.get(name);
    if (buffer === undefined || buffer.length < 2) return;
    const scale = (0.42 * rowHeight) / Math.max(1e-6, 3 * buffer.std());
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    let started = false;
    buffer.forEach((t, value) => {
      if (t < t0) return;
      const x = toX(t);
      const yy = mid - Math.max(-0.5 * rowHeight, Math.min(0.5 * rowHeight, value * scale));
      if (started) ctx.lineTo(x, yy);
      else ctx.moveTo(x, yy);
      started = true;
    });
    ctx.stroke();
  });
}
