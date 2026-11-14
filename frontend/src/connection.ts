/** WebSocket client that keeps trying: the server may start after the page. */

import { parseMessage } from "./protocol";
import type { ServerMessage } from "./types";

export type ConnectionStatus = "connecting" | "open" | "closed";

export interface ConnectionHandlers {
  onMessage(message: ServerMessage): void;
  onStatus(status: ConnectionStatus): void;
}

const RETRY_MS = 1500;

export function websocketUrl(location: Pick<Location, "protocol" | "host">): string {
  return `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;
}

export function connect(url: string, handlers: ConnectionHandlers): void {
  handlers.onStatus("connecting");
  const socket = new WebSocket(url);
  socket.onopen = () => handlers.onStatus("open");
  socket.onmessage = (event: MessageEvent<unknown>) => {
    if (typeof event.data !== "string") return;
    const message = parseMessage(event.data);
    if (message !== null) handlers.onMessage(message);
  };
  socket.onclose = () => {
    handlers.onStatus("closed");
    window.setTimeout(() => connect(url, handlers), RETRY_MS);
  };
  socket.onerror = () => socket.close();
}
