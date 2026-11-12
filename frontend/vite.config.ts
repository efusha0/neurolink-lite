import { defineConfig } from "vite";

export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      // Forward WebSocket traffic to the FastAPI server during development.
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
    },
  },
});
