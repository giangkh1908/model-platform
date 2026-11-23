import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Khi phát triển (npm run dev), mọi request /api được chuyển tới backend FastAPI ở cổng 8100.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8100" },
  },
  build: { outDir: "dist", emptyOutDir: true },
});
