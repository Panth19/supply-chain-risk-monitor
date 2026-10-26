import { defineConfig } from "vite";

// In development the dashboard runs on :5173 and proxies API calls to FastAPI on :8000.
export default defineConfig({
  server: {
    proxy: { "/api": "http://localhost:8000" },
  },
});
