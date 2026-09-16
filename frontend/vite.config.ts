import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const FRONTEND_PORT = parseInt(process.env.VITE_PORT || "3000", 10);
const BACKEND_PORT = parseInt(process.env.VITE_BACKEND_PORT || "8001", 10);

export default defineConfig({
  plugins: [react()],
  server: {
    port: FRONTEND_PORT,
    host: "127.0.0.1",
    strictPort: true,
    proxy: {
      "/api": {
        target: `http://127.0.0.1:${BACKEND_PORT}`,
        changeOrigin: true,
      },
    },
  },
});
