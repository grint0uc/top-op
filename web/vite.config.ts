import react from '@vitejs/plugin-react';
import { defineConfig, loadEnv } from 'vite';
import { mockApi } from './mock/plugin.ts';

const PORT = Number(process.env.VITE_PORT ?? 5173);

export default defineConfig(({ mode }) => {
  // VITE_MOCK=1 serves the whole /api surface from web/mock (no Python server needed).
  const mock = loadEnv(mode, process.cwd(), 'VITE_').VITE_MOCK === '1';
  return {
    plugins: [react(), ...(mock ? [mockApi()] : [])],
    // one dependency cache per dev-server port, so concurrent dev servers (Playwright's mock + real) do not fight over it
    cacheDir: PORT === 5173 ? 'node_modules/.vite' : `node_modules/.vite-${PORT}`,
    build: {
      // FastAPI serves this directory (topop/server/static) at /
      outDir: '../topop/server/static',
      emptyOutDir: true,
      chunkSizeWarningLimit: 1200, // three.js alone is ~700 kB minified
    },
    server: {
      host: '127.0.0.1',
      // VITE_PORT / VITE_API_TARGET let several dev servers coexist (e.g. Playwright's mock and real projects)
      port: PORT,
      strictPort: true,
      proxy: mock
        ? undefined
        : { '/api': { target: process.env.VITE_API_TARGET ?? 'http://127.0.0.1:8000', changeOrigin: true, ws: true } },
    },
  };
});
