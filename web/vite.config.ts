import react from '@vitejs/plugin-react';
import { defineConfig, loadEnv } from 'vite';
import { mockApi } from './mock/plugin.ts';

export default defineConfig(({ mode }) => {
  // VITE_MOCK=1 serves the whole /api surface from web/mock (no Python server needed).
  const mock = loadEnv(mode, process.cwd(), 'VITE_').VITE_MOCK === '1';
  return {
    plugins: [react(), ...(mock ? [mockApi()] : [])],
    build: {
      // FastAPI serves this directory (topop/server/static) at /
      outDir: '../topop/server/static',
      emptyOutDir: true,
      chunkSizeWarningLimit: 1200, // three.js alone is ~700 kB minified
    },
    server: {
      host: '127.0.0.1',
      port: 5173,
      strictPort: true,
      proxy: mock ? undefined : { '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true, ws: true } },
    },
  };
});
