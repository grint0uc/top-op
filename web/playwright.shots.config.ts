import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { defineConfig, devices } from '@playwright/test';
import base from './playwright.config';

// GUI screenshots for the docs (docs/img/gui-*.png), against the real server like the `real` e2e project:
//   npx playwright test -c playwright.shots.config.ts
// Separate from playwright.config.ts so the e2e suites never rewrite the committed images.
const REAL_PORT = 5174;
const API_PORT = 8765;
if (process.env.TEST_WORKER_INDEX === undefined) process.env.TOPOP_E2E_DATA_DIR ??= mkdtempSync(join(tmpdir(), 'topop-e2e-'));
const launchOptions = base.projects?.find((p) => p.name === 'real')?.use?.launchOptions;

export default defineConfig({
  testDir: 'shots',
  testMatch: /\.shots\.ts$/,
  globalTeardown: './e2e/global-teardown.ts',
  reporter: 'list',
  timeout: 180_000,
  use: { headless: true },
  projects: [{ name: 'shots', use: { ...devices['Desktop Chrome'], baseURL: `http://127.0.0.1:${REAL_PORT}`, launchOptions } }],
  webServer: [
    {
      command: `uv run topop serve --port ${API_PORT}`,
      cwd: '..',
      env: { TOPOP_DATA_DIR: process.env.TOPOP_E2E_DATA_DIR ?? '' },
      url: `http://127.0.0.1:${API_PORT}/api/health`,
      timeout: 120_000,
    },
    {
      command: 'npm run dev',
      env: { VITE_PORT: String(REAL_PORT), VITE_API_TARGET: `http://127.0.0.1:${API_PORT}` },
      url: `http://127.0.0.1:${REAL_PORT}/api/health`,
      timeout: 60_000,
    },
  ],
});
