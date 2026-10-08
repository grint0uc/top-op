import { existsSync, mkdtempSync, readdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { chromium, defineConfig, devices } from '@playwright/test';

// Sandbox fallback: when the Chromium revision this Playwright wants is not installed (and
// `playwright install` is unavailable), use whatever chromium-* exists under
// PLAYWRIGHT_BROWSERS_PATH. PW_CHROMIUM_PATH overrides explicitly. Normal installs are untouched.
function chromiumPath(): string | undefined {
  if (process.env.PW_CHROMIUM_PATH) return process.env.PW_CHROMIUM_PATH;
  if (existsSync(chromium.executablePath())) return undefined;
  const root = process.env.PLAYWRIGHT_BROWSERS_PATH;
  if (!root || !existsSync(root)) return undefined;
  const dir = readdirSync(root)
    .filter((d) => /^chromium-\d+$/.test(d))
    .sort()
    .pop();
  const exe = dir ? join(root, dir, 'chrome-linux', 'chrome') : undefined;
  return exe && existsSync(exe) ? exe : undefined;
}

// Two projects:
//   chromium  the UI against the in-process mock backend (web/mock) on :5173
//   real      the UI through Vite's dev proxy (:5174) against `uv run topop serve` on :8765, which uses a fresh data
//             dir per run, so the suite never sees (or pollutes) ~/.cache/topop
// Playwright starts every webServer entry whatever project is selected; the filter below skips the ones the
// selected projects do not need (`--project=real` never starts the mock server and vice versa).
const MOCK_PORT = 5173;
const REAL_PORT = 5174;
const API_PORT = 8765;
const reuse = process.env.PW_REUSE_SERVER === '1';

const picked = process.argv.flatMap((a, i, all) => (a === '--project' ? [all[i + 1] ?? ''] : a.startsWith('--project=') ? [a.slice(10)] : []));
const wants = (name: string) => picked.length === 0 || picked.includes(name);

// The config is loaded again in every worker (where --project is not in argv): only the main process creates the
// fresh data dir for the real server, and global-teardown removes it.
const inWorker = process.env.TEST_WORKER_INDEX !== undefined;
if (!inWorker && wants('real')) process.env.TOPOP_E2E_DATA_DIR ??= mkdtempSync(join(tmpdir(), 'topop-e2e-'));

const webServer: NonNullable<Parameters<typeof defineConfig>[0]['webServer']> = [
  ...(wants('chromium')
    ? [
        {
          // `npm run dev` with the mock plugin: the whole /api surface is served from web/mock
          command: 'npm run dev',
          env: { VITE_MOCK: '1', VITE_PORT: String(MOCK_PORT) },
          url: `http://127.0.0.1:${MOCK_PORT}/api/health`,
          reuseExistingServer: reuse,
          timeout: 60_000,
        },
      ]
    : []),
  ...(wants('real')
    ? [
        {
          command: `uv run topop serve --port ${API_PORT}`,
          cwd: '..',
          env: { TOPOP_DATA_DIR: process.env.TOPOP_E2E_DATA_DIR ?? '' },
          url: `http://127.0.0.1:${API_PORT}/api/health`,
          reuseExistingServer: reuse,
          timeout: 120_000,
        },
        {
          command: 'npm run dev',
          env: { VITE_PORT: String(REAL_PORT), VITE_API_TARGET: `http://127.0.0.1:${API_PORT}` },
          url: `http://127.0.0.1:${REAL_PORT}/api/health`, // through the proxy: checks the wiring too
          reuseExistingServer: reuse,
          timeout: 60_000,
        },
      ]
    : []),
];

export default defineConfig({
  testDir: 'e2e',
  globalTeardown: './e2e/global-teardown.ts',
  reporter: 'list',
  use: { headless: true },
  projects: [
    {
      name: 'chromium',
      testIgnore: /real\.spec\.ts/,
      use: {
        ...devices['Desktop Chrome'],
        baseURL: `http://127.0.0.1:${MOCK_PORT}`,
        launchOptions: { executablePath: chromiumPath() },
      },
    },
    {
      name: 'real',
      testMatch: /real\.spec\.ts/,
      use: {
        ...devices['Desktop Chrome'],
        baseURL: `http://127.0.0.1:${REAL_PORT}`,
        launchOptions: { executablePath: chromiumPath() },
      },
    },
  ],
  webServer,
});
