import { existsSync, readdirSync } from 'node:fs';
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

export default defineConfig({
  testDir: 'e2e',
  reporter: 'list',
  use: {
    baseURL: 'http://127.0.0.1:5173',
    headless: true,
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'], launchOptions: { executablePath: chromiumPath() } },
    },
  ],
  // The e2e suite runs against the in-process mock backend (web/mock). Set PW_REUSE_SERVER=1 to attach to a
  // dev server you started yourself with `VITE_MOCK=1 npm run dev`.
  webServer: {
    command: 'npm run dev',
    env: { VITE_MOCK: '1' },
    url: 'http://127.0.0.1:5173/api/health',
    reuseExistingServer: process.env.PW_REUSE_SERVER === '1',
    timeout: 60_000,
  },
});
