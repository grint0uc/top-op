import { rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, dirname } from 'node:path';

// Removes the throwaway data dir the `real` project's server wrote to (see playwright.config.ts).
export default function globalTeardown(): void {
  const dir = process.env.TOPOP_E2E_DATA_DIR;
  if (dir && dirname(dir) === tmpdir() && basename(dir).startsWith('topop-e2e-')) {
    rmSync(dir, { recursive: true, force: true });
  }
}
