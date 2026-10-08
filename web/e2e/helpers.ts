import { fileURLToPath } from 'node:url';
import { expect, type Page } from '@playwright/test';

export const BRACKET = fileURLToPath(new URL('../../examples/bracket.stl', import.meta.url));

type StoreState = ReturnType<NonNullable<Window['__topop']>['getState']>;

/** Reads (a projection of) the zustand store inside the page. `fn` must be self-contained. */
export const state = <T>(page: Page, fn: (s: StoreState) => T): Promise<T> =>
  page.evaluate(`(${fn.toString()})(window.__topop.getState())`) as Promise<T>;

export const viewportStats = (page: Page) => page.evaluate(() => window.__topopViewport!.stats());

/** Uploads examples/bracket.stl as the design mesh and waits until the viewport shows it. */
export async function importBracket(page: Page): Promise<void> {
  await page.goto('/');
  await page.getByTestId('mesh-file-input').setInputFiles(BRACKET);
  await expect
    .poll(async () => (await page.evaluate(() => window.__topopViewport?.stats().designFaces)) ?? 0, { message: 'design mesh in viewport' })
    .toBeGreaterThan(0);
}

export async function canvasCentre(page: Page): Promise<{ x: number; y: number }> {
  const box = await page.locator('canvas#viewport').boundingBox();
  if (!box) throw new Error('canvas not laid out');
  return { x: box.x + box.width / 2, y: box.y + box.height / 2 };
}

/** pick mode, click the viewport centre, shift+click to grow the flat face; resolves once the grown set is in the store. */
export async function selectFlatFaceAtCentre(page: Page): Promise<number> {
  await page.getByTestId('mode-pick').click();
  const { x, y } = await canvasCentre(page);
  await page.mouse.click(x, y);
  await expect.poll(() => state(page, (s) => s.selection.faceIds.length)).toBe(1);
  await page.keyboard.down('Shift');
  await page.mouse.click(x, y);
  await page.keyboard.up('Shift');
  await expect.poll(() => state(page, (s) => s.selection.faceIds.length)).toBeGreaterThan(1);
  return state(page, (s) => s.selection.faceIds.length);
}

/** One load on the flat face under the viewport centre, one support from a box primitive. */
export async function defineBoundaries(page: Page): Promise<void> {
  await selectFlatFaceAtCentre(page);
  await page.getByTestId('add-load').click();
  await page.getByTestId('add-box').click();
  await page.getByTestId('add-support').click();
  await expect.poll(() => state(page, (s) => s.project.loads.length + s.project.supports.length)).toBe(2);
}
