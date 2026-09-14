import { test, expect } from '@playwright/test';
import { readFile } from 'node:fs/promises';

async function ready(page) {
  await expect(page.locator('.py-runtime')).toHaveText('Python + NumPy ready', { timeout: 20000 });
}

test('angle units link circle coordinates, tangent gaps, and vector similarity', async ({ page }) => {
  await page.goto('trigonometry.html');
  await ready(page);
  const circle = page.getByRole('group', { name: 'Unit circle with sine, cosine and tangent' });
  await circle.scrollIntoViewIfNeeded();
  const bounds = await circle.boundingBox();
  const handle = page.getByRole('slider', { name: 'Unit circle angle' });
  const start = await handle.boundingBox();
  await page.mouse.move(start.x + start.width / 2, start.y + start.height / 2);
  await page.mouse.down();
  await page.mouse.move(bounds.x + bounds.width * 230 / 480, bounds.y + bounds.height * 62 / 340, { steps: 8 });
  await page.mouse.up();
  await expect(page.getByLabel('Angle conversion')).toHaveText('90° = π/2 rad');
  await expect(page.locator('.trig-tan-text')).toHaveText('tan θ = undefined');
  await expect(page.getByLabel('Cosine similarity', { exact: true })).toHaveText('0');
  const path = await page.locator('.trig-wave.trig-tan').getAttribute('d');
  expect(path.match(/M/g).length).toBe(5);
  await page.getByRole('radio', { name: 'Radians', exact: true }).check();
  await handle.focus();
  await page.keyboard.press('End');
  await expect(page.getByLabel('Angle conversion')).toHaveText('360° = 2π rad');
  await page.getByLabel('Reference angle').selectOption('180');
  await expect(page.getByLabel('Angle conversion')).toHaveText('180° = π rad');
  await expect(page.getByLabel('Cosine similarity', { exact: true })).toHaveText('-1');
  await page.getByRole('slider', { name: 'Length of v' }).fill('2');
  await expect(page.locator('.py-output-log')).toContainText('u dot v = -3.0000');
  await expect(page.getByLabel('Cosine similarity', { exact: true })).toHaveText('-1');
  await page.getByRole('slider', { name: 'Length of u' }).fill('0');
  await expect(page.getByLabel('Cosine similarity', { exact: true })).toHaveText('undefined');
});

test('parameter updates reuse setup while edited equations rebuild the session', async ({ page }) => {
  await page.goto('trigonometry.html');
  await ready(page);
  const source = await readFile(new URL('./lesson.py', import.meta.url), 'utf8');
  const editor = page.locator('.cm-content');
  await editor.focus();
  await page.keyboard.press('ControlOrMeta+A');
  await page.keyboard.insertText(source + '\nprint("setup complete")\n');
  await page.getByRole('button', { name: 'Run Python', exact: true }).click();
  await expect(page.locator('.py-output-log')).toContainText('setup complete');
  await ready(page);
  await page.getByLabel('Reference angle').selectOption('180');
  await expect(page.locator('.py-output-log')).toContainText('u dot v = -1.5000');
  await expect(page.locator('.py-output-log')).not.toContainText('setup complete');
  await editor.focus();
  await page.keyboard.press('ControlOrMeta+A');
  await page.keyboard.insertText(source.replace('dot = u @ v', 'dot = 2 * (u @ v)').replace('similarity = cosine if', 'similarity = np.float32(0.25) if') + '\nprint("new setup")\n');
  await page.getByLabel('Reference angle').selectOption('180');
  await expect(page.locator('.py-output-log')).toContainText('u dot v = -3.0000');
  await expect(page.locator('.py-output-log')).toContainText('new setup');
  await expect(page.getByLabel('Cosine similarity', { exact: true })).toHaveText('0.25');
  await ready(page);
  await page.getByLabel('Reference angle').selectOption('0');
  await expect(page.locator('.py-output-log')).toContainText('u dot v = 3.0000');
  await expect(page.locator('.py-output-log')).not.toContainText('new setup');
  await page.getByRole('button', { name: 'Reset Python code' }).click();
  await expect(page.locator('.py-output-log')).toContainText('u dot v = 1.0607');
});

test('chapter navigation and diagrams work on desktop and mobile', async ({ page }, testInfo) => {
  await page.goto('./');
  await page.getByRole('link', { name: /Trigonometry/ }).click();
  await ready(page);
  for (const width of [1440, 390, 320]) {
    await page.setViewportSize({ width, height: 900 });
    await expect(page.locator('.trig-scenes svg')).toHaveCount(4);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.screenshot({ path: testInfo.outputPath(`trigonometry-${width}.png`), fullPage: true });
  }
  const circle = page.getByRole('group', { name: 'Unit circle with sine, cosine and tangent' });
  await circle.scrollIntoViewIfNeeded();
  const bounds = await circle.boundingBox();
  const handle = await page.getByRole('slider', { name: 'Unit circle angle' }).boundingBox();
  const touch = await page.context().newCDPSession(page);
  await touch.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x: handle.x + handle.width / 2, y: handle.y + handle.height / 2 }] });
  for (const [x, y] of [[230, 62], [122, 170], [230, 278]]) {
    await touch.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x: bounds.x + bounds.width * x / 480, y: bounds.y + bounds.height * y / 340 }] });
  }
  await touch.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
  await touch.detach();
  await expect(page.getByLabel('Angle conversion')).toHaveText('270° = 3π/2 rad');
  await expect(page.locator('.trig-sin-text')).toHaveText('sin θ = -1');
  await expect(page.locator('.trig-values')).toContainText(['sin θ = -1', 'Signed cross z = -1.5']);
  await expect(page.locator('.trig-point-control')).not.toHaveClass(/dragging/);
  await page.getByRole('link', { name: 'Reference', exact: true }).click();
  await expect(page.locator('#_keep_in_mind')).toBeFocused();
  await page.locator('.chapter-pagination').getByRole('link', { name: 'Calculus' }).click();
  await expect(page).toHaveURL(/calculus.html$/);
});