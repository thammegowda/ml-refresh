import { test, expect } from '@playwright/test';

test('neural network trains entirely in JupyterLite with interactive plots', async ({ page }, testInfo) => {
  test.setTimeout(180000);
  const origin = new URL(testInfo.project.use.baseURL).origin;
  const external = [];
  await page.route('**/*', route => {
    const url = route.request().url();
    if (/^https?:/.test(url) && new URL(url).origin !== origin) {
      external.push(url);
      return route.abort();
    }
    return route.continue();
  });
  await page.goto('./');
  await page.getByRole('link', { name: /Neural Networks/ }).click();
  const notebook = page.frameLocator('.notebook-frame');
  await expect(notebook.locator('.jp-Cell')).toHaveCount(19, { timeout: 30000 });
  await notebook.getByRole('menuitem', { name: 'Run', exact: true }).click();
  await notebook.getByRole('menuitem', { name: 'Run All Cells', exact: true }).click();
  await expect(notebook.locator('.jp-OutputArea').last()).toContainText('Final accuracy gap:', { timeout: 120000 });
  await expect(notebook.locator('.jp-CodeCell').filter({ hasText: 'clean_epochs = 120' })).toContainText('accuracy=92.5%');
  await expect(notebook.locator('.jp-CodeCell').filter({ hasText: 'memorization_epochs = 1500' })).toContainText('accuracy=100.0%');
  const plots = notebook.locator('.js-plotly-plot');
  await expect(plots).toHaveCount(3);
  const learning = plots.nth(1);
  await learning.scrollIntoViewIfNeeded();
  const range = await learning.evaluate(plot => plot._fullLayout.xaxis.range[1] - plot._fullLayout.xaxis.range[0]);
  await learning.locator('[data-title="Zoom in"]').click();
  await expect.poll(() => learning.evaluate(plot => plot._fullLayout.xaxis.range[1] - plot._fullLayout.xaxis.range[0])).toBeLessThan(range);
  await learning.locator('[data-title="Reset axes"]').click();
  await expect.poll(() => learning.evaluate(plot => plot._fullLayout.xaxis.range[1] - plot._fullLayout.xaxis.range[0])).toBeGreaterThanOrEqual(range);
  await expect(learning.locator('.scatterlayer .js-line')).toHaveCount(4);
  for (const width of [1440, 390, 320]) {
    await page.setViewportSize({ width, height: 900 });
    await notebook.locator('.jp-MarkdownCell').first().scrollIntoViewIfNeeded();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    const header = await page.locator('.site-header').boundingBox();
    const frame = await page.locator('.notebook-frame').boundingBox();
    expect(frame.width).toBe(width);
    expect(frame.y).toBeCloseTo(header.y + header.height, 0);
    expect(frame.y + frame.height).toBeCloseTo(900, 0);
    await page.screenshot({ path: testInfo.outputPath(`tutorial-${width}.png`) });
    for (const [index, label] of [[1, 'learning'], [2, 'memorization']]) {
      const plot = plots.nth(index);
      await plot.scrollIntoViewIfNeeded();
      await expect.poll(() => plot.evaluate(element => element.getBoundingClientRect().height)).toBeGreaterThan(500);
      const bounds = await plot.boundingBox();
      expect(bounds.x + bounds.width).toBeLessThanOrEqual(width + 1);
      await page.screenshot({ path: testInfo.outputPath(`${label}-${width}.png`) });
    }
  }
  expect(external).toEqual([]);
  const download = page.waitForEvent('download');
  await page.getByRole('link', { name: 'Download notebook', exact: true }).click();
  expect((await download).suggestedFilename()).toBe('neural-network.ipynb');
});