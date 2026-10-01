import { test, expect } from '@playwright/test';

test('neural network companion notebook runs offline in JupyterLite', async ({ page }, testInfo) => {
  test.setTimeout(180000);
  const origin = new URL(testInfo.project.use.baseURL).origin;
  const external = [];
  await page.route('**/*', (route) => {
    const url = route.request().url();
    if (/^https?:/.test(url) && new URL(url).origin !== origin) {
      external.push(url);
      return route.abort();
    }
    return route.continue();
  });
  await page.goto('./neural-network.html');
  const download = page.waitForEvent('download');
  await page.getByRole('link', { name: 'Download notebook', exact: true }).click();
  expect((await download).suggestedFilename()).toBe('neural-network.ipynb');
  await page.getByRole('link', { name: 'Run the companion notebook' }).click();
  await expect(page.locator('.jp-Cell')).toHaveCount(19, { timeout: 30000 });
  await page.getByRole('menuitem', { name: 'Run', exact: true }).click();
  await page.getByRole('menuitem', { name: 'Run All Cells', exact: true }).click();
  await expect(page.locator('.jp-OutputArea').last()).toContainText('Final accuracy gap:', { timeout: 120000 });
  await expect(page.locator('.jp-CodeCell').filter({ hasText: 'clean_epochs = 120' })).toContainText('accuracy=92.5%');
  await expect(page.locator('.jp-CodeCell').filter({ hasText: 'memorization_epochs = 1500' })).toContainText('accuracy=100.0%');
  const plots = page.locator('.js-plotly-plot');
  await expect(plots).toHaveCount(3);
  const learning = plots.nth(1);
  await learning.scrollIntoViewIfNeeded();
  const range = await learning.evaluate((plot) => plot._fullLayout.xaxis.range[1] - plot._fullLayout.xaxis.range[0]);
  await learning.locator('[data-title="Zoom in"]').click();
  await expect.poll(() => learning.evaluate((plot) => plot._fullLayout.xaxis.range[1] - plot._fullLayout.xaxis.range[0])).toBeLessThan(range);
  await learning.locator('[data-title="Reset axes"]').click();
  await expect.poll(() => learning.evaluate((plot) => plot._fullLayout.xaxis.range[1] - plot._fullLayout.xaxis.range[0])).toBeGreaterThanOrEqual(range);
  await expect(learning.locator('.scatterlayer .js-line')).toHaveCount(4);
  await expect(page.locator('.jp-OutputArea-output[data-mime-type="application/vnd.jupyter.stderr"]')).toHaveCount(0);
  expect(external).toEqual([]);
});
