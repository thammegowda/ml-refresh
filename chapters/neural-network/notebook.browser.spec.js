import { test, expect } from '@playwright/test';

test('neural network companion notebook opens and runs offline in JupyterLite', async ({ page }, testInfo) => {
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
  // The CPython notebook test checks the training results; here, check that the notebook runs in
  // the browser: its first cells install Plotly offline and draw the dataset.
  await page.getByRole('menuitem', { name: 'Run', exact: true }).click();
  await page.getByRole('menuitem', { name: 'Run All Cells', exact: true }).click();
  await expect(page.locator('.js-plotly-plot').first()).toBeVisible({ timeout: 120000 });
  await expect(page.locator('.jp-OutputArea-output[data-mime-type="application/vnd.jupyter.stderr"]')).toHaveCount(0);
  expect(external).toEqual([]);
});
