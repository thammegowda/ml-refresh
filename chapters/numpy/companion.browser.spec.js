import { test, expect } from '@playwright/test';

test('the companion notebook imports the scratch package and runs offline in JupyterLite', async ({ page }, testInfo) => {
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
  await page.goto('./numpy.html');
  await page.getByRole('link', { name: 'Run the companion notebook' }).click();
  await expect(page.locator('.jp-Cell')).toHaveCount(12, { timeout: 30000 });
  await page.getByRole('menuitem', { name: 'Run', exact: true }).click();
  await page.getByRole('menuitem', { name: 'Run All Cells', exact: true }).click();
  await expect(page.locator('.jp-OutputArea').last()).toContainText('tanh gradient check passed', { timeout: 120000 });
  await expect(page.locator('.jp-CodeCell').filter({ hasText: 'running_sum' })).toContainText('bfloat16: 0.5000');
  await expect(page.locator('.jp-OutputArea-output[data-mime-type="application/vnd.jupyter.stderr"]')).toHaveCount(0);
  expect(external).toEqual([]);
});
