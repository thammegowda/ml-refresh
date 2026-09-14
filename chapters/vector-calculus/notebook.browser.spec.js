import { test, expect } from '@playwright/test';

test('JupyterLite runs and edits the notebook with only local static assets', async ({ page }, testInfo) => {
  test.setTimeout(180000);
  const external = [];
  await page.route('**/*', route => {
    const url = route.request().url();
    if (/^https?:/.test(url) && !url.startsWith('http://localhost:1414/')) {
      external.push(url);
      return route.abort();
    }
    return route.continue();
  });
  await page.goto('./vector-calculus.html');
  const notebook = page.frameLocator('iframe.notebook-frame');
  await expect(notebook.locator('.jp-Notebook .jp-Cell')).toHaveCount(14, { timeout: 30000 });
  await notebook.getByRole('menuitem', { name: 'Run', exact: true }).click();
  await notebook.getByRole('menuitem', { name: 'Run All Cells', exact: true }).click();
  await expect(notebook.locator('.jp-OutputArea').last()).toContainText('Gradient, Hessian, and Jacobian checks passed.', { timeout: 120000 });
  const plots = notebook.locator('.js-plotly-plot');
  await expect(plots).toHaveCount(4, { timeout: 30000 });
  const surface = plots.nth(0);
  await expect(surface.locator('canvas')).toHaveCount(1);
  await surface.locator('canvas').scrollIntoViewIfNeeded();
  const cameraBefore = await surface.evaluate(plot => JSON.stringify(plot._fullLayout.scene._scene.getCamera()));
  const canvas = await surface.locator('canvas').boundingBox();
  await page.mouse.move(canvas.x + canvas.width * 0.4, canvas.y + canvas.height * 0.4);
  await page.mouse.down();
  await page.mouse.move(canvas.x + canvas.width * 0.65, canvas.y + canvas.height * 0.6, { steps: 20 });
  await page.mouse.up();
  await expect.poll(() => surface.evaluate(plot => JSON.stringify(plot._fullLayout.scene._scene.getCamera()))).not.toBe(cameraBefore);
  const slice = plots.nth(2);
  await slice.scrollIntoViewIfNeeded();
  const rangeBefore = await slice.evaluate(plot => plot._fullLayout.xaxis.range[1] - plot._fullLayout.xaxis.range[0]);
  await slice.locator('[data-title="Zoom in"]').click();
  await expect.poll(() => slice.evaluate(plot => plot._fullLayout.xaxis.range[1] - plot._fullLayout.xaxis.range[0])).toBeLessThan(rangeBefore);
  await slice.locator('.nsewdrag').scrollIntoViewIfNeeded();
  const hoverTarget = await slice.locator('.nsewdrag').boundingBox();
  const hoverPoint = await slice.evaluate(plot => ({ x: plot._fullLayout.xaxis.l2p(0), y: plot._fullLayout.yaxis.l2p(plot.data[3].y[0]) }));
  await page.mouse.move(hoverTarget.x + hoverPoint.x, hoverTarget.y + hoverPoint.y);
  await expect(slice.locator('.hoverlayer')).toContainText('exact');
  const parameters = notebook.locator('.jp-CodeCell').nth(1).locator('.cm-content');
  await parameters.click();
  await page.keyboard.press('ControlOrMeta+A');
  const original = await parameters.innerText();
  await page.keyboard.insertText(original.replace("example = 'bowl'", "example = 'saddle'"));
  await notebook.getByRole('menuitem', { name: 'Run', exact: true }).click();
  await notebook.getByRole('menuitem', { name: 'Run All Cells', exact: true }).click();
  await expect(notebook.locator('.jp-CodeCell').nth(1).locator('.jp-OutputArea')).toContainText('saddle: a=1, b=0, c=-1, q=0');
  await expect(notebook.locator('.jp-OutputArea').last()).toContainText('checks passed', { timeout: 30000 });
  expect(external).toEqual([]);
  for (const width of [1440, 390, 320]) {
    await page.setViewportSize({ width, height: 900 });
    await notebook.locator('.jp-MarkdownCell').first().scrollIntoViewIfNeeded();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    const header = await page.locator('.site-header').boundingBox();
    const frame = await page.locator('.notebook-frame').boundingBox();
    expect(frame.x).toBe(0);
    expect(frame.width).toBe(width);
    expect(frame.y).toBeCloseTo(header.y + header.height, 0);
    expect(frame.y + frame.height).toBeCloseTo(900, 0);
    await expect(page.locator('.site-footer')).toBeHidden();
    await expect(page.locator('.chapter-pagination')).toBeHidden();
    await expect(page.locator('#_keep_in_mind')).toHaveCount(0);
    await page.screenshot({ path: testInfo.outputPath(`notebook-${width}.png`), fullPage: true });
    await surface.locator('canvas').scrollIntoViewIfNeeded();
    const canvasImage = await surface.locator('canvas').screenshot();
    expect(await page.evaluate(async encoded => {
      const image = new Image();
      image.src = `data:image/png;base64,${encoded}`;
      await image.decode();
      const copy = document.createElement('canvas');
      copy.width = image.width;
      copy.height = image.height;
      const context = copy.getContext('2d');
      context.drawImage(image, 0, 0);
      const pixels = context.getImageData(0, 0, copy.width, copy.height).data;
      let colored = 0;
      for (let offset = 0; offset < pixels.length; offset += 4) {
        if (pixels[offset + 3] && Math.max(pixels[offset], pixels[offset + 1], pixels[offset + 2]) - Math.min(pixels[offset], pixels[offset + 1], pixels[offset + 2]) > 30) colored++;
      }
      return colored > 100;
    }, canvasImage.toString('base64'))).toBe(true);
    await page.screenshot({ path: testInfo.outputPath(`notebook-plot-${width}.png`), fullPage: true });
  }
  const download = page.waitForEvent('download');
  await page.getByRole('link', { name: 'Download notebook', exact: true }).click();
  expect((await download).suggestedFilename()).toBe('vector-calculus.ipynb');
  await expect(page.getByRole('link', { name: 'Open notebook full-screen' })).toHaveAttribute('href', './jupyter/notebooks/index.html?path=vector-calculus.ipynb');
});