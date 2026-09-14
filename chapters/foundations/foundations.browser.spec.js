import { test, expect } from '@playwright/test';

test('foundations disclosures, search, tables, and diagrams work on desktop and mobile', async ({ page }, testInfo) => {
  const requests = [];
  const origin = new URL(testInfo.project.use.baseURL).origin;
  await page.route('**/*', route => {
    const url = route.request().url();
    requests.push(url);
    return /^https?:/.test(url) && new URL(url).origin !== origin ? route.abort() : route.continue();
  });
  await page.goto('./');
  await expect(page.locator('.chapter-link').first()).toContainText('Mathematical Foundations');
  await page.locator('.chapter-link').first().click();
  await expect(page.locator('.foundation-list details')).toHaveCount(71);
  await expect(page.locator('details[open]')).toHaveCount(0);
  await page.locator('#square-sum > summary').focus();
  await page.keyboard.press('Enter');
  await expect(page.locator('#square-sum')).toHaveAttribute('open', '');
  const canvas = page.locator('[data-diagram="square"]');
  const before = await canvas.evaluate(element => element.toDataURL());
  await page.getByRole('slider', { name: 'Side length b' }).focus();
  await page.keyboard.press('ArrowRight');
  await expect(page.locator('[data-square-value]')).toHaveText('2.5');
  await expect.poll(() => canvas.evaluate(element => element.toDataURL())).not.toBe(before);
  await page.getByRole('button', { name: 'Collapse all reminders' }).click();
  await expect(page.locator('details[open]')).toHaveCount(0);
  const search = page.getByRole('searchbox', { name: 'Search mathematical reminders' });
  await search.fill('sine rule');
  await expect(page.locator('.foundation-count')).toHaveText('1 reminder');
  await expect(page.locator('#triangle-laws')).toBeVisible();
  await search.fill('no-such-mathematical-topic');
  await expect(page.getByText('No matching reminders.', { exact: true })).toBeVisible();
  await page.getByRole('link', { name: 'Numbers worth knowing', exact: true }).click();
  await expect(search).toHaveValue('');
  await page.locator('#multiplication > summary').click();
  const grid = page.locator('.multiplication-grid');
  const headers = Array.from({ length: 20 }, (_, index) => String(index + 1));
  await expect(grid.locator('th[scope="col"]')).toHaveText(headers);
  await expect(grid.locator('th[scope="row"]')).toHaveText(headers.slice(0, 10));
  await expect(grid.locator('tbody td')).toHaveCount(200);
  await expect(grid.locator('tbody td').last()).toHaveText('200');
  expect(await grid.locator('tbody tr').evaluateAll(rows => rows.every((row, rowIndex) =>
    [...row.querySelectorAll('td')].every((cell, columnIndex) => Number(cell.textContent) === (rowIndex + 1) * (columnIndex + 1))))).toBe(true);
  expect(await grid.locator('th').evaluateAll(headers => headers.every(header => getComputedStyle(header).fontWeight === '700'))).toBe(true);
  await expect(grid.locator('thead th').nth(10)).toHaveCSS('border-right-width', '3px');
  await expect(grid.locator('tbody tr').nth(9).locator('td').first()).toHaveCSS('border-bottom-width', '1px');
  await expect(grid.locator('tbody tr').nth(8).locator('td').first()).toHaveCSS('border-bottom-width', '1px');
  await expect(page.locator('#cubes')).toHaveCount(0);
  await page.locator('#squares > summary').click();
  await expect(page.locator('#squares tbody tr')).toHaveCount(25);
  await expect(page.locator('#squares tbody tr').last()).toHaveText('25625');
  await page.locator('#primes > summary').click();
  await expect(page.locator('.prime-list li')).toHaveCount(168);
  await expect(page.locator('.prime-list li').last()).toHaveText('997');
  await page.goto('./foundations.html#log-definition');
  await expect(page.locator('#log-definition')).toHaveAttribute('open', '');
  for (const width of [1440, 390, 320]) {
    await page.setViewportSize({ width, height: 900 });
    if (width < 720) {
      await page.getByRole('combobox', { name: 'Jump to topic' }).selectOption('geometry');
      await expect(page).toHaveURL(/#geometry$/);
      await page.getByRole('combobox', { name: 'Jump to topic' }).selectOption('');
      await expect(page).not.toHaveURL(/#/);
    }
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({ path: testInfo.outputPath(`intro-${width}.png`) });
    await page.locator('#multiplication').evaluate(element => { element.open = true; element.scrollIntoView(); });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.screenshot({ path: testInfo.outputPath(`multiplication-${width}.png`) });
    if (width < 720) {
      await grid.evaluate(element => { element.scrollLeft = element.scrollWidth; });
      await expect(grid.locator('tbody td').last()).toBeInViewport();
      await expect(grid.locator('th[scope="row"]').last()).toBeInViewport();
      await page.screenshot({ path: testInfo.outputPath(`multiplication-scrolled-${width}.png`) });
      await grid.evaluate(element => { element.scrollLeft = 0; });
    }
    for (const id of ['square-sum', 'log-definition', 'derivatives', 'angle-addition']) {
      const details = page.locator(`#${id}`);
      await details.evaluate(element => { element.open = true; element.scrollIntoView(); });
      await expect(details.locator('math').first()).toBeVisible();
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      if (id === 'square-sum' || id === 'log-definition') {
        const drawing = details.locator('canvas');
        await expect.poll(() => drawing.evaluate(element => {
          const pixels = element.getContext('2d').getImageData(0, 0, element.width, element.height).data;
          let colored = 0;
          for (let offset = 0; offset < pixels.length; offset += 4) {
            if (pixels[offset + 3] && Math.max(pixels[offset], pixels[offset + 1], pixels[offset + 2]) - Math.min(pixels[offset], pixels[offset + 1], pixels[offset + 2]) > 25) colored++;
          }
          return colored;
        })).toBeGreaterThan(100);
      }
      await page.screenshot({ path: testInfo.outputPath(`${id}-${width}.png`) });
    }
  }
  expect(requests.filter(url => /^https?:/.test(url) && new URL(url).origin !== origin)).toEqual([]);
  expect(requests.some(url => /pyodide|jupyter|katex.*\.js/.test(url))).toBe(false);
});

test('reference equations and disclosures remain usable without JavaScript', async ({ browser }, testInfo) => {
  const context = await browser.newContext({ javaScriptEnabled: false });
  const page = await context.newPage();
  await page.goto(new URL('foundations.html', testInfo.project.use.baseURL).href);
  await expect(page.locator('.foundation-tools')).toBeHidden();
  await expect(page.locator('.foundation-list details')).toHaveCount(71);
  await page.locator('#binomial > summary').click();
  await expect(page.locator('#binomial')).toHaveAttribute('open', '');
  await expect(page.locator('#binomial table')).toBeVisible();
  await expect(page.locator('#binomial math').first()).toBeVisible();
  await context.close();
});