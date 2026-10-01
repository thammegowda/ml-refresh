import { test, expect } from '@playwright/test';
import { chapters, parts } from '../app.js';

const published = chapters.filter((chapter) => chapter.status === 'published');

async function offline(page, testInfo) {
  const origin = new URL(testInfo.project.use.baseURL).origin;
  const failures = [];
  await page.route('**/*', (route) => (new URL(route.request().url()).origin === origin ? route.continue() : route.abort()));
  page.on('requestfailed', (request) => failures.push(request.url()));
  page.on('response', (response) => { if (response.status() >= 400) failures.push(`${response.status()} ${response.url()}`); });
  return failures;
}

test('contents groups chapters into parts and links both printable formats', async ({ page }, testInfo) => {
  const failures = await offline(page, testInfo);
  await page.goto('./');
  await expect(page.locator('.part-heading')).toHaveCount(parts.length);
  await expect(page.locator('.chapter-link')).toHaveCount(published.length);
  await expect(page.locator('.chapter-planned')).toHaveCount(chapters.length - published.length);
  await expect(page.getByRole('link', { name: 'Single-page edition' })).toHaveAttribute('href', './book.html');
  await expect(page.getByRole('link', { name: 'Download PDF' })).toHaveAttribute('href', './refresh.pdf');
  await page.getByRole('link', { name: /NumPy for Deep Learning/ }).click();
  await expect(page.getByRole('heading', { level: 1, name: 'NumPy for Deep Learning' })).toBeVisible();
  expect(failures).toEqual([]);
});

test('a longform chapter renders numbered math and figures, and exercises link to solutions', async ({ page }, testInfo) => {
  const failures = await offline(page, testInfo);
  await page.goto('./numpy.html');
  await expect(page.locator('.chapter-body h2').first()).toHaveText(/^B\.1 /);
  expect(await page.locator('.katex').count()).toBeGreaterThan(40);
  expect(await page.locator('main').innerText()).not.toMatch(/\\\(|\\\[/);
  await expect(page.locator('#eq-logsumexp .katex-display')).toContainText('(B.3)');
  const images = page.locator('.imageblock img');
  await expect(images).toHaveCount(3);
  for (const image of await images.all()) {
    await image.scrollIntoViewIfNeeded();
    await expect.poll(() => image.evaluate((element) => element.complete && element.naturalWidth > 0)).toBe(true);
  }
  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 900 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.screenshot({ path: testInfo.outputPath(`numpy-${width}.png`), fullPage: false });
  }
  await page.locator('#ex-numpy-keepdims .exercise-solution-link a').click();
  await expect(page).toHaveURL(/solutions\.html#sol-ex-numpy-keepdims$/);
  await expect(page.locator('#sol-ex-numpy-keepdims')).toContainText('Solution B.2');
  await page.locator('#sol-ex-numpy-keepdims .solution-back-link a').click();
  await expect(page).toHaveURL(/numpy\.html#ex-numpy-keepdims$/);
  expect(failures).toEqual([]);
});

test('the single-page edition holds every published chapter and fits a printed page', async ({ page }, testInfo) => {
  const failures = await offline(page, testInfo);
  await page.goto('./book.html');
  await expect(page.locator('article.book-chapter')).toHaveCount(published.length);
  await expect(page.locator('iframe, noscript')).toHaveCount(0);
  await expect(page.locator('details:not([open])')).toHaveCount(0);
  expect(await page.locator('.print-note').count()).toBeGreaterThanOrEqual(published.filter((chapter) => chapter.interactive).length);
  const ids = await page.evaluate(() => [...document.querySelectorAll('[id]')].map((element) => element.id));
  expect(new Set(ids).size).toBe(ids.length);
  const broken = await page.evaluate(() => [...document.querySelectorAll('a[href^="#"]')].map((link) => link.getAttribute('href').slice(1)).filter((id) => id && !document.getElementById(id)));
  expect(broken).toEqual([]);
  await page.emulateMedia({ media: 'print' });
  await page.setViewportSize({ width: 662, height: 900 });
  await expect(page.locator('.edition-bar')).toBeHidden();
  const overflowing = await page.evaluate(() => [...document.querySelectorAll('.longform-content :is(pre, .katex-display, table, img, .stemblock)')]
    .filter((element) => element.getBoundingClientRect().right > document.documentElement.clientWidth + 1 || element.scrollWidth > element.clientWidth + 1)
    .map((element) => `${element.closest('[id]')?.id}: ${element.tagName}.${element.className}`));
  expect(overflowing).toEqual([]);
  expect(failures).toEqual([]);
});

test('equation numbers never overlap their equations, on screen or at print width', async ({ page }) => {
  const collisions = async () => page.evaluate(() => [...document.querySelectorAll('.katex-display .katex-tag')].filter((tag) => {
    const html = tag.closest('.katex-html');
    const right = Math.max(...[...html.children].filter((child) => child.classList.contains('katex-base')).map((base) => base.getBoundingClientRect().right));
    return right > tag.getBoundingClientRect().left + 1;
  }).map((tag) => tag.closest('[id]')?.id));
  for (const chapter of chapters.filter((entry) => entry.status === 'published' && entry.layout === 'longform' && !entry.generated)) {
    await page.goto(`./${chapter.id}.html`);
    await page.setViewportSize({ width: 1440, height: 900 });
    expect(await collisions(), chapter.id).toEqual([]);
  }
  await page.goto('./book.html');
  expect(await page.locator('.katex-display .katex-tag').count()).toBeGreaterThan(20);
  await page.emulateMedia({ media: 'print' });
  await page.setViewportSize({ width: 662, height: 900 });
  expect(await collisions()).toEqual([]);
});

test('every figure loads, on chapter pages and in the single-page edition', async ({ page }) => {
  const broken = async () => page.evaluate(async () => {
    const images = [...document.querySelectorAll('.imageblock img')];
    for (const image of images) image.loading = 'eager';
    await Promise.all(images.map((image) => (image.complete ? null : new Promise((resolve) => { image.onload = image.onerror = resolve; }))));
    return { count: images.length, broken: images.filter((image) => !image.naturalWidth).map((image) => image.getAttribute('src')) };
  });
  let count = 0;
  for (const chapter of published) {
    await page.goto(`./${chapter.id}.html`);
    const result = await broken();
    count += result.count;
    expect(result.broken, chapter.id).toEqual([]);
  }
  expect(count).toBeGreaterThan(5);
  await page.goto('./book.html');
  expect((await broken()).broken).toEqual([]);
});

test('notebook chapters print a static rendering instead of the embedded notebook', async ({ page }, testInfo) => {
  await offline(page, testInfo);
  await page.goto('./vector-calculus.html');
  await expect(page.locator('.notebook-print')).toBeHidden();
  await page.emulateMedia({ media: 'print' });
  await expect(page.locator('.notebook-frame')).toBeHidden();
  await expect(page.locator('.notebook-print')).toBeVisible();
  expect(await page.locator('.notebook-print .katex').count()).toBeGreaterThan(10);
  await expect(page.locator('.print-note').first()).toBeVisible();
});

test('the PDF is served beside the book', async ({ request }) => {
  const response = await request.get('./refresh.pdf');
  expect(response.status()).toBe(200);
  expect(response.headers()['content-type']).toBe('application/pdf');
  expect((await response.body()).subarray(0, 5).toString()).toBe('%PDF-');
});
