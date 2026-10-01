import { chromium } from '@playwright/test';
import { readFile, writeFile } from 'node:fs/promises';
import { PDFDocument, PDFDict, PDFHexString, PDFName } from 'pdf-lib';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { book } from '../app.js';
import { basePath, serve } from './serve.mjs';

const root = fileURLToPath(new URL('..', import.meta.url));
const destination = path.join(root, 'dist');
const output = path.join(destination, 'refresh.pdf');
const format = process.env.REFRESH_PDF_FORMAT ?? 'Letter';

const { server, url } = await serve(destination, { base: basePath });
const browser = await chromium.launch();
let headings;
try {
  const page = await browser.newPage();
  const failures = [];
  await page.route('**/*', (route) => (route.request().url().startsWith(url) ? route.continue() : route.abort()));
  page.on('requestfailed', (request) => failures.push(`${request.failure()?.errorText} ${request.url()}`));
  page.on('response', (response) => { if (response.status() >= 400) failures.push(`${response.status()} ${response.url()}`); });
  await page.goto(`${url}book.html`, { waitUntil: 'networkidle' });
  await page.evaluate(async () => {
    await document.fonts.ready;
    await Promise.all([...document.images].map((image) => (image.complete ? null : new Promise((resolve) => { image.onload = image.onerror = resolve; }))));
  });
  if (failures.length) throw new Error(`book.html did not load completely:\n${failures.join('\n')}`);
  // Relative links would resolve to local files inside a PDF; point them at the online book instead.
  await page.evaluate((online) => {
    for (const link of document.querySelectorAll('a[href]')) {
      const href = link.getAttribute('href');
      if (!href.startsWith('#') && !/^[a-z][a-z0-9+.-]*:/i.test(href)) link.href = new URL(href, online).href;
    }
  }, book.url);
  await page.emulateMedia({ media: 'print' });
  headings = await page.evaluate(() => [...document.querySelectorAll('h1, h2, h3, h4, h5, h6')].filter((heading) => heading.getClientRects().length).map((heading) => heading.textContent.replace(/\s+/g, ' ').trim()));
  await page.pdf({
    path: output,
    format,
    printBackground: true,
    outline: true,
    tagged: true,
    displayHeaderFooter: true,
    headerTemplate: '<span></span>',
    footerTemplate: '<div style="width:100%;text-align:center;font:8px Georgia,serif;color:#64716d"><span class="pageNumber"></span></div>',
    margin: { top: '0.6in', bottom: '0.65in', left: '0.75in', right: '0.75in' },
  });
} finally {
  await browser.close();
  server.close();
}

// Chromium drops the spaces where a heading wraps when it builds the outline; restore them from the DOM.
const compact = (text) => text.replace(/\s+/g, '');
const document = await PDFDocument.load(await readFile(output));
const outlines = document.catalog.lookup(PDFName.of('Outlines'), PDFDict);
let next = 0;
let repaired = 0;
function repair(item) {
  for (let node = item; node; node = node.lookupMaybe(PDFName.of('Next'), PDFDict)) {
    const title = node.lookup(PDFName.of('Title')).decodeText();
    const index = headings.findIndex((heading, position) => position >= next && compact(heading) === compact(title));
    if (index >= 0) {
      next = index + 1;
      if (headings[index] !== title) {
        node.set(PDFName.of('Title'), PDFHexString.fromText(headings[index]));
        repaired++;
      }
    }
    const child = node.lookupMaybe(PDFName.of('First'), PDFDict);
    if (child) repair(child);
  }
}
if (outlines) repair(outlines.lookupMaybe(PDFName.of('First'), PDFDict));
const bytes = await document.save({ useObjectStreams: false });
await writeFile(output, bytes);
console.log(`Wrote ${path.relative(root, output)}: ${document.getPageCount()} pages, ${(bytes.length / 1e6).toFixed(1)} MB, ${format}, ${repaired} outline titles repaired`);
