import { chromium } from '@playwright/test';
import { readFile, writeFile } from 'node:fs/promises';
import { PDFDocument, PDFDict, PDFHexString, PDFName, PDFString, StandardFonts, rgb } from 'pdf-lib';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { book } from '../app.js';
import { basePath, serve } from './serve.mjs';

const root = fileURLToPath(new URL('..', import.meta.url));
const destination = path.join(root, 'dist');
const output = path.join(destination, book.pdf);
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
    displayHeaderFooter: false,
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

// Chromium's footer templates cannot hold clickable links, so draw every footer here:
// the disclaimer notice with a link to the full text, and the page number.
const font = await document.embedFont(StandardFonts.Helvetica);
const gray = rgb(0.39, 0.44, 0.43);
const note = 'AI-generated, not reviewed by experts, and likely to contain errors. Disclaimer:';
const linkText = book.disclaimer.replace(/^https:\/\//, '');
const side = 54; // 0.75 in, the side margin above
const baseline = 22;
let size = 7;
const contentWidth = document.getPage(0).getWidth() - 2 * side;
while (size > 5 && font.widthOfTextAtSize(note + linkText, size) + font.widthOfTextAtSize('0000', 8) + 18 > contentWidth) size -= 0.25;
for (const [index, page] of document.getPages().entries()) {
  const number = String(index + 1);
  const linkX = side + font.widthOfTextAtSize(`${note}  `, size);
  const linkWidth = font.widthOfTextAtSize(linkText, size);
  page.drawText(note, { x: side, y: baseline, size, font, color: gray });
  page.drawText(linkText, { x: linkX, y: baseline, size, font, color: rgb(0.03, 0.5, 0.45) });
  page.drawText(number, { x: page.getWidth() - side - font.widthOfTextAtSize(number, 8), y: baseline, size: 8, font, color: gray });
  page.node.addAnnot(document.context.register(document.context.obj({
    Type: 'Annot',
    Subtype: 'Link',
    Rect: [linkX, baseline - 2, linkX + linkWidth, baseline + size],
    Border: [0, 0, 0],
    A: { Type: 'Action', S: 'URI', URI: PDFString.of(book.disclaimer) },
  })));
}
const bytes = await document.save({ useObjectStreams: false });
await writeFile(output, bytes);
console.log(`Wrote ${path.relative(root, output)}: ${document.getPageCount()} pages, ${(bytes.length / 1e6).toFixed(1)} MB, ${format}, ${repaired} outline titles repaired`);
