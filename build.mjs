import { execFileSync, spawn } from 'node:child_process';
import { availableParallelism } from 'node:os';
import { cp, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { build } from 'esbuild';
import { parse, compileScript } from '@vue/compiler-sfc';
import { loadPyodide } from 'pyodide';
import { createHash } from 'node:crypto';
import { book, chapters, chapterLabels, escapeHtml, partLabels, parts, renderContents, renderChapterNavigation, renderNotice, validateChapters } from './app.js';
import { functionCode } from './chapters/calculus/expressions.js';
import { renderReference } from './chapters/foundations/reference.js';
import { enhanceLongform, longformAttributes, renderAsciidocAsync, solutionBlocks } from './book/render.mjs';
import { renderNotebook } from './book/notebook.mjs';
import { fragmentIds, printableFragment, printNote, renderEdition, renderOpener, resolveLinks } from './book/edition.mjs';
import { renderBibliography, renderFormulaSheets, renderSolutions } from './book/generated.mjs';
import { buildFigures } from './book/figures.mjs';

const root = fileURLToPath(new URL('.', import.meta.url));
const destination = path.join(root, 'dist');
validateChapters(chapters);
const published = chapters.filter((chapter) => chapter.status === 'published');
const shell = await readFile(path.join(root, 'shell.html'), 'utf8');
const renderPage = (values) => shell.replace(/\{\{(\w+)\}\}/g, (_, key) => ({ notice: renderNotice(), bookTitle: escapeHtml(book.title), ...values })[key] ?? '');
await rm(destination, { recursive: true, force: true });
await mkdir(destination, { recursive: true });
// JupyterLite and its Pyodide build in a separate process while this one renders the book.
const notebooksBuilt = new Promise((resolve, reject) => {
  const child = spawn(process.execPath, [path.join(root, 'jupyter/build.mjs'), destination], { stdio: 'inherit' });
  child.on('error', reject);
  child.on('exit', (code) => (code === 0 ? resolve() : reject(new Error(`jupyter/build.mjs exited with code ${code}`))));
});
notebooksBuilt.catch(() => {});
const bundles = await build({
  entryPoints: Object.fromEntries([
    ['app', path.join(root, 'app.js')],
    ...published.filter((chapter) => chapter.interactive).map((chapter) => [`chapters/${chapter.id}`, path.join(root, 'chapters', chapter.id, 'index.js')]),
  ]),
  outdir: destination,
  bundle: true,
  loader: { '.py': 'text', '.py.in': 'text' },
  plugins: [{
    name: 'vue',
    setup(builder) {
      builder.onLoad({ filter: /\.vue$/ }, async ({ path: filename }) => {
        const source = await readFile(filename, 'utf8');
        const { descriptor, errors } = parse(source, { filename });
        if (errors.length) throw errors[0];
        if (descriptor.styles.length) throw new Error('Import chapter CSS from index.js instead of using SFC style blocks.');
        const compiled = compileScript(descriptor, {
          id: path.relative(root, filename),
          inlineTemplate: true,
          isProd: true,
        });
        return { contents: compiled.content, loader: 'js', resolveDir: path.dirname(filename) };
      });
    },
  }],
  define: {
    __VUE_OPTIONS_API__: 'false',
    __VUE_PROD_DEVTOOLS__: 'false',
    __VUE_PROD_HYDRATION_MISMATCH_DETAILS__: 'false',
  },
  splitting: true,
  chunkNames: 'chunks/[name]-[hash]',
  metafile: true,
  minify: true,
  format: 'esm',
  target: 'es2022',
  legalComments: 'eof',
});
const labels = chapterLabels(chapters);
const partNames = partLabels();
const planned = new Set(chapters.filter((chapter) => chapter.status === 'planned').map((chapter) => chapter.id));
const solutionsChapter = chapters.find((chapter) => chapter.generated === 'solutions');
const exists = (filename) => readFile(filename).then(() => true, () => false);
const classicHeader = (title, lessonTitle = '') => `<h1>${title} <span class="lesson-title">${lessonTitle}</span></h1>`;
const pagePrintNote = (chapter) => printNote('Interactive version', `${book.url}${chapter.id}.html`).replace('class="print-note"', 'class="print-only print-note"');
const figuresBuilt = buildFigures({ root, destination, chapters: published.filter((chapter) => chapter.layout === 'longform' && !chapter.generated) });
figuresBuilt.catch(() => {});

/** Runs fn over items with at most `limit` in flight, returning results in input order. */
async function mapConcurrent(items, limit, fn) {
  const results = new Array(items.length);
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (next < items.length) {
      const index = next++;
      results[index] = await fn(items[index]);
    }
  }));
  return results;
}

// Stage 1: render every authored page. Longform sources fail on warnings, invalid math, and broken anchors.
// Asciidoctor runs in child processes, one per core; pages are stored in book order.
const pages = new Map();
const renderedPages = await mapConcurrent(published.filter((entry) => !entry.generated), availableParallelism(), async (chapter) => {
  const label = labels.get(chapter.id);
  const directory = path.join(root, 'chapters', chapter.id);
  const source = path.join(directory, 'content.adoc');
  if (chapter.layout === 'longform') {
    const [content, solutionsHtml] = await Promise.all([
      renderAsciidocAsync(source, longformAttributes(chapter.id), { strict: true }),
      exists(path.join(directory, 'solutions.adoc')).then((found) => (found ? renderAsciidocAsync(path.join(directory, 'solutions.adoc'), longformAttributes(chapter.id), { strict: true }) : null)),
    ]);
    const result = enhanceLongform(content, {
      label: label.label,
      solutionLink: (id) => ({ href: `./${solutionsChapter.id}.html#sol-${id}`, text: `Solution in Appendix ${labels.get(solutionsChapter.id).label}` }),
    });
    const solutions = solutionsHtml === null ? []
      : solutionBlocks(enhanceLongform(solutionsHtml, { label: label.label, numberEquations: false, numberSectionHeadings: false, numberCaptions: false }).html);
    return { chapter, label, ...result, solutions };
  }
  let html = await renderAsciidocAsync(source);
  if (chapter.id === 'foundations') html = html.replace('<div data-foundations></div>', renderReference());
  const notebook = chapter.notebook ? renderNotebook(JSON.parse(await readFile(path.join(directory, `${chapter.id}.ipynb`), 'utf8')), chapter.id) : '';
  return { chapter, label, html, notebook, labels: new Map(), exercises: [], solutions: [], keyEquations: null, bibliography: [] };
});
for (const page of renderedPages) pages.set(page.chapter.id, page);

// Stage 2: generated appendices collect exercises, key equations, and sources in book order.
const authored = [...pages.values()];
const generators = { solutions: renderSolutions, 'formula-sheets': renderFormulaSheets, bibliography: renderBibliography };
for (const chapter of published.filter((entry) => entry.generated)) {
  pages.set(chapter.id, { chapter, label: labels.get(chapter.id), html: generators[chapter.generated](authored), labels: new Map() });
}

// Stage 3: validate and label links between pages.
const targets = new Map([...pages.values()].map((page) => [page.chapter.id, { ids: fragmentIds(page.html), labels: page.labels, name: `${page.label.kind} ${page.label.label}` }]));
for (const page of pages.values()) page.html = resolveLinks(page.html, { pages: targets, planned, source: page.chapter.id });

// Stage 4: write chapter pages and their downloadable sources.
for (const chapter of published) {
  const page = pages.get(chapter.id);
  const longform = chapter.layout === 'longform';
  const script = `chapters/${chapter.id}.js`;
  const output = Object.entries(bundles.metafile.outputs).find(([filename]) => path.resolve(filename) === path.join(destination, script))?.[1];
  const stylesheet = output?.cssBundle ? path.relative(destination, path.resolve(output.cssBundle)).split(path.sep).join('/') : null;
  const assets = (chapter.interactive ? `<script type="module" src="./${script}"></script>` : '')
    + (stylesheet ? `<link rel="stylesheet" href="./${stylesheet}">` : '')
    + (longform ? '<link rel="stylesheet" href="./katex/katex.min.css"><link rel="stylesheet" href="./longform.css">' : '')
    + (page.notebook ? '<link rel="stylesheet" href="./katex/katex.min.css" media="print"><link rel="stylesheet" href="./longform.css" media="print">' : '');
  const navigation = `<a class="contents-link" href="./">Contents</a>${chapter.reference ? `<a class="reference-link" href="#${escapeHtml(chapter.reference)}">Reference</a>` : ''}`;
  const links = [
    chapter.companion ? `<a href="./jupyter/notebooks/index.html?path=${chapter.id}.ipynb">Run the companion notebook</a><a href="./${chapter.id}.ipynb" download>Download notebook</a>` : '',
    chapter.generated ? '' : `<a href="./${chapter.id}.adoc" download>AsciiDoc source</a>`,
  ].join('');
  const content = longform
    ? `<article class="longform-chapter longform-content" aria-labelledby="${chapter.id}-title">${renderOpener({ chapter, label: page.label, links })}<div class="chapter-body">${page.html}</div></article>`
    : (chapter.interactive ? pagePrintNote(chapter) : '') + page.html + (page.notebook ? `<section class="print-only notebook-print longform-content" aria-label="Printable notebook">${page.notebook}</section>` : '');
  const headerTitle = longform
    ? `<p class="header-title">${escapeHtml(page.label.kind)} ${escapeHtml(page.label.label)} <span class="lesson-title">${escapeHtml(chapter.title)}</span></p>`
    : classicHeader(escapeHtml(chapter.title), escapeHtml(chapter.lessonTitle ?? ''));
  const html = renderPage({ title: escapeHtml(chapter.title), description: escapeHtml(chapter.description), content, assets, navigation, headerTitle, bodyClass: longform ? 'longform' : '', pagination: renderChapterNavigation(chapters, chapter.id) });
  await writeFile(path.join(destination, `${chapter.id}.html`), html);
  if (!chapter.generated) await cp(path.join(root, 'chapters', chapter.id, 'content.adoc'), path.join(destination, `${chapter.id}.adoc`));
}
await writeFile(path.join(destination, 'index.html'), renderPage({ title: 'Contents', headerTitle: classicHeader('Contents'), description: `${escapeHtml(book.title)}: ${escapeHtml(book.subtitle.toLowerCase())}, chapter by chapter.`, content: renderContents(chapters) }));
await cp(path.join(root, 'style.css'), path.join(destination, 'style.css'));
await cp(path.join(root, 'book/longform.css'), path.join(destination, 'longform.css'));
const katexDirectory = path.dirname(fileURLToPath(import.meta.resolve('katex')));
await mkdir(path.join(destination, 'katex/fonts'), { recursive: true });
await cp(path.join(katexDirectory, 'katex.min.css'), path.join(destination, 'katex/katex.min.css'));
for (const font of (await readdir(path.join(katexDirectory, 'fonts'))).filter((filename) => filename.endsWith('.woff2'))) {
  await cp(path.join(katexDirectory, 'fonts', font), path.join(destination, 'katex/fonts', font));
}

// Stage 5: the single-page edition that the PDF is printed from.
const chapterStylesheets = Object.values(bundles.metafile.outputs).filter((entry) => entry.cssBundle).map((entry) => path.relative(destination, path.resolve(entry.cssBundle)).split(path.sep).join('/'));
const fragments = new Map(published.map((chapter) => {
  const page = pages.get(chapter.id);
  const note = chapter.notebook ? printNote('Runnable notebook', `${book.url}${chapter.id}.html`)
    : chapter.companion ? printNote('Runnable companion notebook', `${book.url}jupyter/notebooks/index.html?path=${chapter.id}.ipynb`)
    : chapter.interactive ? printNote('Interactive version', `${book.url}${chapter.id}.html`) : '';
  const html = page.notebook ? `<div class="notebook-print longform-content">${page.notebook}</div>` : page.html;
  return [chapter.id, printableFragment(chapter.layout === 'longform' ? `<div class="longform-content">${html}</div>` : html, { note })];
}));
await writeFile(path.join(destination, 'book.html'), renderEdition({
  book, parts, chapters, labels, partNames, fragments,
  stylesheets: ['style.css', 'katex/katex.min.css', 'longform.css', ...new Set(chapterStylesheets)],
  date: new Date().toISOString().slice(0, 10),
}));
const pythonDirectory = path.join(destination, 'python');
await mkdir(pythonDirectory, { recursive: true });
await cp(path.join(root, 'python/worker.js'), path.join(pythonDirectory, 'worker.js'));
const runtimeDirectory = path.dirname(fileURLToPath(import.meta.resolve('pyodide')));
const deployedRuntime = path.join(destination, 'pyodide');
await mkdir(deployedRuntime, { recursive: true });
const runtime = await loadPyodide();
await runtime.loadPackage('numpy');
const lock = JSON.parse(await readFile(path.join(runtimeDirectory, 'pyodide-lock.json'), 'utf8'));
const packageNames = new Set();
function includePackage(name) {
  if (packageNames.has(name)) return;
  packageNames.add(name);
  for (const dependency of lock.packages[name].depends) includePackage(dependency);
}
includePackage('numpy');
for (const filename of ['pyodide.mjs', 'pyodide.asm.mjs', 'pyodide.asm.wasm', 'python_stdlib.zip', 'pyodide-lock.json']) {
  await cp(path.join(runtimeDirectory, filename), path.join(deployedRuntime, filename));
}
for (const name of packageNames) {
  const entry = lock.packages[name];
  const wheel = await readFile(path.join(runtimeDirectory, entry.file_name));
  if (createHash('sha256').update(wheel).digest('hex') !== entry.sha256) throw new Error(`Invalid package checksum: ${name}`);
  await writeFile(path.join(deployedRuntime, entry.file_name), wheel);
}
await cp(path.join(root, 'chapters/linear-algebra/lesson.py'), path.join(destination, 'linear-algebra.py'));
await cp(path.join(root, 'chapters/trigonometry/lesson.py'), path.join(destination, 'trigonometry.py'));
await Promise.all([notebooksBuilt, figuresBuilt]);
await writeFile(path.join(destination, 'calculus.py'), functionCode(await readFile(path.join(root, 'chapters/calculus/lesson.py.in'), 'utf8'), 'power'));
const notices = [];
const modules = path.join(root, 'node_modules');
for (const entry of await readdir(modules, { withFileTypes: true })) {
  if (!entry.isDirectory() || entry.name.startsWith('.')) continue;
  const packages = entry.name.startsWith('@')
    ? (await readdir(path.join(modules, entry.name))).map((name) => `${entry.name}/${name}`)
    : [entry.name];
  for (const name of packages) {
    const directory = path.join(modules, name);
    for (const filename of await readdir(directory)) {
      if (/^licen[sc]e(\..*)?$/i.test(filename)) {
        notices.push(`${name}\n${await readFile(path.join(directory, filename), 'utf8')}`);
      }
    }
  }
}
await writeFile(path.join(destination, 'THIRD-PARTY.txt'), notices.join('\n\n---\n\n'));
console.log(`Built contents and ${published.length} chapters in ${destination}`);