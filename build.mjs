import { execFileSync } from 'node:child_process';
import { cp, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { build } from 'esbuild';
import { parse, compileScript } from '@vue/compiler-sfc';
import { loadPyodide } from 'pyodide';
import { createHash } from 'node:crypto';
import { chapters, escapeHtml, renderContents, renderChapterNavigation, validateChapters } from './app.js';
import { functionCode } from './chapters/calculus/expressions.js';
import { buildNotebooks } from './jupyter/build.mjs';
import { renderReference } from './chapters/foundations/reference.js';

const root = fileURLToPath(new URL('.', import.meta.url));
const destination = path.join(root, 'dist');
validateChapters(chapters);
const published = chapters.filter((chapter) => chapter.status === 'published');
const shell = await readFile(path.join(root, 'shell.html'), 'utf8');
const renderPage = (values) => shell.replace(/\{\{(\w+)\}\}/g, (_, key) => values[key] ?? '');
await rm(destination, { recursive: true, force: true });
await mkdir(destination, { recursive: true });
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
for (const chapter of published) {
  const source = path.join(root, 'chapters', chapter.id, 'content.adoc');
  let content = execFileSync('asciidoctor', ['-s', '-o', '-', source], { encoding: 'utf8' });
  if (chapter.id === 'foundations') content = content.replace('<div data-foundations></div>', renderReference());
  const script = `chapters/${chapter.id}.js`;
  const output = Object.entries(bundles.metafile.outputs).find(([filename]) => path.resolve(filename) === path.join(destination, script))?.[1];
  const stylesheet = output?.cssBundle ? path.relative(destination, path.resolve(output.cssBundle)).split(path.sep).join('/') : null;
  const assets = (chapter.interactive ? `<script type="module" src="./${script}"></script>` : '')
    + (stylesheet ? `<link rel="stylesheet" href="./${stylesheet}">` : '');
  const navigation = `<a class="contents-link" href="./">Contents</a>${chapter.reference ? `<a class="reference-link" href="#${escapeHtml(chapter.reference)}">Reference</a>` : ''}`;
  const html = renderPage({ title: escapeHtml(chapter.title), lessonTitle: escapeHtml(chapter.lessonTitle ?? ''), description: escapeHtml(chapter.description), content, assets, navigation, pagination: renderChapterNavigation(chapters, chapter.id) });
  await writeFile(path.join(destination, `${chapter.id}.html`), html);
  await cp(source, path.join(destination, `${chapter.id}.adoc`));
}
await writeFile(path.join(destination, 'index.html'), renderPage({ title: 'Contents', description: 'refresh: mathematics and statistics, chapter by chapter.', content: renderContents(chapters) }));
await cp(path.join(root, 'style.css'), path.join(destination, 'style.css'));
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
await buildNotebooks(root, destination);
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