import { execFileSync } from 'node:child_process';
import { cp, mkdir, readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { loadPyodide } from 'pyodide-notebook';
import { chapters } from '../app.js';

export async function buildNotebooks(root, destination) {
  const contents = path.join(root, 'jupyter/.cache/contents');
  const wheels = path.join(root, 'jupyter/.cache/wheels');
  const jupyter = process.env.JUPYTER ?? path.join(root, '.venv/bin/jupyter');
  await mkdir(contents, { recursive: true });
  for (const chapter of chapters.filter(chapter => chapter.status === 'published' && chapter.notebook)) {
    const filename = `${chapter.id}.ipynb`;
    await cp(path.join(root, 'chapters', chapter.id, filename), path.join(contents, filename));
    await cp(path.join(contents, filename), path.join(destination, filename));
  }
  const wheelVersions = { comm: '0.2.3', plotly: '6.3.1', narwhals: '2.5.0', nbformat: '5.10.4', fastjsonschema: '2.21.2', jupyter_core: '5.8.1', platformdirs: '4.4.0' };
  execFileSync(path.join(path.dirname(jupyter), 'python'), ['-m', 'pip', 'download', '--no-deps', '--dest', wheels, ...Object.entries(wheelVersions).map(([name, version]) => `${name}==${version}`)], { stdio: 'inherit' });
  execFileSync(jupyter, ['lite', 'build', '--lite-dir', path.join(root, 'jupyter'),
    '--contents', contents, '--output-dir', path.join(destination, 'jupyter'),
    ...Object.entries(wheelVersions).flatMap(([name, version]) => ['--piplite-wheels', path.join(wheels, `${name}-${version}-py3-none-any.whl`)]),
    '--apps', 'notebooks', '--apps', 'tree', '--no-sourcemaps'], { stdio: 'inherit' });
  const runtimeDirectory = path.dirname(fileURLToPath(import.meta.resolve('pyodide-notebook')));
  const runtime = await loadPyodide();
  const packages = ['numpy', 'ipython', 'jedi', 'micropip', 'ssl', 'sqlite3', 'jsonschema'];
  await runtime.loadPackage(packages);
  const lock = JSON.parse(await readFile(path.join(runtimeDirectory, 'pyodide-lock.json'), 'utf8'));
  const included = new Set();
  function include(name) {
    name = name.toLowerCase().replace(/[-_.]+/g, '-');
    if (included.has(name)) return;
    included.add(name);
    lock.packages[name].depends.forEach(include);
  }
  packages.forEach(include);
  const deployed = path.join(destination, 'jupyter/pyodide');
  await mkdir(deployed, { recursive: true });
  for (const filename of ['pyodide.js', 'pyodide.asm.js', 'pyodide.asm.wasm', 'python_stdlib.zip', 'pyodide-lock.json']) {
    await cp(path.join(runtimeDirectory, filename), path.join(deployed, filename));
  }
  for (const name of included) {
    const entry = lock.packages[name];
    const filename = path.join(runtimeDirectory, entry.file_name);
    if (createHash('sha256').update(await readFile(filename)).digest('hex') !== entry.sha256) throw new Error(`Invalid notebook package checksum: ${name}`);
    await cp(filename, path.join(deployed, entry.file_name));
  }
}