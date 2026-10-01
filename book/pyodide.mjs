import test from 'node:test';
import { readFile, readdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadPyodide } from 'pyodide';

const root = fileURLToPath(new URL('..', import.meta.url));
let runtime;

async function copyPython(runtimeFS, source, target) {
  runtimeFS.mkdirTree(target);
  for (const entry of await readdir(source, { withFileTypes: true })) {
    if (entry.name === '__pycache__' || /^test_.*\.py$/.test(entry.name)) continue;
    const from = path.join(source, entry.name);
    if (entry.isDirectory()) await copyPython(runtimeFS, from, `${target}/${entry.name}`);
    else if (entry.name.endsWith('.py')) runtimeFS.writeFile(`${target}/${entry.name}`, await readFile(from, 'utf8'));
  }
}

/** A Pyodide runtime with NumPy and the shared `scratch` package importable, as in the notebooks. */
export async function pythonRuntime() {
  if (runtime) return runtime;
  runtime = await loadPyodide();
  await runtime.loadPackage('numpy', { messageCallback: () => {} });
  await copyPython(runtime.FS, path.join(root, 'scratch'), '/book/scratch');
  runtime.runPython("import sys; sys.path.insert(0, '/book')");
  return runtime;
}

/**
 * Registers every `test_*` function in a chapter's Python test file as a node:test case.
 * The chapter's code/ directory is importable, so tests exercise the exact listing sources.
 */
export async function registerPythonTests(testFileUrl) {
  const filename = fileURLToPath(testFileUrl);
  const directory = path.dirname(filename);
  const python = await pythonRuntime();
  const mount = `/book/chapters/${path.basename(directory)}`;
  await copyPython(python.FS, path.join(directory, 'code'), `${mount}/code`).catch((error) => { if (error.code !== 'ENOENT') throw error; });
  python.runPython(`import sys; sys.path.insert(0, ${JSON.stringify(`${mount}/code`)})`);
  const namespace = python.toPy({ __name__: path.basename(filename, '.py') });
  python.runPython(await readFile(filename, 'utf8'), { globals: namespace, filename });
  const names = [...namespace.keys()].filter((name) => name.startsWith('test_'));
  if (!names.length) throw new Error(`${filename} defines no test_ functions`);
  for (const name of names) {
    const fn = namespace.get(name);
    if (typeof fn !== 'function') continue;
    test(`${path.basename(filename)}::${name}`, () => { fn(); });
  }
}
