import assert from 'node:assert/strict';
import { before, describe, it } from 'node:test';
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

// Chapters share one interpreter, so each suite first drops the previous chapter's modules
// (several chapters have a code/solutions.py, for example) and its import path.
const forgetChapterModules = `
import sys
for name, module in list(sys.modules.items()):
    if (getattr(module, '__file__', None) or '').startswith('/book/chapters/'):
        del sys.modules[name]
sys.path[:] = [entry for entry in sys.path if not entry.startswith('/book/chapters/')]
`;

/**
 * Registers each Python test file as a suite of node:test cases that share one Pyodide
 * runtime. Booting Pyodide is the most expensive step, so a process boots it once.
 * entries: [{ id, file }], where id is a chapter ID or 'scratch'.
 */
export async function registerPythonFiles(entries) {
  for (const { id, file } of entries) {
    const source = await readFile(file, 'utf8');
    const names = [...source.matchAll(/^def (test_\w+)\(/gm)].map((match) => match[1]);
    if (!names.length) throw new Error(`${file} defines no test_ functions`);
    describe(path.relative(root, file), () => {
      let namespace;
      before(async () => {
        const python = await pythonRuntime();
        python.runPython(forgetChapterModules);
        if (id !== 'scratch') {
          const mount = `/book/chapters/${id}/code`;
          await copyPython(python.FS, path.join(path.dirname(file), 'code'), mount).catch((error) => { if (error.code !== 'ENOENT') throw error; });
          python.runPython(`import sys; sys.path.insert(0, ${JSON.stringify(mount)})`);
        }
        namespace = python.toPy({ __name__: path.basename(file, '.py') });
        python.runPython(source, { globals: namespace, filename: file });
        const defined = [...namespace.keys()].filter((name) => name.startsWith('test_')).sort();
        assert.deepEqual(defined, [...names].sort(), `${file}: top-level test_ functions`);
      });
      for (const name of names) it(name, () => { namespace.get(name)(); });
    });
  }
}

/** Every Python test file, in book order: one per chapter that has one, then the scratch package. */
export async function pythonTestFiles(chapterIds) {
  const entries = [];
  for (const id of chapterIds) {
    const directory = path.join(root, 'chapters', id);
    const files = (await readdir(directory).catch(() => [])).filter((name) => /^test_.*\.py$/.test(name)).sort();
    entries.push(...files.map((name) => ({ id, file: path.join(directory, name) })));
  }
  return [...entries, { id: 'scratch', file: path.join(root, 'scratch/test_scratch.py') }];
}
