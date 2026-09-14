import { readFile } from 'node:fs/promises';
import { loadPyodide } from 'pyodide';

const runtime = await loadPyodide();
await runtime.loadPackage('numpy');
const driver = await readFile(new URL('./runner.py', import.meta.url), 'utf8');

export function runPython(code, emit = () => {}) {
  const session = createPythonSession();
  try { return session.run({ code }, emit); }
  finally { session.destroy(); }
}

export function createPythonSession() {
  const globals = runtime.toPy({});
  runtime.runPython(driver, { globals });
  const request = globals.get('run_request');
  return {
    run(payload, emit = () => {}) { return JSON.parse(request(JSON.stringify(payload), emit)); },
    destroy() { request.destroy(); globals.destroy(); },
  };
}