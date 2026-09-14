import { loadPyodide } from '../pyodide/pyodide.mjs';

let ready;
let currentDriver;
let driverGlobals;
let dispatch;
self.onmessage = async ({ data: request }) => {
  let initialized = false;
  try {
    ready ??= loadPyodide({
      indexURL: new URL('../pyodide/', import.meta.url).href,
      stdin: () => { throw new Error('Interactive stdin is not available; edit values in the Python source.'); },
    }).then(async (runtime) => {
      await runtime.loadPackage('numpy');
      const numpy = runtime.pyimport('numpy');
      numpy.destroy();
      return runtime;
    });
    const runtime = await ready;
    initialized = true;
    self.postMessage({ id: request.id, type: 'running' });
    currentDriver = request.driver ?? currentDriver;
    const emit = (stream, text) => {
      self.postMessage({ id: request.id, type: 'output', stream, text });
    };
    if (request.persistent) {
      if (!dispatch) {
        driverGlobals = runtime.toPy({});
        runtime.runPython(currentDriver, { globals: driverGlobals });
        dispatch = driverGlobals.get('run_request');
      }
      const output = dispatch(JSON.stringify(request.payload), emit);
      self.postMessage({ id: request.id, type: 'result', output: JSON.parse(output) });
      return;
    }
    const globals = runtime.toPy({ payload: JSON.stringify(request.payload) });
    globals.set('emit_output', emit);
    try {
      const output = runtime.runPython(currentDriver, { globals });
      self.postMessage({ id: request.id, type: 'result', output: JSON.parse(output) });
    } finally {
      globals.destroy();
    }
  } catch (error) {
    self.postMessage({ id: request.id, type: 'error', startup: !initialized, error: String(error).slice(-8000) });
  }
};