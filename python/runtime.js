import defaultDriver from './runner.py';

export function createPythonRunner() {
  let worker;
  let loadedDriver;
  let pending;
  let sequence = 0;
  function stop(message = 'Execution stopped. Run again to restart Python.') {
    worker?.terminate();
    worker = undefined;
    loadedDriver = undefined;
    if (pending) {
      clearTimeout(pending.timer);
      pending.reject(new Error(message));
      pending = undefined;
    }
  }
  function run(driver, payload, onState = () => {}, onOutput = () => {}) {
    if (pending) return Promise.reject(new Error('Python is already running.'));
    return new Promise((resolve, reject) => {
      const id = ++sequence;
      const loading = !worker;
      pending = { id, resolve, reject, timer: setTimeout(() => stop('Python startup timed out. Try Run again.'), 60000) };
      try {
        worker ??= new Worker(new URL('./python/worker.js', document.baseURI), { type: 'module' });
        onState(loading ? 'Loading Python + NumPy' : 'Running Python');
        worker.onerror = (event) => stop(event.message || 'Python worker failed. Try Run again.');
        worker.onmessage = ({ data: message }) => {
          if (message.id !== pending?.id) return;
          if (message.type === 'output') {
            onOutput({ stream: message.stream, text: message.text });
            return;
          }
          clearTimeout(pending.timer);
          if (message.type === 'running') {
            onState('Running Python');
            pending.timer = setTimeout(() => stop('Execution exceeded 5 seconds. Python was restarted; edit the code before running again.'), 5000);
            return;
          }
          if (message.type === 'error' && message.startup) {
            stop(message.error);
            return;
          }
          const current = pending;
          pending = undefined;
          if (message.type === 'error') current.reject(new Error(message.error));
          else current.resolve(message.output);
        };
        worker.postMessage({ id, driver: loadedDriver === driver ? undefined : driver, persistent: driver === defaultDriver, payload });
        loadedDriver = driver;
      } catch (error) {
        stop(error.message);
      }
    });
  }
  return { run, stop };
}