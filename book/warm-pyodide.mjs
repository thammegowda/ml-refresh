// Pyodide downloads its NumPy wheel on first use. Fetch it once before the tests start, so that
// parallel test processes on a fresh CI install do not all download it at the same time.
import { loadPyodide } from 'pyodide';

const python = await loadPyodide();
await python.loadPackage('numpy', { messageCallback: () => {} });
