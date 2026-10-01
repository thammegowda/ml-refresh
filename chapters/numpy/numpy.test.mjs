import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_numpy.py', import.meta.url));
