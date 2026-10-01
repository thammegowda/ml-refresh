import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_optimization.py', import.meta.url));
