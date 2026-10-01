import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_learning.py', import.meta.url));
