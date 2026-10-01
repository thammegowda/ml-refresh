import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_notation.py', import.meta.url));
