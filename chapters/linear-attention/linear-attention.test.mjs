import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_linear_attention.py', import.meta.url));
