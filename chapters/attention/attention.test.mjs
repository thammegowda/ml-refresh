import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_attention.py', import.meta.url));
