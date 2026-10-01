import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_gpt.py', import.meta.url));
