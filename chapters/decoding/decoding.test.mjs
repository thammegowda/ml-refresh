import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_decoding.py', import.meta.url));
