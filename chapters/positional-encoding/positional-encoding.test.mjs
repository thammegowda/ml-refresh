import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_positional_encoding.py', import.meta.url));
