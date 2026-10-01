import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_normalization.py', import.meta.url));
