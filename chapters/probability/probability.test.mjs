import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_probability.py', import.meta.url));
