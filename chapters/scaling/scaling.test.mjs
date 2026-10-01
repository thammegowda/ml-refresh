import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_scaling.py', import.meta.url));
