import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_fine_tuning.py', import.meta.url));
