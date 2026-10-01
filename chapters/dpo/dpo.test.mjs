import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_dpo.py', import.meta.url));
