import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_grpo.py', import.meta.url));
