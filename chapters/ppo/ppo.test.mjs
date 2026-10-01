import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_ppo.py', import.meta.url));
