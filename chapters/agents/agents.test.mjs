import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_agents.py', import.meta.url));
