import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_agent_systems.py', import.meta.url));
