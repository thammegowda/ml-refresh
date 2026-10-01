import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_reinforcement_learning.py', import.meta.url));
