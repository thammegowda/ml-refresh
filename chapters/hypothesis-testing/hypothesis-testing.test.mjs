import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_hypothesis.py', import.meta.url));
