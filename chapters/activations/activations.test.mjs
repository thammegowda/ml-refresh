import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_activations.py', import.meta.url));
