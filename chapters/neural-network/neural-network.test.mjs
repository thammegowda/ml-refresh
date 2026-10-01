import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_neural_network.py', import.meta.url));
