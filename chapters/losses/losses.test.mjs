import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_losses.py', import.meta.url));
