import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_language_models.py', import.meta.url));
