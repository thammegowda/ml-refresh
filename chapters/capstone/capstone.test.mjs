import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_capstone.py', import.meta.url));
