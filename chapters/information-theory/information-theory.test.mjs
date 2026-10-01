import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_information.py', import.meta.url));
