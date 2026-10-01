import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_matrix_calculus.py', import.meta.url));
