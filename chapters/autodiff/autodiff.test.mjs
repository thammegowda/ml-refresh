import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_autodiff.py', import.meta.url));
