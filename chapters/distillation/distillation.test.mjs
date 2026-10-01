import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_distillation.py', import.meta.url));
