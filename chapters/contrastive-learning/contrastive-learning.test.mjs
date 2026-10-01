import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_contrastive.py', import.meta.url));
