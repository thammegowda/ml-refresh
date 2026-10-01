import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_tokenization.py', import.meta.url));
