import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_multi_head_attention.py', import.meta.url));
