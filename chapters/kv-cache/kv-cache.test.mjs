import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_kv_cache.py', import.meta.url));
