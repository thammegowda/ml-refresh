import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_flash_attention.py', import.meta.url));
