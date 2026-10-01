import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_quantization.py', import.meta.url));
