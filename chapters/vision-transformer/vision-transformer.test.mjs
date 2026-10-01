import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_vision_transformer.py', import.meta.url));
