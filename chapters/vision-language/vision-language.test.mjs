import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_vision_language.py', import.meta.url));
