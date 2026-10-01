import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_distributed_training.py', import.meta.url));
