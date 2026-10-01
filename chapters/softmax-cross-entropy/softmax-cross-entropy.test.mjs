import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_softmax_cross_entropy.py', import.meta.url));
