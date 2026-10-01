import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_mixture_of_experts.py', import.meta.url));
