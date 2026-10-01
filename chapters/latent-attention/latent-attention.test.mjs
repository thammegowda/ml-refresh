import { registerPythonTests } from '../../book/pyodide.mjs';

await registerPythonTests(new URL('./test_latent_attention.py', import.meta.url));
