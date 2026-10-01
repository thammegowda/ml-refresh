import test from 'node:test';
import { execFileSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

test('NumPy companion notebook runs top to bottom against the shared scratch package', async () => {
  const source = await readFile(new URL('./numpy.ipynb', import.meta.url), 'utf8');
  const root = fileURLToPath(new URL('../../', import.meta.url));
  execFileSync(`${root}.venv/bin/python`, ['-c', `
import sys
import nbformat
from nbclient import NotebookClient
notebook = nbformat.reads(sys.stdin.read(), as_version=4)
nbformat.validate(notebook)
NotebookClient(notebook, timeout=120, kernel_name='python3', resources={'metadata': {'path': ${JSON.stringify(root)}}}).execute()
text = '\\n'.join(output.get('text', '') for cell in notebook.cells if cell.cell_type == 'code' for output in cell.outputs)
for expected in ['(4, 1, 3) + (5, 1) -> (4, 5, 3)', '(4, 3) + (5, 1) -> error', 'naive inf  stable 1002.4', 'bfloat16: 0.5000', ' float16: 4.0000', 'tanh gradient check passed']:
    assert expected in text, (expected, text)
`], { input: source, env: { ...process.env, PATH: `${root}.venv/bin:${process.env.PATH}` }, timeout: 180000 });
});
