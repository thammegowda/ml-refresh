import test from 'node:test';
import { execFileSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

test('vector calculus notebook executes all examples and derivative checks', async () => {
  const source = await readFile(new URL('./vector-calculus.ipynb', import.meta.url), 'utf8');
  const python = fileURLToPath(new URL('../../.venv/bin/python', import.meta.url));
  // One kernel runs every example: each pass re-executes all cells in order, and starting a
  // kernel costs more than the cells themselves on a small CI machine.
  execFileSync(python, ['-c', `
import sys
import nbformat
from nbclient import NotebookClient
notebook = nbformat.reads(sys.stdin.read(), as_version=4)
nbformat.validate(notebook)
originals = [cell.source for cell in notebook.cells]
client = NotebookClient(notebook, timeout=120, kernel_name='python3')
with client.setup_kernel():
    for example in ['bowl', 'coupled bowl', 'saddle', 'quartic']:
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != 'code':
                continue
            cell.source = originals[index].replace("example = 'bowl'", f"example = '{example}'")
            cell.outputs = []
            client.execute_cell(cell, index)
        assert 'checks passed' in notebook.cells[-1].outputs[0].text, example
        figures = [output.data['application/vnd.plotly.v1+json'] for cell in notebook.cells if cell.cell_type == 'code' for output in cell.outputs if 'application/vnd.plotly.v1+json' in output.get('data', {})]
        assert len(figures) == 4, example
        assert figures[0]['data'][0]['type'] == 'surface'
        assert figures[0]['data'][0]['z'] == figures[1]['data'][0]['z']
`], { input: source, env: { ...process.env, PATH: `${fileURLToPath(new URL('../../.venv/bin', import.meta.url))}:${process.env.PATH}` }, timeout: 180000 });
});
