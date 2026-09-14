import test from 'node:test';
import { execFileSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

test('vector calculus notebook executes all examples and derivative checks', async () => {
  const source = await readFile(new URL('./vector-calculus.ipynb', import.meta.url), 'utf8');
  const python = fileURLToPath(new URL('../../.venv/bin/python', import.meta.url));
  execFileSync(python, ['-c', `
import copy, json, sys
import nbformat
from nbclient import NotebookClient
notebook = nbformat.reads(sys.stdin.read(), as_version=4)
nbformat.validate(notebook)
for example in ['bowl', 'coupled bowl', 'saddle', 'quartic']:
    current = copy.deepcopy(notebook)
    for cell in current.cells:
        if cell.cell_type == 'code':
            cell.source = cell.source.replace("example = 'bowl'", f"example = '{example}'")
    NotebookClient(current, timeout=120, kernel_name='python3').execute()
    assert 'checks passed' in current.cells[-1].outputs[0].text
    figures = [output.data['application/vnd.plotly.v1+json'] for cell in current.cells if cell.cell_type == 'code' for output in cell.outputs if 'application/vnd.plotly.v1+json' in output.get('data', {})]
    assert len(figures) == 4
    assert figures[0]['data'][0]['type'] == 'surface'
    assert figures[0]['data'][0]['z'] == figures[1]['data'][0]['z']
`], { input: source, env: { ...process.env, PATH: `${fileURLToPath(new URL('../../.venv/bin', import.meta.url))}:${process.env.PATH}` }, timeout: 180000 });
});