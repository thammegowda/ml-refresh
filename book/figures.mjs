import { execFileSync } from 'node:child_process';
import { access, cp, mkdir, readdir } from 'node:fs/promises';
import path from 'node:path';

const exists = (filename) => access(filename).then(() => true, () => false);

export function buildPython(root) {
  return process.env.PYTHON ?? path.join(path.dirname(process.env.JUPYTER ?? path.join(root, '.venv/bin/jupyter')), 'python');
}

/** Browsers refuse to render SVG that is not well-formed XML (an unescaped "&", for example). */
export function validateSvg(files, python) {
  if (!files.length) return;
  try {
    execFileSync(python, ['-c', 'import sys, xml.dom.minidom as m\nfor f in sys.argv[1:]:\n    try: m.parse(f)\n    except Exception as e: sys.exit(f"{f}: {e}")', ...files], { stdio: 'pipe' });
  } catch (error) {
    throw new Error(`Invalid SVG (not well-formed XML): ${String(error.stderr).trim()}`);
  }
}

/**
 * Runs chapters/<id>/figures.py (deterministic matplotlib SVGs) and copies hand-authored
 * chapters/<id>/diagrams/*.svg into dist/figures/<id>/.
 */
export async function buildFigures({ root, destination, chapters, python = buildPython(root) }) {
  for (const chapter of chapters) {
    const directory = path.join(root, 'chapters', chapter.id);
    const output = path.join(destination, 'figures', chapter.id);
    const script = path.join(directory, 'figures.py');
    const diagrams = path.join(directory, 'diagrams');
    const generated = new Set();
    if (await exists(script)) {
      await mkdir(output, { recursive: true });
      execFileSync(python, [script, output], { stdio: 'inherit', env: { ...process.env, PYTHONPATH: [root, path.join(root, 'book')].join(path.delimiter), PYTHONHASHSEED: '0' } });
      for (const filename of await readdir(output)) generated.add(filename);
    }
    if (await exists(diagrams)) {
      await mkdir(output, { recursive: true });
      validateSvg((await readdir(diagrams)).filter((name) => name.endsWith('.svg')).map((name) => path.join(diagrams, name)), python);
      for (const filename of await readdir(diagrams)) {
        if (!filename.endsWith('.svg')) continue;
        if (generated.has(filename)) throw new Error(`Figure ${chapter.id}/${filename} is both generated and hand-authored`);
        await cp(path.join(diagrams, filename), path.join(output, filename));
      }
    }
  }
}
