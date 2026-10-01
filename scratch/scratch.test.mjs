import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, readdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chapters, parts } from '../app.js';
import { registerPythonTests } from '../book/pyodide.mjs';

const root = fileURLToPath(new URL('..', import.meta.url));
const order = new Map(chapters.map((chapter, index) => [chapter.id, index]));
const appendices = new Set(chapters.filter((chapter) => parts.find((part) => part.id === chapter.part).appendix).map((chapter) => chapter.id));

function importedModules(source) {
  const modules = [];
  for (const match of source.matchAll(/^\s*from\s+scratch\.(\w+)\s+import\b|^\s*import\s+scratch\.(\w+)|^\s*from\s+scratch\s+import\s+([\w, ()]+)/gm)) {
    if (match[1] || match[2]) modules.push(match[1] ?? match[2]);
    else modules.push(...match[3].replace(/[()]/g, '').split(',').map((name) => name.trim()).filter(Boolean));
  }
  return modules;
}

test('scratch modules name a published owner and only import earlier chapters or appendices', async () => {
  const owners = new Map();
  const sources = new Map();
  for (const filename of (await readdir(path.join(root, 'scratch'))).filter((name) => name.endsWith('.py') && name !== '__init__.py' && !name.startsWith('test_'))) {
    const source = await readFile(path.join(root, 'scratch', filename), 'utf8');
    const owner = source.match(/^__chapter__ = "([a-z0-9-]+)"$/m)?.[1];
    assert.ok(owner, `${filename} must declare __chapter__`);
    assert.equal(chapters.find((chapter) => chapter.id === owner)?.status, 'published', `${filename} belongs to unpublished chapter ${owner}`);
    owners.set(filename.slice(0, -3), owner);
    sources.set(`scratch/${filename}`, { owner, source });
  }
  for (const directory of await readdir(path.join(root, 'chapters'))) {
    const code = path.join(root, 'chapters', directory, 'code');
    const files = await readdir(code).catch(() => []);
    for (const filename of files.filter((name) => name.endsWith('.py'))) sources.set(`chapters/${directory}/code/${filename}`, { owner: directory, source: await readFile(path.join(code, filename), 'utf8') });
  }
  for (const [filename, { owner, source }] of sources) {
    for (const module of importedModules(source)) {
      const introducedBy = owners.get(module);
      assert.ok(introducedBy, `${filename} imports unknown scratch module ${module}`);
      assert.ok(appendices.has(introducedBy) || order.get(introducedBy) <= order.get(owner), `${filename} (${owner}) imports scratch.${module} from later chapter ${introducedBy}`);
    }
  }
  assert.deepEqual(importedModules('from scratch.gradcheck import check_gradient\nimport scratch.arrays\nfrom scratch import (precision, arrays)'), ['gradcheck', 'arrays', 'precision', 'arrays']);
});

test('printed listing sources fit an 88-column page', async () => {
  const sources = [...(await readdir(path.join(root, 'scratch'))).map((name) => path.join('scratch', name))];
  for (const directory of await readdir(path.join(root, 'chapters'))) {
    for (const name of await readdir(path.join(root, 'chapters', directory, 'code')).catch(() => [])) sources.push(path.join('chapters', directory, 'code', name));
  }
  for (const filename of sources.filter((name) => name.endsWith('.py') && !path.basename(name).startsWith('test_'))) {
    (await readFile(path.join(root, filename), 'utf8')).split('\n').forEach((line, index) => {
      assert.ok(line.length <= 88, `${filename}:${index + 1} is ${line.length} characters; listings wrap in print beyond 88`);
    });
  }
});

await registerPythonTests(new URL('./test_scratch.py', import.meta.url));
