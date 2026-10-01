import test from 'node:test';
import assert from 'node:assert/strict';
import { chapters } from '../app.js';
import { checkChapter } from './check.mjs';
import { pythonTestFiles, registerPythonFiles } from './pyodide.mjs';

// The chapter checks and Python tests are split across a few processes (book/tests/shard-*.test.mjs),
// so a small CI machine boots Pyodide a few times, not once per chapter. Set REFRESH_CHAPTERS to a
// comma-separated list of chapter IDs (or "scratch") to run only those.
export const SHARDS = 3;

export async function registerShard(shard) {
  const only = process.env.REFRESH_CHAPTERS?.split(/[\s,]+/).filter(Boolean);
  const ids = [...chapters.map((chapter) => chapter.id), 'scratch'];
  const mine = (id) => ids.indexOf(id) % SHARDS === shard - 1 && (!only || only.includes(id));

  for (const chapter of chapters.filter((entry) => entry.status === 'published' && entry.layout === 'longform' && !entry.generated && mine(entry.id))) {
    test(`${chapter.id} is complete, consistent, and brief`, () => {
      const outcome = checkChapter(chapter.id, { figures: false });
      assert.deepEqual(Array.isArray(outcome) ? outcome : outcome.problems, []);
    });
  }
  await registerPythonFiles((await pythonTestFiles(chapters.map((chapter) => chapter.id))).filter((entry) => mine(entry.id)));
}
