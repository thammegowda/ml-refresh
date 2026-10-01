import test from 'node:test';
import assert from 'node:assert/strict';
import { chapters } from '../app.js';
import { checkChapter } from './check.mjs';

// The build runs every figure script; this suite checks structure, links, and brevity quickly.
for (const chapter of chapters.filter((entry) => entry.status === 'published' && entry.layout === 'longform' && !entry.generated)) {
  test(`${chapter.id} is complete, consistent, and brief`, () => {
    const outcome = checkChapter(chapter.id, { figures: false });
    assert.deepEqual(Array.isArray(outcome) ? outcome : outcome.problems, []);
  });
}
