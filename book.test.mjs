import test from 'node:test';
import assert from 'node:assert/strict';
import { chapters, renderContents, renderChapterNavigation, validateChapters } from './app.js';

test('contents links only published chapters and preserves catalog order', () => {
  validateChapters(chapters);
  const html = renderContents(chapters);
  assert.match(html, /href="\.\/calculus.html"/);
  assert.match(html, /data-state-key="family"/);
  assert.equal((html.match(/<li>/g) ?? []).length, chapters.length);
  for (const chapter of chapters.filter((entry) => entry.status === 'planned')) {
    assert.ok(html.includes(chapter.title));
    assert.ok(!html.includes(`href="./${chapter.id}.html"`));
  }
  assert.ok(html.indexOf('Linear Algebra') < html.indexOf('Probability Theory'));
});

test('publishing another chapter adds its link and navigation without subject branches', () => {
  const extended = [...chapters, { id: 'example', title: 'A & B', description: '<example>', status: 'published' }];
  validateChapters(extended);
  const html = renderContents(extended);
  assert.match(html, /href="\.\/example.html"/);
  assert.match(html, /A &amp; B/);
  assert.match(html, /&lt;example&gt;/);
  const lastPublished = chapters.filter((chapter) => chapter.status === 'published').at(-1);
  assert.match(renderChapterNavigation(extended, lastPublished.id), /rel="next" href="\.\/example.html"/);
  assert.ok(renderChapterNavigation(extended, 'example').includes(`rel="prev" href="./${lastPublished.id}.html"`));
  assert.match(renderChapterNavigation(chapters, 'calculus'), /rel="next" href="\.\/linear-algebra.html"/);
  assert.match(renderChapterNavigation(chapters, 'linear-algebra'), /rel="prev" href="\.\/calculus.html"/);
  assert.equal(renderChapterNavigation(extended, 'probability'), '');
});

test('chapter IDs are unique URL-safe names and statuses are explicit', () => {
  assert.throws(() => validateChapters([...chapters, chapters[0]]), /duplicate/);
  assert.throws(() => validateChapters([{ ...chapters[0], id: '../outside' }]), /Invalid/);
  assert.throws(() => validateChapters([{ ...chapters[0], id: 'index' }]), /Invalid/);
  assert.throws(() => validateChapters([{ ...chapters[0], id: undefined }]), /Invalid/);
  assert.throws(() => validateChapters([{ ...chapters[0], status: 'draft' }]), /Incomplete/);
});