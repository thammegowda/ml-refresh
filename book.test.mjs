import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { book, chapters, parts, chapterLabels, escapeHtml, partLabels, renderContents, renderChapterNavigation, renderNotice, validateChapters } from './app.js';

test('contents links only published chapters and preserves catalog order', () => {
  validateChapters(chapters);
  const html = renderContents(chapters);
  assert.match(html, /href="\.\/calculus.html"/);
  assert.match(html, /data-state-key="family"/);
  assert.equal((html.match(/<li>/g) ?? []).length, chapters.length);
  for (const chapter of chapters.filter((entry) => entry.status === 'planned')) {
    assert.ok(html.includes(escapeHtml(chapter.title)));
    assert.ok(!html.includes(`href="./${chapter.id}.html"`));
  }
  assert.ok(html.indexOf('Linear Algebra') < html.indexOf('Probability Theory'));
  for (const chapter of chapters) assert.ok(html.includes(`title="${escapeHtml(chapter.description)}"`), `${chapter.id} keeps its description as a tooltip`);
  assert.ok(!html.includes('<h4'), 'compact rows use spans, not one heading per chapter');
  assert.match(html, /href="\.\/book.html"/);
  assert.ok(html.includes(`href="./${book.pdf}" download`));
});

test('parts group chapters under numbered headings with continuous chapter numbers', () => {
  const html = renderContents(chapters);
  const headings = [...html.matchAll(/<h3 class="part-heading"[^>]*>(.*?)<\/h3>/g)].map((match) => match[1].replace(/<[^>]+>/g, ' ').trim());
  assert.deepEqual(headings.slice(0, 2), ['Part I Mathematics for Machine Learning', 'Part II Neural Network Fundamentals']);
  assert.equal(headings.at(-1), 'Appendices');
  assert.ok(html.indexOf('Vector Calculus') < html.indexOf('Information Theory'));
  assert.ok(html.indexOf('Information Theory') < html.indexOf('Neural Networks'));
  const labels = chapterLabels(chapters);
  assert.deepEqual(labels.get('foundations'), { kind: 'Chapter', label: '1' });
  assert.deepEqual(labels.get('neural-network'), { kind: 'Chapter', label: String(chapters.findIndex((chapter) => chapter.id === 'neural-network') + 1) });
  assert.deepEqual(labels.get('notation'), { kind: 'Appendix', label: 'A' });
  assert.deepEqual(labels.get('numpy'), { kind: 'Appendix', label: 'B' });
  assert.match(html, /<span class="chapter-number">B<\/span>/);
  assert.equal(partLabels().get('agents'), 'Part VIII');
  assert.equal(partLabels().get('appendices'), '');
  const withoutAgents = chapters.filter((chapter) => chapter.part !== 'agents');
  assert.ok(!renderContents(withoutAgents).includes('part-agents'));
});

test('publishing another chapter adds its link and navigation without subject branches', () => {
  const extended = [...chapters, { id: 'example', title: 'A & B', description: '<example>', status: 'published', part: 'appendices' }, { id: 'later', title: 'Later', description: 'Not yet written.', status: 'planned', part: 'appendices' }];
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
  assert.equal(renderChapterNavigation(extended, 'later'), '');
  assert.match(html, /<div class="chapter-row chapter-planned"[^>]*>.*Later/);
});

test('chapter IDs are unique URL-safe names and statuses, parts, and layouts are explicit', () => {
  assert.throws(() => validateChapters([...chapters, chapters[0]]), /duplicate/);
  assert.throws(() => validateChapters([{ ...chapters[0], id: '../outside' }]), /Invalid/);
  assert.throws(() => validateChapters([{ ...chapters[0], id: 'index' }]), /Invalid/);
  assert.throws(() => validateChapters([{ ...chapters[0], id: 'book' }]), /Invalid/);
  assert.throws(() => validateChapters([{ ...chapters[0], id: undefined }]), /Invalid/);
  assert.throws(() => validateChapters([{ ...chapters[0], status: 'draft' }]), /Incomplete/);
  assert.throws(() => validateChapters([{ ...chapters[0], part: 'missing' }]), /Unknown part/);
  assert.throws(() => validateChapters([{ ...chapters[0], part: 'agents' }, { ...chapters[1], part: 'mathematics' }]), /out of part order/);
  assert.throws(() => validateChapters([{ ...chapters[0], layout: 'poster' }]), /Unknown layout/);
  assert.throws(() => validateChapters([{ ...chapters[0], generated: 'index' }]), /Unknown generated/);
  assert.ok(parts.every((part) => chapters.some((chapter) => chapter.part === part.id)));
});

test('the disclaimer exists in the repository and every page links to it', async () => {
  const disclaimer = await readFile(new URL('./DISCLAIMER.md', import.meta.url), 'utf8');
  assert.match(disclaimer, /^# Disclaimer and Content Policy/);
  assert.ok(disclaimer.includes(book.issues));
  for (const topic of ['AI-generated content', 'no warranty', 'Copyright and ownership', 'Trademarks', 'Privacy']) assert.ok(disclaimer.includes(topic), topic);
  assert.equal(book.disclaimer, `${book.repository}/blob/main/DISCLAIMER.md`);
  const notice = renderNotice();
  assert.ok(notice.includes(`href="${book.disclaimer}"`) && notice.includes(`href="${book.issues}"`));
  assert.match(notice, /AI-generated/);
  assert.ok((await readFile(new URL('./shell.html', import.meta.url), 'utf8')).includes('{{notice}}'));
});
