import test from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { enhanceLongform, longformAttributes, parseFragment, renderAsciidoc, renderMath, solutionBlocks } from './render.mjs';

const fixture = (name) => fileURLToPath(new URL(`./fixtures/${name}`, import.meta.url));
const raw = renderAsciidoc(fixture('sample.adoc'), longformAttributes('sample'), { strict: true });
const result = enhanceLongform(raw, { label: 'B', solutionLink: (id) => ({ href: `./solutions.html#sol-${id}`, text: 'Solution in Appendix D' }) });
const { root } = parseFragment(result.html);

test('math renders with shared macros, numbering, and cross-references', () => {
  assert.equal(root.querySelectorAll('.stemblock .katex-display').length, 3);
  assert.match(root.querySelector('#eq-softmax').textContent, /\(B\.1\)/);
  assert.ok(!root.querySelector('.stemblock:not([id])').textContent.includes('(B.'));
  assert.equal(root.querySelector('a[href="#eq-softmax"]').textContent, '(B.1)');
  assert.ok(root.querySelector('.paragraph .katex'));
  const prose = root.cloneNode(true);
  for (const listing of prose.querySelectorAll('pre')) listing.remove();
  assert.ok(!prose.innerHTML.includes('\\('));
  assert.match(root.querySelector('pre').textContent, /\\\(literal\\\)/);
  assert.throws(() => renderMath('\\frac{1}{'), /Invalid LaTeX/);
});

test('sections, captions, and exercises are numbered by chapter label', () => {
  const headings = [...root.querySelectorAll('h2, h3')].map((heading) => heading.textContent.trim());
  assert.deepEqual(headings, ['B.1 Softmax', 'B.1.1 Details', 'B.2 Exercises', 'References']);
  assert.match(root.querySelector('.listingblock .title').textContent, /^Listing B\.1 Pattern/);
  assert.match(root.querySelector('#fig-sample .title').textContent, /^Figure B\.1 A tiny figure/);
  assert.equal(root.querySelector('a[href="#fig-sample"]').textContent, 'Figure B.1');
  assert.equal(root.querySelector('a[href="#ex-sample-sum"]').textContent, 'Exercise B.1');
  assert.equal(result.labels.get('_details'), 'Section B.1.1');
  assert.equal(root.querySelector('a[href="#_details"]').textContent, 'Section B.1.1');
  assert.match(root.querySelector('#ex-sample-sum .title').textContent, /^Exercise B\.1 ★ Sum/);
  assert.equal(root.querySelector('#ex-sample-sum .exercise-solution-link a').getAttribute('href'), './solutions.html#sol-ex-sample-sum');
  assert.deepEqual(result.exercises, [{ id: 'ex-sample-sum', number: 'B.1', title: '★ Sum' }]);
  assert.equal(root.querySelector('img').getAttribute('src'), 'figures/sample/diagram.svg');
});

test('key equations and bibliography are extracted for generated appendices', () => {
  assert.ok(result.keyEquations.includes('katex'));
  assert.ok(!/\sid="/.test(result.keyEquations));
  assert.deepEqual(result.bibliography.map((entry) => entry.id), ['vaswani2017']);
  assert.match(result.bibliography[0].html, /Attention is all you need/);
});

test('strict AsciiDoc rendering fails on missing includes and solution IDs are validated', () => {
  assert.throws(() => renderAsciidoc(fixture('broken.adoc'), longformAttributes('broken'), { strict: true }));
  assert.deepEqual(solutionBlocks('<div id="sol-ex-a" class="exampleblock solution"><div class="content"><p>Yes</p></div></div>'), [{ id: 'sol-ex-a', exerciseId: 'ex-a', content: '<p>Yes</p>' }]);
  assert.throws(() => solutionBlocks('<div id="ex-a" class="exampleblock solution"><div class="content"></div></div>'), /sol-/);
  assert.throws(() => enhanceLongform('<p>See <a href="#eq-missing">[eq-missing]</a>.</p>', { label: 'X' }), /Broken reference #eq-missing/);
  // An unescaped "]" inside stem:[...] ends the macro early; both symptoms must fail the build.
  assert.throws(() => enhanceLongform('<p>\\(a\\)=\\sum_y b].</p>', { label: 'X' }), /Unrendered LaTeX/);
  assert.throws(() => enhanceLongform('<p>\\(x[i\\)] here.</p>', { label: 'X' }), /stray "\]"/);
  assert.doesNotThrow(() => enhanceLongform('<p>Fine \\(x[i]\\) and [a link].</p>', { label: 'X' }));
  const bibliography = '<div class="ulist bibliography"><ul class="bibliography"><li><p><a id="a2020"></a>[a2020] A.</p></li></ul></div>';
  assert.throws(() => enhanceLongform(`<p>See <a href="#a2020">b2021</a>.</p>${bibliography}`, { label: 'X' }), /cite several sources/);
  assert.doesNotThrow(() => enhanceLongform(`<p>See <a href="#a2020">[a2020]</a>.</p>${bibliography}`, { label: 'X' }));
});
