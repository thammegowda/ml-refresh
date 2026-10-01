import { execFile, execFileSync } from 'node:child_process';
import { promisify } from 'node:util';
import katex from 'katex';
import { parseHTML } from 'linkedom';
import { macros } from './macros.js';

export function escapeText(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character]);
}

export function renderMath(tex, displayMode = false) {
  try {
    return katex.renderToString(tex, { displayMode, output: 'htmlAndMathml', throwOnError: true, strict: 'error', macros: { ...macros } });
  } catch (error) {
    throw new Error(`Invalid LaTeX ${displayMode ? 'block' : 'inline'} math "${tex}": ${error.message}`);
  }
}

export function longformAttributes(chapterId) {
  return [
    'stem=latexmath', 'source-highlighter=rouge', 'rouge-css=style', 'rouge-style=github',
    'example-caption!', 'table-caption!', 'figure-caption!', 'listing-caption!', 'icons!',
    `imagesdir=figures/${chapterId}`,
  ];
}

// Longform sources fail on any Asciidoctor warning, including missing includes and invalid references.
export function renderAsciidoc(filename, attributes = [], { strict = false } = {}) {
  const options = strict ? ['--failure-level=WARN', '--verbose'] : [];
  return execFileSync('asciidoctor', ['-s', ...options, ...attributes.flatMap((attribute) => ['-a', attribute]), '-o', '-', filename], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
}

const execFileAsync = promisify(execFile);

/** renderAsciidoc in a child process, so several documents can render at once. */
export async function renderAsciidocAsync(filename, attributes = [], { strict = false } = {}) {
  const options = strict ? ['--failure-level=WARN', '--verbose'] : [];
  const { stdout } = await execFileAsync('asciidoctor', ['-s', ...options, ...attributes.flatMap((attribute) => ['-a', attribute]), '-o', '-', filename], { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
  return stdout;
}

export function parseFragment(html) {
  const { document } = parseHTML(`<!doctype html><html><body><div id="fragment-root">${html}</div></body></html>`);
  return { document, root: document.getElementById('fragment-root') };
}

const childWithClass = (element, name) => [...element.children].find((child) => child.classList.contains(name));
const childrenWithClass = (element, name) => [...element.children].filter((child) => child.classList.contains(name));
const skipText = new Set(['PRE', 'CODE', 'SCRIPT', 'STYLE', 'TEXTAREA']);

function textNodes(node, found = []) {
  for (const child of [...node.childNodes]) {
    if (child.nodeType === 3) found.push(child);
    else if (child.nodeType === 1 && !skipText.has(child.tagName) && !child.classList.contains('katex')) textNodes(child, found);
  }
  return found;
}

function replaceWithHtml(document, node, html) {
  const holder = document.createElement('span');
  holder.innerHTML = html;
  node.replaceWith(...holder.childNodes);
}

export function renderInlineMath(document, root) {
  root.normalize();
  for (const node of textNodes(root)) {
    const text = node.textContent;
    if (!text.includes('\\(')) continue;
    let html = '';
    let last = 0;
    for (const match of text.matchAll(/\\\(([\s\S]+?)\\\)/g)) {
      html += escapeText(text.slice(last, match.index)) + renderMath(match[1].trim());
      last = match.index + match[0].length;
    }
    replaceWithHtml(document, node, html + escapeText(text.slice(last)));
  }
  // A "]" inside stem:[...] ends the macro early and leaks the rest of the formula as text.
  for (const node of textNodes(root)) {
    const leak = node.textContent.match(/\\[a-zA-Z]{2,}|[_^]\{/);
    if (leak) throw new Error(`Unrendered LaTeX "${leak[0]}" in text: "${node.textContent.trim().slice(0, 120)}" (escape "]" inside stem:[...] as "\\]")`);
  }
  for (const math of root.querySelectorAll('.katex')) {
    if (math.closest('.katex-display')) continue;
    const next = math.nextSibling;
    if (next?.nodeType === 3 && next.textContent.startsWith(']')) {
      throw new Error(`Inline math "${math.querySelector('annotation')?.textContent}" is followed by a stray "]" (escape "]" inside stem:[...] as "\\]")`);
    }
  }
}

function prependLabel(document, element, className, text) {
  const label = document.createElement('span');
  label.className = className;
  label.textContent = text;
  element.prepend(label, document.createTextNode(' '));
}

function numberSections(document, root, label, titles) {
  const sections = new Map();
  let first = 0;
  for (const sect1 of childrenWithClass(root, 'sect1')) {
    const body = childWithClass(sect1, 'sectionbody');
    if (sect1.classList.contains('unnumbered') || (body && childWithClass(body, 'bibliography'))) continue;
    const number = `${label}.${++first}`;
    titles.set(sect1.firstElementChild.id, sect1.firstElementChild.textContent.trim());
    prependLabel(document, sect1.firstElementChild, 'section-number', number);
    if (sect1.firstElementChild.id) sections.set(sect1.firstElementChild.id, `Section ${number}`);
    let second = 0;
    for (const sect2 of body ? childrenWithClass(body, 'sect2') : []) {
      if (sect2.classList.contains('unnumbered')) continue;
      const nested = `${number}.${++second}`;
      titles.set(sect2.firstElementChild.id, sect2.firstElementChild.textContent.trim());
      prependLabel(document, sect2.firstElementChild, 'section-number', nested);
      if (sect2.firstElementChild.id) sections.set(sect2.firstElementChild.id, `Section ${nested}`);
    }
  }
  return sections;
}

/**
 * Renders math and assigns chapter-scoped numbers to sections, equations, exercises,
 * figures, listings, and tables. Returns HTML plus the metadata other book pages need.
 */
export function enhanceLongform(html, { label, numberEquations = true, numberSectionHeadings = true, numberCaptions = true, solutionLink } = {}) {
  const { document, root } = parseFragment(html);
  const references = new Map();
  const titles = new Map();
  let equation = 0;
  for (const block of root.querySelectorAll('.stemblock')) {
    const content = childWithClass(block, 'content');
    const source = content.textContent.trim().replace(/^\\\[/, '').replace(/\\\]$/, '').trim();
    const numbered = numberEquations && block.id;
    const number = numbered ? `${label}.${++equation}` : null;
    content.innerHTML = renderMath(number ? `${source}\\tag{${number}}` : source, true);
    if (number) references.set(block.id, `(${number})`);
  }
  renderInlineMath(document, root);
  const sections = numberSectionHeadings ? numberSections(document, root, label, titles) : new Map();

  const exercises = [];
  for (const block of root.querySelectorAll('.exampleblock.exercise')) {
    if (!block.id) throw new Error(`Exercise without an ID in ${label}`);
    const number = `${label}.${exercises.length + 1}`;
    const title = childWithClass(block, 'title');
    if (!title) throw new Error(`Exercise ${block.id} needs a title`);
    const heading = title.textContent.trim();
    titles.set(block.id, heading);
    prependLabel(document, title, 'exercise-label', `Exercise ${number}`);
    if (solutionLink) {
      const { href, text } = solutionLink(block.id);
      const link = document.createElement('p');
      link.className = 'exercise-solution-link';
      link.innerHTML = `<a href="${escapeText(href)}">${escapeText(text)}</a>`;
      childWithClass(block, 'content').append(link);
    }
    exercises.push({ id: block.id, number, title: heading });
    references.set(block.id, `Exercise ${number}`);
  }
  const captioned = [['Figure', '.imageblock'], ['Listing', '.listingblock'], ['Table', 'table.tableblock']];
  for (const [kind, selector] of numberCaptions ? captioned : []) {
    let count = 0;
    for (const block of root.querySelectorAll(selector)) {
      const title = [...block.children].find((child) => child.classList.contains('title'));
      if (!title) continue;
      const number = `${kind} ${label}.${++count}`;
      if (block.id) titles.set(block.id, title.textContent.trim());
      prependLabel(document, title, 'caption-label', number);
      if (block.id) references.set(block.id, number);
    }
  }
  for (const link of root.querySelectorAll('a[href^="#"]')) {
    const id = link.getAttribute('href').slice(1);
    if (!root.querySelector(`[id="${id}"]`)) throw new Error(`Broken reference #${id} in ${label}`);
    const text = link.textContent.trim();
    const target = references.get(id) ?? sections.get(id);
    if (target && (text === `[${id}]` || text === titles.get(id))) link.textContent = target;
  }

  const keyEquations = root.querySelector('.key-equations');
  const summary = keyEquations ? keyEquations.cloneNode(true) : null;
  if (summary) {
    for (const element of [summary, ...summary.querySelectorAll('[id]')]) element.removeAttribute('id');
  }
  const sources = new Set([...root.querySelectorAll('ul.bibliography a[id]')].map((anchor) => anchor.id));
  for (const link of root.querySelectorAll('a[href^="#"]')) {
    const id = link.getAttribute('href').slice(1);
    if (!sources.has(id)) continue;
    if (link.textContent.trim() !== `[${id}]`) throw new Error(`Citation of ${id} shows "${link.textContent.trim()}"; cite several sources as <<a>> <<b>>, not <<a,b>>`);
    link.classList.add('citation-link');
  }
  const bibliography = [...root.querySelectorAll('ul.bibliography > li')].map((item) => {
    const anchor = item.querySelector('a[id]');
    if (!anchor) throw new Error(`Bibliography entry without an anchor in ${label}`);
    item.classList.add('bibliography-entry');
    return { id: anchor.id, html: item.innerHTML.trim() };
  });
  const labels = new Map([...sections, ...references]);
  return { html: root.innerHTML, exercises, references, labels, keyEquations: summary?.outerHTML ?? null, bibliography };
}

export function solutionBlocks(html) {
  const { root } = parseFragment(html);
  return [...root.querySelectorAll('.exampleblock.solution')].map((block) => {
    if (!block.id?.startsWith('sol-')) throw new Error(`Solution blocks need an ID of the form sol-<exercise-id>: ${block.id}`);
    return { id: block.id, exerciseId: block.id.slice(4), content: childWithClass(block, 'content')?.innerHTML ?? '' };
  });
}
