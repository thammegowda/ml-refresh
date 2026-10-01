// Validates long-form chapters without touching dist/, so several chapters can be checked at once.
// Usage: node book/check.mjs <chapter-id> [...]
import { execFileSync } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chapterLabels, chapters } from '../app.js';
import { buildPython, validateSvg } from './figures.mjs';
import { renderSolutions } from './generated.mjs';
import { enhanceLongform, longformAttributes, parseFragment, renderAsciidoc, solutionBlocks } from './render.mjs';
import { sourceEntries } from './sources.mjs';

const root = fileURLToPath(new URL('..', import.meta.url));
const labels = chapterLabels(chapters);
const sources = sourceEntries(readFileSync(path.join(root, 'book/sources.adoc'), 'utf8'));
export const proseWordLimit = 1800;
const rendered = new Map();

function render(id) {
  if (!rendered.has(id)) {
    const chapter = chapters.find((entry) => entry.id === id);
    const file = path.join(root, 'chapters', id, 'content.adoc');
    if (!chapter || chapter.layout !== 'longform' || chapter.generated || !existsSync(file)) rendered.set(id, null);
    else rendered.set(id, enhanceLongform(renderAsciidoc(file, longformAttributes(id), { strict: true }), {
      label: labels.get(id).label,
      solutionLink: (exercise) => ({ href: `./solutions.html#sol-${exercise}`, text: 'Solution' }),
    }));
  }
  return rendered.get(id);
}

export function proseWords(html) {
  const { root: fragment } = parseFragment(html);
  for (const element of fragment.querySelectorAll('pre, .katex, .bibliography, .exercise, .key-equations, .teach')) element.remove();
  return fragment.textContent.split(/\s+/).filter(Boolean).length;
}

export function checkChapter(id, { figures = true } = {}) {
  const problems = [];
  const chapter = chapters.find((entry) => entry.id === id);
  if (!chapter) return [`unknown chapter ${id}`];
  const directory = path.join(root, 'chapters', id);
  let result;
  try {
    result = render(id);
  } catch (error) {
    return [String(error.stderr || error.message)];
  }
  if (!result) return [`${id} has no long-form content.adoc`];
  const label = labels.get(id).label;
  const { root: fragment } = parseFragment(result.html);

  const solutionsFile = path.join(directory, 'solutions.adoc');
  try {
    const solutions = existsSync(solutionsFile)
      ? solutionBlocks(enhanceLongform(renderAsciidoc(solutionsFile, longformAttributes(id), { strict: true }), { label, numberEquations: false, numberSectionHeadings: false, numberCaptions: false }).html)
      : [];
    renderSolutions([{ chapter, label: labels.get(id), exercises: result.exercises, solutions }]);
    const { root: solutionFragment } = parseFragment(solutions.map((solution) => solution.content).join(''));
    for (const link of solutionFragment.querySelectorAll('a[href^="#"]')) problems.push(`solutions.adoc links to ${link.getAttribute('href')} on its own page; use xref:${id}.adoc#anchor[] instead`);
    fragment.append(...solutionFragment.querySelectorAll('a[href]'));
  } catch (error) {
    problems.push(String(error.stderr || error.message));
  }
  if (result.exercises.length < 4) problems.push(`only ${result.exercises.length} exercises; write at least 4`);
  if (!result.keyEquations) problems.push('missing the [.key-equations#key-equations] sidebar');
  if (!fragment.querySelector('.sect1.teach')) problems.push('missing the [.teach] "Teach it" section');
  if (!result.bibliography.length) problems.push('missing a [bibliography] References section');

  for (const link of fragment.querySelectorAll('a[href]')) {
    const match = link.getAttribute('href').match(/^(?:\.\/)?([a-z][a-z0-9-]*)\.html(?:#(.+))?$/);
    if (!match || ['solutions', 'formula-sheets', 'bibliography'].includes(match[1])) continue;
    const [, target, anchor] = match;
    if (!chapters.some((entry) => entry.id === target)) problems.push(`link to unknown chapter ${target}`);
    else if (anchor) {
      let targetResult = null;
      try { targetResult = target === id ? result : render(target); } catch { targetResult = null; }
      if (!targetResult && !existsSync(path.join(root, 'chapters', target, 'content.adoc'))) problems.push(`link to ${target}.html#${anchor}, but ${target} is not written yet; link to the chapter without an anchor`);
      else if (targetResult && !parseFragment(targetResult.html).root.querySelector(`[id="${anchor}"]`)) problems.push(`link to missing anchor ${target}.html#${anchor}`);
    }
  }

  for (const entry of result.bibliography) {
    const body = entry.html.replace(/^<p>\s*<a id="[^"]+"><\/a>\s*/, '').replace(/<\/p>$/, '').trim();
    const shared = sources.get(entry.id);
    if (shared !== undefined && shared !== body) problems.push(`bibliography entry ${entry.id} differs from book/sources.adoc; include it from there`);
  }

  const diagrams = path.join(directory, 'diagrams');
  if (existsSync(diagrams)) {
    try {
      validateSvg(readdirSync(diagrams).filter((name) => name.endsWith('.svg')).map((name) => path.join(diagrams, name)), buildPython(root));
    } catch (error) {
      problems.push(error.message);
    }
  }
  const images = [...fragment.querySelectorAll('img')].map((image) => image.getAttribute('src'));
  if (images.length && figures) {
    const output = mkdtempSync(path.join(tmpdir(), `figures-${id}-`));
    try {
      if (existsSync(path.join(directory, 'figures.py'))) {
        execFileSync(buildPython(root), [path.join(directory, 'figures.py'), output], { stdio: 'pipe', env: { ...process.env, PYTHONPATH: [root, path.join(root, 'book')].join(path.delimiter) } });
      }
      const available = new Set([...readdirSync(output), ...(existsSync(path.join(directory, 'diagrams')) ? readdirSync(path.join(directory, 'diagrams')) : [])]);
      for (const source of images) if (!available.has(path.basename(source))) problems.push(`image ${source} is neither generated by figures.py nor in diagrams/`);
    } catch (error) {
      problems.push(`figures.py failed: ${String(error.stderr || error.message).slice(-2000)}`);
    } finally {
      rmSync(output, { recursive: true, force: true });
    }
  }

  const code = [path.join(directory, 'code'), path.join(root, 'scratch')];
  for (const folder of code.filter(existsSync)) {
    for (const name of readdirSync(folder).filter((file) => file.endsWith('.py') && !file.startsWith('test_'))) {
      const file = path.join(folder, name);
      if (folder.endsWith('scratch') && !readFileSync(file, 'utf8').includes(`__chapter__ = "${id}"`)) continue;
      readFileSync(file, 'utf8').split('\n').forEach((line, index) => {
        if (line.length > 88) problems.push(`${path.relative(root, file)}:${index + 1} is ${line.length} characters (limit 88)`);
      });
    }
  }

  const words = proseWords(result.html);
  if (words > proseWordLimit) problems.push(`${words} words of prose; keep chapters brief (at most ${proseWordLimit})`);
  return { problems, words, exercises: result.exercises.length };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  let failed = false;
  for (const id of process.argv.slice(2)) {
    const outcome = checkChapter(id);
    const problems = Array.isArray(outcome) ? outcome : outcome.problems;
    if (problems.length) {
      failed = true;
      console.log(`✖ ${id}\n${problems.map((problem) => `  - ${problem}`).join('\n')}`);
    } else console.log(`✔ ${id}: ${outcome.words} words of prose, ${outcome.exercises} exercises`);
  }
  process.exit(failed ? 1 : 0);
}
