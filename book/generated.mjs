import { escapeText } from './render.mjs';

const chapterHeading = ({ chapter, label }) => `<h2 id="${chapter.id}-heading"><span class="section-number">${escapeText(label.label)}</span> ${escapeText(chapter.title)}</h2>`;

/** entries: [{ chapter, label, exercises, solutions }] in book order. */
export function renderSolutions(entries) {
  const ids = new Set();
  const sections = [];
  for (const entry of entries) {
    const solutions = new Map();
    for (const solution of entry.solutions) {
      if (solutions.has(solution.exerciseId)) throw new Error(`Duplicate solution for ${solution.exerciseId}`);
      solutions.set(solution.exerciseId, solution);
    }
    for (const exercise of entry.exercises) {
      if (ids.has(exercise.id)) throw new Error(`Exercise ID ${exercise.id} is used by more than one chapter`);
      ids.add(exercise.id);
      if (!solutions.has(exercise.id)) throw new Error(`Missing solution for ${exercise.id} in ${entry.chapter.id}`);
    }
    for (const exerciseId of solutions.keys()) {
      if (!entry.exercises.some((exercise) => exercise.id === exerciseId)) throw new Error(`Solution sol-${exerciseId} has no matching exercise in ${entry.chapter.id}`);
    }
    if (!entry.exercises.length) continue;
    sections.push(`<section class="generated-section solutions-chapter" aria-labelledby="${entry.chapter.id}-heading">${chapterHeading(entry)}${entry.exercises.map((exercise) => `
      <div id="sol-${exercise.id}" class="exampleblock solution"><div class="title"><span class="exercise-label">Solution ${escapeText(exercise.number)}</span> ${escapeText(exercise.title)}</div>
      <div class="content">${solutions.get(exercise.id).content}<p class="solution-back-link"><a href="./${entry.chapter.id}.html#${exercise.id}">Back to Exercise ${escapeText(exercise.number)}</a></p></div></div>`).join('')}</section>`);
  }
  return sections.length ? sections.join('\n') : '<p class="generated-empty">No exercises have been published yet.</p>';
}

/** entries: [{ chapter, label, keyEquations }] in book order. */
export function renderFormulaSheets(entries) {
  const sections = entries.filter((entry) => entry.keyEquations).map((entry) => `<section class="generated-section formula-sheet" aria-labelledby="${entry.chapter.id}-heading">${chapterHeading(entry)}${entry.keyEquations}<p class="formula-sheet-source"><a href="./${entry.chapter.id}.html">Read ${escapeText(entry.label.kind)} ${escapeText(entry.label.label)}</a></p></section>`);
  return sections.length ? sections.join('\n') : '<p class="generated-empty">No formula sheets have been published yet.</p>';
}

/** entries: [{ chapter, label, bibliography: [{ id, html }] }]; one source must use one key everywhere. */
export function renderBibliography(entries) {
  const sources = new Map();
  for (const entry of entries) {
    for (const item of entry.bibliography) {
      const body = item.html.replace(/^<p>\s*<a id="[^"]+"><\/a>\s*/, '').replace(/<\/p>$/, '').trim();
      const existing = sources.get(item.id);
      if (existing && existing.body !== body) throw new Error(`Bibliography key ${item.id} has different text in different chapters`);
      const source = existing ?? { id: item.id, body, citedBy: [] };
      source.citedBy.push(entry);
      sources.set(item.id, source);
    }
  }
  if (!sources.size) return '<p class="generated-empty">No sources have been cited yet.</p>';
  const sorted = [...sources.values()].sort((left, right) => left.id.localeCompare(right.id));
  return `<div class="ulist bibliography"><ul class="bibliography">${sorted.map((source) => `<li id="${source.id}"><p>${source.body}</p><p class="cited-by">Cited in ${source.citedBy.map(({ chapter, label }) => `<a href="./${chapter.id}.html#${source.id}">${escapeText(label.kind)} ${escapeText(label.label)}</a>`).join(', ')}</p></li>`).join('')}</ul></div>`;
}
