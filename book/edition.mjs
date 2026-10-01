import { renderNotice } from '../app.js';
import { escapeText, parseFragment } from './render.mjs';

const pageLink = /^(?:\.\/)?([a-z][a-z0-9-]*)\.html(?:#(.+))?$/;

export function fragmentIds(html) {
  return new Set([...parseFragment(html).root.querySelectorAll('[id]')].map((element) => element.id));
}

/**
 * Validates links between book pages and replaces Asciidoctor's default cross-document
 * link text (for example "numpy.html") with the target's label.
 * pages: Map<id, { ids: Set, labels: Map, name: string }>; planned: Set of planned chapter IDs.
 */
export function resolveLinks(html, { pages, planned, source }) {
  const { root } = parseFragment(html);
  for (const link of root.querySelectorAll('a[href]')) {
    const match = link.getAttribute('href').match(pageLink);
    if (!match) continue;
    const [, pageId, hash] = match;
    if (planned.has(pageId)) throw new Error(`${source} links to planned chapter ${pageId}`);
    const target = pages.get(pageId);
    if (!target) continue;
    if (hash && !target.ids.has(hash)) throw new Error(`${source} links to missing anchor ${pageId}.html#${hash}`);
    if (link.textContent.trim() === `${pageId}.html`) link.textContent = (hash && target.labels.get(hash)) || target.name;
  }
  return root.innerHTML;
}

export function renderOpener({ chapter, label, level = 1, links = '' }) {
  return `<header class="chapter-opener"><p class="chapter-label">${escapeText(label.kind)} ${escapeText(label.label)}</p>
    <h${level} class="chapter-title" id="${chapter.id}-title">${escapeText(chapter.title)}</h${level}>
    <p class="chapter-summary">${escapeText(chapter.description)}</p>${links ? `<p class="chapter-links">${links}</p>` : ''}</header>`;
}

// Removes interactive-only elements and opens disclosures so the fragment reads well on paper.
export function printableFragment(html, { note = '' } = {}) {
  const { root } = parseFragment(html);
  for (const element of root.querySelectorAll('noscript, iframe, .notebook-links')) element.remove();
  const mounts = [...root.querySelectorAll('div')].filter((element) => !element.children.length && !element.textContent.trim() && [...element.attributes].some((attribute) => attribute.name.startsWith('data-')));
  for (const mount of mounts) mount.remove();
  for (const details of root.querySelectorAll('details')) details.setAttribute('open', '');
  return note + root.innerHTML;
}

export function printNote(label, url) {
  return `<p class="print-note">${escapeText(label)}: <a href="${escapeText(url)}">${escapeText(url)}</a></p>`;
}

function renameElement(document, element, tag) {
  const replacement = document.createElement(tag);
  for (const { name, value } of element.attributes) replacement.setAttribute(name, value);
  replacement.append(...element.childNodes);
  element.replaceWith(replacement);
}

/** Prefixes IDs with the chapter ID, rewrites links to in-book anchors, and demotes section headings one level. */
export function namespaceFragment(html, chapterId, included) {
  const { document, root } = parseFragment(html);
  const prefix = (id) => `${chapterId}--${id}`;
  for (const element of root.querySelectorAll('[id]')) element.id = prefix(element.id);
  for (const attribute of ['for', 'aria-labelledby', 'aria-describedby', 'aria-controls']) {
    for (const element of root.querySelectorAll(`[${attribute}]`)) element.setAttribute(attribute, element.getAttribute(attribute).split(/\s+/).map(prefix).join(' '));
  }
  for (const link of root.querySelectorAll('a[href]')) {
    const href = link.getAttribute('href');
    if (href.startsWith('#')) {
      link.setAttribute('href', `#${prefix(href.slice(1))}`);
      continue;
    }
    const match = href.match(pageLink);
    if (match && included.has(match[1])) link.setAttribute('href', match[2] ? `#${match[1]}--${match[2]}` : `#${match[1]}`);
  }
  for (let level = 5; level >= 2; level--) {
    for (const heading of root.querySelectorAll(`.sect${level - 1} > h${level}, .notebook-markdown h${level}, .generated-section > h${level}`)) renameElement(document, heading, `h${level + 1}`);
  }
  return root.innerHTML;
}

export function renderEdition({ book, parts, chapters, labels, partNames, fragments, stylesheets, date }) {
  const included = new Set(fragments.keys());
  const toc = parts.map((part) => {
    const entries = chapters.filter((chapter) => chapter.part === part.id);
    const partLabel = partNames.get(part.id);
    const heading = `${partLabel ? `<span class="toc-part-label">${partLabel}</span> ` : ''}${escapeText(part.title)}`;
    const hasChapters = entries.some((chapter) => included.has(chapter.id));
    return `<li class="toc-part">${hasChapters ? `<a href="#part-${part.id}">${heading}</a>` : `<span>${heading}</span>`}<ol>${entries.map((chapter) => {
      const number = `<span class="toc-number">${escapeText(labels.get(chapter.id).label)}</span>`;
      return included.has(chapter.id)
        ? `<li><a href="#${chapter.id}">${number}${escapeText(chapter.title)}</a></li>`
        : `<li class="toc-planned">${number}${escapeText(chapter.title)} <span class="toc-status">planned</span></li>`;
    }).join('')}</ol></li>`;
  }).join('');
  const body = parts.map((part) => {
    const entries = chapters.filter((chapter) => chapter.part === part.id && included.has(chapter.id));
    if (!entries.length) return '';
    const partLabel = partNames.get(part.id);
    return `<section class="book-part" id="part-${part.id}" aria-labelledby="part-${part.id}-title"><header class="part-opener">${partLabel ? `<p class="part-label">${partLabel}</p>` : ''}<h1 id="part-${part.id}-title">${escapeText(part.title)}</h1><p class="part-description">${escapeText(part.description)}</p></header>
      ${entries.map((chapter) => `<article class="book-chapter longform-chapter" id="${chapter.id}" aria-labelledby="${chapter.id}-title">${renderOpener({ chapter, label: labels.get(chapter.id), level: 2 })}<div class="chapter-body">${namespaceFragment(fragments.get(chapter.id), chapter.id, included)}</div></article>`).join('\n')}</section>`;
  }).join('\n');
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="${escapeText(book.title)}: ${escapeText(book.subtitle)}, as a single printable page.">
  <title>${escapeText(book.title)}: ${escapeText(book.subtitle)}</title>
  ${stylesheets.map((href) => `<link rel="stylesheet" href="./${href}">`).join('\n  ')}
</head>
<body class="longform book-edition">
  ${renderNotice()}
  <nav class="edition-bar" aria-label="Book edition"><a class="wordmark" href="./">${escapeText(book.title)}</a><a href="./">Contents</a><a href="./${escapeText(book.pdf)}" download>Download PDF</a></nav>
  <main id="main">
    <section class="title-page" aria-label="Title page"><p class="book-title">${escapeText(book.title)}</p><p class="book-subtitle">${escapeText(book.subtitle)}</p>
      <p class="edition-note">Printed edition, ${escapeText(date)}. Interactive labs and runnable notebooks: <a href="${escapeText(book.url)}">${escapeText(book.url)}</a></p>
      <aside class="title-disclaimer" aria-label="Disclaimer">
        <p><strong>Disclaimer.</strong> AI tools generated most of this book, and experts have not reviewed or fact-checked it. It very likely contains errors: check anything important against the cited sources, and use the book at your own risk. We claim no ownership of its AI-generated content and intend no infringement of anyone's rights.</p>
        <p>Full disclaimer and content policy: <a href="${escapeText(book.disclaimer)}">${escapeText(book.disclaimer)}</a><br>Report errors or copyright concerns: <a href="${escapeText(book.issues)}">${escapeText(book.issues)}</a></p>
      </aside></section>
    <nav class="book-toc" aria-labelledby="toc-heading"><h1 id="toc-heading">Contents</h1><ol>${toc}</ol></nav>
    ${body}
  </main>
</body>
</html>
`;
}
