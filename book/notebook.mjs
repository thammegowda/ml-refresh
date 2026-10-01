import { execFileSync } from 'node:child_process';
import MarkdownIt from 'markdown-it';
import { longformAttributes, renderMath } from './render.mjs';

const markdown = new MarkdownIt({ html: true, linkify: false, typographer: false });
const defaultHeading = markdown.renderer.rules.heading_open ?? ((tokens, index, options, env, self) => self.renderToken(tokens, index, options));
markdown.renderer.rules.heading_open = (tokens, index, options, env, self) => {
  tokens[index].tag = `h${Math.min(6, Number(tokens[index].tag.slice(1)) + 1)}`;
  return defaultHeading(tokens, index, options, env, self);
};
markdown.renderer.rules.heading_close = (tokens, index) => `</h${Math.min(6, Number(tokens[index].tag.slice(1)) + 1)}>\n`;

// Math is replaced by placeholders before Markdown parsing so emphasis and escapes cannot corrupt LaTeX.
export function renderMarkdownWithMath(source) {
  const protectedCode = [];
  const math = [];
  let text = source.replace(/(`+)([\s\S]*?[^`])\1(?!`)/g, (match) => `@@CODE${protectedCode.push(match) - 1}@@`);
  text = text.replace(/\$\$([\s\S]+?)\$\$|\$([^$\n]+?)\$/g, (_, display, inline) => `@@MATH${math.push(display !== undefined ? renderMath(display.trim(), true) : renderMath(inline.trim())) - 1}@@`);
  text = text.replace(/@@CODE(\d+)@@/g, (_, index) => protectedCode[index]);
  return markdown.render(text).replace(/@@MATH(\d+)@@/g, (_, index) => math[index]);
}

function delimiter(source, character) {
  let length = 4;
  while (source.split('\n').some((line) => line === character.repeat(length))) length++;
  return character.repeat(length);
}

export function renderNotebook(notebook, chapterId) {
  const blocks = [];
  for (const cell of notebook.cells) {
    const source = Array.isArray(cell.source) ? cell.source.join('') : cell.source;
    if (!source.trim()) continue;
    if (cell.cell_type === 'markdown') {
      const html = renderMarkdownWithMath(source);
      const fence = delimiter(html, '+');
      blocks.push(`${fence}\n<div class="notebook-markdown">${html}</div>\n${fence}`);
    } else if (cell.cell_type === 'code') {
      const fence = delimiter(source, '-');
      blocks.push(`[source,python]\n${fence}\n${source.replace(/\s+$/, '')}\n${fence}`);
    }
  }
  const document = `= Notebook\n\n${blocks.join('\n\n')}\n`;
  return execFileSync('asciidoctor', ['-s', '--failure-level=WARN', ...longformAttributes(chapterId).flatMap((attribute) => ['-a', attribute]), '-o', '-', '-'], { input: document, encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'] });
}
