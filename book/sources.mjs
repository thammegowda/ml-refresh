import { execFileSync } from 'node:child_process';
import { parseFragment } from './render.mjs';

/** Renders the shared bibliography (book/sources.adoc) and maps each key to its entry HTML. */
export function sourceEntries(adoc) {
  const html = execFileSync('asciidoctor', ['-s', '--failure-level=WARN', '-o', '-', '-'], { input: adoc, encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'] });
  const { root } = parseFragment(html);
  return new Map([...root.querySelectorAll('ul.bibliography > li')].map((item) => {
    const anchor = item.querySelector('a[id]');
    const body = item.innerHTML.trim().replace(/^<p>\s*<a id="[^"]+"><\/a>\s*/, '').replace(/<\/p>$/, '').trim();
    return [anchor.id, body];
  }));
}
