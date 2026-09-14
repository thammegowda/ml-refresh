import { createElement, Search, ChevronsDownUp } from 'lucide';
import './style.css';

const root = document.querySelector('.foundations');
root.dataset.enhanced = '';
for (const [name, icon] of [['search', Search], ['collapse', ChevronsDownUp]]) {
  root.querySelector(`[data-foundation-icon="${name}"]`).append(createElement(icon, { width: 18, height: 18, 'aria-hidden': 'true' }));
}
const search = root.querySelector('input[type="search"]');
const topic = root.querySelector('.foundation-topic-picker select');
const sections = [...root.querySelectorAll('.foundation-section')];
const rows = sections.flatMap(section => [...section.querySelectorAll('.foundation-list > li')].map(element => ({
  element, section, text: `${section.querySelector('h3').textContent} ${element.textContent}`.toLowerCase(),
})));

function filter() {
  const terms = search.value.toLowerCase().trim().split(/\s+/).filter(Boolean);
  let count = 0;
  for (const row of rows) {
    row.element.hidden = !terms.every(term => row.text.includes(term));
    if (!row.element.hidden) count++;
  }
  for (const section of sections) section.hidden = !rows.some(row => row.section === section && !row.element.hidden);
  root.querySelector('.foundation-count').textContent = `${count} ${count === 1 ? 'reminder' : 'reminders'}`;
  root.querySelector('.foundation-empty').hidden = count > 0;
}
search.addEventListener('input', filter);
root.querySelector('[data-collapse]').addEventListener('click', () => {
  for (const details of root.querySelectorAll('details[open]')) details.open = false;
});

function revealHash() {
  let id;
  try { id = decodeURIComponent(location.hash.slice(1)); } catch { return; }
  const target = document.getElementById(id);
  if (!target || !root.contains(target)) return;
  topic.value = target.closest('.foundation-section')?.id ?? '';
  search.value = '';
  filter();
  if (target.matches('details')) target.open = true;
  target.scrollIntoView({ block: 'start' });
}
window.addEventListener('hashchange', revealHash);
topic.addEventListener('change', () => {
  search.value = '';
  filter();
  if (topic.value) {
    location.hash = topic.value;
    revealHash();
  } else {
    history.replaceState(null, '', location.pathname + location.search);
    root.scrollIntoView();
  }
});
for (const link of root.querySelectorAll('.foundation-nav a')) link.addEventListener('click', () => {
  search.value = '';
  filter();
});

function draw(canvas) {
  const width = canvas.clientWidth;
  if (!width) return;
  const height = width * 440 / 720;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext('2d');
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.fillStyle = '#ffffff';
  context.fillRect(0, 0, width, height);
  context.font = '14px Georgia';
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  if (canvas.dataset.diagram === 'square') {
    const base = 3;
    const added = Number(root.querySelector('[data-square-size]').value);
    const side = height - 36;
    const start = (width - side) / 2;
    const split = side * base / (base + added);
    const blocks = [[start, 18, split, split, '#beded3', 'a\u00b2'], [start + split, 18, side - split, split, '#f1d49e', 'ab'], [start, 18 + split, split, side - split, '#f1d49e', 'ab'], [start + split, 18 + split, side - split, side - split, '#eac6d2', 'b\u00b2']];
    for (const [left, top, blockWidth, blockHeight, color, label] of blocks) {
      context.fillStyle = color;
      context.fillRect(left, top, blockWidth, blockHeight);
      context.strokeStyle = '#ffffff';
      context.strokeRect(left, top, blockWidth, blockHeight);
      context.fillStyle = '#203a31';
      context.fillText(label, left + blockWidth / 2, top + blockHeight / 2);
    }
    context.fillText(`a = ${base}, b = ${added}`, width / 2, height - 8);
  } else {
    const left = 30;
    const right = width - 14;
    const top = 16;
    const bottom = height - 25;
    const projectX = value => left + (value + 3) / 7 * (right - left);
    const projectY = value => bottom - (value + 3) / 7 * (bottom - top);
    context.strokeStyle = '#dce3df';
    context.fillStyle = '#64716d';
    context.font = '11px sans-serif';
    for (const tick of [-2, 0, 2, 4]) {
      context.beginPath(); context.moveTo(projectX(tick), top); context.lineTo(projectX(tick), bottom); context.stroke();
      context.beginPath(); context.moveTo(left, projectY(tick)); context.lineTo(right, projectY(tick)); context.stroke();
      context.fillText(String(tick), projectX(tick), bottom + 13);
      context.fillText(String(tick), left - 14, projectY(tick));
    }
    context.save();
    context.beginPath(); context.rect(left, top, right - left, bottom - top); context.clip();
    for (const [fn, color, dash] of [[Math.exp, '#087f72', []], [Math.log, '#b04e75', []], [value => value, '#828b85', [4, 4]]]) {
      context.strokeStyle = color; context.lineWidth = 2; context.setLineDash(dash); context.beginPath();
      let started = false;
      for (let step = 0; step <= 700; step++) {
        const input = -3 + step / 100;
        const output = fn(input);
        if (!Number.isFinite(output)) { started = false; continue; }
        if (started) context.lineTo(projectX(input), projectY(output));
        else context.moveTo(projectX(input), projectY(output));
        started = true;
      }
      context.stroke();
    }
    context.restore();
  }
}
const canvases = [...root.querySelectorAll('canvas')];
const observer = new ResizeObserver(entries => entries.forEach(entry => draw(entry.target)));
for (const canvas of canvases) {
  observer.observe(canvas);
  canvas.closest('details').addEventListener('toggle', () => draw(canvas));
}
root.querySelector('[data-square-size]').addEventListener('input', event => {
  root.querySelector('[data-square-value]').value = event.target.value;
  draw(root.querySelector('[data-diagram="square"]'));
});
revealHash();