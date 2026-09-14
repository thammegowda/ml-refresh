import { createElement, ArrowUpRight, ArrowLeft, ArrowRight } from 'lucide';

export const chapters = [
	{
		id: 'foundations',
		title: 'Mathematical Foundations',
		description: 'A return to arithmetic, algebra, geometry, trigonometry, and the ideas that connect them.',
		status: 'published',
		interactive: true,
	},
	{
		id: 'trigonometry',
		title: 'Trigonometry',
		description: 'Radians, the unit circle, trigonometric functions, and vector similarity.',
		status: 'published',
		interactive: true,
		reference: '_keep_in_mind',
	},
	{
		id: 'calculus',
		title: 'Calculus',
		description: 'Functions, derivatives, signed integrals, and activation functions.',
		status: 'published',
		interactive: true,
		reference: '_keep_in_mind',
		legacyStateKey: 'family',
	},
	{
		id: 'linear-algebra',
		title: 'Linear Algebra',
		lessonTitle: 'Matrix operations & backprop',
		description: 'Dot products, matrix multiplication, affine layers, and their backward passes.',
		status: 'published',
		interactive: true,
		reference: '_keep_in_mind',
	},
	{
		id: 'vector-calculus',
		title: 'Vector Calculus',
		description: 'Scalar fields, gradients, Jacobians, Hessians, and local approximations.',
		status: 'published',
		interactive: true,
		notebook: true,
	},
	{
		id: 'neural-network',
		title: 'Neural Networks',
		description: 'Build a two-layer classifier: forward propagation, backpropagation, Adam, and overfitting.',
		status: 'published',
		interactive: true,
		notebook: true,
	},
	{
		id: 'probability',
		title: 'Probability Theory',
		description: 'Random variables, distributions, expectation, and conditional probability.',
		status: 'planned',
	},
	{
		id: 'hypothesis-testing',
		title: 'Hypothesis Testing',
		description: 'Sampling, confidence intervals, significance, and statistical power.',
		status: 'planned',
	},
];

export function escapeHtml(value) {
	return String(value).replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character]);
}

export function validateChapters(chapters) {
	const ids = new Set();
	for (const chapter of chapters) {
		if (typeof chapter.id !== 'string' || !/^[a-z][a-z0-9-]*$/.test(chapter.id) || chapter.id === 'index' || ids.has(chapter.id)) throw new Error(`Invalid or duplicate chapter ID: ${chapter.id}`);
		if (!chapter.title || !chapter.description || !['planned', 'published'].includes(chapter.status)) throw new Error(`Incomplete chapter: ${chapter.id}`);
		ids.add(chapter.id);
	}
}

export function renderContents(chapters) {
	return `<section class="book-contents" aria-labelledby="contents-heading">
		<div class="book-heading"><p class="book-title">refresh</p><p class="book-subtitle">Mathematics &amp; statistics</p></div>
		<div class="contents-heading"><h2 id="contents-heading">Contents</h2><span>${chapters.length} chapters</span></div>
		<ol class="chapter-list">${chapters.map((chapter, index) => {
			const available = chapter.status === 'published';
			const body = `<span class="chapter-number">${String(index + 1).padStart(2, '0')}</span><div class="chapter-description"><h3>${escapeHtml(chapter.title)}</h3><p>${escapeHtml(chapter.description)}</p></div><span class="chapter-status">${available ? '<span data-icon="open" aria-hidden="true"></span>' : 'Planned'}</span>`;
			const stateKey = chapter.legacyStateKey ? ` data-state-key="${escapeHtml(chapter.legacyStateKey)}"` : '';
			return `<li>${available ? `<a class="chapter-row chapter-link" href="./${chapter.id}.html"${stateKey}>${body}</a>` : `<div class="chapter-row chapter-planned">${body}</div>`}</li>`;
		}).join('')}</ol>
	</section>`;
}

export function renderChapterNavigation(chapters, currentId) {
	const published = chapters.filter((chapter) => chapter.status === 'published');
	const current = published.findIndex((chapter) => chapter.id === currentId);
	if (current < 0) return '';
	const previous = published[current - 1];
	const next = published[current + 1];
	return `<nav class="chapter-pagination" aria-label="Chapter navigation">
		${previous ? `<a rel="prev" href="./${previous.id}.html"><span data-icon="previous" aria-hidden="true"></span>${escapeHtml(previous.title)}</a>` : '<a href="./"><span data-icon="previous" aria-hidden="true"></span>Contents</a>'}
		${next ? `<a rel="next" href="./${next.id}.html">${escapeHtml(next.title)}<span data-icon="next" aria-hidden="true"></span></a>` : ''}
	</nav>`;
}

if (typeof document !== 'undefined') initializePage();

function initializePage() {
	const icons = { open: ArrowUpRight, previous: ArrowLeft, next: ArrowRight };
	for (const element of document.querySelectorAll('[data-icon]')) {
		const icon = icons[element.dataset.icon];
		if (icon) element.append(createElement(icon, { width: 18, height: 18, 'aria-hidden': 'true' }));
	}

	document.querySelector('.reference-link')?.addEventListener('click', (event) => {
		const reference = document.getElementById(event.currentTarget.hash.slice(1));
		if (!reference) return;
		event.preventDefault();
		reference.tabIndex = -1;
		reference.focus({ preventScroll: true });
		reference.scrollIntoView({ block: 'start' });
	});

	const state = new URLSearchParams(location.hash.slice(1));
	for (const chapter of document.querySelectorAll('.chapter-link[data-state-key]')) {
		if (state.has(chapter.dataset.stateKey)) {
			location.replace(chapter.getAttribute('href') + location.hash);
			break;
		}
	}
}