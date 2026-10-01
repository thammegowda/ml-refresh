import { createElement, ArrowUpRight, ArrowLeft, ArrowRight } from 'lucide';

export const book = {
	title: 'ML Refresher',
	subtitle: 'Mathematics, statistics & deep learning',
	url: 'https://gowda.ai/ml-refresh/',
	repository: 'https://github.com/thammegowda/ml-refresh',
	disclaimer: 'https://github.com/thammegowda/ml-refresh/blob/main/DISCLAIMER.md',
	issues: 'https://github.com/thammegowda/ml-refresh/issues',
	pdf: 'ml-refresher.pdf',
};

export const parts = [
	{ id: 'mathematics', title: 'Mathematics for Machine Learning', description: 'Numbers, functions, vectors, change, chance, and information.' },
	{ id: 'fundamentals', title: 'Neural Network Fundamentals', description: 'Losses, gradients, activations, optimizers, and the training loop, derived and built in NumPy.' },
	{ id: 'transformers', title: 'Transformers from Scratch', description: 'Tokens, attention, positions, and a GPT trained end to end.' },
	{ id: 'architecture', title: 'Modern LLM Architecture', description: 'The attention, expert, and recurrence variants inside 2026 language models.' },
	{ id: 'multimodal', title: 'Representation & Multimodal Learning', description: 'Contrastive objectives, image patches, and vision-language models.' },
	{ id: 'post-training', title: 'Post-Training & Reinforcement Learning', description: 'Fine-tuning, preferences, policy gradients, PPO, DPO, and GRPO.' },
	{ id: 'systems', title: 'Inference & Systems', description: 'Decoding, quantization, serving, and training at scale, simulated in NumPy.' },
	{ id: 'agents', title: 'Agents', description: 'Tool use, agent loops, retrieval, memory, and evaluation.' },
	{ id: 'appendices', title: 'Appendices', description: 'Reference material, solutions, and formula sheets.', appendix: true },
];

const published = (part, id, title, description) => ({ id, title, description, status: 'published', part, layout: 'longform' });

export const chapters = [
	{
		id: 'foundations',
		title: 'Mathematical Foundations',
		description: 'A return to arithmetic, algebra, geometry, trigonometry, and the ideas that connect them.',
		status: 'published',
		part: 'mathematics',
		interactive: true,
	},
	{
		id: 'trigonometry',
		title: 'Trigonometry',
		description: 'Radians, the unit circle, trigonometric functions, and vector similarity.',
		status: 'published',
		part: 'mathematics',
		interactive: true,
		reference: '_keep_in_mind',
	},
	{
		id: 'calculus',
		title: 'Calculus',
		description: 'Functions, derivatives, signed integrals, and activation functions.',
		status: 'published',
		part: 'mathematics',
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
		part: 'mathematics',
		interactive: true,
		reference: '_keep_in_mind',
	},
	{
		id: 'vector-calculus',
		title: 'Vector Calculus',
		description: 'Scalar fields, gradients, Jacobians, Hessians, and local approximations.',
		status: 'published',
		part: 'mathematics',
		interactive: true,
		notebook: true,
	},
	{
		id: 'probability',
		title: 'Probability Theory',
		description: 'Random variables, distributions, expectation, Bayes, maximum likelihood, and sampling.',
		status: 'published',
		part: 'mathematics',
		layout: 'longform',
	},
	{ id: 'information-theory', title: 'Information Theory', description: 'Surprisal, entropy, cross-entropy, KL divergence, mutual information, and perplexity.', status: 'published', part: 'mathematics', layout: 'longform' },
	published('mathematics', 'hypothesis-testing', 'Hypothesis Testing', 'Confidence intervals, the bootstrap, significance tests, and how many examples a model comparison needs.'),
	published('fundamentals', 'learning-from-data', 'Learning from Data', 'Linear and logistic regression, MSE and cross-entropy from maximum likelihood, and gradient descent.'),
	published('fundamentals', 'autodiff', 'Automatic Differentiation', 'Computational graphs, reverse mode, vector-Jacobian products, and a tiny NumPy autograd.'),
	published('fundamentals', 'activations', 'Activation Functions', 'Sigmoid, tanh, ReLU, GELU, SiLU, and the gated GLU family: GeGLU, SwiGLU, and ReGLU.'),
	published('fundamentals', 'softmax-cross-entropy', 'Softmax & Cross-Entropy', 'Temperature, log-sum-exp stability, the softmax Jacobian, and why the gradient is p minus y.'),
	published('fundamentals', 'losses', 'Loss Functions & Divergences', 'MSE, Huber, cross-entropy, forward and reverse KL, focal loss, and knowledge distillation.'),
	{ id: 'neural-network', title: 'Neural Networks from Scratch', description: 'A multilayer perceptron derived and built by hand: forward pass, backpropagation, initialization, and overfitting.', status: 'published', part: 'fundamentals', layout: 'longform', companion: true },
	published('fundamentals', 'optimization', 'Optimizers & Schedules', 'SGD, momentum, Adam, AdamW, warmup, cosine and WSD schedules, clipping, and Muon.'),
	published('fundamentals', 'normalization', 'Normalization, Residuals & Precision', 'BatchNorm, LayerNorm, RMSNorm, residual streams, dropout, and bf16/fp8 arithmetic.'),
	published('transformers', 'tokenization', 'Tokenization & Embeddings', 'Byte-level BPE from scratch, embedding lookups and their gradients, and weight tying.'),
	published('transformers', 'language-models', 'Language Modeling', 'Autoregressive factorization, n-gram and neural language models, and perplexity.'),
	published('transformers', 'attention', 'Scaled Dot-Product Attention', 'Queries, keys, and values; why divide by the square root of d; masking; and the backward pass.'),
	published('transformers', 'multi-head-attention', 'Multi-Head Attention', 'Heads as subspaces, reshapes and einsum, parameter and FLOP counts.'),
	published('transformers', 'positional-encoding', 'Positional Encoding & RoPE', 'Sinusoidal and learned positions, rotary embeddings, ALiBi, and context extension with YaRN.'),
	published('transformers', 'transformer', 'The Transformer Block', 'Pre-norm residual blocks, RMSNorm, SwiGLU feed-forward layers, and the decoder-only stack.'),
	published('transformers', 'gpt', 'Training a GPT from Scratch', 'A complete NumPy training loop, evaluation, and text generation.'),
	published('architecture', 'kv-cache', 'KV Cache & Grouped-Query Attention', 'Prefill and decode, cache memory, MQA, GQA, sliding windows, sinks, and QK-norm.'),
	published('architecture', 'latent-attention', 'Multi-Head Latent Attention', 'Low-rank KV compression, weight absorption, decoupled RoPE, and sparse attention.'),
	published('architecture', 'flash-attention', 'Online Softmax & FlashAttention', 'Tiled attention with a running maximum and normalizer, and why memory traffic matters.'),
	published('architecture', 'mixture-of-experts', 'Mixture of Experts', 'Top-k routing, load-balancing losses, capacity, shared experts, and auxiliary-loss-free balancing.'),
	published('architecture', 'linear-attention', 'Linear Attention & State-Space Models', 'Attention as recurrence, DeltaNet, Mamba, and hybrid linear/full-attention stacks.'),
	published('architecture', 'scaling', 'Scaling Laws & Pretraining Recipes', 'Compute budgets, Chinchilla, muP, data curation, multi-token prediction, and model case studies.'),
	published('multimodal', 'contrastive-learning', 'Contrastive & Metric Learning', 'Contrastive, triplet, InfoNCE, CLIP, and SigLIP losses, and embedding retrieval.'),
	published('multimodal', 'vision-transformer', 'Vision Transformers', 'Images as patches, patch embeddings, 2-D positions, and a tiny ViT.'),
	published('multimodal', 'vision-language', 'Vision-Language Models', 'CLIP zero-shot, projectors and resamplers, M-RoPE, dynamic resolution, and training stages.'),
	published('post-training', 'fine-tuning', 'Supervised Fine-Tuning & LoRA', 'Chat templates, loss masking, packing, and low-rank adaptation.'),
	published('post-training', 'reinforcement-learning', 'Reinforcement Learning Foundations', 'MDPs, the policy-gradient theorem, REINFORCE, baselines, importance sampling, and GAE.'),
	published('post-training', 'ppo', 'Reward Models, PPO & RLHF', 'Bradley-Terry reward models, the clipped PPO objective, and KL-regularized RLHF.'),
	published('post-training', 'dpo', 'Direct Preference Optimization', 'From the KL-regularized optimum to the DPO loss, its gradient, and its variants.'),
	published('post-training', 'grpo', 'GRPO & Verifiable Rewards', 'Group-relative advantages, RLVR, and the DAPO, Dr. GRPO, and GSPO refinements.'),
	published('post-training', 'distillation', 'Distillation & Reasoning Models', 'Forward and on-policy distillation, test-time compute, and process rewards.'),
	published('systems', 'decoding', 'Decoding & Speculative Sampling', 'Greedy, beam, top-k, top-p, min-p, constrained decoding, and speculative sampling.'),
	published('systems', 'quantization', 'Quantization & Serving', 'Integer and low-bit float formats, GPTQ, AWQ, paged attention, batching, and rooflines.'),
	published('systems', 'distributed-training', 'Training at Scale', 'Memory accounting, data, tensor, pipeline, and expert parallelism, ZeRO, and FP8.'),
	published('agents', 'agents', 'Tool Use & Agent Loops', 'Function calling, JSON schemas, the ReAct loop, and the Model Context Protocol.'),
	published('agents', 'agent-systems', 'Retrieval, Memory, Planning & Evaluation', 'Retrieval, context engineering, reflection, search, benchmarks, and prompt injection.'),
	published('agents', 'capstone', 'Capstone: An LLM End to End', 'Tokenizer, pretraining, fine-tuning, preference training, decoding, and a tool-using agent.'),
	{ id: 'notation', title: 'Notation & Shapes', description: 'Symbols, typography, and the shape conventions used throughout the book.', status: 'published', part: 'appendices', layout: 'longform' },
	{ id: 'numpy', title: 'NumPy for Deep Learning', description: 'Broadcasting, reductions, indexing, einsum, floating point, stability, and gradient checking.', status: 'published', part: 'appendices', layout: 'longform', companion: true },
	published('appendices', 'matrix-calculus', 'Matrix Calculus Cookbook', 'Vector-Jacobian products for the operations used in this book.'),
	{ id: 'solutions', title: 'Solutions to Exercises', description: 'Worked solutions for every published exercise, grouped by chapter.', status: 'published', part: 'appendices', layout: 'longform', generated: 'solutions' },
	{ id: 'formula-sheets', title: 'Formula Sheets', description: 'The key equations from each chapter, collected for review.', status: 'published', part: 'appendices', layout: 'longform', generated: 'formula-sheets' },
	{ id: 'bibliography', title: 'Bibliography', description: 'Every cited source, with links to the primary papers.', status: 'published', part: 'appendices', layout: 'longform', generated: 'bibliography' },
];

const generatedPages = ['solutions', 'formula-sheets', 'bibliography'];

export function escapeHtml(value) {
	return String(value).replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character]);
}

/** The disclaimer banner shown at the top of every web page. */
export function renderNotice() {
	return `<aside class="site-notice" aria-label="Disclaimer"><p><strong>AI-generated</strong> and not reviewed by experts: it likely contains errors. Use at your own risk. <a href="${escapeHtml(book.disclaimer)}">Read the disclaimer</a> · <a href="${escapeHtml(book.issues)}">Report an issue</a></p></aside>`;
}

export function validateChapters(chapters, bookParts = parts) {
	const partIds = new Set(bookParts.map((part) => part.id));
	const ids = new Set();
	let partIndex = 0;
	for (const chapter of chapters) {
		if (typeof chapter.id !== 'string' || !/^[a-z][a-z0-9-]*$/.test(chapter.id) || ['index', 'book'].includes(chapter.id) || ids.has(chapter.id)) throw new Error(`Invalid or duplicate chapter ID: ${chapter.id}`);
		if (!chapter.title || !chapter.description || !['planned', 'published'].includes(chapter.status)) throw new Error(`Incomplete chapter: ${chapter.id}`);
		if (!partIds.has(chapter.part)) throw new Error(`Unknown part for chapter ${chapter.id}: ${chapter.part}`);
		const index = bookParts.findIndex((part) => part.id === chapter.part);
		if (index < partIndex) throw new Error(`Chapter ${chapter.id} is out of part order`);
		partIndex = index;
		if (chapter.layout !== undefined && chapter.layout !== 'longform') throw new Error(`Unknown layout for chapter ${chapter.id}`);
		if (chapter.generated !== undefined && !generatedPages.includes(chapter.generated)) throw new Error(`Unknown generated page for chapter ${chapter.id}`);
		if (chapter.companion && chapter.layout !== 'longform') throw new Error(`Only longform chapters have companion notebooks: ${chapter.id}`);
		ids.add(chapter.id);
	}
}

const romanNumerals = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII'];

export function chapterLabels(chapters, bookParts = parts) {
	const appendixParts = new Set(bookParts.filter((part) => part.appendix).map((part) => part.id));
	const labels = new Map();
	let number = 0;
	let letter = 0;
	for (const chapter of chapters) {
		labels.set(chapter.id, appendixParts.has(chapter.part)
			? { kind: 'Appendix', label: String.fromCharCode(65 + letter++) }
			: { kind: 'Chapter', label: String(++number) });
	}
	return labels;
}

export function partLabels(bookParts = parts) {
	const labels = new Map();
	let number = 0;
	for (const part of bookParts) labels.set(part.id, part.appendix ? '' : `Part ${romanNumerals[number++]}`);
	return labels;
}

export function renderContents(chapters, bookParts = parts) {
	const labels = chapterLabels(chapters, bookParts);
	const partNames = partLabels(bookParts);
	const appendixCount = chapters.filter((chapter) => labels.get(chapter.id).kind === 'Appendix').length;
	const count = `${chapters.length - appendixCount} chapters${appendixCount ? ` · ${appendixCount} appendices` : ''}`;
	return `<section class="book-contents" aria-labelledby="contents-heading">
		<div class="book-heading"><p class="book-title">${escapeHtml(book.title)}</p><p class="book-subtitle">${escapeHtml(book.subtitle)}</p>
		<p class="book-formats"><a href="./book.html">Single-page edition</a><a href="./${escapeHtml(book.pdf)}" download>Download PDF</a></p></div>
		<div class="contents-heading"><h2 id="contents-heading">Contents</h2><span>${count}</span></div>
		<div class="contents-parts">${bookParts.map((part) => {
			const entries = chapters.filter((chapter) => chapter.part === part.id);
			if (!entries.length) return '';
			const label = partNames.get(part.id);
			return `<section class="contents-part" aria-labelledby="part-${part.id}"><h3 class="part-heading" id="part-${part.id}">${label ? `<span class="part-label">${label}</span>` : ''}${escapeHtml(part.title)}</h3>
			<ol class="chapter-list">${entries.map((chapter) => {
				const available = chapter.status === 'published';
				const { kind, label: number } = labels.get(chapter.id);
				const body = `<span class="chapter-number">${kind === 'Chapter' ? number.padStart(2, '0') : number}</span><span class="chapter-title">${escapeHtml(chapter.title)}</span>${available ? '' : '<span class="chapter-status">Planned</span>'}`;
				const stateKey = chapter.legacyStateKey ? ` data-state-key="${escapeHtml(chapter.legacyStateKey)}"` : '';
				// One line per chapter keeps the contents compact; the description is a hover tooltip.
				const summary = ` title="${escapeHtml(chapter.description)}"`;
				return `<li>${available ? `<a class="chapter-row chapter-link" href="./${chapter.id}.html"${summary}${stateKey}>${body}</a>` : `<div class="chapter-row chapter-planned"${summary}>${body}</div>`}</li>`;
			}).join('')}</ol></section>`;
		}).join('')}</div>
	</section>`;
}

export function renderBookNavigation(chapters, currentId = '', { singlePage = false, trigger = 'Contents', triggerClass = '' } = {}) {
	const labels = chapterLabels(chapters);
	const partNames = partLabels();
	const published = chapters.filter((chapter) => chapter.status === 'published');
	return `<div class="navigation-menu">
		<button class="panel-toggle${triggerClass ? ` ${triggerClass}` : ''}" type="button" aria-expanded="false" aria-controls="book-navigation-panel">${trigger}</button>
		<div class="navigation-panel" id="book-navigation-panel" hidden>
			<nav aria-label="Book contents">
				<p class="panel-heading">Book contents</p>
				<ol class="panel-part-list">${parts.map((part) => {
					const entries = published.filter((chapter) => chapter.part === part.id);
					if (!entries.length) return '';
					const partLabel = partNames.get(part.id);
					return `<li class="panel-part"><p class="panel-part-heading">${partLabel ? `<span>${partLabel}</span>` : '<span>Appendix</span>'}${escapeHtml(part.title)}</p>
						<ol class="panel-chapter-list">${entries.map((chapter) => {
							const { kind, label } = labels.get(chapter.id);
							const href = singlePage ? `#${chapter.id}` : `./${chapter.id}.html`;
							const current = chapter.id === currentId ? ' aria-current="page"' : '';
							return `<li><a href="${href}"${current}><span>${kind === 'Chapter' ? label.padStart(2, '0') : label}</span>${escapeHtml(chapter.title)}</a></li>`;
						}).join('')}</ol>
					</li>`;
				}).join('')}</ol>
			</nav>
		</div>
	</div>`;
}

export function renderChapterTocNavigation() {
	return `<div class="navigation-menu chapter-toc-menu">
		<button class="panel-toggle" type="button" aria-expanded="false" aria-controls="chapter-toc-panel">On this page</button>
		<div class="navigation-panel chapter-toc-panel" id="chapter-toc-panel" hidden>
			<nav class="chapter-toc" aria-label="On this page" data-chapter-toc hidden><p class="panel-heading">On this page</p></nav>
		</div>
	</div>`;
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

	const closeNavigation = () => {
		const toggle = document.querySelector('.panel-toggle[aria-expanded="true"]');
		if (!toggle) return;
		toggle.setAttribute('aria-expanded', 'false');
		document.getElementById(toggle.getAttribute('aria-controls'))?.setAttribute('hidden', '');
	};
	for (const toggle of document.querySelectorAll('.panel-toggle')) {
		toggle.addEventListener('click', () => {
			const panel = document.getElementById(toggle.getAttribute('aria-controls'));
			const opening = toggle.getAttribute('aria-expanded') !== 'true';
			closeNavigation();
			toggle.setAttribute('aria-expanded', String(opening));
			panel?.toggleAttribute('hidden', !opening);
		});
	}
	document.addEventListener('click', (event) => {
		if (!event.target.closest('.navigation-menu')) closeNavigation();
	});
	for (const link of document.querySelectorAll('.navigation-panel a[href^="#"]')) {
		link.addEventListener('click', closeNavigation);
	}
	document.addEventListener('keydown', (event) => {
		if (event.key === 'Escape') closeNavigation();
	});

	const chapterToc = document.querySelector('[data-chapter-toc]');
	if (chapterToc) {
		const headings = [...document.querySelectorAll('main h2[id], main h3[id]')];
		if (headings.length) {
			const list = document.createElement('ol');
			let subsectionList = null;
			for (const heading of headings) {
				const item = document.createElement('li');
				const link = document.createElement('a');
				link.href = `#${heading.id}`;
				link.textContent = heading.textContent.trim();
				item.append(link);
				if (heading.tagName === 'H2') {
					list.append(item);
					subsectionList = document.createElement('ol');
					item.append(subsectionList);
				} else if (subsectionList) {
					subsectionList.append(item);
				} else {
					list.append(item);
				}
			}
			for (const nested of list.querySelectorAll('ol:empty')) nested.remove();
			chapterToc.append(list);
			chapterToc.hidden = false;
		} else {
			chapterToc.closest('.chapter-toc-menu').hidden = true;
		}
	}

	let activeCitation = null;
	const citationPopover = document.createElement('aside');
	citationPopover.className = 'citation-popover';
	citationPopover.setAttribute('role', 'dialog');
	citationPopover.setAttribute('aria-label', 'Citation details');
	citationPopover.hidden = true;
	document.body.append(citationPopover);
	const closeCitation = () => {
		activeCitation?.setAttribute('aria-expanded', 'false');
		activeCitation = null;
		citationPopover.hidden = true;
		citationPopover.replaceChildren();
	};
	for (const link of document.querySelectorAll('a.citation-link')) {
		link.setAttribute('aria-haspopup', 'dialog');
		link.setAttribute('aria-expanded', 'false');
		link.addEventListener('click', (event) => {
			const id = link.hash.slice(1);
			const entry = document.getElementById(id)?.closest('li');
			if (!entry) return;
			event.preventDefault();
			closeCitation();
			activeCitation = link;
			link.setAttribute('aria-expanded', 'true');
			const content = entry.cloneNode(true);
			for (const element of [content, ...content.querySelectorAll('[id]')]) element.removeAttribute('id');
			const close = document.createElement('button');
			close.type = 'button';
			close.className = 'citation-close';
			close.setAttribute('aria-label', 'Close citation');
			close.textContent = '\u00d7';
			close.addEventListener('click', () => {
				closeCitation();
				link.focus();
			});
			citationPopover.append(close, content);
			citationPopover.hidden = false;
			const anchor = link.getBoundingClientRect();
			const width = citationPopover.offsetWidth;
			const left = Math.min(Math.max(12, anchor.left), innerWidth - width - 12);
			const top = anchor.bottom + 8 + citationPopover.offsetHeight <= innerHeight
				? anchor.bottom + 8
				: Math.max(12, anchor.top - citationPopover.offsetHeight - 8);
			citationPopover.style.left = `${left}px`;
			citationPopover.style.top = `${top}px`;
			close.focus();
		});
	}
	document.addEventListener('click', (event) => {
		if (activeCitation && !event.target.closest('.citation-popover, .citation-link')) closeCitation();
	});
	document.addEventListener('keydown', (event) => {
		if (event.key === 'Escape' && activeCitation) {
			const link = activeCitation;
			closeCitation();
			link.focus();
		}
	});

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
