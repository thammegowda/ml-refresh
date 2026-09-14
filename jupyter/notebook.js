import { createElement, Download, Maximize } from 'lucide';
import './notebook.css';

const links = document.querySelector('.notebook-links');
const header = document.querySelector('.page-navigation');
if (links && header) {
	for (const link of links.querySelectorAll('a')) {
		const label = link.textContent.trim();
		link.setAttribute('aria-label', label);
		link.title = label;
		link.classList.add('notebook-action');
		link.replaceChildren(createElement(link.hasAttribute('download') ? Download : Maximize, { width: 16, height: 16, 'aria-hidden': 'true' }));
		header.append(link);
	}
	links.remove();
}