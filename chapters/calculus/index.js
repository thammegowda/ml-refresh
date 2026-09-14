import { createApp } from 'vue';
import Calculus from './PythonCalculus.vue';
import './python.css';

const container = document.querySelector('[data-calculus]');
if (container) createApp(Calculus).mount(container);
