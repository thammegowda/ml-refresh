import { createApp } from 'vue';
import Trigonometry from './Trigonometry.vue';
import './style.css';

const container = document.querySelector('[data-trigonometry]');
if (container) createApp(Trigonometry).mount(container);