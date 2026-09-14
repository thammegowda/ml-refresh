import { createApp } from 'vue';
import PythonLab from '../../python/PythonLab.vue';
import defaultCode from './lesson.py';

const container = document.querySelector('[data-linear-algebra]');
if (container) createApp(PythonLab, { defaultCode, frameworkStart: '# Display-only exports;', title: 'Matrix operations & backprop', download: './linear-algebra.py' }).mount(container);