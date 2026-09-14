<script setup>
import { computed, ref } from 'vue';
import PythonLab from '../../python/PythonLab.vue';
import PointControl from './PointControl.vue';
import template from './lesson.py.in';
import composition from './composition.py.in';
import comparison from './comparison.py.in';
import { families, rules, functionCode, compositionCode, comparisonCode } from './expressions.js';

const params = new URLSearchParams(location.hash.slice(1));
const initialFamily = Object.hasOwn(families, params.get('family')) ? params.get('family') : 'power';
const example = ref(params.get('compare') === 'true' ? 'comparison' : Object.hasOwn(rules, params.get('rule')) ? `composition:${params.get('rule')}` : initialFamily);
const parameters = {};
for (const key of ['coefficient', 'exponent', 'point', 'lower']) {
  const raw = params.get(key);
  if (raw !== null && raw.trim() && Number.isFinite(Number(raw))) parameters[key] = Number(raw);
}
const initial = ref(true);
const source = computed(() => example.value === 'comparison' ? comparisonCode(comparison) : example.value.startsWith('composition:')
  ? compositionCode(composition, example.value.split(':')[1])
  : functionCode(template, example.value, initial.value ? parameters : {}));
function choose(event) {
  initial.value = false;
  example.value = event.target.value;
  const hash = example.value === 'comparison' ? 'family=sigmoid&compare=true' : example.value.startsWith('composition:') ? `rule=${example.value.split(':')[1]}` : `family=${example.value}`;
  history.replaceState(null, '', `${location.pathname}${location.search}#${hash}`);
}
</script>

<template>
  <PythonLab :key="example" :default-code="source" :framework-start="example === 'comparison' ? 'PLOTS = ' : example.startsWith('composition:') ? 'with np.errstate' : 'def connected(x):'" title="Calculus" download="./calculus.py">
    <template #controls></template>
    <template #plots="{ source: draft, result, preserveView, updateSource }">
      <PointControl :source="draft" :result="result" :preserve-view="preserveView" :plot-title="example === 'comparison' ? 'Activations' : example.startsWith('composition:') ? 'h(x)' : 'Function'" @change="updateSource" />
    </template>
    <template #toolbar>
      <label class="calculus-example">Example<select :value="example" aria-label="Calculus example" @change="choose">
        <optgroup label="Functions"><option v-for="(family, key) in families" :key="key" :value="key">{{ family.name }}</option></optgroup>
        <optgroup label="Derivative composition"><option v-for="(rule, key) in rules" :key="key" :value="`composition:${key}`">{{ rule.name }} rule</option></optgroup>
        <option value="comparison">Compare activations</option>
      </select></label>
    </template>
  </PythonLab>
</template>