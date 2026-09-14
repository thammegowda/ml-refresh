<script setup>
import { computed } from 'vue';
import { numericParameters } from '../../python/parameters.js';
import PythonPlots from '../../python/PythonPlots.vue';

const props = defineProps({ source: String, result: Object, preserveView: Boolean, plotTitle: String });
const emit = defineEmits(['change']);
const parameter = computed(() => numericParameters(props.source, ['point']).point);
const bounds = computed(() => {
  const points = props.result?.plots?.[0]?.series?.[0]?.points;
  return {
    min: Math.min(points?.[0]?.[0] ?? -5, parameter.value?.value ?? -5),
    max: Math.max(points?.at(-1)?.[0] ?? 5, parameter.value?.value ?? 5),
  };
});
function change(value) {
  if (!parameter.value) return;
  if (!Number.isFinite(value)) return;
  emit('change', props.source.slice(0, parameter.value.from) + value + props.source.slice(parameter.value.to), { point: value });
}
</script>

<template>
  <PythonPlots :plots="result.plots" :preserve-view="preserveView" :point="{ plot: plotTitle, value: parameter?.value ?? result.parameters?.point ?? 0, disabled: !parameter, min: bounds.min, max: bounds.max }" @point-change="change">
    <template #controls>
      <label class="py-point-input">Point<input type="number" aria-label="Evaluation point" step="0.01" :value="parameter?.value" :disabled="!parameter" :title="parameter ? 'Evaluation point' : 'Point must be a single numeric assignment'" @input="change($event.target.valueAsNumber)"></label>
    </template>
  </PythonPlots>
</template>