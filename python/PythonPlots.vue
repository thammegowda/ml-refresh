<script setup>
import { nextTick, onBeforeUnmount, onMounted, ref, shallowRef, useId, watch } from 'vue';
import { ZoomIn, ZoomOut, Scan } from '@lucide/vue';
import { curveColors, curveDashes, detachPlot, fitPlotDomains, renderPlot } from './plots.js';
import './plots.css';

const props = defineProps({ plots: { type: Array, required: true }, preserveView: Boolean, point: Object });
const emit = defineEmits(['point-change']);
const id = useId();
const root = ref(null);
const containers = new Map();
const domains = new Map();
const view = shallowRef({ k: 1, x: 0, y: 0 });
const hidden = ref(new Map());
let observer;
function formatNumber(value) {
  return value === null ? 'undefined' : Number(value.toPrecision(6)).toString();
}
function draw() {
  props.plots.forEach((plot, index) => {
    if (!domains.has(plot.title)) domains.set(plot.title, fitPlotDomains(plot));
    renderPlot(containers.get(plot.title), {
      plot, domains: domains.get(plot.title), id: `${id}-plot-${index}`, view: view.value, hidden: hidden.value.get(plot.title) ?? new Set(),
      onViewChange: next => { view.value = next; }, changeZoom,
      point: props.point?.plot === plot.title ? props.point : null, onPointChange: value => emit('point-change', value),
    });
  });
}
function changeZoom(factor) {
  if (factor === null) { view.value = { k: 1, x: 0, y: 0 }; return; }
  const current = view.value;
  const scale = Math.max(0.25, Math.min(32, current.k * factor));
  const ratio = scale / current.k;
  view.value = { k: scale, x: 0.5 - (0.5 - current.x) * ratio, y: 0.5 - (0.5 - current.y) * ratio };
}
function toggle(title, name) {
  const selected = hidden.value.get(title) ?? new Set();
  if (selected.has(name)) selected.delete(name);
  else selected.add(name);
  hidden.value.set(title, selected);
  draw();
}
watch(() => props.plots, async () => {
  if (!props.preserveView) { domains.clear(); hidden.value = new Map(); changeZoom(null); }
  await nextTick();
  draw();
});
watch(view, draw, { flush: 'post' });
watch(() => props.point, draw, { flush: 'post' });
onMounted(() => { observer = new ResizeObserver(draw); observer.observe(root.value); draw(); });
onBeforeUnmount(() => { observer?.disconnect(); for (const container of containers.values()) detachPlot(container); });
</script>

<template>
  <div ref="root" class="py-plots">
    <div class="py-plot-controls" role="group" aria-label="Synchronized plot controls">
      <slot name="controls" />
      <output aria-label="Plot zoom level">{{ Math.round(view.k * 100) }}%</output>
      <button class="py-icon" type="button" aria-label="Zoom in all plots" title="Zoom in all plots" :disabled="view.k >= 32" @click="changeZoom(1.5)"><ZoomIn :size="16" /></button>
      <button class="py-icon" type="button" aria-label="Zoom out all plots" title="Zoom out all plots" :disabled="view.k <= 0.25" @click="changeZoom(1 / 1.5)"><ZoomOut :size="16" /></button>
      <button class="py-icon" type="button" aria-label="Reset all plot views" title="Reset all plot views" @click="changeZoom(null)"><Scan :size="16" /></button>
    </div>
    <div class="py-plot-panels">
      <section v-for="plot in plots" :key="plot.title" class="py-plot-panel" :aria-label="plot.title">
        <h3>{{ plot.title }}</h3>
        <div v-for="series in plot.series.filter(series => series.area)" :key="series.name" class="py-area-summary">
          <output :aria-label="`${plot.title}: Signed integral`">Integral [{{ formatNumber(series.area.lower) }}, {{ formatNumber(series.area.upper) }}] = {{ formatNumber(series.area.value) }}</output>
          <span><i :style="{ background: curveColors[0] }" aria-hidden="true"></i>Positive contribution</span>
          <span><i :style="{ background: curveColors[1] }" aria-hidden="true"></i>Negative contribution</span>
        </div>
        <div :ref="element => element ? containers.set(plot.title, element) : containers.delete(plot.title)" class="py-chart"></div>
        <div class="py-plot-legend">
          <label v-for="(series, index) in plot.series" :key="series.name"><input type="checkbox" :aria-label="`${plot.title}: ${series.name}`" :checked="!hidden.get(plot.title)?.has(series.name)" @change="toggle(plot.title, series.name)"><svg width="22" height="8" aria-hidden="true"><line x1="0" x2="22" y1="4" y2="4" :stroke="curveColors[index]" :stroke-dasharray="curveDashes[index]" stroke-width="2" /></svg>{{ series.name }}</label>
        </div>
      </section>
    </div>
  </div>
</template>