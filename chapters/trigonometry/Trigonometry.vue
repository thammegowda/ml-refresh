<script setup>
import { computed, ref, useId } from 'vue';
import { line, scaleLinear } from 'd3';
import { numericParameters } from '../../python/parameters.js';
import PythonWorkspace from '../../python/PythonWorkspace.vue';
import lesson from './lesson.py';

const workspace = ref(null);
const execution = ref({ source: lesson, result: null });
const unit = ref('degrees');
const dragging = ref(false);
let drag = null;
const id = useId();
const landmarks = [
  { degrees: -180, radians: '-π' }, { degrees: -90, radians: '-π/2' },
  { degrees: 0, radians: '0' }, { degrees: 30, radians: 'π/6' },
  { degrees: 45, radians: 'π/4' }, { degrees: 60, radians: 'π/3' },
  { degrees: 90, radians: 'π/2' }, { degrees: 180, radians: 'π' },
  { degrees: 270, radians: '3π/2' }, { degrees: 360, radians: '2π' },
];
const parameters = computed(() => numericParameters(execution.value.source ?? lesson, ['angle', 'length_u', 'length_v']));
function change(name, value) {
  const parameter = parameters.value[name];
  if (!parameter || !Number.isFinite(value)) return;
  const source = execution.value.source ?? lesson;
  workspace.value.updateSource(source.slice(0, parameter.from) + value + source.slice(parameter.to), { [name]: value });
}
const angle = computed(() => parameters.value.angle?.value ?? 0);
const displayedAngle = computed(() => unit.value === 'degrees' ? angle.value : angle.value * Math.PI / 180);
function pointerAngle(event) {
  const matrix = event.currentTarget.ownerSVGElement.getScreenCTM();
  if (!matrix) return null;
  const position = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
  if (Math.hypot(position.x - 230, position.y - 170) < 8) return null;
  return Math.atan2(170 - position.y, position.x - 230) * 180 / Math.PI;
}
function startDrag(event) {
  if (event.button !== 0 || !parameters.value.angle || drag) return;
  const initial = pointerAngle(event);
  if (initial === null) return;
  event.preventDefault();
  event.currentTarget.focus();
  event.currentTarget.setPointerCapture(event.pointerId);
  drag = { pointerId: event.pointerId, previous: initial, angle: angle.value };
  dragging.value = true;
}
function movePoint(event) {
  if (!drag || drag.pointerId !== event.pointerId) return;
  const next = pointerAngle(event);
  if (next === null) return;
  const delta = ((next - drag.previous + 540) % 360) - 180;
  drag.previous = next;
  drag.angle += delta;
  if (drag.angle > 360) drag.angle -= 360;
  if (drag.angle < -360) drag.angle += 360;
  change('angle', Math.round(drag.angle));
}
function endDrag(event) {
  if (drag?.pointerId !== event.pointerId) return;
  if (event.type === 'pointerup') movePoint(event);
  drag = null;
  dragging.value = false;
  if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
}
function moveWithKeyboard(event) {
  if (!parameters.value.angle || !['ArrowLeft', 'ArrowDown', 'ArrowRight', 'ArrowUp', 'Home', 'End'].includes(event.key)) return;
  event.preventDefault();
  const step = event.shiftKey ? 15 : 1;
  const next = event.key === 'Home' ? 0 : event.key === 'End' ? 360 : angle.value + (['ArrowUp', 'ArrowRight'].includes(event.key) ? step : -step);
  change('angle', Math.max(-360, Math.min(360, next)));
}
const angleLabel = computed(() => {
  const landmark = landmarks.find(mark => Math.abs(mark.degrees - angle.value) < 0.001);
  return `${format(angle.value)}° = ${landmark?.radians ?? format(angle.value * Math.PI / 180)} rad`;
});
const values = computed(() => Object.fromEntries((execution.value.result?.outputs ?? []).map(entry => [entry.name, entry.values.flat()])));
const vectors = computed(() => Object.fromEntries((execution.value.result?.inputs ?? []).map(entry => [entry.name, entry.values.flat()])));
const point = computed(() => values.value['Unit point'] ?? [1, 0]);
const vectorU = computed(() => vectors.value.u ?? [0, 0]);
const vectorV = computed(() => vectors.value.v ?? [0, 0]);
const similarity = computed(() => values.value['Cosine similarity']?.[0]);
const plotX = scaleLinear().domain([-360, 360]).range([44, 466]);
const plotTicks = [-360, -270, -180, -90, 0, 90, 180, 270, 360];
const radianTicks = ['-2π', '-3π/2', '-π', '-π/2', '0', 'π/2', 'π', '3π/2', '2π'];
function scaleY(title) { return scaleLinear().domain(title === 'Tangent' ? [-4, 4] : [-1.2, 1.2]).range([194, 16]); }
function path(series, title) { return line().defined(entry => Number.isFinite(entry[1])).x(entry => plotX(entry[0])).y(entry => scaleY(title)(entry[1]))(series.points); }
const plots = computed(() => execution.value.result?.plots ?? []);
const wavePlots = computed(() => plots.value.map(plot => ({ ...plot, series: plot.series.map(series => ({ ...series, path: path(series, plot.title) })) })));
function currentValue(name) { return name === 'sin' ? point.value[1] : name === 'cos' ? point.value[0] : values.value['Tangent defined']?.[0] ? values.value.Tangent?.[0] : null; }
function format(value) { return value == null ? 'undefined' : Number(value.toFixed(4)).toString(); }
function polygon() {
  const [ux, uy] = vectorU.value;
  const [vx, vy] = vectorV.value;
  return [[0, 0], [ux, uy], [ux + vx, uy + vy], [vx, vy]].map(([x, y]) => `${220 + 36 * x},${170 - 36 * y}`).join(' ');
}
</script>

<template>
  <PythonWorkspace ref="workspace" :default-code="lesson" title="Trigonometry" download="./trigonometry.py" framework-start="PARAMETERS =" @state="execution = $event" />
  <section class="trig-explorer" aria-label="Trigonometry explorer" :aria-busy="execution.busy">
    <div class="trig-controls">
      <fieldset><legend>Angle unit</legend><label><input v-model="unit" type="radio" value="degrees">Degrees</label><label><input v-model="unit" type="radio" value="radians">Radians</label></fieldset>
      <label class="trig-angle">Angle<input type="number" aria-label="Angle" step="any" :value="Number(displayedAngle.toFixed(4))" :disabled="!parameters.angle" @change="change('angle', unit === 'degrees' ? $event.target.valueAsNumber : $event.target.valueAsNumber * 180 / Math.PI)"></label>
      <output aria-label="Angle conversion">{{ angleLabel }}</output>
      <label>Reference angle<select aria-label="Reference angle" :value="angle" :disabled="!parameters.angle" @change="change('angle', Number($event.target.value))"><option :value="angle" disabled v-if="!landmarks.some(mark => mark.degrees === angle)">{{ format(angle) }}°</option><option v-for="mark in landmarks" :key="mark.degrees" :value="mark.degrees">{{ mark.degrees }}° = {{ mark.radians }}</option></select></label>
      <span class="trig-status" role="status">{{ execution.stale || execution.error ? 'Current angle not applied' : execution.busy ? 'Running...' : 'Current code' }}</span>
    </div>
    <div v-if="execution.result" class="trig-scenes">
      <section aria-label="Unit circle">
        <h2>Unit circle</h2>
        <svg class="trig-unit-circle" viewBox="0 0 480 340" role="group" aria-label="Unit circle with sine, cosine and tangent">
          <defs><clipPath :id="id + '-circle'"><rect x="15" y="20" width="450" height="300" /></clipPath></defs>
          <line class="trig-axis" x1="65" x2="420" y1="170" y2="170" /><line class="trig-axis" x1="230" x2="230" y1="20" y2="320" />
          <circle class="trig-circle" cx="230" cy="170" r="108" />
          <circle class="trig-angle-arc" cx="0" cy="0" r="30" :transform="`translate(230 170) scale(1 ${values.Angle?.[1] < 0 ? 1 : -1})`" :stroke-dasharray="`${30 * Math.min(2 * Math.PI, Math.abs(values.Angle?.[1] ?? 0))} ${60 * Math.PI}`" />
          <text x="353" y="188">0 / 2π</text><text x="238" y="49">π/2</text><text x="99" y="188">π</text><text x="238" y="298">3π/2</text>
          <text x="410" y="162">x</text><text x="238" y="23">y</text>
          <g :clip-path="`url(#${id}-circle)`">
            <line class="trig-guide" x1="338" x2="338" y1="20" y2="320" />
            <line v-if="values['Tangent defined']?.[0]" class="trig-guide" x1="230" y1="170" x2="338" :y2="170 - 108 * values.Tangent[0]" />
            <line v-if="values['Tangent defined']?.[0]" class="trig-tan" x1="338" x2="338" y1="170" :y2="170 - 108 * values.Tangent[0]" />
            <line class="trig-cos" x1="230" :x2="230 + 108 * point[0]" y1="170" y2="170" />
            <line class="trig-sin" :x1="230 + 108 * point[0]" :x2="230 + 108 * point[0]" y1="170" :y2="170 - 108 * point[1]" />
            <line class="trig-radius" x1="230" y1="170" :x2="230 + 108 * point[0]" :y2="170 - 108 * point[1]" />
          </g>
          <g class="trig-point-control" :class="{ dragging }" role="slider" :tabindex="parameters.angle ? 0 : -1" aria-label="Unit circle angle" aria-valuemin="-360" aria-valuemax="360" :aria-valuenow="angle" :aria-valuetext="angleLabel" :aria-disabled="!parameters.angle" :transform="`translate(${230 + 108 * point[0]} ${170 - 108 * point[1]})`" @pointerdown="startDrag" @pointermove="movePoint" @pointerup="endDrag" @pointercancel="endDrag" @lostpointercapture="endDrag" @keydown="moveWithKeyboard">
            <title>Drag to change angle; arrow keys adjust by 1 degree</title>
            <circle class="trig-point-hit" r="8" />
            <circle class="trig-point" r="6" />
          </g>
          <text x="20" y="333">Radius = 1; circumference = 2π</text>
        </svg>
        <div class="trig-values"><span class="trig-sin-text">sin θ = {{ format(point[1]) }}</span><span class="trig-cos-text">cos θ = {{ format(point[0]) }}</span><span class="trig-tan-text">tan θ = {{ format(currentValue('tan')) }}</span></div>
      </section>
      <section aria-label="Vector geometry">
        <h2>Projection and oriented area</h2>
        <div class="trig-lengths"><label v-for="name in ['length_u', 'length_v']" :key="name">|{{ name === 'length_u' ? 'u' : 'v' }}|<input type="range" :aria-label="name === 'length_u' ? 'Length of u' : 'Length of v'" min="0" max="3" step="0.1" :value="parameters[name]?.value ?? 0" :disabled="!parameters[name]" @input="change(name, Number($event.target.value))"><output>{{ format(parameters[name]?.value) }}</output></label></div>
        <svg viewBox="0 0 480 300" role="img" aria-label="Vectors with projection and cross product parallelogram">
          <defs><marker :id="id + '-arrow'" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke" /></marker></defs>
          <line class="trig-axis" x1="10" x2="470" y1="170" y2="170" /><line class="trig-axis" x1="220" x2="220" y1="35" y2="290" />
          <polygon class="trig-parallelogram" :class="{ negative: values['Cross z']?.[0] < 0 }" :points="polygon()" />
          <line class="trig-guide" :x1="220 + 36 * vectorV[0]" :x2="220 + 36 * vectorV[0]" y1="170" :y2="170 - 36 * vectorV[1]" />
          <line class="trig-projection" x1="220" :x2="220 + 36 * vectorV[0]" y1="178" y2="178" />
          <line class="trig-cos" :marker-end="`url(#${id}-arrow)`" x1="220" y1="170" :x2="220 + 36 * vectorU[0]" :y2="170 - 36 * vectorU[1]" />
          <line class="trig-sin" :marker-end="`url(#${id}-arrow)`" x1="220" y1="170" :x2="220 + 36 * vectorV[0]" :y2="170 - 36 * vectorV[1]" />
          <text :x="225 + 36 * vectorU[0]" y="164">u</text><text :x="225 + 36 * vectorV[0]" :y="160 - 36 * vectorV[1]">v</text>
          <text x="16" y="26">u · v = |u| |v| cos θ</text><text x="16" y="48">(u × v)z = |u| |v| sin θ</text>
        </svg>
        <div class="trig-values"><span>Dot = {{ format(values['Dot product']?.[0]) }}</span><span>Cosine similarity = <output aria-label="Cosine similarity">{{ format(similarity) }}</output></span><span>Signed cross z = {{ format(values['Cross z']?.[0]) }}</span></div>
      </section>
      <section v-for="plot in wavePlots" :key="plot.title" :aria-label="plot.title">
        <h2>{{ plot.title }}</h2>
        <svg viewBox="0 0 480 240" role="img" :aria-label="plot.title + ' versus angle in radians'">
          <defs><clipPath :id="id + plot.title.replaceAll(' ', '')"><rect x="44" y="16" width="422" height="178" /></clipPath></defs>
          <g v-for="(tick, index) in plotTicks" :key="tick"><line class="trig-grid" :x1="plotX(tick)" :x2="plotX(tick)" y1="16" y2="194" /><text :x="plotX(tick)" y="212" text-anchor="middle">{{ radianTicks[index] }}</text></g>
          <g v-for="tick in plot.title === 'Tangent' ? [-4, -2, 0, 2, 4] : [-1, 0, 1]" :key="tick"><line class="trig-grid" x1="44" x2="466" :y1="scaleY(plot.title)(tick)" :y2="scaleY(plot.title)(tick)" /><text x="36" :y="scaleY(plot.title)(tick) + 4" text-anchor="end">{{ tick }}</text></g>
          <g :clip-path="`url(#${id + plot.title.replaceAll(' ', '')})`">
            <line v-if="plot.title === 'Tangent'" v-for="pole in [-270, -90, 90, 270]" :key="pole" class="trig-asymptote" :x1="plotX(pole)" :x2="plotX(pole)" y1="16" y2="194" />
            <path v-for="series in plot.series" :key="series.name" :class="'trig-wave trig-' + series.name" :d="series.path" />
            <line class="trig-guide" :x1="plotX(values.Angle?.[0] ?? 0)" :x2="plotX(values.Angle?.[0] ?? 0)" y1="16" y2="194" />
            <template v-for="series in plot.series" :key="series.name"><circle v-if="currentValue(series.name) !== null" :class="'trig-marker trig-' + series.name" :cx="plotX(values.Angle?.[0] ?? 0)" :cy="scaleY(plot.title)(currentValue(series.name))" r="4" /></template>
          </g>
          <text x="252" y="234" text-anchor="middle">θ (radians)</text>
        </svg>
      </section>
    </div>
  </section>
</template>