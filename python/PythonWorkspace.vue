<script setup>
import { computed, onBeforeUnmount, onMounted, ref, useId, watchEffect } from 'vue';
import { RotateCcw, Play, Square, Download, Code, Terminal, Trash2 } from '@lucide/vue';
import PythonEditor from './PythonEditor.vue';
import defaultDriver from './runner.py';
import { createPythonRunner } from './runtime.js';
import './workspace.css';

const props = defineProps({ defaultCode: String, title: String, download: String, frameworkStart: String, driver: { type: String, default: defaultDriver } });
const emit = defineEmits(['state']);
const id = useId();
const source = ref(props.defaultCode);
const appliedCode = ref(null);
const preserveView = ref(false);
const result = ref(null);
const busy = ref(false);
const runtimeState = ref('Loading Python + NumPy');
const error = ref('');
const output = ref([]);
const editorTab = ref('code');
const frameworkExpanded = ref(false);
const workbench = ref(null);
const split = ref(null);
const codeWidth = ref(65);
const workspaceHeight = ref(null);
const measuredHeight = ref(630);
const stale = computed(() => Boolean(result.value && source.value !== appliedCode.value));
const runner = createPythonRunner();
let sizeObserver;
let resizing = null;
let disposed = false;
let scheduledSource = null;
let runTimer = null;
let revision = 0;
let liveSource = null;
let liveParameters = null;

watchEffect(() => emit('state', { source: source.value, result: result.value, preserveView: preserveView.value, busy: busy.value, stale: stale.value, error: error.value, modified: source.value !== props.defaultCode }));

function scheduleRun() {
  if (runTimer !== null || disposed) return;
  runTimer = setTimeout(() => {
    runTimer = null;
    if (busy.value) return;
    const submitted = scheduledSource;
    scheduledSource = null;
    if (submitted !== null && submitted === source.value) execute({ preserveView: true });
  }, 0);
}
function updateSource(nextSource, parameters = null) {
  if (nextSource === source.value && !error.value) return;
  if (parameters && liveSource === source.value && liveParameters && Object.keys(parameters).every(name => Object.hasOwn(liveParameters, name) && Number.isFinite(parameters[name]))) {
    liveParameters = { ...liveParameters, ...parameters };
  } else liveParameters = null;
  liveSource = nextSource;
  source.value = nextSource;
  scheduledSource = nextSource;
  scheduleRun();
}
function cancelScheduledRun() {
  clearTimeout(runTimer);
  runTimer = null;
  scheduledSource = null;
}
function stop() {
  cancelScheduledRun();
  runner.stop();
}
defineExpose({ updateSource });

async function execute(options = {}) {
  if (busy.value || disposed) return;
  const submittedCode = source.value;
  const updating = options.preserveView && liveParameters && liveSource === submittedCode;
  const payload = updating ? { parameters: { ...liveParameters }, revision } : { code: submittedCode, revision: ++revision };
  if (!updating) liveParameters = null;
  busy.value = true;
  error.value = '';
  output.value = [];
  try {
    const response = await runner.run(props.driver, payload, (state) => { runtimeState.value = state; }, (entry) => {
      if (disposed) return;
      const previous = output.value.at(-1);
      if (previous?.stream === entry.stream) previous.text += entry.text;
      else output.value.push(entry);
    });
    if (disposed) return;
    if (source.value !== submittedCode && !(updating && liveParameters && liveSource === source.value)) {
      runtimeState.value = 'Code changed during run; run the current draft';
      return;
    }
    let plots = response.plots ?? result.value?.plots ?? [];
    if (response.plotUpdates) {
      plots = plots.map(plot => {
        const update = response.plotUpdates.find(update => update.title === plot.title);
        return update ? { ...plot, series: plot.series.map(series => {
          const change = update.series.find(change => change.name === series.name);
          return change ? { ...series, ...change } : series;
        }) } : plot;
      });
    }
    result.value = { ...response, plots };
    if (!updating) {
      liveSource = submittedCode;
      liveParameters = Object.keys(response.parameters ?? {}).length ? response.parameters : null;
    }
    preserveView.value = options.preserveView === true;
    appliedCode.value = submittedCode;
    runtimeState.value = 'Python + NumPy ready';
  } catch (failure) {
    if (!disposed) {
      liveParameters = null;
      error.value = failure.message;
      appliedCode.value = null;
      editorTab.value = 'output';
      runtimeState.value = 'Run did not complete';
    }
  } finally {
    busy.value = false;
    if (scheduledSource !== null) scheduleRun();
  }
}
function resetCode() {
  cancelScheduledRun();
  source.value = props.defaultCode;
  editorTab.value = 'code';
  execute();
}
function switchTab(event) {
  if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
  event.preventDefault();
  editorTab.value = event.key === 'Home' ? 'code' : event.key === 'End' ? 'output' : editorTab.value === 'code' ? 'output' : 'code';
  event.currentTarget.querySelector(`[data-tab="${editorTab.value}"]`)?.focus();
}
function resizePanel(axis, value) {
  if (axis === 'width') codeWidth.value = Math.max(30, Math.min(80, value));
  else workspaceHeight.value = Math.max(360, Math.min(1600, value));
}
function startResize(event, axis) {
  if (event.button !== 0) return;
  event.preventDefault();
  event.currentTarget.setPointerCapture(event.pointerId);
  resizing = { axis, startY: event.clientY, height: measuredHeight.value };
}
function moveResize(event) {
  if (!resizing) return;
  if (resizing.axis === 'width') {
    const bounds = split.value.getBoundingClientRect();
    resizePanel('width', (event.clientX - bounds.left) / bounds.width * 100);
  } else resizePanel('height', resizing.height + event.clientY - resizing.startY);
}
function resizeWithKeyboard(event, axis) {
  const decrease = axis === 'width' ? 'ArrowLeft' : 'ArrowUp';
  const increase = axis === 'width' ? 'ArrowRight' : 'ArrowDown';
  if (![decrease, increase, 'Home', 'End'].includes(event.key)) return;
  event.preventDefault();
  const current = axis === 'width' ? codeWidth.value : measuredHeight.value;
  const step = axis === 'width' ? 2 : 40;
  resizePanel(axis, event.key === 'Home' ? 0 : event.key === 'End' ? Infinity : current + (event.key === increase ? step : -step));
}
onMounted(() => {
  sizeObserver = new ResizeObserver(([entry]) => {
    if (entry.contentRect.height > 0) measuredHeight.value = Math.round(entry.contentRect.height);
  });
  sizeObserver.observe(workbench.value);
  execute();
});
onBeforeUnmount(() => { disposed = true; sizeObserver?.disconnect(); stop(); });
</script>

<template>
  <section class="python-workspace" :aria-label="title || 'Python workspace'">
    <div ref="workbench" class="py-workbench" :style="{ '--py-code-width': codeWidth + '%', height: workspaceHeight === null ? undefined : workspaceHeight + 'px' }">
      <div class="py-editor-toolbar">
        <div class="py-desktop-label"><Code :size="16" aria-hidden="true" />Code</div>
        <div class="py-editor-tabs" role="tablist" aria-label="Python workspace" @keydown="switchTab">
          <button :id="id + '-code-tab'" data-tab="code" type="button" role="tab" :aria-controls="id + '-code-panel'" :aria-selected="editorTab === 'code'" :tabindex="editorTab === 'code' ? 0 : -1" @click="editorTab = 'code'"><Code :size="16" aria-hidden="true" />Code</button>
          <button :id="id + '-output-tab'" data-tab="output" type="button" role="tab" aria-label="Output" :aria-controls="id + '-output-panel'" :aria-selected="editorTab === 'output'" :tabindex="editorTab === 'output' ? 0 : -1" @click="editorTab = 'output'"><Terminal :size="16" aria-hidden="true" />Output<span v-if="error" class="py-error-badge">Error</span><span v-else-if="output.length" class="py-output-dot" aria-hidden="true"></span></button>
        </div>
        <slot name="toolbar" />
        <label v-if="frameworkStart" class="py-framework-toggle"><input v-model="frameworkExpanded" type="checkbox"> Show framework</label>
        <span class="py-runtime" role="status">{{ runtimeState }}{{ busy ? '...' : '' }}</span>
        <div class="py-actions">
          <a v-if="download" :href="download" download title="Download default Python lesson" aria-label="Download default Python lesson"><Download :size="17" aria-hidden="true" /></a>
          <button class="py-icon" type="button" :disabled="busy" title="Reset Python code" aria-label="Reset Python code" @click="resetCode"><RotateCcw :size="17" aria-hidden="true" /></button>
          <button v-if="busy" class="py-command" type="button" @click="stop"><Square :size="16" aria-hidden="true" />Stop Python</button>
          <button v-else class="py-command" type="button" title="Run Python (Ctrl/Cmd+Enter)" @click="execute"><Play :size="16" aria-hidden="true" />Run Python</button>
        </div>
      </div>
      <div ref="split" class="py-workspace" :data-active-tab="editorTab">
        <section :id="id + '-code-panel'" class="py-code" aria-label="Python code"><PythonEditor v-model="source" v-model:framework-expanded="frameworkExpanded" :active="editorTab === 'code'" :framework-start="frameworkStart" @run="execute" /></section>
        <div class="py-resize py-resize-width" role="separator" tabindex="0" aria-label="Code panel width" aria-orientation="vertical" :aria-controls="id + '-code-panel'" :aria-valuenow="codeWidth" aria-valuemin="30" aria-valuemax="80" title="Resize code width" @pointerdown="startResize($event, 'width')" @pointermove="moveResize" @pointerup="resizing = null" @pointercancel="resizing = null" @lostpointercapture="resizing = null" @keydown="resizeWithKeyboard($event, 'width')" @dblclick="codeWidth = 65"></div>
        <section :id="id + '-output-panel'" class="py-output-panel" aria-label="Python terminal" tabindex="0">
          <div class="py-output-toolbar"><span class="py-terminal-title"><Terminal :size="16" aria-hidden="true" />Output</span><span>{{ busy ? 'Running...' : error ? 'Failed' : 'Last run' }}</span><button class="py-icon" type="button" aria-label="Clear output" title="Clear output" @click="output = []; error = ''"><Trash2 :size="16" aria-hidden="true" /></button></div>
          <div class="py-output-log" role="log" aria-label="Python output" aria-live="polite" aria-relevant="additions text">
            <p v-if="!output.length && !error" class="py-empty">{{ busy ? 'Awaiting output...' : 'No output' }}</p>
            <div v-for="(entry, index) in output" :key="index" class="py-output-entry" :data-stream="entry.stream"><span>{{ entry.stream.toUpperCase() }}</span><pre>{{ entry.text }}</pre></div>
            <pre v-if="error" class="py-error" role="alert">{{ error }}</pre>
          </div>
        </section>
      </div>
    </div>
    <div class="py-resize py-resize-height" role="separator" tabindex="0" aria-label="Workspace height" aria-orientation="horizontal" :aria-valuenow="measuredHeight" aria-valuemin="360" aria-valuemax="1600" title="Resize workspace height" @pointerdown="startResize($event, 'height')" @pointermove="moveResize" @pointerup="resizing = null" @pointercancel="resizing = null" @lostpointercapture="resizing = null" @keydown="resizeWithKeyboard($event, 'height')" @dblclick="workspaceHeight = null"></div>
  </section>
</template>