<script setup>
import { computed, ref } from 'vue';
import { numericParameters } from './parameters.js';
import PythonWorkspace from './PythonWorkspace.vue';
import PythonPlots from './PythonPlots.vue';
import ValueGrid from './ValueGrid.vue';
import './style.css';

const props = defineProps({ defaultCode: String, title: String, download: String, frameworkStart: String });
const emit = defineEmits(['state']);
const workspace = ref(null);
const execution = ref({ result: null, busy: false, stale: false, error: '' });
const groups = [{ key: 'inputs', title: 'Inputs' }, { key: 'outputs', title: 'Outputs' }];
const parameters = computed(() => {
  const source = execution.value.source ?? props.defaultCode;
  const names = Object.keys(execution.value.result?.parameters ?? {});
  const parsed = numericParameters(source, names);
  return names.map(name => ({ name, label: (name[0].toUpperCase() + name.slice(1)).replaceAll('_', ' '), value: null, ...parsed[name] }));
});
function changeParameter(parameter, event) {
  const value = event.target.valueAsNumber;
  if (parameter.value === null || !Number.isFinite(value)) return;
  const source = execution.value.source ?? props.defaultCode;
  workspace.value.updateSource(source.slice(0, parameter.from) + value + source.slice(parameter.to), { [parameter.name]: value });
}
</script>

<template>
  <section class="python-lab">
    <PythonWorkspace ref="workspace" :default-code="defaultCode" :title="title" :download="download" :framework-start="frameworkStart" @state="execution = $event; emit('state', $event)"><template #toolbar><slot name="toolbar" /></template></PythonWorkspace>
    <section class="py-visualization" aria-label="Python visualization" :aria-busy="execution.busy">
      <header class="py-result-heading"><h2>Visualization</h2><span v-if="execution.stale || execution.error" class="py-stale">{{ execution.result ? 'Last successful run; current code not applied' : 'No successful run' }}</span><span v-else-if="execution.busy">Running...</span><span v-else>Current code</span></header>
      <slot name="controls" :source="execution.source ?? defaultCode" :result="execution.result" :update-source="(nextSource, parameters) => workspace?.updateSource(nextSource, parameters)">
        <div v-if="!$slots.controls && parameters.length" class="py-parameters"><label v-for="parameter in parameters" :key="parameter.name">{{ parameter.label }}<input type="number" step="any" :aria-label="parameter.label" :value="parameter.value" :disabled="parameter.value === null" @change="changeParameter(parameter, $event)"></label></div>
      </slot>
      <p v-if="!execution.result" class="py-empty">Awaiting Python results</p>
      <template v-else>
        <slot v-if="execution.result.plots?.length" name="plots" :source="execution.source ?? defaultCode" :result="execution.result" :preserve-view="execution.preserveView" :update-source="(nextSource, parameters) => workspace?.updateSource(nextSource, parameters)">
          <PythonPlots :plots="execution.result.plots" :preserve-view="execution.preserveView" />
        </slot>
        <template v-if="execution.result.inputs.length || execution.result.outputs.length || !execution.result.plots?.length">
          <section v-for="group in groups" :key="group.key" class="py-value-group" :data-group="group.key" :aria-label="group.title">
            <header><h2>{{ group.title }}</h2><span>{{ execution.result[group.key].length }} values</span></header>
            <p v-if="!execution.result[group.key].length" class="py-empty">No exported values</p>
            <div class="py-variables"><ValueGrid v-for="variable in execution.result[group.key]" :key="variable.name" :variable="variable" /></div>
          </section>
          <div class="py-color-key"><i></i>Positive<i></i>Negative</div>
        </template>
      </template>
    </section>
  </section>
</template>