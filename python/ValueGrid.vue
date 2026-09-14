<script setup>
import { computed } from 'vue';

const props = defineProps({ variable: Object });
const maximum = computed(() => Math.max(1, ...props.variable.values.flat().map(Math.abs)));
const shape = computed(() => props.variable.shape.length === 0 ? 'scalar' : `(${props.variable.shape.join(', ')}${props.variable.shape.length === 1 ? ',' : ''})`);
const columns = computed(() => props.variable.values[0]?.length || 1);
function format(value) {
  if (Math.abs(value) >= 1e6) return value.toExponential(3);
  return Number(value.toFixed(3)).toString();
}
function cellLabel(row, column) {
  if (!props.variable.shape.length) return props.variable.name;
  return `${props.variable.name}[${props.variable.shape.length === 1 ? column : `${row},${column}`}]`;
}
function color(value) {
  return { backgroundColor: `rgba(${value < 0 ? '176,78,117' : '8,127,114'},${0.04 + Math.abs(value) / maximum.value * 0.2})` };
}
</script>

<template>
  <article class="py-variable" :data-variable="variable.name">
    <header><h3>{{ variable.name }}</h3><span class="py-shape">{{ shape }}</span></header>
    <div class="py-variable-meta"><span>{{ variable.dtype }}</span><span v-if="variable.truncated">Preview {{ variable.values.length }} &times; {{ columns }} of {{ variable.shape.join(' x ') }}</span></div>
    <p v-if="!variable.size" class="py-empty">Empty array</p>
    <div v-else class="py-grid-scroll" tabindex="0" role="group" :aria-label="`${variable.name} values`">
      <div class="py-value-grid" :style="{ gridTemplateColumns: `repeat(${columns}, 72px)` }">
        <template v-for="(row, rowIndex) in variable.values" :key="rowIndex">
          <output v-for="(value, column) in row" :key="column" :aria-label="cellLabel(rowIndex, column)" :title="`${cellLabel(rowIndex, column)} = ${value}`" :style="color(value)">{{ format(value) }}</output>
        </template>
      </div>
    </div>
  </article>
</template>