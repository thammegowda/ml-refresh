<script setup>
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { basicSetup } from 'codemirror';
import { python } from '@codemirror/lang-python';
import { foldEffect, foldedRanges, unfoldEffect, HighlightStyle, syntaxHighlighting } from '@codemirror/language';
import { tags } from '@lezer/highlight';
import { EditorState, Prec } from '@codemirror/state';
import { Decoration, EditorView, keymap } from '@codemirror/view';

const props = defineProps({ modelValue: String, active: Boolean, frameworkStart: String, frameworkExpanded: Boolean });
const emit = defineEmits(['update:modelValue', 'update:frameworkExpanded', 'run']);
const container = ref(null);
let view;

function foldFramework() {
  if (!view || !props.frameworkStart) return;
  const effects = [];
  const source = view.state.doc.toString();
  const start = source.indexOf('\n' + props.frameworkStart);
  foldedRanges(view.state).between(0, view.state.doc.length, (from, to) => {
    if (from === start && to === source.length) effects.push(unfoldEffect.of({ from, to }));
  });
  if (!props.frameworkExpanded && start >= 0) effects.push(foldEffect.of({ from: start, to: source.length }));
  view.dispatch({ effects });
}

onMounted(() => {
  view = new EditorView({
    parent: container.value,
    state: EditorState.create({
      doc: props.modelValue,
      extensions: [
        basicSetup,
        python(),
        syntaxHighlighting(HighlightStyle.define([
          { tag: tags.keyword, color: '#a62646' },
          { tag: [tags.variableName, tags.self], color: '#273b43' },
          { tag: [tags.function(tags.variableName), tags.function(tags.propertyName)], color: '#245fa5' },
          { tag: [tags.className, tags.typeName], color: '#87520b' },
          { tag: tags.propertyName, color: '#805a25' },
          { tag: [tags.string, tags.special(tags.string)], color: '#267044' },
          { tag: [tags.number, tags.bool, tags.null], color: '#a15312' },
          { tag: [tags.operator, tags.punctuation], color: '#5e5378' },
          { tag: [tags.meta, tags.escape], color: '#955078' },
          { tag: tags.comment, color: '#707070' },
          { tag: tags.invalid, color: '#b42318', textDecoration: 'underline wavy' },
        ])),
        EditorState.tabSize.of(4),
        EditorView.lineWrapping,
        EditorView.decorations.compute(['selection'], state => Decoration.set(state.selection.ranges
          .filter((range, index) => !range.empty && index !== state.selection.mainIndex)
          .map(range => Decoration.mark({ class: 'cm-secondary-selection' }).range(range.from, range.to)))),
        EditorView.contentAttributes.of({ 'aria-label': 'Python source', 'aria-multiline': 'true', spellcheck: 'false' }),
        Prec.highest(keymap.of([{ key: 'Mod-Enter', run: () => { emit('run'); return true; } }])),
        EditorView.updateListener.of((update) => {
          if (update.docChanged) emit('update:modelValue', update.state.doc.toString());
          else if (props.frameworkStart && update.transactions.some(transaction => transaction.effects.some(effect => effect.is(foldEffect) || effect.is(unfoldEffect)))) {
            const start = update.state.doc.toString().indexOf('\n' + props.frameworkStart);
            let folded = false;
            foldedRanges(update.state).between(0, update.state.doc.length, (from, to) => {
              if (from === start && to === update.state.doc.length) folded = true;
            });
            if (start >= 0 && props.frameworkExpanded === folded) emit('update:frameworkExpanded', !folded);
          }
        }),
        EditorView.theme({
          '&': { height: '360px', color: '#202e2b', backgroundColor: '#fbfcfb', fontSize: '13px' },
          '.cm-scroller': { overflow: 'auto', fontFamily: 'Menlo, Consolas, monospace', lineHeight: '1.65' },
          '.cm-content': { padding: '12px 0', minHeight: '100%' },
          '.cm-line': { padding: '0 12px' },
          '.cm-gutters': { backgroundColor: '#edf2ef', color: '#64716d', borderRight: '1px solid #dce3df' },
          '.cm-activeLine, .cm-activeLineGutter': { backgroundColor: '#e5efea' },
          '&.cm-focused .cm-selectionBackground, .cm-selectionBackground': { backgroundColor: '#263238 !important' },
          '&.cm-editor .cm-content::selection, &.cm-editor .cm-content ::selection': { backgroundColor: '#263238 !important', color: '#ffffff !important' },
          '.cm-secondary-selection, .cm-secondary-selection *': { color: '#ffffff !important' },
          '&.cm-focused': { outline: '2px solid #087f72', outlineOffset: '-2px' },
        }),
      ],
    }),
  });
  foldFramework();
});
watch(() => props.modelValue, (source) => {
  if (view && source !== view.state.doc.toString()) {
    view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: source } });
    foldFramework();
  }
});
watch(() => props.frameworkExpanded, foldFramework);
watch(() => props.active, async (active) => {
  if (active) { await nextTick(); view?.requestMeasure(); }
});
onBeforeUnmount(() => view?.destroy());
</script>

<template><div ref="container" class="python-editor"></div></template>