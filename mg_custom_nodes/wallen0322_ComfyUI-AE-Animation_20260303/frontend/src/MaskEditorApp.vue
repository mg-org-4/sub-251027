<template>
  <div class="mask-editor-app">
    <MaskEditorContent v-if="node" :node="node" />
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import MaskEditorContent from '@/components/maskeditor/MaskEditorContent.vue'
import type { LGraphNode } from '@/lib/litegraph/src/litegraph'

const props = defineProps<{
  nodeData?: LGraphNode
}>()

const node = ref<LGraphNode | null>(null)

onMounted(() => {
  if (props.nodeData) {
    node.value = props.nodeData
  }
})
</script>

<style>
.mask-editor-app {
  width: 100%;
  height: 100%;
  position: relative;
}
</style>
