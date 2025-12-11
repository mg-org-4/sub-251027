import { createApp } from 'vue'
import { createPinia } from 'pinia'

import MaskEditorApp from './MaskEditorApp.vue'

export interface MaskEditorMountOptions {
  nodeData?: any
  props?: Record<string, unknown>
}

export function createMaskEditorApp(
  container: HTMLElement,
  options: MaskEditorMountOptions = {}
) {
  const app = createApp(MaskEditorApp, {
    nodeData: options.nodeData,
    ...options.props
  })
  const pinia = createPinia()
  app.use(pinia)

  app.mount(container)
  return app
}

// Export to global for timeline.js to access
declare global {
  interface Window {
    createMaskEditorApp: typeof createMaskEditorApp
  }
}

window.createMaskEditorApp = createMaskEditorApp
