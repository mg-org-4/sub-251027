import { createApp } from 'vue'
import { createPinia } from 'pinia'
import './timeline.css'
import TimelineApp from './TimelineApp.vue'

export interface TimelineAppOptions {
  node: any
}

export function createTimelineApp(
  container: HTMLElement,
  options: TimelineAppOptions
) {
  const app = createApp(TimelineApp, {
    node: options.node
  })
  
  const pinia = createPinia()
  app.use(pinia)

  app.mount(container)
  return app
}

// Export to global for timeline.js to access
declare global {
  interface Window {
    createTimelineApp: typeof createTimelineApp
  }
}

window.createTimelineApp = createTimelineApp
