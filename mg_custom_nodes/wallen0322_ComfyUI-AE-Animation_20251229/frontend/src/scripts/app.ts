import type { LGraphNode } from '@/lib/litegraph/src/litegraph'

interface ComfyGraph {
  setDirtyCanvas(dirty: boolean): void
}

class ComfyApp {
  graph: ComfyGraph = {
    setDirtyCanvas: () => {}
  }

  nodeOutputs: Record<string, any> = {}
  nodePreviewImages: Record<string, string[]> = {}

  getPreviewFormatParam(): string {
    return '&preview=true'
  }

  getRandParam(): string {
    return `&rand=${Math.random()}`
  }
}

export const app = new ComfyApp()
