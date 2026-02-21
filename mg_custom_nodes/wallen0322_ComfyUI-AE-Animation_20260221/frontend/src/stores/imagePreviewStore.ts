import { defineStore } from 'pinia'
import { api } from '@/scripts/api'
import { app } from '@/scripts/app'
import type { LGraphNode } from '@/lib/litegraph/src/litegraph'

// Simplified store for mask editor
export const useNodeOutputStore = defineStore('nodeOutput', () => {
  function getNodeOutputs(node: LGraphNode): any | undefined {
    return app.nodeOutputs[String(node.id)]
  }

  function getNodePreviews(node: LGraphNode): string[] | undefined {
    return app.nodePreviewImages[String(node.id)]
  }

  function getNodeImageUrls(node: LGraphNode): string[] | undefined {
    const previews = getNodePreviews(node)
    if (previews?.length) return previews

    const outputs = getNodeOutputs(node)
    if (!outputs?.images?.length) return

    const rand = app.getRandParam()
    return outputs.images.map((image: any) => {
      const imgUrlPart = new URLSearchParams(image)
      return api.apiURL(`/view?${imgUrlPart}${rand}`)
    })
  }

  function updateNodeImages(node: LGraphNode) {
    // Stub for compatibility
  }

  return {
    getNodeOutputs,
    getNodePreviews,
    getNodeImageUrls,
    updateNodeImages
  }
})
