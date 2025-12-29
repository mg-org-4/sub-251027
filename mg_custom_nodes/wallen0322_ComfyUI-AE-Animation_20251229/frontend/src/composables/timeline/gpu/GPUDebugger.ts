/**
 * GPUDebugger - Debugging utilities for GPU Timeline Renderer
 * 
 * Provides tools for debugging GPU rendering issues
 */

export interface GPUDebugInfo {
  adapter: string
  device: string
  limits: Record<string, number>
  features: string[]
  canvasFormat: string
}

export class GPUDebugger {
  private device: GPUDevice
  private adapter: GPUAdapter | null = null

  constructor(device: GPUDevice, adapter?: GPUAdapter) {
    this.device = device
    this.adapter = adapter || null
  }

  /**
   * Get GPU device information
   */
  async getDeviceInfo(): Promise<GPUDebugInfo> {
    const limits: Record<string, number> = {}
    const features: string[] = []

    // Get device limits
    if (this.device.limits) {
      for (const [key, value] of Object.entries(this.device.limits)) {
        if (typeof value === 'number') {
          limits[key] = value
        }
      }
    }

    // Get device features
    if (this.device.features) {
      this.device.features.forEach((feature) => {
        features.push(feature)
      })
    }

    // Get adapter info
    let adapterName = 'Unknown'
    if (this.adapter) {
      try {
        // requestAdapterInfo is not standard yet, use type assertion
        const info = await (this.adapter as any).requestAdapterInfo?.()
        adapterName = info?.description || info?.vendor || 'WebGPU Adapter'
      } catch {
        adapterName = 'WebGPU Adapter'
      }
    }

    return {
      adapter: adapterName,
      device: 'WebGPU Device',
      limits,
      features,
      canvasFormat: navigator.gpu?.getPreferredCanvasFormat() || 'unknown'
    }
  }

  /**
   * Log device information to console
   */
  async logDeviceInfo(): Promise<void> {
    const info = await this.getDeviceInfo()
    console.group('[GPUDebugger] Device Information')
    console.log('Adapter:', info.adapter)
    console.log('Device:', info.device)
    console.log('Canvas Format:', info.canvasFormat)
    console.log('Features:', info.features)
    console.log('Limits:', info.limits)
    console.groupEnd()
  }

  /**
   * Check if device supports required features
   */
  checkRequiredFeatures(): { supported: boolean; missing: string[] } {
    const required: string[] = []
    const missing: string[] = []

    for (const feature of required) {
      if (!this.device.features.has(feature)) {
        missing.push(feature)
      }
    }

    return {
      supported: missing.length === 0,
      missing
    }
  }

  /**
   * Get memory usage estimate
   */
  getMemoryEstimate(textureCount: number, avgTextureSize: number): string {
    const bytesPerPixel = 4 // RGBA8
    const totalBytes = textureCount * avgTextureSize * bytesPerPixel
    const mb = totalBytes / (1024 * 1024)
    return `~${Math.round(mb)}MB`
  }

  /**
   * Validate render pipeline configuration
   */
  validatePipeline(pipeline: GPURenderPipeline | null): boolean {
    if (!pipeline) {
      console.error('[GPUDebugger] Pipeline is null')
      return false
    }
    return true
  }

  /**
   * Log texture cache statistics
   */
  logCacheStats(stats: any): void {
    console.group('[GPUDebugger] Texture Cache Stats')
    console.log('Total Textures:', stats.totalTextures)
    console.log('Cache Hits:', stats.cacheHits)
    console.log('Cache Misses:', stats.cacheMisses)
    console.log('Hit Rate:', `${stats.hitRate}%`)
    console.log('Memory Usage:', stats.memoryUsage)
    console.groupEnd()
  }

  /**
   * Log performance statistics
   */
  logPerformanceStats(stats: any): void {
    console.group('[GPUDebugger] Performance Stats')
    console.log('FPS:', stats.fps)
    console.log('Frame Time:', `${stats.frameTime}ms`)
    console.log('Avg Frame Time:', `${stats.avgFrameTime}ms`)
    console.log('Min Frame Time:', `${stats.minFrameTime}ms`)
    console.log('Max Frame Time:', `${stats.maxFrameTime}ms`)
    console.log('Total Frames:', stats.totalFrames)
    console.log('Dropped Frames:', stats.droppedFrames)
    console.groupEnd()
  }

  /**
   * Create a debug overlay for performance monitoring
   */
  createDebugOverlay(container: HTMLElement): HTMLDivElement {
    const overlay = document.createElement('div')
    overlay.style.position = 'absolute'
    overlay.style.top = '10px'
    overlay.style.right = '10px'
    overlay.style.padding = '10px'
    overlay.style.backgroundColor = 'rgba(0, 0, 0, 0.7)'
    overlay.style.color = '#00ff00'
    overlay.style.fontFamily = 'monospace'
    overlay.style.fontSize = '12px'
    overlay.style.zIndex = '10000'
    overlay.style.pointerEvents = 'none'
    container.appendChild(overlay)
    return overlay
  }

  /**
   * Update debug overlay with current stats
   */
  updateDebugOverlay(
    overlay: HTMLDivElement,
    perfStats: any,
    cacheStats: any
  ): void {
    overlay.innerHTML = `
      <div><strong>GPU Timeline Renderer</strong></div>
      <div>FPS: ${perfStats.fps}</div>
      <div>Frame: ${perfStats.frameTime}ms</div>
      <div>Avg: ${perfStats.avgFrameTime}ms</div>
      <div>Dropped: ${perfStats.droppedFrames}</div>
      <div>---</div>
      <div>Textures: ${cacheStats.totalTextures}</div>
      <div>Hit Rate: ${cacheStats.hitRate}%</div>
    `
  }
}
