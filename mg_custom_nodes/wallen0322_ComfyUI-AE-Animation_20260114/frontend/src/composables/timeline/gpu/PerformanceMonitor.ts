/**
 * PerformanceMonitor - GPU Rendering Performance Tracking
 * 
 * Monitors frame times, FPS, and GPU memory usage
 */

export interface PerformanceStats {
  fps: number
  frameTime: number
  avgFrameTime: number
  minFrameTime: number
  maxFrameTime: number
  totalFrames: number
  droppedFrames: number
}

export class PerformanceMonitor {
  private frameTimes: number[] = []
  private lastFrameTime: number = 0
  private frameCount: number = 0
  private droppedFrames: number = 0
  private startTime: number = 0
  private maxSamples: number = 60

  constructor(maxSamples: number = 60) {
    this.maxSamples = maxSamples
    this.startTime = performance.now()
  }

  /**
   * Mark the start of a frame
   */
  startFrame(): void {
    this.lastFrameTime = performance.now()
  }

  /**
   * Mark the end of a frame
   */
  endFrame(): void {
    const now = performance.now()
    const frameTime = now - this.lastFrameTime

    this.frameTimes.push(frameTime)
    if (this.frameTimes.length > this.maxSamples) {
      this.frameTimes.shift()
    }

    this.frameCount++

    // Count dropped frames (> 16.67ms = below 60 FPS)
    if (frameTime > 16.67) {
      this.droppedFrames++
    }
  }

  /**
   * Get current performance statistics
   */
  getStats(): PerformanceStats {
    if (this.frameTimes.length === 0) {
      return {
        fps: 0,
        frameTime: 0,
        avgFrameTime: 0,
        minFrameTime: 0,
        maxFrameTime: 0,
        totalFrames: 0,
        droppedFrames: 0
      }
    }

    const avgFrameTime =
      this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length
    const minFrameTime = Math.min(...this.frameTimes)
    const maxFrameTime = Math.max(...this.frameTimes)
    const fps = 1000 / avgFrameTime

    return {
      fps: Math.round(fps * 10) / 10,
      frameTime: Math.round(this.frameTimes[this.frameTimes.length - 1] * 100) / 100,
      avgFrameTime: Math.round(avgFrameTime * 100) / 100,
      minFrameTime: Math.round(minFrameTime * 100) / 100,
      maxFrameTime: Math.round(maxFrameTime * 100) / 100,
      totalFrames: this.frameCount,
      droppedFrames: this.droppedFrames
    }
  }

  /**
   * Reset all statistics
   */
  reset(): void {
    this.frameTimes = []
    this.frameCount = 0
    this.droppedFrames = 0
    this.startTime = performance.now()
  }

  /**
   * Get formatted stats string
   */
  getFormattedStats(): string {
    const stats = this.getStats()
    return `FPS: ${stats.fps} | Frame: ${stats.frameTime}ms | Avg: ${stats.avgFrameTime}ms | Dropped: ${stats.droppedFrames}`
  }

  /**
   * Check if performance is good (>= 55 FPS)
   */
  isPerformanceGood(): boolean {
    const stats = this.getStats()
    return stats.fps >= 55
  }

  /**
   * Check if performance is acceptable (>= 30 FPS)
   */
  isPerformanceAcceptable(): boolean {
    const stats = this.getStats()
    return stats.fps >= 30
  }
}
