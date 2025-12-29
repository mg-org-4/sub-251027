// @ts-nocheck
// 这里使用 require 避免在缺少类型定义时阻塞构建（运行时由打包好的依赖提供）
// eslint-disable-next-line @typescript-eslint/no-var-requires
const TheatreCore = require('@theatre/core') as typeof import('@theatre/core')
// eslint-disable-next-line @typescript-eslint/no-var-requires
const TheatreStudio = require('@theatre/studio') as typeof import('@theatre/studio')

const { getProject, onChange, types: t, val } = TheatreCore
const studio = TheatreStudio.default || TheatreStudio
import { ref, shallowRef, onMounted, onUnmounted } from 'vue'
import type { IProject, ISheet, ISheetObject } from '@theatre/core'

// Theatre.js 项目状态
let theatreInitialized = false
let project: IProject | null = null
let sheet: ISheet | null = null

// 图层对象映射
const layerObjects = new Map<string, ISheetObject<any>>()

// 导出的响应式状态
export const isTheatreReady = ref(false)
export const currentPosition = ref(0)

// 初始化 Theatre Studio
export async function initTheatre() {
  if (theatreInitialized) return
  
  try {
    // 初始化 studio（可视化编辑器）
    studio.initialize({
      usePersistentStorage: false // 不使用持久化存储
    })
    
    // 创建项目
    project = getProject('AE-Animation', {
      // 可以传入已保存的状态
      // state: savedState
    })
    
    // 创建主时间轴 sheet
    sheet = project.sheet('Main Timeline')
    
    theatreInitialized = true
    isTheatreReady.value = true
    
    console.log('[Theatre] Initialized successfully')
    
    // 监听播放位置变化
  const sequence = sheet.sequence
  onChange(sequence.pointer.position, (pos: number) => {
      currentPosition.value = pos
    })
    
  } catch (e) {
    console.error('[Theatre] Failed to initialize:', e)
  }
}

// 创建图层对象
export function createLayerObject(
  layerId: string,
  layerName: string,
  layerType: 'foreground' | 'background',
  initialProps: {
    x?: number
    y?: number
    scale?: number
    rotation?: number
    opacity?: number
    rotationX?: number
    rotationY?: number
  } = {}
): ISheetObject<any> | null {
  if (!sheet) {
    console.warn('[Theatre] Sheet not ready')
    return null
  }
  
  // 如果已存在，先移除
  if (layerObjects.has(layerId)) {
    // Theatre.js 不支持直接删除对象，需要重新创建 sheet
    // 这里我们只是更新引用
  }
  
  // 定义属性 schema
  const propsSchema = {
    // 2D 变换
    position: t.compound({
      x: t.number(initialProps.x ?? 0, { range: [-2000, 2000] }),
      y: t.number(initialProps.y ?? 0, { range: [-2000, 2000] })
    }),
    scale: t.number(initialProps.scale ?? 1, { range: [0.1, 5] }),
    rotation: t.number(initialProps.rotation ?? 0, { range: [-180, 180] }),
    opacity: t.number(initialProps.opacity ?? 1, { range: [0, 1] }),
    
    // 3D 变换（仅前景图层）
    ...(layerType === 'foreground' ? {
      rotation3D: t.compound({
        x: t.number(initialProps.rotationX ?? 0, { range: [-90, 90] }),
        y: t.number(initialProps.rotationY ?? 0, { range: [-90, 90] })
      })
    } : {})
  }
  
  // 创建 Theatre 对象
  const obj = sheet.object(layerName, propsSchema)
  layerObjects.set(layerId, obj)
  
  console.log('[Theatre] Created layer object:', layerName)
  
  return obj
}

// 获取图层对象
export function getLayerObject(layerId: string) {
  return layerObjects.get(layerId) || null
}

// 移除图层对象
export function removeLayerObject(layerId: string) {
  // Theatre.js 不支持直接删除对象
  // 只能从我们的映射中移除
  layerObjects.delete(layerId)
}

// 订阅图层属性变化
export function subscribeToLayer(
  layerId: string,
  callback: (props: any) => void
) {
  const obj = layerObjects.get(layerId)
  if (!obj) return () => {}
  
  return onChange(obj.props, callback)
}

// 获取当前属性值
export function getLayerProps(layerId: string) {
  const obj = layerObjects.get(layerId)
  if (!obj) return null
  
  // 使用 Theatre 的 val 函数获取当前 props 的实时值
  return val(obj.props)
}

// 播放控制
export function play() {
  if (!sheet) return
  sheet.sequence.play({
    iterationCount: Infinity,
    range: [0, sheet.sequence.pointer.length as any]
  })
}

export function pause() {
  if (!sheet) return
  sheet.sequence.pause()
}

export function setPosition(time: number) {
  if (!sheet) return
  sheet.sequence.position = time
}

export function getPosition() {
  if (!sheet) return 0
  return sheet.sequence.position
}

// 获取/设置时间轴长度
export function setDuration(duration: number) {
  // Theatre.js 的 duration 是通过关键帧自动确定的
  // 这里可以通过设置一个虚拟关键帧来控制
  console.log('[Theatre] Duration set to:', duration)
}

// 导出状态
export function exportState() {
  if (!project) return null
  return studio.createContentOfSaveFile(project.address.projectId)
}

// 导入状态
export function importState(state: any) {
  // 需要重新创建项目来加载状态
  // getProject('AE-Animation', { state })
}

// 隐藏/显示 Studio UI
export function hideStudio() {
  studio.ui.hide()
}

export function showStudio() {
  studio.ui.restore()
}

// 清理
export function destroyTheatre() {
  layerObjects.clear()
  theatreInitialized = false
  isTheatreReady.value = false
}

// Vue Composable
export function useTheatre() {
  onMounted(() => {
    initTheatre()
  })
  
  onUnmounted(() => {
    // 不销毁，因为可能被其他组件使用
  })
  
  return {
    isReady: isTheatreReady,
    position: currentPosition,
    createLayerObject,
    getLayerObject,
    removeLayerObject,
    subscribeToLayer,
    getLayerProps,
    play,
    pause,
    setPosition,
    getPosition,
    exportState,
    hideStudio,
    showStudio
  }
}
