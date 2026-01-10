<template>
  <div class="theatre-root" ref="rootRef">
    <!-- Canvas 渲染区域 -->
    <div class="canvas-container">
      <canvas 
        ref="canvasRef"
        :width="project.width"
        :height="project.height"
        @mousedown="onMouseDown"
        @mousemove="onMouseMove"
        @mouseup="onMouseUp"
        @wheel.prevent="onWheel"
      />
      <div class="canvas-info">
        {{ project.width }}x{{ project.height }} | {{ position.toFixed(2) }}s
      </div>
    </div>
    
    <!-- 控制栏 -->
    <div class="controls">
      <button @click="addLayer('foreground')">+ Layer</button>
      <button @click="addLayer('background')">+ BG</button>
      <span class="sep"></span>
      <button @click="togglePlay">{{ isPlaying ? '■' : '▶' }}</button>
      <button @click="seekTo(0)">|◀</button>
      <span class="sep"></span>
      <button @click="saveProject">Save</button>
      <button @click="$emit('close')">✕</button>
    </div>
    
    <!-- Theatre Studio 会自动挂载到 body -->
    
    <!-- 隐藏的文件输入 -->
    <input ref="fileInput" type="file" accept="image/*" hidden @change="onFileSelect" />
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, onUnmounted, watch } from 'vue'
import { getProject, onChange, types as t, val } from '@theatre/core'
import studio from '@theatre/studio'
import type { IProject, ISheet, ISheetObject } from '@theatre/core'

const props = defineProps<{ node: any }>()
const emit = defineEmits(['close'])

// DOM refs
const rootRef = ref<HTMLDivElement>()
const canvasRef = ref<HTMLCanvasElement>()
const fileInput = ref<HTMLInputElement>()

// Project settings
const project = reactive({
  width: 1280,
  height: 720,
  fps: 30,
  duration: 5
})

// Theatre.js state
let theatreProject: IProject | null = null
let sheet: ISheet | null = null
const layerObjects = new Map<string, ISheetObject<any>>()

// Playback state
const position = ref(0)
const isPlaying = ref(false)

// Layers
interface Layer {
  id: string
  name: string
  type: 'foreground' | 'background'
  image?: HTMLImageElement
  imageData?: string
  theatreObj?: ISheetObject<any>
}
const layers = ref<Layer[]>([])
const currentLayerIndex = ref(-1)

// Render context
let ctx: CanvasRenderingContext2D | null = null
let animFrameId: number | null = null

// Pending layer type for file selection
let pendingLayerType: 'foreground' | 'background' = 'foreground'

onMounted(async () => {
  // Initialize canvas
  if (canvasRef.value) {
    ctx = canvasRef.value.getContext('2d', { alpha: false })
  }
  
  // Initialize Theatre.js
  await initTheatre()
  
  // Load saved data
  loadFromNode()
  
  // Start render loop
  startRenderLoop()
})

onUnmounted(() => {
  if (animFrameId) cancelAnimationFrame(animFrameId)
  // Don't destroy Theatre - it's a singleton
})

async function initTheatre() {
  try {
    // Studio 已在 timeline-main.ts 中初始化
    // 创建项目
    theatreProject = getProject('AE-Animation')
    
    // 创建主 sheet
    sheet = theatreProject.sheet('Main')
    
    // 监听播放位置变化
    onChange(sheet.sequence.pointer.position, (pos) => {
      position.value = pos
    })
    
    console.log('[Theatre] Project ready')
  } catch (e) {
    console.error('[Theatre] Init failed:', e)
  }
}

function createTheatreObject(layer: Layer) {
  if (!sheet) return
  
  const objName = `${layer.type}_${layer.id}`
  
  // Define props schema based on layer type
  const propsSchema = layer.type === 'background' 
    ? {
        opacity: t.number(1, { range: [0, 1] }),
        scale: t.number(1, { range: [0.1, 3] })
      }
    : {
        x: t.number(0, { range: [-2000, 2000] }),
        y: t.number(0, { range: [-2000, 2000] }),
        scale: t.number(1, { range: [0.1, 5] }),
        rotation: t.number(0, { range: [-180, 180] }),
        opacity: t.number(1, { range: [0, 1] })
      }
  
  const obj = sheet.object(objName, propsSchema)
  layer.theatreObj = obj
  layerObjects.set(layer.id, obj)
  
  // Subscribe to changes
  onChange(obj.props, () => {
    // Trigger re-render when props change
  })
  
  return obj
}

function addLayer(type: 'foreground' | 'background') {
  pendingLayerType = type
  fileInput.value?.click()
}

function onFileSelect(e: Event) {
  const input = e.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) return
  
  const reader = new FileReader()
  reader.onload = (e) => {
    const imageData = e.target?.result as string
    const img = new Image()
    img.onload = () => {
      const layer: Layer = {
        id: `layer_${Date.now()}`,
        name: `Layer ${layers.value.length + 1}`,
        type: pendingLayerType,
        image: img,
        imageData
      }
      
      // Create Theatre object for this layer
      createTheatreObject(layer)
      
      layers.value.push(layer)
      currentLayerIndex.value = layers.value.length - 1
    }
    img.src = imageData
  }
  reader.readAsDataURL(file)
  
  // Reset input
  input.value = ''
}

function togglePlay() {
  if (!sheet) return
  
  if (isPlaying.value) {
    sheet.sequence.pause()
    isPlaying.value = false
  } else {
    sheet.sequence.play({
      iterationCount: Infinity,
      range: [0, project.duration]
    })
    isPlaying.value = true
  }
}

function seekTo(time: number) {
  if (!sheet) return
  sheet.sequence.position = time
}

function startRenderLoop() {
  function render() {
    if (!ctx || !canvasRef.value) {
      animFrameId = requestAnimationFrame(render)
      return
    }
    
    // Clear
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, project.width, project.height)
    
    // Draw layers
    layers.value.forEach((layer, i) => {
      if (!layer.image || !layer.theatreObj) return
      
      // Get current values from Theatre
      const props = val(layer.theatreObj.props)
      
      ctx!.save()
      
      if (layer.type === 'background') {
        // Background layer
        ctx!.globalAlpha = props.opacity ?? 1
        const scale = props.scale ?? 1
        const w = layer.image.width * scale
        const h = layer.image.height * scale
        ctx!.drawImage(
          layer.image,
          (project.width - w) / 2,
          (project.height - h) / 2,
          w, h
        )
      } else {
        // Foreground layer
        const x = props.x ?? 0
        const y = props.y ?? 0
        const scale = props.scale ?? 1
        const rotation = props.rotation ?? 0
        const opacity = props.opacity ?? 1
        
        ctx!.globalAlpha = opacity
        ctx!.translate(project.width / 2 + x, project.height / 2 + y)
        ctx!.rotate(rotation * Math.PI / 180)
        ctx!.scale(scale, scale)
        
        const w = layer.image.width
        const h = layer.image.height
        ctx!.drawImage(layer.image, -w / 2, -h / 2, w, h)
        
        // Draw selection border
        if (i === currentLayerIndex.value) {
          ctx!.strokeStyle = '#4a7bc8'
          ctx!.lineWidth = 2 / scale
          ctx!.strokeRect(-w / 2 - 2, -h / 2 - 2, w + 4, h + 4)
        }
      }
      
      ctx!.restore()
    })
    
    animFrameId = requestAnimationFrame(render)
  }
  
  render()
}

// Mouse interaction
let isDragging = false
let dragStartX = 0
let dragStartY = 0
let dragStartLayerX = 0
let dragStartLayerY = 0

function onMouseDown(e: MouseEvent) {
  const layer = layers.value[currentLayerIndex.value]
  if (!layer || layer.type === 'background' || !layer.theatreObj) return
  
  isDragging = true
  const rect = canvasRef.value!.getBoundingClientRect()
  const scaleX = project.width / rect.width
  const scaleY = project.height / rect.height
  
  dragStartX = (e.clientX - rect.left) * scaleX
  dragStartY = (e.clientY - rect.top) * scaleY
  
  const props = val(layer.theatreObj.props)
  dragStartLayerX = props.x ?? 0
  dragStartLayerY = props.y ?? 0
}

function onMouseMove(e: MouseEvent) {
  if (!isDragging) return
  
  const layer = layers.value[currentLayerIndex.value]
  if (!layer || !layer.theatreObj) return
  
  const rect = canvasRef.value!.getBoundingClientRect()
  const scaleX = project.width / rect.width
  const scaleY = project.height / rect.height
  
  const x = (e.clientX - rect.left) * scaleX
  const y = (e.clientY - rect.top) * scaleY
  
  const dx = x - dragStartX
  const dy = y - dragStartY
  
  // Update via Theatre scrub
  studio.transaction(({ set }) => {
    set(layer.theatreObj!.props.x, dragStartLayerX + dx)
    set(layer.theatreObj!.props.y, dragStartLayerY + dy)
  })
}

function onMouseUp() {
  isDragging = false
}

function onWheel(e: WheelEvent) {
  const layer = layers.value[currentLayerIndex.value]
  if (!layer || !layer.theatreObj) return
  
  const delta = e.deltaY > 0 ? -0.05 : 0.05
  const props = val(layer.theatreObj.props)
  
  if (e.shiftKey) {
    // Rotate
    const newRotation = (props.rotation ?? 0) + (delta > 0 ? 5 : -5)
    studio.transaction(({ set }) => {
      set(layer.theatreObj!.props.rotation, newRotation)
    })
  } else {
    // Scale
    const newScale = Math.max(0.1, Math.min(5, (props.scale ?? 1) + delta))
    studio.transaction(({ set }) => {
      set(layer.theatreObj!.props.scale, newScale)
    })
  }
}

function saveProject() {
  if (!theatreProject) return
  
  // Get Theatre state
  const theatreState = studio.createContentOfSaveFile(theatreProject.address.projectId)
  
  // Build our save data
  const saveData = {
    project: { ...project },
    layers: layers.value.map(l => ({
      id: l.id,
      name: l.name,
      type: l.type,
      imageData: l.imageData
    })),
    theatreState
  }
  
  // Save to localStorage
  const key = `ae_animation_${props.node?.id || 'default'}`
  localStorage.setItem(key, JSON.stringify(saveData))
  
  console.log('[Theatre] Project saved')
}

function loadFromNode() {
  const key = `ae_animation_${props.node?.id || 'default'}`
  const saved = localStorage.getItem(key)
  if (!saved) return
  
  try {
    const data = JSON.parse(saved)
    
    // Restore project settings
    if (data.project) {
      Object.assign(project, data.project)
    }
    
    // Restore layers
    if (data.layers) {
      data.layers.forEach((l: any) => {
        const img = new Image()
        img.onload = () => {
          const layer: Layer = {
            id: l.id,
            name: l.name,
            type: l.type,
            image: img,
            imageData: l.imageData
          }
          createTheatreObject(layer)
          layers.value.push(layer)
        }
        if (l.imageData) img.src = l.imageData
      })
    }
    
    console.log('[Theatre] Loaded saved data')
  } catch (e) {
    console.error('[Theatre] Failed to load:', e)
  }
}
</script>

<style scoped>
.theatre-root {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  background: #1a1a1a;
  color: #ccc;
  font: 12px -apple-system, sans-serif;
}

.canvas-container {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #111;
  position: relative;
  overflow: hidden;
}

canvas {
  max-width: calc(100% - 20px);
  max-height: calc(100% - 20px);
  box-shadow: 0 4px 20px rgba(0,0,0,0.5);
}

.canvas-info {
  position: absolute;
  bottom: 8px;
  left: 50%;
  transform: translateX(-50%);
  background: rgba(0,0,0,0.7);
  padding: 4px 12px;
  border-radius: 4px;
  font-size: 11px;
  color: #888;
}

.controls {
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
  padding: 0 12px;
  background: #252525;
  border-top: 1px solid #333;
}

.controls button {
  padding: 4px 10px;
  background: #333;
  border: 1px solid #444;
  border-radius: 3px;
  color: #aaa;
  cursor: pointer;
  font-size: 11px;
}

.controls button:hover {
  background: #444;
  color: #fff;
}

.sep {
  width: 1px;
  height: 20px;
  background: #444;
  margin: 0 4px;
}
</style>
