# AE Animation Frontend

Vue 3 + TypeScript + Pinia + WebGPU 前端应用

## 开发

```bash
# 安装依赖
pnpm install

# 开发模式
pnpm dev

# 构建生产版本
pnpm build

# 类型检查
pnpm type-check
```

## 架构

### 应用入口
- `src/timeline-main.ts` - 时间轴应用入口
- `src/main.ts` - 遮罩编辑器应用入口

### 核心组件
- `src/TimelineApp.vue` - 主时间轴应用
- `src/MaskEditorApp.vue` - GPU 遮罩编辑器
- `src/components/timeline/` - 时间轴 UI 组件
- `src/components/maskeditor/` - 遮罩编辑器 UI 组件

### 状态管理 (Pinia)
- `src/stores/timelineStore.ts` - 时间轴状态
- `src/stores/maskEditorStore.ts` - 遮罩编辑器状态
- `src/stores/maskEditorDataStore.ts` - 遮罩数据管理

### GPU 加速
- `src/composables/maskeditor/gpu/` - WebGPU 渲染器
- `src/composables/maskeditor/useBrushDrawing.ts` - GPU 画笔

## 构建产物

构建后的文件位于 `../js/vue-dist/`:
- `timeline.js` - 时间轴应用
- `mask-editor.js` - GPU 遮罩编辑器
- `assets/` - CSS 和共享模块

这些文件会被 ComfyUI 通过 `/extensions/ComfyUI-AE-Animation/vue-dist/` 路径提供。
