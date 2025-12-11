<template>
  <div class="project-settings">
    <div class="section-title">📐 项目</div>
    
    <div class="setting-row">
      <label>宽度</label>
      <input type="number" :value="store.project.width" @input="updateWidth" min="64" max="8192" />
    </div>
    <div class="setting-row">
      <label>高度</label>
      <input type="number" :value="store.project.height" @input="updateHeight" min="64" max="8192" />
    </div>
    <div class="setting-row">
      <label>帧率</label>
      <input type="number" :value="store.project.fps" @input="updateFps" min="1" max="120" />
    </div>
    <div class="setting-row">
      <label>帧数</label>
      <input type="number" :value="store.project.total_frames" @input="updateFrames" min="1" max="9999" />
    </div>

    <!-- 渲染模式 -->
    <div class="section-title">🖥️ 预览渲染</div>
    <div class="setting-row">
      <label>GPU加速</label>
      <input type="checkbox" :checked="!gpuDisabled" @change="toggleGPU" />
    </div>
    <div class="setting-note" v-if="gpuDisabled">
      <small style="color: #f90; font-size: 10px;">
        ⚠️ 已禁用GPU，使用Canvas 2D渲染
      </small>
    </div>

    <!-- 3D 摄像机 -->
    <div class="section-title">🎥 3D 摄像机</div>
    <div class="setting-row">
      <label>启用</label>
      <input type="checkbox" :checked="store.project.cam_enable" @change="updateCamEnable" />
    </div>
    
    <template v-if="store.project.cam_enable">
      <div class="subsection-title">位置</div>
      <div class="setting-row">
        <label>X</label>
        <input type="number" :value="store.project.cam_pos_x" @input="updateCamPosX" step="10" />
      </div>
      <div class="setting-row">
        <label>Y</label>
        <input type="number" :value="store.project.cam_pos_y" @input="updateCamPosY" step="10" />
      </div>
      <div class="setting-row">
        <label>Z (距离)</label>
        <input type="number" :value="store.project.cam_pos_z" @input="updateCamPosZ" step="50" />
      </div>

      <div class="subsection-title">旋转</div>
      <div class="setting-row">
        <label>Yaw (Y轴)</label>
        <input type="number" :value="store.project.cam_yaw" @input="updateCamYaw" step="5" />
      </div>
      <div class="setting-row">
        <label>Pitch (X轴)</label>
        <input type="number" :value="store.project.cam_pitch" @input="updateCamPitch" step="5" />
      </div>
      <div class="setting-row">
        <label>Roll (Z轴)</label>
        <input type="number" :value="store.project.cam_roll" @input="updateCamRoll" step="5" />
      </div>

      <div class="subsection-title">投影</div>
      <div class="setting-row">
        <label>FOV</label>
        <input type="number" :value="store.project.cam_fov" @input="updateCamFov" min="1" max="179" step="5" />
      </div>

      <div class="subsection-title">预览模式</div>
      <div class="setting-row">
        <label>模式</label>
        <select :value="store.project.preview_mode || '2d'" @change="updatePreviewMode" style="width: 80px; padding: 4px; background: #1a1a1a; border: 1px solid #444; border-radius: 3px; color: #fff; font-size: 11px;">
          <option value="2d">2D 近似</option>
          <option value="3d-css">3D CSS (实验性)</option>
        </select>
      </div>
      <div class="setting-note">
        <small style="color: #666; font-size: 10px; line-height: 1.4;">
          ⚠️ 预览仅为近似效果<br/>
          最终渲染使用完整3D透视
        </small>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useTimelineStore } from '@/stores/timelineStore'

const store = useTimelineStore()

// GPU开关状态
const gpuDisabled = ref(false)

onMounted(() => {
  gpuDisabled.value = localStorage.getItem('timeline_disable_gpu') === 'true'
})

function toggleGPU(e: Event) {
  const enabled = (e.target as HTMLInputElement).checked
  if (enabled) {
    localStorage.removeItem('timeline_disable_gpu')
    gpuDisabled.value = false
  } else {
    localStorage.setItem('timeline_disable_gpu', 'true')
    gpuDisabled.value = true
  }
  // 提示用户需要刷新页面
  alert('GPU设置已更改，请刷新页面以应用更改。')
}

function updateWidth(e: Event) {
  store.setProject({ width: parseInt((e.target as HTMLInputElement).value) || 1280 })
}
function updateHeight(e: Event) {
  store.setProject({ height: parseInt((e.target as HTMLInputElement).value) || 720 })
}
function updateFps(e: Event) {
  store.setProject({ fps: parseInt((e.target as HTMLInputElement).value) || 30 })
}
function updateFrames(e: Event) {
  store.setProject({ total_frames: parseInt((e.target as HTMLInputElement).value) || 150 })
}
function updateCamEnable(e: Event) {
  store.setProject({ cam_enable: (e.target as HTMLInputElement).checked })
}
function updateCamPosX(e: Event) {
  store.setProject({ cam_pos_x: parseFloat((e.target as HTMLInputElement).value) || 0 })
}
function updateCamPosY(e: Event) {
  store.setProject({ cam_pos_y: parseFloat((e.target as HTMLInputElement).value) || 0 })
}
function updateCamPosZ(e: Event) {
  store.setProject({ cam_pos_z: parseFloat((e.target as HTMLInputElement).value) || 0 })
}
function updateCamYaw(e: Event) {
  store.setProject({ cam_yaw: parseFloat((e.target as HTMLInputElement).value) || 0 })
}
function updateCamPitch(e: Event) {
  store.setProject({ cam_pitch: parseFloat((e.target as HTMLInputElement).value) || 0 })
}
function updateCamRoll(e: Event) {
  store.setProject({ cam_roll: parseFloat((e.target as HTMLInputElement).value) || 0 })
}
function updateCamFov(e: Event) {
  store.setProject({ cam_fov: parseFloat((e.target as HTMLInputElement).value) || 90 })
}
function updatePreviewMode(e: Event) {
  const mode = (e.target as HTMLSelectElement).value as '2d' | '3d-css'
  store.setProject({ preview_mode: mode })
}
</script>

<style scoped>
.project-settings {
  padding: 12px;
}

.section-title {
  font-size: 11px;
  font-weight: 600;
  color: #888;
  margin-bottom: 12px;
  padding-bottom: 6px;
  border-bottom: 1px solid #333;
}

.setting-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}

.setting-row label {
  font-size: 11px;
  color: #888;
}

.setting-row input[type="number"] {
  width: 80px;
  padding: 4px 6px;
  background: #1a1a1a;
  border: 1px solid #444;
  border-radius: 3px;
  color: #fff;
  font-size: 11px;
  font-family: monospace;
  text-align: right;
}

.setting-row input[type="number"]:focus {
  outline: none;
  border-color: #3a7bc8;
}

.setting-row input[type="checkbox"] {
  width: 18px;
  height: 18px;
  padding: 0;
  margin: 0;
  background: transparent;
  border: none;
  border-radius: 0;
  color: inherit;
  font-family: inherit;
  text-align: left;
  cursor: pointer;
  accent-color: #3a7bc8;
  flex-shrink: 0;
}

.setting-row input[type="checkbox"]:focus {
  outline: 2px solid #3a7bc8;
  outline-offset: 2px;
}

.subsection-title {
  font-size: 10px;
  color: #666;
  margin: 8px 0 6px 0;
  padding-left: 8px;
}

.setting-note {
  margin: 8px 0;
  padding: 8px;
  background: rgba(255, 152, 0, 0.1);
  border-left: 2px solid #ff9800;
  border-radius: 3px;
}
</style>
