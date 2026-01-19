import * as d from 'typegpu/data'

/**
 * GPU Schema Definitions for Timeline Renderer
 * 
 * These schemas define the data structures used in GPU shaders
 * for rendering timeline layers with transformations, camera, and effects.
 */

// Global layer uniforms for transformations and rendering
export const LayerUniforms = d.struct({
  // Layer transform
  position: d.vec2f,
  scale: d.vec2f,
  rotation: d.f32,
  opacity: d.f32,
  
  // Screen dimensions
  screenSize: d.vec2f,
  
  // 3D transform
  rotationX: d.f32,
  rotationY: d.f32,
  rotationZ: d.f32,
  perspective: d.f32,
  
  // Camera
  cameraOffset: d.vec2f,
  cameraScale: d.f32,
  cameraYaw: d.f32,
  cameraPitch: d.f32,
  cameraRoll: d.f32,
  cameraFOV: d.f32,
  
  // Layer specific
  zDepth: d.f32,
  anchorOffset: d.vec2f,
  
  // Background mode: 0=fit, 1=fill, 2=stretch
  backgroundMode: d.u32
})

// Camera-specific uniforms for panorama and 3D rendering
export const CameraUniforms = d.struct({
  position: d.vec3f,
  offset: d.vec2f,
  yaw: d.f32,
  pitch: d.f32,
  roll: d.f32,
  fov: d.f32,
  scale: d.f32,
  enabled: d.u32 // 0=disabled, 1=enabled
})

// Panorama-specific uniforms for equirectangular projection
export const PanoramaUniforms = d.struct({
  yaw: d.f32,
  pitch: d.f32,
  roll: d.f32,
  fov: d.f32,
  aspectRatio: d.f32,
  screenSize: d.vec2f
})

// Vertex output structure shared across shaders
export const VertexOutput = d.struct({
  position: d.builtin.position,
  uv: d.location(0, d.vec2f),
  opacity: d.location(1, d.f32)
})
