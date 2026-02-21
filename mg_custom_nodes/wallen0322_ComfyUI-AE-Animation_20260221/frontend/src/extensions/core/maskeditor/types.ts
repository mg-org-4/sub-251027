export enum BrushShape {
  Arc = 'arc',
  Rect = 'rect'
}

export type ImageLayer = 'mask' | 'rgb'

export interface Brush {
  type: BrushShape
  size: number
  opacity: number
  hardness: number
  stepSize: number
}

export interface Point {
  x: number
  y: number
}

export interface Offset {
  x: number
  y: number
}

export enum Tools {
  MaskPen = 'mask-pen',
  Eraser = 'eraser',
  PaintPen = 'paint-pen',
  MaskBucket = 'mask-bucket',
  MaskColorFill = 'mask-color-fill'
}

export const allTools: Tools[] = [
  Tools.MaskPen,
  Tools.Eraser,
  Tools.PaintPen,
  Tools.MaskBucket,
  Tools.MaskColorFill
]

export enum MaskBlendMode {
  Black = 'black',
  White = 'white',
  Negative = 'negative'
}

export enum ColorComparisonMethod {
  Simple = 'simple',
  Lab = 'lab'
}

export enum CompositionOperation {
  SourceOver = 'source-over',
  DestinationOut = 'destination-out'
}

export interface ToolInternalSettings {
  newActiveLayerOnSet?: ImageLayer
  cursor?: string
}
