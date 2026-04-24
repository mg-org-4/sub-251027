const i = [];
let l = !1;
function m() {
  l && (i.sort((e, a) => (a.priority ?? 0) - (e.priority ?? 0)), l = !1);
}
function A(e) {
  if (!(e != null && e.name)) {
    console.warn("[NodeStream] Cannot register adapter without a name");
    return;
  }
  const a = i.findIndex((t) => t.name === e.name);
  a >= 0 && i.splice(a, 1), i.push(e), l = !0, console.debug(
    `[NodeStream] Adapter registered: ${e.name} (priority ${e.priority ?? 0})`
  );
}
function B() {
  return m(), i.map((e) => ({
    name: e.name,
    priority: e.priority ?? 0,
    description: e.description ?? ""
  }));
}
function s(e) {
  return {
    priority: 0,
    description: "",
    canHandle: () => !1,
    extractMedia: () => null,
    ...e
  };
}
const d = /* @__PURE__ */ new Set([".png", ".jpg", ".jpeg", ".webp", ".avif", ".gif", ".bmp", ".tiff"]);
function g(e) {
  if (!e) return !1;
  const a = String(e).lastIndexOf(".");
  return a >= 0 && d.has(String(e).slice(a).toLowerCase());
}
s({
  name: "default-image",
  priority: 0,
  description: "Standard image output (images: [{filename, subfolder, type}])",
  canHandle(e, a) {
    var o;
    const t = a == null ? void 0 : a.images;
    return Array.isArray(t) && t.length > 0 && !!((o = t[0]) != null && o.filename);
  },
  extractMedia(e, a, t) {
    const o = a == null ? void 0 : a.images;
    if (!Array.isArray(o) || !o.length) return null;
    const n = [];
    for (const r of o)
      r != null && r.filename && n.push({
        filename: r.filename,
        subfolder: r.subfolder || "",
        type: r.type || "output",
        kind: g(r.filename) ? "image" : void 0,
        _nodeId: t,
        _classType: e
      });
    return n.length ? n : null;
  }
});
const c = /* @__PURE__ */ new Set([".mp4", ".webm", ".mov", ".avi", ".mkv"]);
function I(e) {
  if (!e) return !1;
  const a = String(e).lastIndexOf(".");
  return a >= 0 && c.has(String(e).slice(a).toLowerCase());
}
function y(e) {
  var o, n;
  const a = e == null ? void 0 : e.gifs;
  if (Array.isArray(a) && a.length && ((o = a[0]) != null && o.filename)) return a;
  const t = e == null ? void 0 : e.videos;
  return Array.isArray(t) && t.length && ((n = t[0]) != null && n.filename) ? t : null;
}
s({
  name: "video-output",
  priority: 10,
  description: "Video output (gifs/videos: [{filename, subfolder, type}])",
  canHandle(e, a) {
    return !!y(a);
  },
  extractMedia(e, a, t) {
    const o = y(a);
    if (!o) return null;
    const n = [];
    for (const r of o)
      r != null && r.filename && n.push({
        filename: r.filename,
        subfolder: r.subfolder || "",
        type: r.type || "output",
        kind: I(r.filename) ? "video" : "image",
        _nodeId: t,
        _classType: e
      });
    return n.length ? n : null;
  }
});
const C = /* @__PURE__ */ new Set([
  "SaveImage",
  "PreviewImage",
  "SaveAnimatedWEBP",
  "SaveAnimatedPNG",
  "SaveVideo",
  "SaveWEBM",
  "SaveGif",
  "SaveImageWebSocket"
]), L = /* @__PURE__ */ new Set([
  "SaveImageWithAlpha",
  "SaveImageKJ",
  "PreviewAnimation",
  "FastPreview",
  "ImageAndMaskPreview",
  "PreviewImageOrMask"
]), f = /* @__PURE__ */ new Set(["ImagePreviewFromLatent"]), u = /* @__PURE__ */ new Set(["LayerUtility: ImageTaggerSave"]), S = /* @__PURE__ */ new Set([
  // Color correction
  "LayerColor: ColorCorrectBrightnessAndContrast",
  "LayerColor: ColorCorrectHSV",
  "LayerColor: ColorCorrectLAB",
  "LayerColor: ColorCorrectRGB",
  "LayerColor: ColorCorrectYUV",
  "LayerColor: ColorCorrectAutoAdjust",
  "LayerColor: ColorCorrectAutoAdjustV2",
  "LayerColor: ColorCorrectAutoBrightness",
  "LayerColor: ColorCorrectColorBalance",
  "LayerColor: ColorCorrectColorTemperature",
  "LayerColor: ColorCorrectExposure",
  "LayerColor: ColorCorrectGamma",
  "LayerColor: ColorCorrectLevels",
  "LayerColor: ColorCorrectShadowAndHighlight",
  "LayerColor: ColorCorrectLUTapply",
  // Color mapping / gradients
  "LayerColor: GradientMap",
  "LayerColor: GradientOverlay",
  "LayerColor: GradientOverlayV2",
  "LayerColor: ColorOverlay",
  "LayerColor: ColorOverlayV2",
  "LayerColor: ColorNegative",
  "LayerColor: ColorAdapter",
  // Filters & effects
  "LayerFilter: GaussianBlur",
  "LayerFilter: GaussianBlurV2",
  "LayerFilter: AddGrain",
  "LayerFilter: ChannelShake",
  "LayerFilter: FilmPost",
  "LayerFilter: FilmPostV2",
  "LayerFilter: Halftone",
  "LayerFilter: HDREffects",
  "LayerFilter: MotionBlur",
  "LayerFilter: DistortDisplace",
  "LayerFilter: LightLeak",
  // Layer style effects
  "LayerStyle: DropShadow",
  "LayerStyle: DropShadowV2",
  "LayerStyle: DropShadowV3",
  "LayerStyle: InnerShadow",
  "LayerStyle: InnerShadowV2",
  "LayerStyle: InnerGlow",
  "LayerStyle: InnerGlowV2",
  "LayerStyle: OuterGlow",
  // Blending & composition
  "LayerUtility: ImageBlend",
  "LayerUtility: ImageBlendV2",
  "LayerUtility: ImageBlendV3",
  "LayerUtility: ImageBlendAdvance",
  "LayerUtility: ImageBlendAdvanceV2",
  "LayerUtility: ImageBlendAdvanceV3",
  "LayerUtility: ImageCompositeHandleMask",
  "LayerUtility: ImageCombineAlpha",
  "LayerUtility: ImageRemoveAlpha",
  "LayerUtility: ImageOpacity",
  "LayerUtility: ImageShift",
  // Canvas & crop
  "LayerUtility: ExtendCanvas",
  "LayerUtility: ExtendCanvasV2",
  "LayerUtility: CropByMask",
  "LayerUtility: CropByMaskV2",
  "LayerUtility: CropByMaskV3",
  // Transform
  "LayerUtility: ImageScaleRestore",
  "LayerUtility: ImageScaleRestoreV2",
  "LayerUtility: ImageScaleByAspectRatio",
  "LayerUtility: ImageScaleByAspectRatioV2",
  "LayerUtility: LayerImageTransform",
  // Mask nodes
  "LayerMask: MaskByColor",
  "LayerMask: MaskGradient",
  "LayerMask: MaskInvert",
  "LayerMask: MaskStroke",
  "LayerMask: MaskGrain",
  "LayerMask: MaskGrow",
  "LayerMask: MaskMotionBlur",
  "LayerMask: MaskEdgeShrink",
  "LayerMask: MaskEdgeUltraDetail",
  "LayerMask: MaskEdgeUltraDetailV2",
  "LayerMask: MaskEdgeUltraDetailV3",
  "LayerMask: CreateGradientMask",
  // Channel
  "LayerUtility: ImageChannelSplit",
  "LayerUtility: ImageChannelMerge",
  "LayerUtility: ImageToMask"
]), M = /* @__PURE__ */ new Set([
  "ColorMatch",
  "ColorMatchV2",
  "ImageConcanate",
  "ImageConcatFromBatch",
  "ImageConcatMulti",
  "ImageGridComposite2x2",
  "ImageGridComposite3x3",
  "ImageBatchTestPattern",
  "ImageGrabPIL",
  "Screencap_mss",
  "AddLabel",
  "ImagePass",
  "ImageResizeKJ",
  "ImageResizeKJv2",
  "ImageUpscaleWithModelBatched",
  "CrossFadeImages",
  "CrossFadeImagesMulti",
  "TransitionImagesMulti",
  "TransitionImagesInBatch",
  "ImageBatchJoinWithTransition",
  "ImageBatchRepeatInterleaving",
  "ImageBatchExtendWithOverlap",
  "ImageBatchFilter",
  "ImageBatchMulti",
  "ImageAddMulti",
  "ImageNormalize_Neg1_To_1",
  "RemapImageRange",
  "ReverseImageBatch",
  "ShuffleImageBatch",
  "ReplaceImagesInBatch",
  "GetImagesFromBatchIndexed",
  "GetImageRangeFromBatch",
  "InsertImagesToBatchIndexed",
  "ImageCropByMask",
  "ImageCropByMaskAndResize",
  "ImageCropByMaskBatch",
  "ImageUncropByMask",
  "ImagePadForOutpaintMasked",
  "ImagePadForOutpaintTargetSize",
  "ImagePadKJ",
  "ImagePrepForICLora",
  "PadImageBatchInterleaved",
  "LoadAndResizeImage",
  "SplitImageChannels",
  "MergeImageChannels",
  "DrawMaskOnImage",
  "ImageNoiseAugmentation",
  // Mask nodes
  "ColorToMask",
  "BlockifyMask",
  "CreateGradientMask",
  "CreateTextMask",
  "CreateAudioMask",
  "CreateFadeMask",
  "CreateFadeMaskAdvanced",
  "CreateFluidMask",
  "CreateShapeMask",
  "CreateVoronoiMask",
  "CreateMagicMask",
  "GrowMaskWithBlur",
  "OffsetMask",
  "RemapMaskRange",
  "ResizeMask",
  "RoundMask"
]), h = /* @__PURE__ */ new Set([
  "ImageResize+",
  "ImageCrop+",
  "ImageFlip+",
  "ImageDesaturate+",
  "ImagePosterize+",
  "ImageEnhanceDifference+",
  "ImageExpandBatch+",
  "ImageFromBatch+",
  "ImageRemoveBackground+",
  "MaskBlur+",
  "MaskPreview+",
  "TransitionMask+"
]), p = /* @__PURE__ */ new Set([
  ...C,
  ...L,
  ...f,
  ...u
]);
[
  ...S,
  ...M,
  ...h
];
s({
  name: "known-nodes",
  priority: 5,
  description: "Well-known output nodes from core ComfyUI, KJNodes, Essentials, LayerStyle",
  canHandle(e, a) {
    var o;
    if (!p.has(e)) return !1;
    const t = a == null ? void 0 : a.images;
    return Array.isArray(t) && t.length > 0 && !!((o = t[0]) != null && o.filename);
  },
  extractMedia(e, a, t) {
    const o = a == null ? void 0 : a.images;
    if (!Array.isArray(o) || !o.length) return null;
    const n = [];
    for (const r of o)
      r != null && r.filename && n.push({
        filename: r.filename,
        subfolder: r.subfolder || "",
        type: r.type || "output",
        kind: "image",
        _nodeId: t,
        _classType: e,
        _source: "known-nodes"
      });
    return n.length ? n : null;
  }
});
let k = "selected";
function v(e, a) {
}
function U({ app: e, onOutput: a, onStatus: t } = {}) {
  {
    console.debug("[NodeStream] Disabled by feature flag");
    return;
  }
}
function w(e) {
}
function E() {
  return !1;
}
function T(e) {
}
function F() {
  return k;
}
function O(e) {
}
function _() {
  return null;
}
function G() {
  return null;
}
function P(e) {
  console.debug("[NodeStream] Controller torn down");
}
export {
  E as getNodeStreamActive,
  _ as getPinnedNodeId,
  G as getSelectedNodeId,
  F as getWatchMode,
  U as initNodeStream,
  B as listAdapters,
  v as onNodeOutputs,
  O as pinNode,
  A as registerAdapter,
  w as setNodeStreamActive,
  T as setWatchMode,
  P as teardownNodeStream
};
