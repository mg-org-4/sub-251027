import { app } from "../../../scripts/app.js";

const SINGLE_NODE_TYPE = "Krea2StyleTransfer";
const MULTI_STAGE_NODE_TYPE = "Krea2TwoStyleTransfer";

const SINGLE_CUSTOM_WIDGETS = new Set([
  "style_strength",
  "value_adain_strength",
  "ref_value_mix",
  "ref_k_strength",
  "rf_mode",
  "gamma",
  "beta",
  "high_scale_start",
  "high_scale_end",
  "low_scale_start",
  "low_scale_end",
  "adain_strength",
  "blocks",
]);

const MULTI_STAGE_CUSTOM_WIDGETS = new Set([
  "style_strength",
  "ref_k_1",
  "ref_k_2",
  "first_phase_ratio",
  "stage_focus",
  "ref_value_mix",
  "low_scale_end",
  "rf_mode",
  "gamma",
  "beta",
  "high_scale_start",
  "high_scale_end",
  "low_scale_start",
  "adain_strength",
  "blocks",
]);

const MULTI_STAGE_ALWAYS_HIDDEN_WIDGETS = new Set([
  "stage_blend",
  "token_rms_cap",
  "resolution_gain",
  "delta_clip",
  "late_release",
  "stage_shift",
  "stage_schedule",
  "value_adain_strength",
]);

function widgetName(widget) {
  return widget?.name || widget?.label || "";
}

function recompute(node) {
  try {
    if (typeof node.computeSize === "function") {
      const size = node.computeSize();
      if (Array.isArray(size)) {
        node.size = [Math.max(node.size?.[0] || 0, size[0]), size[1]];
      }
    }
  } catch (_) {}
  try {
    app.graph?.setDirtyCanvas(true, true);
  } catch (_) {
    try {
      node.setDirtyCanvas?.(true, true);
    } catch (_) {}
  }
}

function applyModeVisibility(node, customWidgets, alwaysHiddenWidgets = new Set()) {
  const mode = node.widgets?.find((w) => widgetName(w) === "mode");
  const showCustom = mode?.value === "custom";
  let changed = false;
  for (const widget of node.widgets || []) {
    const name = widgetName(widget);
    if (alwaysHiddenWidgets.has(name)) {
      if (widget.hidden !== true) {
        widget.hidden = true;
        changed = true;
      }
      continue;
    }
    if (!customWidgets.has(name)) {
      continue;
    }
    const nextHidden = !showCustom;
    if (widget.hidden !== nextHidden) {
      widget.hidden = nextHidden;
      changed = true;
    }
  }
  if (changed) {
    recompute(node);
  }
}

function installModeVisibility(nodeType, customWidgets, alwaysHiddenWidgets = new Set()) {
  const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    originalOnNodeCreated?.apply(this, arguments);

    const mode = this.widgets?.find((w) => widgetName(w) === "mode");
    if (mode) {
      const originalCallback = mode.callback;
      mode.callback = (...args) => {
        const result = originalCallback?.apply(mode, args);
        applyModeVisibility(this, customWidgets, alwaysHiddenWidgets);
        return result;
      };
    }
    applyModeVisibility(this, customWidgets, alwaysHiddenWidgets);
  };

  const originalConfigure = nodeType.prototype.configure;
  nodeType.prototype.configure = function () {
    const result = originalConfigure?.apply(this, arguments);
    requestAnimationFrame(() => applyModeVisibility(this, customWidgets, alwaysHiddenWidgets));
    return result;
  };
}

app.registerExtension({
  name: "Krea2StyleTransfer.ControlledModeUI",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name === SINGLE_NODE_TYPE) {
      installModeVisibility(nodeType, SINGLE_CUSTOM_WIDGETS);
    } else if (nodeData?.name === MULTI_STAGE_NODE_TYPE) {
      installModeVisibility(nodeType, MULTI_STAGE_CUSTOM_WIDGETS, MULTI_STAGE_ALWAYS_HIDDEN_WIDGETS);
    }
  },
});
