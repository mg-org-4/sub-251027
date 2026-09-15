const { app } = window.comfyAPI.app;

const HIDDEN_LOADER_WIDGETS = new Set([
  "backend",
  "dtype",
  "quantization",
  "local_files_only",
  "unload_policy",
  "max_memory_gb",
]);

function widget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

function migrateValues(node) {
  const temp = widget(node, "temperature");
  if (!temp) return;

  const tempNumber = Number(temp.value);
  if (!Number.isFinite(tempNumber)) {
    temp.value = 0.45;
  }
}

function positiveNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : null;
}

function hideWidgets(node, names) {
  const widthBeforeHide = positiveNumber(node.size?.[0]);
  for (const w of node.widgets || []) {
    if (!names.has(w.name) || w.__dgHidden) continue;
    w.__dgHidden = true;
    w.hidden = true;
    w.computeSize = () => [0, -4];
  }
  const computedSize = node.computeSize?.() || node.size;
  const width = widthBeforeHide || positiveNumber(computedSize?.[0]);
  const height = positiveNumber(computedSize?.[1]) || positiveNumber(node.size?.[1]);
  if (width && height) node.setSize?.([width, height]);
}

function simplifyLoader(node) {
  migrateValues(node);
  hideWidgets(node, HIDDEN_LOADER_WIDGETS);
}

app.registerExtension({
  name: "DiffusionGemmaPromptBuilder.ModelLoaderUI",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "DiffusionGemmaModelLoader") return;

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      originalOnNodeCreated?.apply(this, args);
      simplifyLoader(this);
    };

    const originalOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (...args) {
      originalOnConfigure?.apply(this, args);
      simplifyLoader(this);
    };
  },
});
