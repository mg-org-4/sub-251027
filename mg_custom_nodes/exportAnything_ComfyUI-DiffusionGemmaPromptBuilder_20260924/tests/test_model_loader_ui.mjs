import assert from "node:assert/strict";

let registeredExtension = null;
globalThis.window = {
  comfyAPI: {
    app: {
      app: {
        registerExtension(extension) {
          registeredExtension = extension;
        },
      },
    },
  },
};

await import(`../web/js/model_loader.js?test=${Date.now()}`);
assert.ok(registeredExtension, "the model-loader frontend extension should register");

class UnrelatedNode {}
await registeredExtension.beforeRegisterNodeDef(UnrelatedNode, { name: "OtherNode" });
assert.equal(
  UnrelatedNode.prototype.onNodeCreated,
  undefined,
  "unrelated node types should not be patched",
);

let createdCalls = 0;
let configuredCalls = 0;

class FakeLoaderNode {}
FakeLoaderNode.prototype.onNodeCreated = function () {
  createdCalls += 1;
};
FakeLoaderNode.prototype.onConfigure = function () {
  configuredCalls += 1;
};

await registeredExtension.beforeRegisterNodeDef(FakeLoaderNode, {
  name: "DiffusionGemmaModelLoader",
});

const hiddenWidgetNames = new Set([
  "backend",
  "dtype",
  "quantization",
  "local_files_only",
  "unload_policy",
  "max_memory_gb",
]);
const node = new FakeLoaderNode();
node.widgets = [
  { name: "model_path", value: "models/diffusiongemma" },
  { name: "temperature", value: "invalid" },
  ...[...hiddenWidgetNames].map((name) => ({ name, value: "advanced" })),
];
node.size = [520, 640];
node.computeSize = () => [360, 240];
node.setSize = (size) => {
  node.size = [...size];
};

node.onNodeCreated();
assert.equal(createdCalls, 1, "the original creation hook should still run");
assert.equal(
  node.widgets.find((entry) => entry.name === "temperature").value,
  0.45,
  "invalid saved temperatures should migrate to the stable default",
);
assert.deepEqual(node.size, [520, 240], "hiding widgets should preserve the user's width");

for (const entry of node.widgets) {
  if (hiddenWidgetNames.has(entry.name)) {
    assert.equal(entry.hidden, true, `${entry.name} should be hidden`);
    assert.equal(entry.__dgHidden, true, `${entry.name} should only be hidden once`);
    assert.deepEqual(entry.computeSize(), [0, -4]);
  } else {
    assert.notEqual(entry.hidden, true, `${entry.name} should remain visible`);
  }
}

node.widgets.find((entry) => entry.name === "temperature").value = 0.7;
node.size = [610, 700];
node.computeSize = () => [380, 260];
node.onConfigure({});
assert.equal(configuredCalls, 1, "the original configuration hook should still run");
assert.equal(node.widgets.find((entry) => entry.name === "temperature").value, 0.7);
assert.deepEqual(node.size, [610, 260], "configuration should keep the saved width");

console.log("DiffusionGemma model-loader UI checks passed");
