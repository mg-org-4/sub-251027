import { app } from "../../../../scripts/app.js";

function removeReferenceInstruction(graph) {
  const removedSlots = new Map();
  const removedLinks = new Set();
  for (const node of graph.nodes ?? []) {
    if (node.type !== "FL_KreaReference") continue;
    if (node.widgets_values?.length === 9) node.widgets_values.splice(2, 1);
    if (node.widgets_values_named) delete node.widgets_values_named.instruction;
    const slot = node.inputs?.findIndex(input => input.name === "instruction") ?? -1;
    if (slot < 0) continue;
    const [input] = node.inputs.splice(slot, 1);
    removedSlots.set(String(node.id), slot);
    if (input.link != null) removedLinks.add(input.link);
  }
  if (removedSlots.size) {
    graph.links = (graph.links ?? []).filter(link => {
      const array = Array.isArray(link);
      const id = array ? link[0] : link.id;
      if (removedLinks.has(id)) return false;
      const slot = removedSlots.get(String(array ? link[3] : link.target_id));
      if (slot !== undefined) {
        if (array && link[4] > slot) link[4]--;
        else if (!array && link.target_slot > slot) link.target_slot--;
      }
      return true;
    });
    for (const node of graph.nodes ?? []) {
      for (const output of node.outputs ?? []) {
        if (output.links) output.links = output.links.filter(id => !removedLinks.has(id));
      }
    }
  }
  for (const subgraph of graph.definitions?.subgraphs ?? []) removeReferenceInstruction(subgraph);
}

app.registerExtension({
  name: "ComfyUI.FL_KreaReference",
  beforeConfigureGraph: removeReferenceInstruction,
  nodeCreated(node) {
    if (node.constructor?.comfyClass !== "FL_KreaReferenceGuider") return;
    const mode = node.widgets.find(widget => widget.name === "blend_mode");
    const amount = node.widgets.find(widget => widget.name === "average_amount");
    const update = () => {
      const connectedMode = node.inputs?.some(input => input.name === "blend_mode" && input.link != null);
      amount.disabled = mode.value !== "average" && !connectedMode;
      node.setDirtyCanvas(true);
    };
    const callback = mode.callback;
    mode.callback = function () {
      const result = callback?.apply(this, arguments);
      update();
      return result;
    };
    const onConfigure = node.onConfigure;
    node.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      update();
      return result;
    };
    const onConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function () {
      const result = onConnectionsChange?.apply(this, arguments);
      update();
      return result;
    };
    update();
  },
});
