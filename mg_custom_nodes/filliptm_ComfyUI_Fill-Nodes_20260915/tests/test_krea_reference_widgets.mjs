import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

const source = await readFile(new URL("../web/nodes/conditioning/FL_KreaReference.js", import.meta.url), "utf8");
let extension;
vm.runInNewContext(source.replace(/^import .*;$/gm, ""), {
  app: { registerExtension: value => { extension = value; } },
});

test("average slider follows the selected blend mode on change and load", () => {
  const mode = { name: "blend_mode", value: "add" };
  const amount = { name: "average_amount", value: .6 };
  const node = {
    constructor: { comfyClass: "FL_KreaReferenceGuider" },
    widgets: [mode, amount], setDirtyCanvas() {},
  };
  extension.nodeCreated(node);
  assert.equal(amount.disabled, true);
  mode.value = "average";
  mode.callback();
  assert.equal(amount.disabled, false);
  assert.equal(amount.value, .6);
  mode.value = "add";
  node.onConfigure();
  assert.equal(amount.disabled, true);
  node.inputs = [{ name: "blend_mode", link: 12 }];
  node.onConnectionsChange();
  assert.equal(amount.disabled, false);
});

test("removing instruction preserves saved reference settings and connected inputs", () => {
  for (const objectLinks of [false, true]) {
    const graph = {
      nodes: [
        { id: 1, outputs: [{ links: [10, 11] }] },
        { id: 2, type: "FL_KreaReference",
          widgets_values: [true, "palette", "Old instruction", .35, 1024, .1, .9, .2, "full"],
          widgets_values_named: { instruction: "Old instruction", weight: .35 },
          inputs: [{ name: "image", link: null }, { name: "instruction", link: 10 }, { name: "weight", link: 11 }] },
      ],
      links: [[10, 1, 0, 2, 1, "STRING"], [11, 1, 0, 2, 2, "FLOAT"]],
    };
    if (objectLinks) graph.links = graph.links.map(([id, origin_id, origin_slot, target_id, target_slot, type]) =>
      ({ id, origin_id, origin_slot, target_id, target_slot, type }));
    extension.beforeConfigureGraph(graph);
    assert.deepEqual(graph.nodes[1].widgets_values, [true, "palette", .35, 1024, .1, .9, .2, "full"]);
    assert.deepEqual(graph.nodes[1].widgets_values_named, { weight: .35 });
    assert.deepEqual(graph.nodes[0].outputs[0].links, [11]);
    assert.equal(graph.links.length, 1);
    assert.equal(objectLinks ? graph.links[0].target_slot : graph.links[0][4], 1);
    assert.deepEqual(graph.nodes[1].inputs.map(input => input.name), ["image", "weight"]);
    const migrated = JSON.stringify(graph);
    extension.beforeConfigureGraph(graph);
    assert.equal(JSON.stringify(graph), migrated);
  }
});
