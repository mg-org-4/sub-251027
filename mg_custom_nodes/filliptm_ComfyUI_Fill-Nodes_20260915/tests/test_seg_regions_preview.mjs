import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

const source = await readFile(
  new URL("../web/nodes/ksamplers/FL_KsamplerSEG_Regions.js", import.meta.url),
  "utf8",
);

test("SEG Regions preview uses stable string node keys", () => {
  assert.match(source, /const nodeKey = \(value\) => String\(value\);/);
  assert.match(source, /INSTANCES\.set\(nodeKey\(node\.id\), inst\);/);
  assert.match(source, /INSTANCES\.get\(nodeKey\(detail\.node\)\)/);
  assert.doesNotMatch(source, /parseInt\(detail\.node/);
});

test("old workflows retain factors and new workflows expose pixel controls", () => {
  let extension;
  vm.runInNewContext(source.replace(/^import .*;$/gm, ""), {
    app: { registerExtension: value => { extension = value; } },
    api: { addEventListener() {} },
    document: { createElement: () => ({ style: {} }) },
    setTimeout() {},
  });
  const names = ["num_regions", "relaxation_iterations", "region_overlap_factor", "edge_softness", "context_padding_factor", "safe_zone_feather_px", "downscale_ratio", "seed", "control_after_generate", "show_preview", "preview_mode", "margin_mode", "overlap_width_px", "feather_width_px", "context_padding_px"];
  const node = {
    constructor: { comfyClass: "FL_KsamplerSEG_Regions" },
    widgets: names.map(name => ({ name, type: "number", options: {}, value: name === "margin_mode" ? "pixels" : 0 })),
    size: [380, 560], setSize() {}, setDirtyCanvas() {},
    addDOMWidget: () => ({}),
  };
  extension.nodeCreated(node);
  const widget = name => node.widgets.find(w => w.name === name);
  assert.equal(widget("region_overlap_factor").hidden, true);
  assert.notEqual(widget("overlap_width_px").hidden, true);
  node.onConfigure({ widgets_values: [4, 5, .15, .1, .2, 8, 8, 1, "fixed", true, "overlay", ""] });
  assert.equal(widget("margin_mode").value, "legacy");
  assert.notEqual(widget("region_overlap_factor").hidden, true);
  assert.equal(widget("overlap_width_px").hidden, true);
  widget("margin_mode").value = "pixels";
  widget("margin_mode").callback();
  assert.equal(widget("region_overlap_factor").hidden, true);
  assert.equal(widget("overlap_width_px").type, "number");
  widget("feather_width_px").value = 32;
  widget("overlap_width_px").value = 16;
  widget("overlap_width_px").callback();
  assert.equal(widget("feather_width_px").value, 8);
  assert.equal(widget("feather_width_px").options.max, 8);
  widget("overlap_width_px").value = 0;
  widget("overlap_width_px").callback();
  assert.equal(widget("feather_width_px").value, 0);
});
