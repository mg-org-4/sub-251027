import assert from "node:assert/strict";
import fs from "node:fs";

const source = fs.readFileSync(new URL("../web/decode_cache_helper.js", import.meta.url), "utf8");
let extension;
globalThis.__h3TestApp = {
    registerExtension(value) { extension = value; },
    graph: { _nodes: [] },
    ui: { settings: { getSettingValue: () => "en" } },
};
const module = await import(`data:text/javascript;base64,${Buffer.from(source.replace(
    'import { app } from "../../scripts/app.js";', 'const app = globalThis.__h3TestApp;',
)).toString("base64")}`);

function createNode(type = "H3DecodeCacheHelper") {
    return {
        comfyClass: type,
        widgets: ["cache_mode", "ram_budget_mb", "disk_budget_gb", "reset_token"].map((name, i) => ({
            name, value: ["Auto", 256, 8, 0][i], type: i === 0 ? "combo" : "number", options: {},
        })),
        addWidget(type, name, value, callback, options) {
            const widget = { type, name, value, callback, options };
            this.widgets.push(widget);
            return widget;
        },
        serialize() { return { widgets_values: this.widgets.map((w) => w.value) }; },
        configure(data) { data.widgets_values.forEach((value, i) => { if (this.widgets[i]) this.widgets[i].value = value; }); },
        setDirtyCanvas() {},
    };
}

const node = createNode();
extension.nodeCreated(node);
const token = node.widgets[3];
const button = node.widgets[4];
assert.equal(token.value, 0);
assert.equal(token.hidden, true);
assert.equal(token.options.hidden, true);
assert.equal(token.options.serialize, undefined);
assert.equal(button.type, "button");
assert.equal(button.name, "Clear cache");
assert.equal(button.options.serialize, false);
assert.equal(button.serialize, false);
assert.match(button.tooltip, /next Queue/);
assert.deepEqual(node.serialize().widgets_values, ["Auto", 256, 8, 0]);
extension.loadedGraphNode(node);
globalThis.__h3TestApp.graph._nodes = [node];
extension.afterConfigureGraph();
assert.equal(node.widgets.length, 5);
assert.equal(token.value, 0);
button.callback();
assert.equal(token.value, 1);
assert.deepEqual(node.serialize().widgets_values, ["Auto", 256, 8, 1]);

// Save/load and the clone path keep the same four backend values.
const clone = createNode();
extension.nodeCreated(clone);
clone.configure(node.serialize());
assert.deepEqual(clone.serialize(), node.serialize());
assert.equal(clone.widgets.length, 5);
clone.widgets[4].callback();
assert.equal(clone.widgets[3].value, 2);
assert.equal(token.value, 1);

// Both official V3.8X2 distribution names preserve the same helper settings.
for (const filename of ["MiniMax_H3_Continuum_V38X2.json", "MiniMax_H3_Continuum_V38X2+Decode_Cache_Helper.json"]) {
    const workflow = JSON.parse(fs.readFileSync(new URL(`../examples/workflows/${filename}`, import.meta.url), "utf8"));
    const saved = workflow.nodes.find((n) => n.type === "H3DecodeCacheHelper");
    const loaded = createNode();
    extension.nodeCreated(loaded);
    loaded.configure(saved);
    assert.deepEqual(loaded.serialize().widgets_values, saved.widgets_values);
    assert.equal(loaded.widgets[3].hidden, true);
}
node.configure({ widgets_values: ["RAM", 128, 16, 2147483647] });
button.callback();
assert.deepEqual(node.serialize().widgets_values, ["RAM", 128, 16, 0]);
const other = createNode("H3ContinuumSamplerV38");
module.configureDecodeCacheHelper(other);
assert.equal(other.widgets.length, 4);
assert.equal(other.widgets[3].hidden, undefined);
// Without JS the native reset_token input remains untouched.
assert.equal(createNode().widgets[3].type, "number");

// ComfyUI's explicit locale setting takes precedence over browser language.
globalThis.__h3TestApp.ui = { settings: { getSettingValue: () => "ja" } };
const japaneseNode = createNode();
extension.nodeCreated(japaneseNode);
assert.equal(japaneseNode.widgets[4].name, "キャッシュをクリア");
assert.match(japaneseNode.widgets[4].tooltip, /次回Queue/);
console.log("Decode Cache Helper frontend lifecycle / serialization / clone / standard workflows: PASS");
