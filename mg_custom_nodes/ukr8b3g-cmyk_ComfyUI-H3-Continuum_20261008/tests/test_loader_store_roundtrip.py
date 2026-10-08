"""Regressions for Core's prototype mode accessor and stable widgets view.

This models the persistence boundary; it is not a real-browser acceptance test.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("store_backed", [False, True], ids=["legacy", "store"])
@pytest.mark.parametrize("kind", ["image", "audio", "video"])
def test_loader_state_survives_draw_save_and_workflow_roundtrip(tmp_path, kind, store_backed):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for frontend regression tests")
    source = (ROOT / "web/easy_bypass_toggle.js").read_text(encoding="utf-8").replace("export ", "")
    script = source + "\n" + r'''
const storeBacked = STORE_BACKED;
const kind = KIND;
const nodeClass = {image: "H3EasyLoadImage", audio: "H3EasyLoadAudio", video: "H3ContinuumLoadVideo"}[kind];
const enableName = {image: "Enable Image", audio: "Enable Audio", video: "enable_video"}[kind];
const fileName = {image: "image", audio: "audio", video: "file"}[kind];
const selectedFile = {image: "selected-b.png", audio: "selected-b.wav", video: "selected-b.mp4"}[kind];
function plainWidget(name, value) { return {name, value, options: {}, serialize: true}; }
class StoreNode {
    constructor() {
        this.comfyClass = nodeClass;
        this._state = {mode: 0};
        this.drawCalls = 0;
        const widgets = [];
        // Like Core widgetsView: assignment mutates the same array/view.
        Object.defineProperty(this, "widgets", {
            configurable: true,
            get() { return widgets; },
            set(value) { widgets.splice(0, widgets.length, ...value); },
        });
        if (!storeBacked) {
            Object.defineProperty(this, "mode", {value: 0, writable: true, configurable: true});
            Object.defineProperty(this, "widgets", {value: [], writable: true, configurable: true});
        }
        if (kind === "video") this.widgets.push(plainWidget(enableName, true));
        this.widgets.push(plainWidget(fileName, "default-file"));
        if (kind === "video") this.widgets.push(plainWidget("force_rate", 30));
    }
    get mode() { return this._state.mode; }
    set mode(value) { this._state.mode = value; }
    setDirtyCanvas() {}
    addWidget(type, name, value, callback, options) {
        const w = {type, name, value, callback, options}; this.widgets.push(w); return w;
    }
    drawWidgets() { this.drawCalls++; }
    serializeFromStoreState(state) {
        const ws = this.widgets.filter(w => w.serialize !== false);
        return {mode: state.mode, widgets_values: ws.map(w => w.value),
            widgets_values_named: Object.fromEntries(ws.map(w => [w.name, w.value]))};
    }
    serialize() { return this.serializeFromStoreState(storeBacked ? this._state : {mode:this.mode}); }
    configure(info) {
        this.mode = info.mode;
        for (const w of this.widgets) {
            if (w.serialize !== false && Object.hasOwn(info.widgets_values_named, w.name))
                w.value = info.widgets_values_named[w.name];
        }
    }
}
function configure(n) {
    if (kind === "video") configureExistingEasyBypassWidgetNode(n, {nodeClass, widgetNames:[enableName]});
    else configureEasyBypassToggleNode(n, {nodeClass, widgetName:enableName, tooltip:"test"});
}
function saveGraph(n) {
    // Core 1.53 may bypass node.serialize() and use the store directly.
    return storeBacked ? n.serializeFromStoreState(n._state) : n.serialize();
}
const a = new StoreNode(); configure(a); configure(a);
a.widgets.find(w => w.name === fileName).value = selectedFile;
findEasyBypassWidget(a).callback(false);
const beforeDraw = a.widgets.map(w => w.name);
for (let i=0; i<4; i++) a.drawWidgets({}, {editorAlpha:0.2});
const afterDraw = a.widgets.map(w => w.name);
const saved = saveGraph(a);
const b = new StoreNode(); configure(b); b.configure(saved); configure(b);
const reopened = saveGraph(b);
b.mode = 0;
const externalOn = findEasyBypassWidget(b).value;
b.mode = 4;
const externalOff = findEasyBypassWidget(b).value;
console.log(JSON.stringify({beforeDraw, afterDraw, liveMode:a.mode, storeMode:a._state.mode,
    saved, reopened, reopenedEnable:findEasyBypassWidget(b).value, externalOn, externalOff,
    selectedFile, fileName, kind, drawCalls:a.drawCalls}));
'''
    script = script.replace("STORE_BACKED", str(store_backed).lower()).replace("KIND", json.dumps(kind))
    file = tmp_path / "loader-store.js"
    file.write_text(script, encoding="utf-8")
    result = subprocess.run([node, str(file)], capture_output=True, text=True, check=True)
    observed = json.loads(result.stdout)
    assert observed["beforeDraw"] == observed["afterDraw"], observed
    assert observed["liveMode"] == observed["saved"]["mode"] == 4, observed
    if store_backed:
        assert observed["storeMode"] == 4, observed
    assert observed["saved"]["widgets_values_named"][observed["fileName"]] == observed["selectedFile"]
    if kind == "video":
        assert observed["saved"]["widgets_values_named"]["force_rate"] == 30
        assert observed["saved"]["widgets_values_named"]["enable_video"] is False
    else:
        assert not any(name.startswith("Enable") for name in observed["saved"]["widgets_values_named"])
    assert observed["reopened"] == observed["saved"]
    assert observed["reopenedEnable"] is False
    assert observed["externalOn"] is True
    assert observed["externalOff"] is False
