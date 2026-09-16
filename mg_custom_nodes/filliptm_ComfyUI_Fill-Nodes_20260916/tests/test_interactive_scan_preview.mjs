import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const mappingSource=await readFile(new URL("../web/nodes/vfx/scan_audio_mapping.js",import.meta.url),"utf8");
const mappingURL=`data:text/javascript;base64,${Buffer.from(mappingSource).toString("base64")}`;
const previsSource=await readFile(new URL("../web/nodes/vfx/scan_previs.js",import.meta.url),"utf8");
const previsURL=`data:text/javascript;base64,${Buffer.from(previsSource).toString("base64")}`;
const controlsSource=(await readFile(new URL("../web/nodes/vfx/scan_controls.js",import.meta.url),"utf8")).replace('"./scan_previs.js"',JSON.stringify(previsURL));
const controlsURL=`data:text/javascript;base64,${Buffer.from(controlsSource).toString("base64")}`;
const source = (await readFile(new URL("../web/nodes/vfx/FL_InteractiveScanFX.js", import.meta.url), "utf8"))
    .replace('"./scan_previs.js"',JSON.stringify(previsURL))
    .replace('"./scan_controls.js"',JSON.stringify(controlsURL))
    .replace('"./scan_audio_mapping.js"',JSON.stringify(mappingURL))
    .replace('import { app } from "../../../../scripts/app.js";', 'const app = {registerExtension() {}};')
    .replace('import { api } from "../../../../scripts/api.js";', 'const api = {};');
const { previewFrame, frameSeekTime } = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);

test("preview meters use the displayed frame and clamp the end", () => {
    assert.equal(previewFrame(0,24,192),0);
    assert.equal(previewFrame(1.5,24,192),36);
    assert.equal(previewFrame(8,24,192),191);
    assert.equal(previewFrame(-1,24,192),0);
    assert.equal(previewFrame(0.999999999/24,24,192),1);
});

const {mappingRange,moveMappingRange}=await import(mappingURL);
const {demoPoint,EFFECT_GROUPS}=await import(previsURL);
const {createPrevis}=await import(previsURL);

test('demo stops scheduling when paused, offscreen, hidden, or disposed', () => {
    const original = Object.fromEntries(['document','IntersectionObserver','requestAnimationFrame','cancelAnimationFrame'].map(k=>[k,globalThis[k]]));
    let visible, visibility;
    const callbacks = new Map();let next = 0;
    const element = () => ({style:{}, children:[], append(...items){this.children.push(...items);}});
    globalThis.document = {hidden:false,createElement:element,
        addEventListener(name,fn){visibility=fn;},removeEventListener(){visibility=null;}};
    globalThis.IntersectionObserver = class {constructor(fn){visible=fn;}observe(){}disconnect(){visible=null;}};
    globalThis.requestAnimationFrame = fn => {callbacks.set(++next,fn);return next;};
    globalThis.cancelAnimationFrame = id => callbacks.delete(id);
    try {
        const demo=createPrevis(()=>({}));
        assert.equal(callbacks.size,0);
        visible([{isIntersecting:true}]);assert.equal(callbacks.size,1);
        demo.element.children[2].onclick();assert.equal(callbacks.size,0);
        demo.element.children[2].onclick();assert.equal(callbacks.size,1);
        visible([{isIntersecting:false}]);assert.equal(callbacks.size,0);
        visible([{isIntersecting:true}]);assert.equal(callbacks.size,1);
        document.hidden=true;visibility();assert.equal(callbacks.size,0);
        document.hidden=false;visibility();assert.equal(callbacks.size,1);
        demo.setActive(false);assert.equal(callbacks.size,0);
        demo.setActive(true);assert.equal(callbacks.size,1);
        demo.dispose();assert.equal(callbacks.size,0);assert.equal(visibility,null);
    } finally {for(const [key,value] of Object.entries(original)){if(value===undefined)delete globalThis[key];else globalThis[key]=value;}}
});
const {mappingConflict,mappingTarget,normalizeCutRange}=await import(controlsURL);
test('cut bounds follow the edited control and repair reversed saved ranges',()=>{
    assert.deepEqual(normalizeCutRange({min_cut_frames:30,max_cut_frames:10},'min_cut_frames'),{min_cut_frames:30,max_cut_frames:30});
    assert.deepEqual(normalizeCutRange({min_cut_frames:30,max_cut_frames:10},'max_cut_frames'),{min_cut_frames:10,max_cut_frames:10});
    assert.deepEqual(normalizeCutRange({min_cut_frames:30,max_cut_frames:10}),{min_cut_frames:10,max_cut_frames:30});
});
test("parameter aliases and mapping overlap rules preserve absolute mappings",()=>{
    assert.equal(mappingTarget('base_brightness'),'brightness');
    const rows=[{target:'dolly',start_frame:0,end_frame:48,enabled:true}];
    assert.equal(mappingConflict(rows,{target:'dolly',start_frame:48,end_frame:null}),false);
    assert.equal(mappingConflict(rows,{target:'dolly',start_frame:47,end_frame:null}),true);
    assert.equal(mappingConflict(rows,{target:'dolly',start_frame:0,end_frame:null},0),false);
    assert.equal(mappingConflict(rows,{target:'relief',start_frame:0,end_frame:null}),false);
    assert.equal(mappingConflict(rows,{target:'dolly',start_frame:0,end_frame:null,enabled:false}),false);
});
test("previs controls have unique owners and depth movement respects strength",()=>{
    const keys=Object.values(EFFECT_GROUPS).flat();assert.equal(keys.length,new Set(keys).size);
    const s={motion_mode:"depth_parallax",parallax_strength:0,depth_relief:1,steady_depth:.5,orbit_degrees:0,dolly:.2,offset_x:.1,offset_y:0,scene_scale:1};
    assert.deepEqual(demoPoint(.5,.5,.9,s,0),[.5,.5]);
    assert.notDeepEqual(demoPoint(.5,.5,.9,{...s,parallax_strength:1},0),[.5,.5]);
    assert.deepEqual(demoPoint(.5,.5,.5,{...s,parallax_strength:1},0),[.5,.5]);
});
test("timeline ranges are end-exclusive and dragging preserves length",()=>{
    assert.deepEqual(mappingRange({start_frame:48,end_frame:null},192),[48,192]);
    assert.deepEqual(moveMappingRange(48,96,-100,192),[0,48]);
    assert.deepEqual(moveMappingRange(48,96,200,192),[144,192]);
});

test("seeks land inside a frame, not on the previous frame boundary", () => {
    for(let frame=0;frame<192;frame++) {
        const time=frameSeekTime(frame,24);
        assert.ok(time>frame/24 && time<(frame+1)/24);
        assert.equal(previewFrame(time,24,192),frame);
    }
});
