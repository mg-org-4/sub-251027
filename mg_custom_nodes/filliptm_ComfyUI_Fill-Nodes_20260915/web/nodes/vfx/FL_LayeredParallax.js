import {app} from "../../../../scripts/app.js";
import {api} from "../../../../scripts/api.js";
import {CAMERA_PRESETS, demoLayers, drawScene, viewport} from "./parallax_preview.js";
import {createDepthPreview} from "./parallax_depth_preview.js";

const CONTROLS = ["motion","travel_x","travel_y","push_in","background_depth","overscan","device","layer_fit","relief_scope","relief_strength","relief_anchor","depth_invert","depth_smoothing"];
const style = document.createElement("style");
style.textContent = `
.fl-parallax-editor{height:100%;box-sizing:border-box;overflow:auto;padding:12px;background:#111a26;color:#e6edf7;font:12px system-ui;border:1px solid #334459;border-radius:9px}
.fl-parallax-editor *{box-sizing:border-box}.fl-parallax-editor .pe-top,.fl-parallax-editor .pe-tabs,.fl-parallax-editor .pe-presets{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.fl-parallax-editor .pe-top{justify-content:space-between;margin-bottom:10px}.fl-parallax-editor .pe-body{display:grid;grid-template-columns:minmax(0,1fr) 235px;gap:12px}
.fl-parallax-editor button,.fl-parallax-editor select,.fl-parallax-editor input[type=number]{background:#223247;color:#e6edf7;border:1px solid #425772;border-radius:5px;padding:5px;font:inherit}
.fl-parallax-editor button{cursor:pointer}.fl-parallax-editor button[aria-pressed=true]{background:#23665f;border-color:#68c6b5}.fl-parallax-editor button:disabled{opacity:.45;cursor:default}
.fl-parallax-editor canvas{display:block;width:100%;background:#0b111c;border:1px solid #34485d;border-radius:6px;touch-action:none}.fl-parallax-editor .pe-scene{cursor:grab;margin:8px 0}
.fl-parallax-editor .pe-pad{height:105px;cursor:crosshair;margin:8px 0}.fl-parallax-editor .pe-hint{font-size:11px;color:#9fb1c7;line-height:1.5;margin:6px 0}
.fl-parallax-editor .pe-field{display:grid;grid-template-columns:minmax(0,1fr) 68px;gap:5px;align-items:center;margin:12px 0}.fl-parallax-editor .pe-field input[type=range]{grid-column:1/-1;width:100%;accent-color:#75cbbb}
.fl-parallax-editor .pe-field input[type=number]{width:68px}.fl-parallax-editor .pe-help{border-radius:50%;padding:0 4px;margin-left:5px;font-size:10px;cursor:help}
.fl-parallax-editor .pe-tabs{border-bottom:1px solid #334459;padding-bottom:8px;margin-bottom:8px}.fl-parallax-editor .pe-warning{color:#f4c184;min-height:30px;font-size:11px;line-height:1.4}
.fl-parallax-editor .pe-status{color:#8fd8c5;font-size:11px;margin-top:8px}.fl-parallax-editor input[type=range]{min-width:0}.fl-parallax-editor [hidden]{display:none!important}
`;
document.head.append(style);

app.registerExtension({
    name: "Fill.LayeredParallax.Editor",
    async beforeRegisterNodeDef(nodeType, data) {
        if (data.name !== "FL_LayeredParallax") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            created?.apply(this,arguments);
            const node=this, widgets=Object.fromEntries(node.widgets.map(w=>[w.name,w]));
            for (const name of CONTROLS) if (widgets[name]) {
                const w=widgets[name];w.type="converted-widget";w.computeSize=()=>[0,-4];w.hidden=true;
                if(w.element)w.element.style.display="none";
            }
            const root=document.createElement("div");root.className="fl-parallax-editor";
            root.innerHTML=`<div class="pe-top"><strong>PARALLAX STUDIO</strong><span>Camera & framing · no diffusion</span></div>
                <div class="pe-body"><div><div class="pe-tabs"><button data-view="scene" aria-pressed="true">Scene</button><button data-view="depth" aria-pressed="false">Depth order</button></div>
                <canvas class="pe-scene" width="480" height="288" aria-label="Parallax preview. Drag to adjust camera travel."></canvas>
                <div class="pe-presets"><button data-play aria-label="Play preview">Play</button><input data-phase aria-label="Preview position" type="range" min="0" max="1" step=".001" value="0" style="flex:1"><button data-center>Center</button></div>
                <p class="pe-hint" data-source-label></p><select data-source aria-label="Preview source"><option value="actual">Actual layers</option><option value="landscape">Demo · landscape</option><option value="design">Demo · typography</option><option value="product">Demo · product shapes</option></select>
                <p class="pe-hint">Drag the scene to set travel. Preview uses small first-frame images and the render's camera math; it is not a final-quality render.</p><p class="pe-warning" role="status"></p><div class="pe-status"></div></div>
                <div><div class="pe-tabs"><button data-tab="camera" aria-pressed="true">Camera</button><button data-tab="framing" aria-pressed="false">Framing</button></div>
                <div data-page="camera"><div class="pe-presets" data-presets></div><p class="pe-hint">Presets change motion, not the image's style.</p><div class="pe-presets" data-motions></div>
                <canvas class="pe-pad" width="235" height="105" aria-label="Camera direction control. Drag the point to set horizontal and vertical travel."></canvas><div data-camera-fields></div></div>
                <div data-page="framing" hidden><div data-framing-fields></div><div class="pe-field"><label>Cutout fit</label><select data-select="layer_fit"><option>cover</option><option>contain</option></select></div><p class="pe-hint">Contain keeps the full cutout when aspect ratios differ. Cover fills the frame. Background edges extend existing pixels; no new scenery is generated.</p>
                <div class="pe-field"><label>Render device</label><select data-select="device"><option>auto</option><option>cpu</option></select></div><p class="pe-hint">Layer depths, placement and visibility come from the connected layer controls. Width, height and frame count remain above this editor.</p></div></div></div>`;
            node.addDOMWidget("parallax_editor","div",root,{serialize:false});
            const reliefPreview=createDepthPreview();
            const tabs=root.querySelector('[data-tab="camera"]').parentElement;
            const reliefTab=document.createElement('button');reliefTab.dataset.tab='relief';reliefTab.textContent='Relief';tabs.append(reliefTab);
            const reliefPage=document.createElement('div');reliefPage.dataset.page='relief';reliefPage.hidden=true;
            reliefPage.innerHTML='<div class="pe-field"><label>Apply to</label><select data-select="relief_scope"><option>off</option><option>background</option><option>background + artwork</option></select></div><div data-relief-fields></div><label><input type="checkbox" data-invert> Invert depth (black is near)</label><p class="pe-hint">Text stays flat. Depth batches must match the current layer stack. Strength and anchor preview live; smoothing requires Run. Preview uses a small depth grid. Relief cannot reveal missing surfaces or change layer draw order.</p>';
            tabs.parentElement.append(reliefPage);
            const scene=root.querySelector(".pe-scene"),pad=root.querySelector(".pe-pad"),phaseInput=root.querySelector("[data-phase]"),play=root.querySelector("[data-play]"),status=root.querySelector(".pe-status"),warning=root.querySelector(".pe-warning"),sourceSelect=root.querySelector("[data-source]"),sourceLabel=root.querySelector("[data-source-label]");
            let phase=0,centered=false,playing=false,visible=false,disposed=false,raf=0,last=0,drawn=0,view="scene",actual=[],demo=demoLayers("landscape"),loadVersion=0;
            const linked=name=>node.inputs?.some(i=>i.name===name&&i.link!=null);
            const settings=()=>Object.fromEntries(["width","height",...CONTROLS].map(k=>[k,(linked(k)?node.properties.parallax_settings?.[k]:undefined)??widgets[k]?.value??(k==="layer_fit"?"cover":0)]));
            const plates=()=>sourceSelect.value==="actual"&&actual.length?actual:demo;
            function write(name,value) {
                if(linked(name)){status.textContent=`${name} is driven by a connected input.`;return;}
                if(!widgets[name])return;
                if(name==="background_depth")value=Math.min(100,Math.max(value,...plates().filter(p=>!p.background).map(p=>p.depth+.25)));
                widgets[name].value=value;node.graph?.change();node.graph?.setDirtyCanvas(true,true);refresh();invalidate();
            }
            function refresh() {
                const s=settings();
                root.querySelectorAll("[data-setting]").forEach(e=>{e.value=s[e.dataset.setting];e.disabled=linked(e.dataset.setting);});
                root.querySelectorAll("[data-select]").forEach(e=>{e.value=s[e.dataset.select];e.disabled=linked(e.dataset.select);});
                root.querySelector('[data-invert]').checked=!!s.depth_invert;root.querySelector('[data-invert]').disabled=linked('depth_invert');
                root.querySelectorAll("[data-motion]").forEach(e=>{e.setAttribute("aria-pressed",String(e.dataset.motion===s.motion));e.disabled=linked("motion");});
                root.querySelectorAll("[data-preset]").forEach(e=>e.setAttribute("aria-pressed",String(Object.entries(CAMERA_PRESETS[e.dataset.preset]).every(([k,v])=>s[k]===v))));
                const nearest=Math.min(...plates().filter(p=>!p.background).map(p=>p.depth),s.background_depth);
                const farthest=Math.max(...plates().filter(p=>!p.background).map(p=>p.depth),0);
                root.querySelectorAll('[data-setting="background_depth"]').forEach(e=>e.min=Math.min(100,Math.max(.25,farthest+.25)));
                warning.textContent=s.background_depth<=farthest?"Background depth must be greater than every cutout depth.":Math.abs(s.push_in)>=nearest?"Reduce push-in: the camera must stay in front of every layer.":Math.hypot(s.travel_x,s.travel_y)/nearest>.08?"Strong motion can reveal missing edges or hidden regions. Reduce travel or prepare wider layers.":"";
                if(s.relief_scope!=="off"&&s.relief_strength>0){
                    if(!plates().some(p=>p.relief))warning.textContent="Run with depth maps once to enable the relief preview. Showing flat camera motion.";
                    else if(!reliefPreview)warning.textContent="WebGL unavailable: preview shows flat layers; depth relief still applies on Run.";
                }
            }
            const ranges=[
                ["travel_x","Horizontal travel",-2,2,.005,"Camera movement in output widths at depth 1. Nearer layers move more.","camera"],
                ["travel_y","Vertical travel",-2,2,.005,"Camera movement in output heights at depth 1.","camera"],
                ["push_in","Push / pull",-.2,.2,.005,"Forward travel in depth units. Positive moves toward the layers.","camera"],
                ["background_depth","Background depth",.25,100,.25,"Must be farther away than every foreground layer. Does not change their depth order.","framing"],
                ["overscan","Background coverage",1,2,.01,"Extra background scale. More coverage reduces visible border extension.","framing"],
                ["relief_strength","Relief strength",0,.5,.01,"Variation within each layer. Zero keeps the original flat-plane rendering.","relief"],
                ["relief_anchor","Anchor depth",0,1,.01,"Depth-map value that follows the original flat plane. White is near unless inverted.","relief"],
                ["depth_smoothing","Depth smoothing",0,16,1,"Smoothing radius in depth-map pixels. Run to apply; cached depth inference is reused.","relief"],
            ];
            for(const [name,label,min,max,step,help,page] of ranges){
                const field=document.createElement("div");field.className="pe-field";
                const title=document.createElement("label");title.textContent=label;
                const tip=document.createElement("button");tip.textContent="?";tip.className="pe-help";tip.title=help;tip.setAttribute("aria-label",help);tip.onclick=()=>status.textContent=help;title.append(tip);field.append(title);
                for(const type of ["number","range"]){const input=document.createElement("input");input.type=type;input.min=min;input.max=max;input.step=step;input.dataset.setting=name;input.setAttribute("aria-label",`${label} ${type}`);input.title=help;
                    input.addEventListener(type==="range"?"input":"change",()=>{const v=Number(input.value);if(input.value!==""&&Number.isFinite(v))write(name,Math.max(min,Math.min(max,v)));else refresh();});field.append(input);}
                root.querySelector(`[data-${page}-fields]`).append(field);
            }
            for(const [label,preset] of Object.entries(CAMERA_PRESETS)){
                const b=document.createElement("button");b.textContent=label;b.dataset.preset=label;b.onclick=()=>{for(const [k,v] of Object.entries(preset))write(k,v);status.textContent=Object.keys(preset).some(linked)?"Preset applied to unconnected controls; connected inputs stay unchanged.":`${label} motion applied. Run to export.`;};root.querySelector("[data-presets]").append(b);
            }
            for(const motion of ["loop","glide","bursts","locked"]){const b=document.createElement("button");b.textContent=motion;b.dataset.motion=motion;b.onclick=()=>write("motion",motion);root.querySelector("[data-motions]").append(b);}
            root.querySelectorAll("[data-select]").forEach(e=>e.onchange=()=>write(e.dataset.select,e.value));
            root.querySelector('[data-invert]').onchange=e=>write('depth_invert',e.target.checked);
            root.querySelectorAll("[data-tab]").forEach(b=>b.onclick=()=>{root.querySelectorAll("[data-tab]").forEach(e=>e.setAttribute("aria-pressed",String(e===b)));root.querySelectorAll("[data-page]").forEach(e=>e.hidden=e.dataset.page!==b.dataset.tab);});
            root.querySelectorAll("[data-view]").forEach(b=>b.onclick=()=>{view=b.dataset.view;if(view==="depth")stop();root.querySelectorAll("[data-view]").forEach(e=>e.setAttribute("aria-pressed",String(e===b)));invalidate();});
            function drawPad(s){const c=pad.getContext("2d"),w=pad.width,h=pad.height;c.clearRect(0,0,w,h);c.strokeStyle="#3b5068";c.beginPath();c.moveTo(w/2,0);c.lineTo(w/2,h);c.moveTo(0,h/2);c.lineTo(w,h/2);c.stroke();
                const x=w/2+Math.max(-.5,Math.min(.5,s.travel_x))*w,y=h/2+Math.max(-.5,Math.min(.5,s.travel_y))*h;c.strokeStyle="#85d5c4";c.beginPath();c.moveTo(w/2,h/2);c.lineTo(x,y);c.stroke();c.fillStyle="#85d5c4";c.beginPath();c.arc(x,y,5,0,7);c.fill();c.font="10px system-ui";c.fillStyle="#b7c9da";c.fillText("CAMERA XY · ±0.5",6,14);}
            function paint(now){raf=0;if(disposed||!visible||document.hidden)return;
                if(playing){if(last)phase=(phase+(now-last)/6000)%1;last=now;}else last=0;
                if(!playing||now-drawn>=50){const s=settings(),height=view==="depth"?Math.max(360,plates().length*26+50):288;if(scene.height!==height)scene.height=height;drawScene(scene,plates(),centered?{...s,motion:"locked"}:s,phase,view==="depth",reliefPreview);drawPad(s);phaseInput.value=phase;drawn=now;}
                if(playing)raf=requestAnimationFrame(paint);
            }
            function invalidate(){if(!raf&&visible&&!document.hidden&&!disposed)raf=requestAnimationFrame(paint);}
            function stop(){playing=false;last=0;play.textContent="Play";play.setAttribute("aria-label","Play preview");if(raf)cancelAnimationFrame(raf);raf=0;}
            play.onclick=()=>{centered=false;playing=!playing;play.textContent=playing?"Pause":"Play";play.setAttribute("aria-label",playing?"Pause preview":"Play preview");last=0;invalidate();};
            phaseInput.oninput=()=>{stop();centered=false;phase=Number(phaseInput.value);invalidate();};
            root.querySelector("[data-center]").onclick=()=>{stop();centered=true;phase=0;invalidate();};
            function drag(canvas,move){let active=false;canvas.onpointerdown=e=>{if(view==="depth"&&canvas===scene)return;active=true;canvas.setPointerCapture(e.pointerId);stop();centered=false;if(settings().motion==="locked")write("motion","loop");phase=settings().motion==="loop"?.25:1;move(e,true);invalidate();e.stopPropagation();};canvas.onpointermove=e=>{if(active){move(e,false);e.stopPropagation();}};canvas.onpointerup=canvas.onpointercancel=e=>{active=false;if(canvas.hasPointerCapture(e.pointerId))canvas.releasePointerCapture(e.pointerId);};}
            drag(pad,e=>{const r=pad.getBoundingClientRect();write("travel_x",Math.max(-.5,Math.min(.5,(e.clientX-r.left)/r.width-.5)));write("travel_y",Math.max(-.5,Math.min(.5,(e.clientY-r.top)/r.height-.5)));});
            let origin;
            drag(scene,(e,start)=>{const s=settings();if(start){origin={x:e.clientX,y:e.clientY,tx:s.travel_x,ty:s.travel_y};return;}const r=scene.getBoundingClientRect(),v=viewport(r.width,r.height,s.width/s.height),depth=Math.min(...plates().filter(p=>!p.background).map(p=>p.depth),s.background_depth);write("travel_x",Math.max(-2,Math.min(2,origin.tx-(e.clientX-origin.x)/v.w*depth)));write("travel_y",Math.max(-2,Math.min(2,origin.ty-(e.clientY-origin.y)/v.h*depth)));});
            function sourceChanged(){if(sourceSelect.value!=="actual")demo=demoLayers(sourceSelect.value);sourceLabel.textContent=sourceSelect.value==="actual"&&actual.length?`${actual.length} actual layers · first-frame preview · upstream changes need Run`:"DEMO GEOMETRY · not generated output";refresh();invalidate();}
            sourceSelect.onchange=sourceChanged;
            async function load(rows){reliefPreview?.clear();const version=++loadVersion;if(!rows?.length){sourceChanged();return;}const loaded=await Promise.all(rows.map(p=>new Promise(resolve=>{const bitmap=new Image();bitmap.decoding="async";bitmap.onload=()=>resolve({...p,bitmap});bitmap.onerror=()=>resolve(null);bitmap.src=api.apiURL(`/view?${new URLSearchParams(p.image)}`);})));if(disposed||version!==loadVersion)return;actual=loaded.every(Boolean)?loaded.sort((a,b)=>a.background?-1:b.background?1:b.depth-a.depth):[];sourceChanged();if(!actual.length)status.textContent="Source previews unavailable; showing demo. Run once to rebuild thumbnails.";}
            function restore(){const rows=node.properties.parallax_layers;if(rows){load(rows);return;}const link=node.inputs?.find(i=>i.name==="layer_stack")?.link;const source=node.graph?.getNodeById(node.graph?.links[link]?.origin_id);const review=source?.properties.poster_review?.poster_layers;
                if(review)load(review.filter(r=>r.kind==="background"||r.visible).map(r=>({...r,image:r.thumbnail,width:widgets.width.value,height:widgets.height.value,background:r.kind==="background",opacity:1})));else sourceChanged();}
            const observer=new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;if(!visible){if(raf)cancelAnimationFrame(raf);raf=0;last=0;}else invalidate();});observer.observe(root);
            const visibility=()=>{if(document.hidden){if(raf)cancelAnimationFrame(raf);raf=0;last=0;}else invalidate();};document.addEventListener("visibilitychange",visibility);
            const progress=e=>{if(String(e.detail.node)===String(node.id))status.textContent=`Rendering ${Math.round(e.detail.value/e.detail.max*100)}%`;};api.addEventListener("progress",progress);
            for(const w of Object.values(widgets)){const callback=w.callback;w.callback=function(){const result=callback?.apply(this,arguments);refresh();invalidate();return result;};}
            const executed=node.onExecuted,configured=node.onConfigure,removed=node.onRemoved;
            node.onExecuted=function(message){executed?.apply(this,arguments);if(message.parallax_settings)node.properties.parallax_settings=message.parallax_settings[0];if(message.parallax_layers){node.properties.parallax_layers=message.parallax_layers;load(message.parallax_layers);status.textContent="Rendered · camera edits preview immediately; Run exports them.";}};
            node.onConfigure=function(){configured?.apply(this,arguments);restore();refresh();};
            node.onRemoved=function(){disposed=true;stop();reliefPreview?.dispose();observer.disconnect();document.removeEventListener("visibilitychange",visibility);api.removeEventListener("progress",progress);root.remove();removed?.apply(this,arguments);};
            queueMicrotask(()=>{if(!disposed){restore();refresh();}});node.setSize([850,750]);
        };
    },
});
