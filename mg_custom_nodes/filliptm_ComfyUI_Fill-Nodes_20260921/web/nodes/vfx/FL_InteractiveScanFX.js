import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";
import { createAudioMappingPanel } from "./scan_audio_mapping.js";
import { createPrevis, EFFECT_GROUPS } from "./scan_previs.js";
import { createScanControls } from "./scan_controls.js";

export function previewFrame(time, fps, count) {
    return Math.max(0, Math.min(count - 1, Math.floor(time * fps + 0.0001)));
}

export function frameSeekTime(frame, fps) {
    return (frame + 0.5) / fps;
}

app.registerExtension({
    name: "Fill.InteractiveScanFX",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "FL_InteractiveScanFX") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            created?.apply(this, arguments);
            const node = this;
            const settings = node.widgets.find(w => w.name === "advanced_settings");
            const inputOptions=nodeData.input.required.advanced_settings[1];
            const defaults=JSON.parse(inputOptions.default);
            const choices=inputOptions.scan_choices;
            settings.hidden = true;
            settings.type = "converted-widget";
            settings.computeSize = () => [0, 0];
            settings.computedHeight = 0;
            if (settings.element) settings.element.style.display = "none";
            const root = document.createElement("div");
            root.className = "fl-interactive-scan";
            root.style.cssText = "background:#141920;color:#dce4ef;padding:10px;border-radius:8px;font:12px system-ui;overflow:auto;box-sizing:border-box;height:100%";
            const layout=document.createElement("div"),visuals=document.createElement("div"),controls=document.createElement("div"),rendered=document.createElement("div"),sources=document.createElement("div");
            layout.style.cssText="display:grid;grid-template-columns:minmax(260px,1fr) minmax(380px,1.1fr);gap:10px;height:100%;min-height:0";
            visuals.style.cssText="min-width:0;overflow:auto";controls.style.cssText="min-width:0;overflow:auto;padding-right:6px";
            sources.style.cssText="display:flex;gap:6px;margin-bottom:10px";
            const readSettings=()=>({...defaults,...JSON.parse(settings.value),...Object.fromEntries(node.widgets.filter(w=>Object.values(EFFECT_GROUPS).flat().includes(w.name)).map(w=>[w.name,w.value]))});
            const previs=createPrevis(readSettings);
            const tabs = document.createElement("div");
            tabs.style.cssText = "display:flex;gap:5px;margin-bottom:7px";
            const canvas = document.createElement("canvas");
            canvas.width = 320; canvas.height = 320;
            canvas.style.cssText = "width:100%;max-height:360px;object-fit:contain;background:#080b10;display:block";
            const video = document.createElement("video");
            video.muted = true; video.playsInline = true; video.preload = "metadata";
            const transport = document.createElement("div");
            transport.style.cssText = "display:flex;gap:6px;align-items:center;margin:8px 0";
            const button = (label, parent, action) => {
                const b = document.createElement("button"); b.textContent = label;
                b.style.cssText = "background:#293345;color:#e9f1ff;border:1px solid #46536c;border-radius:4px;padding:5px 8px;cursor:pointer";
                b.onclick = action; parent.append(b); return b;
            };
            function setSource(value){previs.setActive(value);rendered.hidden=value;liveButton.style.background=value?"#345d91":"#293345";renderButton.style.background=value?"#293345":"#345d91";if(value)video.pause();}
            const liveButton=button("Live demo",sources,()=>setSource(true));
            const renderButton=button("Last render",sources,()=>setSource(false));
            let view = 0, data = null, frameCallback = null, displayedTime = 0, disposed = false;
            const status = document.createElement("div");
            const progressBox=document.createElement("div");
            progressBox.hidden=true;
            const progressLabel=document.createElement("div"),progressBar=document.createElement("progress");
            progressLabel.setAttribute("aria-live","polite");
            progressBar.setAttribute("aria-label","Current render stage progress");
            progressBar.style.cssText="width:100%;height:12px;accent-color:#68b8ff";
            progressBox.append(progressLabel,progressBar);
            let stage="Preparing",frameUpdates=false;
            function showProgress(value,total){
                progressBox.hidden=false;progressBar.max=Math.max(1,total);progressBar.value=value;
                progressLabel.textContent=`${stage} · ${Math.round(100*value/Math.max(1,total))}% (${value}/${total})`;
            }
            const caption = document.createElement("div");
            caption.style.cssText = "color:#aab8cc;margin:6px 0;font-size:11px";
            const labels = ["Final", "Surface", "Mask", "Debug", "Depth"];
            const descriptions = ["Finished effect", "Projected voxel surface · before camera accents", "Reveal alpha · excludes cursor graphics", "Planned cursor trajectories · frame and shot", "Projected depth · before camera accents"];
            let mappingPanel;
            const tabButtons = labels.map((label,i) => button(label,tabs,()=>{view=i;draw();}));
            const scrub = document.createElement("input");scrub.type="range";scrub.min="0";scrub.max="0";scrub.step="1";scrub.style.width="100%";scrub.setAttribute("aria-label","Preview frame");
            const readout = document.createElement("span");readout.style.whiteSpace="nowrap";
            const meters = document.createElement("div");meters.style.cssText="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:8px 0";
            const bars = ["Kick", "Snare", "Hat"].map(label=>{
                const cell=document.createElement("label");cell.textContent=label;
                const meter=document.createElement("meter");meter.min=0;meter.max=1;meter.value=0;meter.style.width="100%";cell.append(meter);meters.append(cell);return meter;
            });
            function draw() {
                tabButtons.forEach((b,i)=>{b.style.background=i===view?"#345d91":"#293345";b.disabled=!!data&&i>=(data.views?.length??4);});
                caption.textContent=descriptions[view];
                if (!data) {status.textContent="Run the node to generate previews.";return;}
                const frame=previewFrame(displayedTime,data.fps,data.frames);
                scrub.value=String(frame);readout.textContent=`${frame+1} / ${data.frames}`;
                bars.forEach((b,i)=>b.value=data.envelopes[i][frame]??0);
                const columns=data.columns??2;
                if(video.readyState>=2)canvas.getContext("2d").drawImage(video,(view%columns)*data.width,Math.floor(view/columns)*data.height,data.width,data.height,0,0,canvas.width,canvas.height);
                play.textContent=video.paused?"Play":"Pause";
                mappingPanel?.update(data,frame);
                controlPanel?.update(data,frame);
            }
            function tick(now, metadata){frameCallback=null;if(disposed)return;displayedTime=metadata.mediaTime;draw();if(!video.paused)frameCallback=video.requestVideoFrameCallback(tick);}
            const play=button("Play",transport,async()=>{
                if(!data)return;
                if(video.paused){if(video.ended)video.currentTime=0;try{await video.play();}catch(error){status.textContent=error.message;return;}if(frameCallback===null)frameCallback=video.requestVideoFrameCallback(tick);}else video.pause();draw();
            });
            button("Stop",transport,()=>{video.pause();video.currentTime=0;displayedTime=0;draw();});
            transport.append(scrub,readout);
            scrub.oninput=()=>{if(data){video.pause();video.currentTime=frameSeekTime(Number(scrub.value),data.fps);}};
            video.onloadeddata=video.onseeked=()=>{displayedTime=video.currentTime;draw();};video.onended=draw;
            video.onerror=()=>{status.textContent="Preview unavailable. Run again to recreate the temporary preview.";};
            let controlPanel;
            function populateSettings(){
                settings.value=JSON.stringify({...defaults,...JSON.parse(settings.value)});
                controlPanel?.rebuild();mappingPanel?.rebuild();
                previs.redraw();
            }
            mappingPanel=createAudioMappingPanel({settings,targets:inputOptions.scan_mapping_targets,
                onChange:()=>{node.graph?.setDirtyCanvas(true,true);controlPanel?.rebuild();status.textContent="Mapping changed · run again to update rendered values.";},
                onSeek:frame=>{if(data){video.pause();video.currentTime=frameSeekTime(frame,data.fps);}}});
            controlPanel=createScanControls({node,settings,options:{...inputOptions,widget_specs:Object.fromEntries(Object.entries(nodeData.input.required).map(([k,v])=>[k,v[1]]))},mappingPanel,
                onChange:()=>{previs.redraw();status.textContent="Settings changed · run again to update rendered values.";},
                onSolo:key=>{setSource(true);previs.select(key);}});
            rendered.append(tabs,canvas,caption,transport,meters,status);
            visuals.append(progressBox,sources,previs.element,rendered);
            visuals.append(mappingPanel.element);controls.append(controlPanel.element);layout.append(visuals,controls);root.append(layout);setSource(true);
            node.addDOMWidget("scan_preview","fl-scan-preview",root,{serialize:false,hideOnZoom:false,getMinHeight:()=>540});
            function load(preview){
                if(!preview)return;
                data=preview;node.properties.fl_scan_preview=preview;
                video.pause();displayedTime=0;video.src=api.apiURL(`/view?${new URLSearchParams({filename:preview.filename,subfolder:preview.subfolder,type:preview.type})}`);
                canvas.width=preview.width;canvas.height=preview.height;scrub.max=String(preview.frames-1);
                status.textContent=`${preview.frames} frames · ${preview.cuts} cuts · ${preview.events} gestures · silent preview from last run`;
                draw();
            }
            const executed=node.onExecuted;
            node.onExecuted=function(message){executed?.apply(this,arguments);load(message.fl_interactive_scan?.[0]);};
            const configured=node.onConfigure;
            node.onConfigure=function(){configured?.apply(this,arguments);populateSettings();load(node.properties.fl_scan_preview);};
            const connectionsChanged=node.onConnectionsChange;
            node.onConnectionsChange=function(){connectionsChanged?.apply(this,arguments);queueMicrotask(()=>{if(!disposed)controlPanel?.rebuild();});};
            const removed=node.onRemoved;
            const stageProgress=event=>{
                const d=event.detail;if(String(d.node)!==String(node.id))return;
                stage=d.stage;frameUpdates=d.frame_updates;showProgress(d.value,d.max);
            };
            const frameProgress=event=>{
                const d=event.detail;
                if(frameUpdates&&String(d.node)===String(node.id))showProgress(d.value,d.max);
            };
            const failed=event=>{
                if(String(event.detail.node_id)!==String(node.id))return;
                frameUpdates=false;progressLabel.textContent="Render stopped · check the execution error";
            };
            api.addEventListener("fl_scan_progress",stageProgress);
            api.addEventListener("progress",frameProgress);
            api.addEventListener("execution_error",failed);
            api.addEventListener("execution_interrupted",failed);
            const executing=event=>{if(String(event.detail?.node??event.detail)===String(node.id))status.textContent="Rendering effect and diagnostic previews…";};
            api.addEventListener("executing",executing);
            node.onRemoved=function(){disposed=true;previs.dispose();api.removeEventListener("executing",executing);api.removeEventListener("fl_scan_progress",stageProgress);api.removeEventListener("progress",frameProgress);api.removeEventListener("execution_error",failed);api.removeEventListener("execution_interrupted",failed);if(frameCallback!==null)video.cancelVideoFrameCallback(frameCallback);video.pause();video.removeAttribute("src");video.load();root.remove();removed?.apply(this,arguments);};
            populateSettings();draw();node.setSize([1000,950]);
        };
    },
});
