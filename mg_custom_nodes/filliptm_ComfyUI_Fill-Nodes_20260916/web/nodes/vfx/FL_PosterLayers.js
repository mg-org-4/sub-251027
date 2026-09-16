import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";
import { ownsPosterEvent, posterLayerId, replacePreviewUrl } from "./poster_preview_events.js";

app.registerExtension({
    name: "Fill.PosterLayers",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "FL_PosterLayers" && nodeData.name !== "FL_PosterLayerPlanner") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            created?.apply(this, arguments);
            const node = this;
            if (nodeData.name === "FL_PosterLayerPlanner") {
                const executed = node.onExecuted;
                node.onExecuted = function (message) {
                    executed?.apply(this, arguments);
                    if (message.poster_plan?.[0]) node.properties.poster_plan = message.poster_plan[0];
                };
                node.addWidget("button", "Use last plan in Manual mode", null, () => {
                    if (!node.properties.poster_plan) return;
                    node.widgets.find(w => w.name === "manual_plan").value = JSON.stringify(node.properties.poster_plan, null, 2);
                    node.widgets.find(w => w.name === "mode").value = "manual";
                    app.graph.change();
                });
                return;
            }
            const widget = node.widgets.find(w => w.name === "overrides");
            widget.type = "converted-widget";
            widget.computeSize = () => [0, -4];
            widget.hidden = true;
            if (widget.element) widget.element.style.display = "none";
            const root = document.createElement("div");
            root.style.cssText = "height:100%;overflow:auto;background:#141820;color:#e9edf5;padding:12px;font:12px system-ui;box-sizing:border-box;border-radius:8px";
            const status = document.createElement("div"), cards = document.createElement("div");
            status.textContent = "Run to extract the planned layers. Layout edits apply on the next Run.";
            status.style.cssText = "padding:0 0 12px;color:#a7b8d2";
            cards.style.cssText = "display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px";
            root.append(status, cards);
            const live=document.createElement("section"),liveLabel=document.createElement("div"),liveImage=document.createElement("img"),liveProgress=document.createElement("progress");
            live.hidden=true;live.style.cssText="background:#101722;border:1px solid #425772;border-radius:6px;padding:10px;margin-bottom:12px";
            liveLabel.setAttribute("aria-live","polite");liveImage.alt="Current layer preview";liveImage.decoding="async";
            liveImage.style.cssText="display:block;width:100%;height:240px;object-fit:contain;margin:8px 0;background:repeating-conic-gradient(#343c48 0% 25%,#252c37 0% 50%) 50%/20px 20px";
            liveImage.hidden=true;liveProgress.max=1;liveProgress.value=0;liveProgress.style.cssText="width:100%;height:10px;accent-color:#77c8bb";
            liveProgress.setAttribute("aria-label","Current layer sampling progress");live.append(liveLabel,liveImage,liveProgress);root.insertBefore(live,cards);
            node.addDOMWidget("poster_review", "div", root, { serialize: false });
            // This node owns its image display; skip Comfy's duplicate canvas image widget.
            node.onDrawBackground=function() {};
            const removeCanvasPreview=()=>{
                const index=node.widgets.findIndex(w=>w.name==="$$canvas-image-preview");
                if(index>=0){node.widgets[index].onRemove?.();node.widgets.splice(index,1);}
                node.imgs=undefined;
            };
            let rows = [], key = "", disposed = false;
            const previewState={url:null};let currentLayer=null;
            const layerImages=new Map();
            const belongs=d=>!disposed&&node.graph===app.graph&&ownsPosterEvent(node.id,d);
            const read = () => {
                const data = JSON.parse(widget.value || "{}");
                return data.plan_key === key ? data : { plan_key: key, layers: {} };
            };
            const edit = (id, field, value) => {
                const data = read();
                data.layers[id] = { ...data.layers[id], [field]: value };
                widget.value = JSON.stringify(data);
                app.graph.change();
                status.textContent = "Edits pending — Run to apply. Only changed extraction prompts/rerolls need diffusion.";
            };
            const button = (label, parent, action) => {
                const b = document.createElement("button");
                b.textContent = label;
                b.style.cssText = "background:#303b51;color:#f2f5fa;border:1px solid #50617d;border-radius:4px;padding:5px 8px;cursor:pointer";
                b.onclick = action; parent.append(b); return b;
            };
            function draw() {
                cards.replaceChildren();
                layerImages.clear();
                for (const row of rows) {
                    const values = { ...row.defaults, ...read().layers[row.id] };
                    const card = document.createElement("section");
                    card.style.cssText = "background:#1e2633;border:1px solid #39465d;border-radius:6px;padding:10px;min-width:0";
                    const heading = document.createElement("strong");
                    heading.textContent = `${row.name} · ${row.kind}`;
                    const image = document.createElement("img");
                    image.src = api.apiURL(`/view?${new URLSearchParams(row.thumbnail)}`);
                    layerImages.set(row.id,image);
                    image.loading = "lazy"; image.decoding = "async"; image.alt = row.name;
                    image.style.cssText = "width:100%;height:160px;object-fit:contain;margin:8px 0;cursor:zoom-in;background:repeating-conic-gradient(#343c48 0% 25%,#252c37 0% 50%) 50%/20px 20px";
                    image.onclick = () => {
                        const dialog = document.createElement("dialog"), full = document.createElement("img");
                        dialog.style.cssText = "background:#18202b;color:white;max-width:85vw;max-height:90vh;border:1px solid #52637a;border-radius:8px";
                        full.src = api.apiURL(`/view?${new URLSearchParams(row.file)}`);
                        full.style.cssText = "display:block;max-width:78vw;max-height:77vh;object-fit:contain";
                        dialog.append(full); button("Close", dialog, () => dialog.close());
                        dialog.addEventListener("close", () => dialog.remove(), { once: true });
                        document.body.append(dialog); dialog.showModal();
                    };
                    const prompt = document.createElement("textarea");
                    prompt.value = values.prompt; prompt.rows = 3;
                    prompt.setAttribute("aria-label", `${row.name} extraction prompt`);
                    prompt.style.cssText = "box-sizing:border-box;width:100%;resize:vertical;background:#111822;color:#e9edf5;border:1px solid #4c5c74;border-radius:4px;padding:6px";
                    prompt.onchange = () => { if (prompt.value.trim()) edit(row.id, "prompt", prompt.value.trim()); else prompt.value = values.prompt; };
                    const fields = document.createElement("div");
                    fields.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:6px;margin:8px 0";
                    for (const [field, label, min, max, step] of [["depth", "Depth", .25, 11.75, .25], ["scale", "Scale", .1, 4, .01], ["offset_x", "X offset", -2, 2, .01], ["offset_y", "Y offset", -2, 2, .01]]) {
                        if (row.kind === "background") continue;
                        const l = document.createElement("label"), input = document.createElement("input");
                        l.textContent = label + " "; input.type = "number"; input.min = min; input.max = max; input.step = step; input.value = values[field];
                        input.title = field === "depth" ? "Smaller = in front and faster. Must be less than camera background depth." : "Applied by the compositor; does not rerun diffusion.";
                        input.style.cssText = "width:70px;background:#111822;color:white;border:1px solid #4c5c74;border-radius:3px";
                        input.onchange = () => { const v = Number(input.value); if (input.value !== "" && Number.isFinite(v)) { input.value = Math.max(min, Math.min(max, v)); edit(row.id, field, Number(input.value)); } };
                        l.append(input); fields.append(l);
                    }
                    const actions = document.createElement("div");
                    actions.style.cssText = "display:flex;gap:6px;align-items:center;flex-wrap:wrap";
                    if (row.kind !== "background") {
                        const label = document.createElement("label"), visible = document.createElement("input");
                        visible.type = "checkbox"; visible.checked = values.visible; visible.onchange = () => edit(row.id, "visible", visible.checked);
                        label.append(visible, " Visible"); actions.append(label);
                    }
                    button("Reroll", actions, async () => {
                        edit(row.id, "revision", (read().layers[row.id]?.revision ?? row.defaults.revision) + 1);
                        status.textContent = `Queuing ${row.name} reroll…`;
                        try { await api.queuePrompt(0, await app.graphToPrompt()); status.textContent = "Reroll queued. Other layers can reuse their cache."; }
                        catch (error) { status.textContent = `Could not queue: ${error.message}`; }
                    });
                    button("Reset", actions, () => {
                        const data = read(); delete data.layers[row.id]; widget.value = JSON.stringify(data); app.graph.change();
                        status.textContent = "Plan defaults restored — Run to apply.";
                        draw();
                    });
                    const coverage = document.createElement("div");
                    const incompleteBackground = row.kind === "background" && row.coverage < .98;
                    coverage.textContent = incompleteBackground ? "Background has missing areas — use a full-frame background prompt." : row.coverage < .001 ? "Nearly empty extraction — check prompt or reroll." : `Visible alpha: ${(row.coverage * 100).toFixed(1)}%`;
                    coverage.style.cssText = `color:${incompleteBackground ? "#f4c184" : "#a8b5c9"};margin-top:8px`;
                    card.append(heading, image, prompt, fields, actions, coverage); cards.append(card);
                }
            }
            function load(message) {
                if (!message?.poster_layers) return;
                rows = message.poster_layers; key = message.poster_plan_key[0];
                node.properties.poster_review = { poster_layers: rows, poster_plan_key: [key] };
                status.textContent = `${rows.length} layers · click a thumbnail for full-resolution RGBA · Run applies layout edits`;
                draw();
            }
            const executed = node.onExecuted, configured = node.onConfigure, removed = node.onRemoved;
            node.onExecuted = function (message) { executed?.apply(this, arguments); load(message); };
            node.onConfigure = function () { configured?.apply(this, arguments); removeCanvasPreview(); load(node.properties.poster_review); };
            const progress = event => {
                const d = event.detail;
                if(!belongs(d))return;
                const id=posterLayerId(d);
                if(!id&&liveProgress.hidden)currentLayer=null;
                if(id&&id!==currentLayer){currentLayer=id;replacePreviewUrl(previewState,null);liveImage.removeAttribute("src");liveImage.hidden=true;}
                live.hidden=false;liveProgress.hidden=false;liveProgress.max=Math.max(1,d.max);liveProgress.value=d.value;
                liveLabel.textContent=`${rows.find(r=>r.id===currentLayer)?.name??currentLayer??"Current layer"} · ${d.value}/${d.max}`;
                status.textContent="Generating layers · previews appear here, inside the review panel.";
            };
            const preview=event=>{
                const d=event.detail;if(!belongs(d)||!(d.blob instanceof Blob))return;
                currentLayer=posterLayerId(d)??currentLayer;live.hidden=false;liveImage.hidden=false;
                liveLabel.textContent=`${rows.find(r=>r.id===currentLayer)?.name??currentLayer??"Current layer"} · denoising preview`;
                liveImage.src=replacePreviewUrl(previewState,d.blob);
            };
            const decoded=event=>{
                const d=event.detail;if(!belongs(d))return;
                for(const ready of d.output?.poster_layer_ready??[]){
                    currentLayer=ready.id;replacePreviewUrl(previewState,null);live.hidden=false;liveImage.hidden=false;liveProgress.hidden=true;
                    liveLabel.textContent=`${rows.find(r=>r.id===ready.id)?.name??ready.id} · decoded RGBA`;
                    liveImage.src=api.apiURL(`/view?${new URLSearchParams(ready.thumbnail)}`);
                    const thumbnail=layerImages.get(ready.id);if(thumbnail)thumbnail.src=liveImage.src;
                }
            };
            const started=()=>{replacePreviewUrl(previewState,null);currentLayer=null;live.hidden=true;liveImage.removeAttribute("src");liveImage.hidden=true;};
            const finished=()=>{if(!live.hidden){replacePreviewUrl(previewState,null);live.hidden=true;liveImage.removeAttribute("src");}};
            const failed=event=>{if(belongs(event.detail)){liveProgress.hidden=true;liveLabel.textContent="Generation stopped · see execution details";}};
            api.addEventListener("progress", progress);
            const listeners={b_preview_with_metadata:preview,executed:decoded,execution_start:started,execution_success:finished,execution_error:failed,execution_interrupted:failed};
            for(const [name,handler] of Object.entries(listeners))api.addEventListener(name,handler);
            node.onRemoved = function () { disposed = true; replacePreviewUrl(previewState,null);api.removeEventListener("progress", progress);for(const [name,handler] of Object.entries(listeners))api.removeEventListener(name,handler); root.remove(); removed?.apply(this, arguments); };
            queueMicrotask(() => { if (!disposed) load(node.properties.poster_review); });
            node.setSize([760, 1100]);
        };
    },
});
