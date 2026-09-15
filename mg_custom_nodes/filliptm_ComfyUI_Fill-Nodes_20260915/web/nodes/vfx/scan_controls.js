import { EFFECT_GROUPS } from "./scan_previs.js";

export function mappingTarget(key) {
    return ({base_brightness:"brightness",base_saturation:"saturation",glow_intensity:"glow"})[key] ?? key;
}

export function normalizeCutRange(settings, changed=null) {
    if(settings.min_cut_frames>settings.max_cut_frames){
        if(changed==='min_cut_frames')settings.max_cut_frames=settings.min_cut_frames;
        else if(changed==='max_cut_frames')settings.min_cut_frames=settings.max_cut_frames;
        else [settings.min_cut_frames,settings.max_cut_frames]=[settings.max_cut_frames,settings.min_cut_frames];
    }
    return settings;
}

export function mappingConflict(rows, candidate, skip=-1) {
    if(candidate.enabled===false)return false;
    return rows.some((r,i)=>i!==skip&&r.enabled!==false&&r.target===candidate.target&&
        (r.start_frame??0)<(candidate.end_frame??Infinity)&&(candidate.start_frame??0)<(r.end_frame??Infinity));
}

export function createScanControls({node,settings,options,onChange,mappingPanel,onSolo}) {
    const element=document.createElement("div"),sources=document.createElement("div"),tabs=document.createElement("div"),body=document.createElement("div"),notice=document.createElement("div");
    element.className="fl-scan-controls";
    const style=document.createElement("style");style.textContent=`
    .fl-scan-controls button{background:#26364a;color:#e7f1ff;border:1px solid #58708f;border-radius:4px;padding:3px 5px;font:11px system-ui;cursor:pointer;min-height:24px}
    .fl-scan-controls button[aria-selected=true],.fl-scan-controls button[aria-pressed=true]{background:#326397;border-color:#91ceff}
    .fl-scan-controls button:disabled{opacity:.4;cursor:default}
    .fl-scan-controls input{background:#101b2a;color:white;border:1px solid #58708f;border-radius:4px;min-width:0;padding:4px;box-sizing:border-box}
    .fl-scan-controls input[type=range]{padding:0;accent-color:#7bbfff;width:100%}
    .fl-scan-controls input:invalid{border-color:#ff8989}
    .fl-scan-controls .scan-body{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:5px;align-items:start}
    .fl-scan-controls .scan-toolbar{grid-column:1/-1;display:flex;flex-wrap:wrap;gap:4px;margin-bottom:3px}
    .fl-scan-controls .scan-card{border:1px solid #35465c;border-radius:5px;padding:6px;min-width:0}
    .fl-scan-controls .scan-card:has(.scan-mapping){grid-column:1/-1}
    .fl-scan-controls .scan-card strong{font-size:11px;font-weight:500;line-height:1.2}
    .fl-scan-controls .scan-card>label:empty{display:none}
    .fl-scan-controls.assigning .scan-card[data-target]{border-color:#91ceff;background:#193149}
    .fl-scan-controls .scan-card.drop{outline:2px solid #c7e5ff}
    .fl-scan-controls .scan-mapping{padding:8px;border-left:3px solid #80bdff;background:#111e2d;margin-top:8px}
    .fl-scan-controls label{font-size:11px;color:#bfd0e5}
    `;
    sources.style.cssText=tabs.style.cssText="display:flex;gap:4px;flex-wrap:wrap;margin-bottom:6px";
    body.className="scan-body";
    tabs.setAttribute("role","tablist");
    notice.style.cssText="font-size:11px;color:#ffd08a;line-height:1.5";notice.setAttribute("role","status");
    element.append(style,sources,tabs,notice,body);
    const read=()=>JSON.parse(settings.value),targets=options.scan_mapping_targets;
    const colors=["#70baff","#ed97ca","#f6cc78"],names=["Kick","Snare","Hat"];
    let group=0,armed=null,expanded=-1;
    const btn=(text,parent,action)=>{const b=document.createElement("button");b.textContent=text;b.type="button";b.onclick=action;parent.append(b);return b;};
    function change(){node.graph?.setDirtyCanvas(true,true);onChange();}
    function commit(config){settings.value=JSON.stringify(config);change();mappingPanel.rebuild();}
    function arm(source){armed=source;element.classList.toggle("assigning",source!==null);[...sources.children].forEach((b,i)=>b.setAttribute("aria-pressed",String(i===source)));notice.textContent=source===null?"":`Click a highlighted parameter to assign ${names[source]}. Escape cancels.`;}
    names.forEach((name,i)=>{
        const b=btn(`${i+1} · ${name}`,sources,()=>arm(armed===i?null:i));b.draggable=true;b.style.borderColor=colors[i];
        b.title="Drag onto a highlighted parameter, or click then select a parameter.";
        b.ondragstart=e=>{e.dataTransfer.setData("application/x-fl-envelope",String(i));e.dataTransfer.effectAllowed="copy";arm(i);};
        b.ondragend=()=>arm(null);
    });
    element.onkeydown=e=>{if(e.key==="Escape")arm(null);};
    const entries=Object.entries(EFFECT_GROUPS),titles=["Voxels","Camera","Reveals","Overlay","Finish","Layers"];
    entries.forEach((entry,i)=>{const b=btn(titles[i],tabs,()=>{group=i;rebuild();});b.setAttribute("role","tab");b.onkeydown=e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();group=e.key==='Home'?0:e.key==='End'?entries.length-1:(i+(e.key==='ArrowRight'?1:entries.length-1))%entries.length;rebuild();tabs.children[group].focus();};});
    function meta(key){
        const spec=options.widget_specs?.[key]??{};
        return {default:spec.default,range:spec.min===undefined?null:[spec.min,spec.max],help:spec.tooltip??`${key.replaceAll("_"," ")}.`,choices:options.scan_choices[key],...options.scan_controls?.[key]};
    }
    const widget=key=>node.widgets.find(w=>w.name===key);
    for(const key of Object.values(EFFECT_GROUPS).flat()){
        const w=widget(key);if(w){w.hidden=true;w.type="converted-widget";w.computeSize=()=>[0,0];w.computedHeight=0;if(w.element)w.element.style.display="none";}
    }
    const value=key=>widget(key)?.value??read()[key];
    const linked=key=>node.inputs?.some(i=>i.name===key&&i.link!=null);
    function setValue(key,v){
        const w=widget(key);
        if(w){w.value=v;w.callback?.(v);}
        else{
            const c=read();c[key]=v;
            if(key==='min_cut_frames'||key==='max_cut_frames'){
                normalizeCutRange(c,key);
                for(const name of ['min_cut_frames','max_cut_frames']){
                    for(const input of element.querySelectorAll(`[aria-label="${name.replaceAll('_',' ')}"], [aria-label="${name} slider"]`))input.value=c[name];
                }
            }
            settings.value=JSON.stringify(c);
        }
        change();
    }
    function newMapping(target,source,start=0,end=null){const bounds=targets[target];const key=Object.values(EFFECT_GROUPS).flat().find(k=>mappingTarget(k)===target);const base=Math.max(bounds[0],Math.min(bounds[1],Number(value(key))));return {enabled:true,source,target,minimum:base,maximum:Math.min(bounds[1],base+(bounds[1]-bounds[0])*.1),start_frame:start,end_frame:end,invert:false,smoothing:0};}
    function assign(target,source,host){
        arm(null);const c=read();c.audio_mappings??=[];
        const matches=c.audio_mappings.map((r,i)=>r.target===target?i:-1).filter(i=>i>=0);
        const add=(replace)=>{
            if(replace){const i=matches.includes(expanded)?expanded:matches[0];c.audio_mappings[i].source=source;expanded=i;}
            else{
                let start=0,end=null;const ranges=c.audio_mappings.filter(r=>r.target===target&&r.enabled!==false).sort((a,b)=>(a.start_frame??0)-(b.start_frame??0));
                for(const r of ranges){if((r.start_frame??0)>start){end=r.start_frame;break;}start=Math.max(start,r.end_frame??Infinity);}
                if(!Number.isFinite(start)){notice.textContent="No unused time range. Shorten an existing range before adding another.";return;}
                const count=mappingPanel.frameCount();if(count&&start>=count){notice.textContent="No unused frames remain. Shorten an existing range first.";return;}
                c.audio_mappings.push(newMapping(target,source,start,end));expanded=c.audio_mappings.length-1;
            }
            commit(c);mappingPanel.select(expanded);rebuild();
        };
        if(matches.length){const choice=document.createElement("div");choice.textContent="Existing assignment: ";btn("Replace selected",choice,()=>add(true));btn("Add time range",choice,()=>add(false));btn("Cancel",choice,()=>choice.remove());host.append(choice);}
        else add(false);
    }
    function mappingEditor(row,index,parent){
        const box=document.createElement("div");box.className="scan-mapping";box.style.borderColor=colors[row.source];parent.append(box);
        const grid=document.createElement("div");grid.style.cssText="display:grid;grid-template-columns:1fr 1fr;gap:7px";box.append(grid);
        for(const [key,label] of [["minimum","Minimum"],["maximum","Maximum"],["start_frame","Start frame"],["end_frame","End frame (exclusive)"],["smoothing","Smoothing (seconds)"],["enabled","Enabled"],["invert","Invert"]]){
            const wrap=document.createElement("label"),input=document.createElement("input");wrap.textContent=label;
            const check=["enabled","invert"].includes(key);input.type=check?"checkbox":"number";
            input.style.width=check?"auto":"100%";input.setAttribute("aria-label",`${row.target} mapping ${index+1} ${label}`);
            if(check)input.checked=row[key]??(key==="enabled");else{input.value=row[key]??"";input.placeholder=key==="end_frame"?"Whole clip":"";input.step=key.includes("frame")?1:.01;
                const range=key==="minimum"||key==="maximum"?targets[row.target]:[0,key==="smoothing"?2:2147483647];[input.min,input.max]=range;}
            input.onchange=()=>{if(!input.checkValidity())return;const c=read(),next={...c.audio_mappings[index],[key]:check?input.checked:input.value===""&&key==="end_frame"?null:Number(input.value)};
                const count=mappingPanel.frameCount();
                if((next.end_frame??Infinity)<=next.start_frame||(count&&(next.start_frame>=count||(next.end_frame??count)>count))||mappingConflict(c.audio_mappings,next,index)){notice.textContent="Invalid or overlapping time range. Change was not applied.";input.value=row[key]??"";if(check)input.checked=row[key];return;}
                c.audio_mappings[index]=next;commit(c);notice.textContent="";rebuild();};
            wrap.append(input);grid.append(wrap);
        }
        btn("Reset mapping",box,()=>{const c=read(),old=c.audio_mappings[index];c.audio_mappings[index]={...newMapping(row.target,row.source,old.start_frame,old.end_frame),enabled:old.enabled};commit(c);rebuild();});
        btn("Remove mapping",box,()=>{const c=read();c.audio_mappings.splice(index,1);expanded=-1;commit(c);rebuild();});
        btn("Edit time range",box,()=>{mappingPanel.select(index);mappingPanel.element.scrollIntoView({block:"nearest"});});
    }
    function rebuild(){
        settings.value=JSON.stringify(normalizeCutRange(read()));
        body.replaceChildren();[...tabs.children].forEach((b,i)=>{b.setAttribute("aria-selected",String(i===group));b.tabIndex=i===group?0:-1;});
        const toolbar=document.createElement("div");toolbar.className="scan-toolbar";body.append(toolbar);
        btn("Solo",toolbar,()=>onSolo(["voxel","depth","cursors","edges","finish","layers"][group]));
        btn("All effects",toolbar,()=>onSolo("all"));
        btn("Reset tab values",toolbar,()=>{const confirm=document.createElement("div");confirm.textContent="Restore factory values? Mappings stay unchanged. ";btn("Confirm reset",confirm,()=>{for(const k of entries[group][1])if(!linked(k))setValue(k,meta(k).default);rebuild();});btn("Cancel",confirm,()=>confirm.remove());toolbar.append(confirm);});
        for(const key of entries[group][1]){
            const w=widget(key);if(w){w.hidden=true;w.type="converted-widget";w.computeSize=()=>[0,0];w.computedHeight=0;if(w.element)w.element.style.display="none";}
            const m=meta(key),v=value(key),target=mappingTarget(key),eligible=!!targets[target]&&!linked(key);
            const card=document.createElement("div");card.className="scan-card";if(eligible)card.dataset.target=target;body.append(card);
            if(eligible){card.tabIndex=0;card.setAttribute("aria-label",`Assign envelope to ${key.replaceAll('_',' ')}`);card.onkeydown=e=>{if(armed!==null&&e.target===card&&['Enter',' '].includes(e.key)){e.preventDefault();assign(target,armed,card);}};}
            const head=document.createElement("div");head.style.cssText="display:flex;align-items:center;gap:4px;margin-bottom:4px";card.append(head);
            const label=document.createElement("strong");label.textContent=key.replaceAll("_"," ");label.style.flex=1;head.append(label);
            const help=btn("?",head,()=>{let text=card.querySelector('.scan-help');if(text){text.remove();return;}text=document.createElement('p');text.className='scan-help';text.textContent=m.help;card.append(text);});help.title=m.help;help.setAttribute("aria-label",`Help: ${key}`);
            const reset=btn("↺",head,()=>{setValue(key,m.default);rebuild();});reset.setAttribute("aria-label",`Reset ${key}`);reset.disabled=!!linked(key);reset.title="Reset base value; keep envelope assignments.";
            if(typeof v==="number"){
                const line=document.createElement("div");line.style.cssText="display:flex;gap:8px;align-items:center";card.append(line);
                const range=m.range??targets[target];const step=["seed","surface_seed","cube_size","cursor_count","min_cut_frames","max_cut_frames","stack_count"].includes(key)?1:.01;
                const number=document.createElement("input");number.type="number";number.value=v;number.step=step;number.style.cssText="width:64px;flex-shrink:0;font-size:11px";number.setAttribute("aria-label",key.replaceAll("_"," "));number.disabled=!!linked(key);
                let slider;if(range){[number.min,number.max]=range;number.title=`Range: ${range[0]} to ${range[1]}`;if(!key.includes("seed")){slider=document.createElement("input");slider.type="range";slider.min=range[0];slider.max=range[1];slider.step=step;slider.value=v;slider.disabled=number.disabled;slider.title=number.title;slider.setAttribute("aria-label",`${key} slider`);line.append(slider);}}
                number.oninput=()=>{if(number.value===""||!number.checkValidity())return;setValue(key,Number(number.value));if(slider)slider.value=number.value;};
                if(slider)slider.oninput=()=>{number.value=slider.value;setValue(key,Number(slider.value));};line.append(number);
            }else{const line=document.createElement("div");line.style.cssText="display:flex;gap:4px;flex-wrap:wrap";card.append(line);for(const choice of m.choices??[]){const b=btn(choice.replaceAll("_"," "),line,()=>{setValue(key,choice);rebuild();});b.setAttribute("aria-pressed",String(v===choice));b.disabled=!!linked(key);}}
            if(linked(key)){const text=document.createElement("p");text.textContent="Controlled by connected input.";card.append(text);}
            card.addEventListener("click",e=>{if(armed!==null&&eligible){e.preventDefault();e.stopPropagation();assign(target,armed,card);}},true);
            card.ondragover=e=>{if(eligible&&e.dataTransfer.types.includes("application/x-fl-envelope")){e.preventDefault();card.classList.add("drop");}};
            card.ondragleave=()=>card.classList.remove("drop");card.ondrop=e=>{e.preventDefault();card.classList.remove("drop");const raw=e.dataTransfer.getData("application/x-fl-envelope");if(eligible&&["0","1","2"].includes(raw))assign(target,Number(raw),card);};
            read().audio_mappings?.forEach((row,index)=>{if(row.target!==target)return;const badge=btn(`${names[row.source]} · ${row.enabled===false?"off":"mapped"} · ${row.start_frame??0}–${row.end_frame??"end"}`,card,()=>{expanded=expanded===index?-1:index;mappingPanel.select(index);rebuild();});badge.style.borderColor=colors[row.source];if(expanded===index)mappingEditor(row,index,card);});
            if(eligible&&read().audio_mappings?.some(r=>r.target===target)){const meter=document.createElement("meter"),text=document.createElement("label");meter.hidden=true;meter.min=targets[target][0];meter.max=targets[target][1];meter.dataset.renderTarget=target;meter.style.width="100%";meter.setAttribute("aria-label",`${target} last rendered value`);text.dataset.renderLabel=target;card.append(meter,text);}
        }
    }
    return {element,rebuild,update(preview,frame){element.querySelectorAll('[data-render-target]').forEach(m=>{const v=preview.curves?.[m.dataset.renderTarget]?.[frame];m.hidden=v===undefined;if(v!==undefined)m.value=v;const label=element.querySelector(`[data-render-label="${m.dataset.renderTarget}"]`);label.textContent=v===undefined?"Not rendered":`Last render: ${v.toFixed(3)} · frame ${frame}`;});}};
}
