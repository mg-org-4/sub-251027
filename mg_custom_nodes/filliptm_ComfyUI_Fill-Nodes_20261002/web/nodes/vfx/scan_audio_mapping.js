export function mappingRange(row, count) {
    return [row.start_frame ?? 0, row.end_frame ?? count];
}

export function moveMappingRange(start, end, delta, count) {
    const length = end-start;
    start=Math.max(0,Math.min(count-length,start+delta));
    return [start,start+length];
}

export function createAudioMappingPanel({settings,targets,onChange,onSeek}) {
    const element=document.createElement("div");
    const summary=document.createElement("div");summary.textContent="Envelope timeline · select a parameter badge to edit its range";
    summary.style.cssText="cursor:pointer;margin:10px 0";
    const timeline=document.createElement("canvas");timeline.width=640;timeline.height=150;
    timeline.style.cssText="width:100%;height:150px;background:#0c111a;display:block;touch-action:none";
    timeline.setAttribute("aria-label","Envelope timeline: drag selected range handles, or click to seek");
    const readout=document.createElement("div");readout.style.cssText="margin:6px 0;color:#adbed5";
    element.append(summary,timeline,readout);
    let data=null,frame=0,selected=0,drag=null;
    const read=()=>JSON.parse(settings.value);
    const commit=value=>{settings.value=JSON.stringify(value);onChange();};
    function draw() {
        const ctx=timeline.getContext("2d"),w=timeline.width,h=timeline.height;
        ctx.clearRect(0,0,w,h);
        if(!data){ctx.fillStyle="#acbbd0";ctx.fillText("Run once to load envelopes and shot boundaries",12,28);return;}
        const colors=["#70baff","#ed97ca","#f6cc78"];
        for(let lane=0;lane<3;lane++){
            ctx.strokeStyle=colors[lane];ctx.beginPath();
            data.envelopes[lane].forEach((v,i)=>{const x=i/Math.max(1,data.frames-1)*w,y=38+lane*43-v*27;i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();
            ctx.fillStyle=colors[lane];ctx.fillText(["Kick","Snare","Hat"][lane],5,12+lane*43);
        }
        for(const shot of (data.segments??[]).filter(s=>s.trigger==="authored_shot")){
            const x=shot.start_frame/data.frames*w;ctx.strokeStyle="#52637b";ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();
            ctx.fillStyle="#bbc8da";ctx.fillText(`Shot ${shot.shot}`,x+4,h-5);
        }
        const mapping=read().audio_mappings?.[selected];
        if(mapping){
            const [a,b]=mappingRange(mapping,data.frames),x=a/data.frames*w,right=b/data.frames*w;
            ctx.fillStyle="#ffd67522";ctx.fillRect(x,0,right-x,h);ctx.strokeStyle="#ffd675";ctx.strokeRect(x,1,right-x,h-2);
            ctx.fillStyle="#ffd675";ctx.fillRect(x,0,5,h);ctx.fillRect(right-5,0,5,h);
            const value=data.curves?.[mapping.target]?.[frame];
            readout.textContent=`${mapping.target}: ${value===undefined?"not rendered":value.toFixed(3)} · frame ${frame} · last render. Range [${a}, ${b})`;
        }else readout.textContent=`Frame ${frame} · Add a mapping, then drag its range on the timeline.`;
        ctx.strokeStyle="#ffffff";ctx.beginPath();ctx.moveTo(frame/data.frames*w,0);ctx.lineTo(frame/data.frames*w,h);ctx.stroke();
    }
    function rebuild() {
        const mappings=read().audio_mappings??[];
        selected=Math.min(selected,Math.max(0,mappings.length-1));draw();
    }
    const position=e=>Math.max(0,Math.min(data.frames,Math.round((e.clientX-timeline.getBoundingClientRect().left)/timeline.getBoundingClientRect().width*data.frames)));
    timeline.onpointerdown=e=>{
        if(!data)return;
        const row=read().audio_mappings?.[selected],p=position(e);
        if(row){const [a,b]=mappingRange(row,data.frames),tolerance=Math.max(1,data.frames*8/timeline.getBoundingClientRect().width);
            const mode=Math.abs(p-a)<=tolerance?"start":Math.abs(p-b)<=tolerance?"end":p>a&&p<b?"move":null;
            if(mode){drag={mode,start:a,end:b,anchor:p};timeline.setPointerCapture(e.pointerId);e.preventDefault();return;}}
        onSeek(Math.min(data.frames-1,p));
    };
    timeline.onpointermove=e=>{
        if(!drag)return;
        const p=position(e),config=read(),row=config.audio_mappings[selected];
        let a=drag.start,b=drag.end;
        if(drag.mode==="start")a=Math.min(p,b-1);
        else if(drag.mode==="end")b=Math.max(p,a+1);
        else [a,b]=moveMappingRange(a,b,p-drag.anchor,data.frames);
        if(config.audio_mappings.some((r,i)=>i!==selected&&r.enabled!==false&&row.enabled!==false&&r.target===row.target&&(r.start_frame??0)<b&&a<(r.end_frame??Infinity))){readout.textContent="Range overlaps an existing assignment; move was not applied.";return;}
        row.start_frame=a;row.end_frame=b;commit(config);draw();
    };
    timeline.onpointerup=timeline.onpointercancel=()=>{if(drag){drag=null;rebuild();}};
    return {element,rebuild,select(index){selected=index;draw();},frameCount(){return data?.frames??null;},update(preview,currentFrame){data=preview;frame=currentFrame;draw();}};
}
