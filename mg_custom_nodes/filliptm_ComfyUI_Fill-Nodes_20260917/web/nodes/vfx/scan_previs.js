export const EFFECT_GROUPS = {
    "Voxel surface": ["cube_size", "relief", "animation", "speed", "surface_seed"],
    "Depth camera": ["motion_mode", "parallax_scope", "orbit_degrees", "depth_relief", "scene_scale", "parallax_strength", "offset_x", "offset_y", "dolly", "steady_depth"],
    "Cursor reveals": ["cursor_count", "cursor_scale", "cursor_activity", "reveal_size", "reveal_strength", "voxel_weight", "edge_weight", "depth_weight", "depth_style", "seed"],
    "Digital overlay": ["normal_mix", "hud_opacity", "pose_opacity", "motion_strength", "min_cut_frames", "max_cut_frames"],
    "Color & glow": ["base_brightness", "brightness_intensity", "base_saturation", "saturation_intensity", "edge_threshold", "glow_intensity", "envelope_intensity", "glow_color", "blend_mode"],
    "Layer stack": ["stack_count", "stack_spacing", "stack_x", "stack_y", "stack_rotation", "stack_opacity", "stack_palette", "window_order", "window_blend", "voxel_opacity", "edge_opacity", "depth_opacity", "window_fade_in", "window_fade_out"],
};

// Small synthetic surface, not model output. Coordinates are normalized to the canvas.
export function demoPoint(x, y, depth, s, time) {
    const native=s.motion_mode==="depth_parallax", strength=native?s.parallax_strength:0;
    const z=1.5+(1-depth)*s.depth_relief, pivot=1.5+(1-s.steady_depth)*s.depth_relief;
    const angle=Math.sin(time*.7)*s.orbit_degrees*Math.PI/180;
    const px=(x-.5)*Math.cos(angle)+(z-pivot)*Math.sin(angle)*.3;
    const zoom=(pivot-strength*s.dolly)/pivot*z/Math.max(.1,z-strength*s.dolly);
    return [.5+(px+strength*s.offset_x*(1/z-1/pivot))*s.scene_scale*zoom,
        .5+((y-.5)+strength*s.offset_y*(1/z-1/pivot))*s.scene_scale*zoom];
}

export function drawPrevis(canvas, s, time, view, buffer = canvas.ownerDocument.createElement("canvas")) {
    let ctx=canvas.getContext("2d");
    const w=canvas.width, h=canvas.height;
    if(buffer.width!==w)buffer.width=w;
    if(buffer.height!==h)buffer.height=h;
    ctx.clearRect(0,0,w,h);ctx.fillStyle="#080f1c";ctx.fillRect(0,0,w,h);
    const grid=Math.max(8,Math.round(s.cube_size/640*w)), cells=Math.ceil(w/grid);
    const pulse=Math.pow(Math.max(0,Math.cos(time*Math.PI*4)),8);
    const surface=(kind,clip=false)=>{
        const destination=ctx;
        ctx=buffer.getContext("2d");ctx.clearRect(0,0,w,h);
        for(let row=0;row<cells;row++)for(let col=0;col<cells;col++){
            const x=(col+.5)/cells,y=(row+.5)/cells;
            const radius=Math.hypot((x-.5)*1.4,(y-.5)*1.05),depth=Math.max(0,1-radius*2.3);
            const cameraSettings=s.parallax_scope==="reveals_only"&&!clip&&kind!=="depth"?{...s,motion_mode:"current"}:s;
            const [px,py]=demoPoint(x,y,depth,cameraSettings,time).map((v,i)=>v*(i?h:w));
            const voxel=kind==="voxel"||(clip&&kind==="all");
            const lift=voxel?depth*s.relief*18+Math.sin(time*s.speed*3+x*9+y*8+s.surface_seed)*s.animation*12:0;
            if(kind==="depth")ctx.fillStyle=s.depth_style==="grayscale"?`rgb(${depth*255} ${depth*255} ${depth*255})`:
                s.depth_style==="contours"?`hsl(190 80% ${Math.floor(depth*12)%2?60:15}%)`:`hsl(${(1-depth)*250} 80% 55%)`;
            else ctx.fillStyle=voxel?`hsl(${210+x*110-y*80} 65% ${30+depth*35}%)`:`hsl(${215-depth*45+s.normal_mix*x*120} ${45+s.normal_mix*35}% ${15+depth*65}%)`;
            if(voxel){ctx.fillStyle="#172940";ctx.fillRect(px-grid/2,py-lift,grid*.93,grid*.85+lift);ctx.fillStyle=`hsl(${210+x*110-y*80} 65% ${30+depth*35}%)`;}
            ctx.fillRect(px-grid/2,py-lift,grid*(voxel?.86:1.04),grid*(voxel?.72:1.04));
        }
        ctx=destination;ctx.save();
        if(kind==="all"||kind==="finish")ctx.filter=`brightness(${Math.max(0,s.base_brightness+s.brightness_intensity*pulse)}) saturate(${Math.max(0,s.base_saturation+s.saturation_intensity*pulse)})`;
        ctx.drawImage(buffer,0,0);ctx.restore();
    };
    ctx.save();
    if(view==="all"){ctx.translate(Math.sin(time*4)*s.motion_strength*4,Math.cos(time*3)*s.motion_strength*3);}
    if(view==="all"||view==="layers"){
        const step=Math.max(2,Math.round(w*(.009+.004*s.depth_relief)))*(s.stack_spacing??1);
        const color=({cobalt:"#0905e6",cyan:"#05b3d9",magenta:"#cc0599",mono:"#a6a6b3"})[s.stack_palette??"cobalt"];
        for(let level=s.stack_count??4;level>0;level--){
            ctx.save();ctx.globalAlpha=s.stack_opacity??1;
            ctx.translate(w/2+level*step*(s.stack_x??1),h/2+level*step*(s.stack_y??1)*(.65+.2*Math.sin(time)));
            ctx.rotate(-level*(s.stack_rotation??0)*Math.PI/180);
            ctx.fillStyle=level%2?"#0a0a0e":color;ctx.strokeStyle="#d9e6ff";
            ctx.fillRect(-w*s.scene_scale/2,-h*s.scene_scale/2,w*s.scene_scale,h*s.scene_scale);
            ctx.strokeRect(-w*s.scene_scale/2,-h*s.scene_scale/2,w*s.scene_scale,h*s.scene_scale);ctx.restore();
        }
    }
    surface(view);
    if(view==="all"||view==="cursors"||view==="layers"){
        const count=Math.min(3,s.cursor_count),period=Math.max(.4,(s.min_cut_frames+s.max_cut_frames)/48);
        const windows=[];
        for(let i=0;i<count;i++){
            const phase=(time/period+i*.31)%1;if(phase>s.cursor_activity)continue;
            const x=(.12+i*.22+Math.sin(s.seed+i)*.06)*w,y=(.18+i*.17)*h;
            const rw=w*.28*s.reveal_size*Math.min(1,phase*3),rh=h*.32*s.reveal_size*Math.min(1,phase*3);
            const total=s.voxel_weight+s.edge_weight+s.depth_weight;
            const pick=((Math.sin(s.seed+i*17+Math.floor(time/period))*43758.5453)%1+1)%1*total;
            const kind=pick<s.voxel_weight?"voxel":pick<s.voxel_weight+s.edge_weight?"edges":"depth";
            windows.push({i,phase,x,y,rw,rh,kind});
        }
        const priority=({voxel_on_top:"voxel",edge_on_top:"edges",depth_on_top:"depth"})[s.window_order];
        if(priority)windows.sort((a,b)=>Number(a.kind===priority)-Number(b.kind===priority));
        if(s.window_order==="random_on_snare")windows.sort((a,b)=>Math.sin(s.seed+a.i*79+Math.floor(time*2)*31)-Math.sin(s.seed+b.i*79+Math.floor(time*2)*31));
        for(const {phase,x,y,rw,rh,kind} of windows){
            const opacity=s[({voxel:"voxel_opacity",edges:"edge_opacity",depth:"depth_opacity"})[kind]]??1;
            const fadeIn=s.window_fade_in?Math.min(1,Math.max(0,phase-.12)*period/s.window_fade_in):1;
            const fadeOut=s.window_fade_out?Math.min(1,(1-phase)*period/s.window_fade_out):1;
            ctx.save();ctx.globalAlpha=s.reveal_strength*opacity*fadeIn*fadeOut;
            ctx.globalCompositeOperation=({normal:"source-over",screen:"screen",add:"lighter"})[s.window_blend??"normal"];
            ctx.beginPath();ctx.rect(x,y,rw,rh);ctx.clip();
            if(kind==="edges"){ctx.strokeStyle="#48f2ff";for(let j=0;j<10;j++)ctx.strokeRect(w*(.2+j*.03),h*(.15+j*.03),w*.3,h*.5);}
            else surface(kind,true);
            ctx.restore();ctx.strokeStyle=kind==="depth"?"#ffd36b":"#78f4ed";ctx.strokeRect(x,y,rw,rh);
            ctx.save();ctx.translate(x+rw,y+rh);ctx.scale(s.cursor_scale,s.cursor_scale);ctx.fillStyle="#fff";ctx.strokeStyle="#0c1726";
            ctx.beginPath();ctx.moveTo(0,0);ctx.lineTo(2,19);ctx.lineTo(7,13);ctx.lineTo(13,20);ctx.lineTo(17,17);ctx.lineTo(11,10);ctx.lineTo(19,8);ctx.closePath();ctx.fill();ctx.stroke();ctx.restore();
        }
    }
    if(view==="all"||view==="edges"){
        ctx.save();ctx.globalAlpha=s.hud_opacity;ctx.strokeStyle="#9ef66e";ctx.strokeRect(w*.3,h*.18,w*.4,h*.62);
        ctx.fillStyle="#9ef66e";ctx.font="12px system-ui";ctx.fillText("SYNTHETIC DEPTH SUBJECT",w*.3,h*.16);
        ctx.globalAlpha=s.pose_opacity;ctx.beginPath();ctx.moveTo(w*.5,h*.3);ctx.lineTo(w*.5,h*.58);ctx.lineTo(w*.38,h*.77);ctx.moveTo(w*.5,h*.58);ctx.lineTo(w*.62,h*.77);ctx.stroke();ctx.restore();
    }
    if(view==="all"||view==="finish"){
        ctx.save();ctx.globalAlpha=Math.min(.8,(s.glow_intensity+s.envelope_intensity*pulse)*.35)*(1-s.edge_threshold);
        ctx.globalCompositeOperation=s.blend_mode==="add"?"lighter":s.blend_mode;
        ctx.shadowBlur=20;ctx.shadowColor=s.glow_color==="original"?"#75d8ff":s.glow_color;
        ctx.strokeStyle=ctx.shadowColor;ctx.lineWidth=3;ctx.strokeRect(w*.28,h*.2,w*.44,h*.6);ctx.restore();
    }
    ctx.restore();
}

export function createPrevis(readSettings) {
    const element=document.createElement("div"), tabs=document.createElement("div"), canvas=document.createElement("canvas"), note=document.createElement("p");
    element.className="fl-scan-previs";canvas.width=320;canvas.height=320;
    const buffer=document.createElement("canvas");
    canvas.style.cssText="width:100%;display:block;background:#080f1c";
    tabs.style.cssText="display:flex;gap:4px;flex-wrap:wrap;margin-bottom:8px";
    let view="all",time=1,playing=true,active=true,visible=false,raf=null,last=0;
    const buttons=[];
    const redraw=()=>{drawPrevis(canvas,readSettings(),time,view,buffer);buttons.forEach(([button,key])=>button.style.background=key===view?"#345d91":"#293345");};
    for(const [key,label] of [["all","All"],["voxel","Voxels"],["depth","Depth"],["cursors","Reveals"],["edges","Overlay"],["finish","Finish"],["layers","Layers"]]){
        const b=document.createElement("button");b.textContent=label;b.style.cssText="color:white;border:1px solid #46536c;border-radius:4px;padding:5px";
        b.onclick=()=>{view=key;redraw();};buttons.push([b,key]);tabs.append(b);
    }
    const pause=document.createElement("button");pause.textContent="Pause demo";pause.onclick=()=>{playing=!playing;pause.textContent=playing?"Pause demo":"Play demo";schedule();};
    note.textContent="LIVE DEMO · Synthetic scene, approximate effects. No model or queue required. Audio pulses are illustrative; mappings, tracking, cut timing and occlusion must be checked in the render.";
    note.style.cssText="color:#bdcce1;font-size:11px;line-height:1.5";
    element.append(tabs,canvas,pause,note);
    function tick(now){
        raf=null;
        if(now-last>=1000/12){time+=last?Math.min(.15,(now-last)/1000):0;last=now;redraw();}
        schedule();
    }
    function schedule(){
        if(active&&visible&&playing&&!document.hidden){if(raf===null)raf=requestAnimationFrame(tick);}
        else {if(raf!==null)cancelAnimationFrame(raf);raf=null;last=0;}
    }
    const observer=new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;schedule();});observer.observe(element);
    document.addEventListener("visibilitychange",schedule);
    return {element,redraw,select(key){view=key;redraw();},setActive(value){active=value;element.hidden=!value;schedule();},dispose(){active=false;schedule();observer.disconnect();document.removeEventListener("visibilitychange",schedule);}};
}
