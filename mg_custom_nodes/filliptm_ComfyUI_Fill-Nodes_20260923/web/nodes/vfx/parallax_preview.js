export const CAMERA_PRESETS = {
    Gentle: {motion: "loop", travel_x: .06, travel_y: .015, push_in: .02},
    Reveal: {motion: "glide", travel_x: .22, travel_y: .025, push_in: .04},
    Punchy: {motion: "bursts", travel_x: .35, travel_y: .035, push_in: .06},
    Still: {motion: "locked", travel_x: 0, travel_y: 0, push_in: 0},
};

export function cameraValue(t, motion) {
    if (motion === "locked") return 0;
    if (motion === "loop") return Math.sin(t * Math.PI * 2);
    const smooth = u => u * u * (3 - 2 * u);
    if (motion === "glide") return 2 * smooth(t) - 1;
    const keys = [[0,-1],[.1,-1],[.28,.4],[.45,.4],[.61,-.3],[.73,-.3],[.92,1],[1,1]];
    for (let i = 1; i < keys.length; i++) {
        const [a, x] = keys[i-1], [b, y] = keys[i];
        if (t <= b) return x + (y-x) * smooth((t-a)/(b-a));
    }
    return 1;
}

export function plateRect(p, s, phase, width, height) {
    const depth = p.background ? s.background_depth : p.depth;
    const value = cameraValue(phase, s.motion);
    const zoom = depth / (depth - value * s.push_in);
    const scale = p.background ? s.overscan : p.scale;
    const fit = !p.background && s.layer_fit === "contain" ? Math.min : Math.max;
    const cover = fit(width / p.width, height / p.height);
    const w = p.width * cover * scale * zoom, h = p.height * cover * scale * zoom;
    return {x: (width-w)/2 + ((p.offset_x || 0)-value*s.travel_x/depth)*zoom*width,
        y: (height-h)/2 + ((p.offset_y || 0)-value*s.travel_y/depth)*zoom*height, w, h};
}

export function viewport(width, height, aspect) {
    const w = Math.min(width, height * aspect), h = w / aspect;
    return {x: (width-w)/2, y: (height-h)/2, w, h};
}

function borderImage(ctx, image, r, width, height) {
    const iw = image.width, ih = image.height;
    // Match the renderer's clamped background edges, not invented scenery.
    if (r.x > 0) ctx.drawImage(image, 0, 0, 1, ih, 0, r.y, r.x, r.h);
    if (r.y > 0) ctx.drawImage(image, 0, 0, iw, 1, r.x, 0, r.w, r.y);
    if (r.x+r.w < width) ctx.drawImage(image, iw-1, 0, 1, ih, r.x+r.w, r.y, width-r.x-r.w, r.h);
    if (r.y+r.h < height) ctx.drawImage(image, 0, ih-1, iw, 1, r.x, r.y+r.h, r.w, height-r.y-r.h);
    for (const [sx, dx, dw] of [[0,0,r.x],[iw-1,r.x+r.w,width-r.x-r.w]])
        for (const [sy, dy, dh] of [[0,0,r.y],[ih-1,r.y+r.h,height-r.y-r.h]])
            if (dw > 0 && dh > 0) ctx.drawImage(image,sx,sy,1,1,dx,dy,dw,dh);
}

export function drawScene(canvas, plates, s, phase, depthView=false, relief=null) {
    const ctx = canvas.getContext("2d"), w = canvas.width, h = canvas.height;
    ctx.clearRect(0,0,w,h); ctx.fillStyle="#0b111c"; ctx.fillRect(0,0,w,h);
    if (depthView) {
        const ordered = [...plates].reverse();
        ordered.forEach((p,i) => {
            const depth = p.background ? s.background_depth : p.depth;
            const x = 18 + depth / s.background_depth * (w-170), y = 30 + i * Math.min(32, (h-60)/ordered.length);
            ctx.fillStyle = `hsl(${190+i*17} 40% 24%)`; ctx.fillRect(x,y,145,24);
            ctx.fillStyle="#e4efff"; ctx.font="12px system-ui"; ctx.fillText(`${p.name.slice(0,15)} · ${depth}`,x+5,y+16);
        });
        ctx.fillStyle="#91a6be"; ctx.font="12px system-ui"; ctx.fillText("NEAR / FASTER",12,h-12); ctx.fillText("FAR / SLOWER",w-110,h-12);
        return;
    }
    const v = viewport(w,h,s.width/s.height);
    ctx.save(); ctx.translate(v.x,v.y); ctx.beginPath(); ctx.rect(0,0,v.w,v.h); ctx.clip();
    for (const p of plates) {
        if (!p.bitmap) continue;
        const r = plateRect(p,s,phase,v.w,v.h);
        ctx.globalAlpha = p.background ? 1 : p.opacity;
        if(relief?.draw(ctx,p,s,cameraValue(phase,s.motion),v.w,v.h))continue;
        if (p.background) borderImage(ctx,p.bitmap,r,v.w,v.h);
        ctx.drawImage(p.bitmap,r.x,r.y,r.w,r.h);
    }
    ctx.restore(); ctx.strokeStyle="#597088"; ctx.strokeRect(v.x+.5,v.y+.5,v.w-1,v.h-1);
}

export function demoLayers(kind) {
    const make = (name, depth, draw, background=false) => {
        const bitmap = document.createElement("canvas"); bitmap.width=320; bitmap.height=200;
        const context=bitmap.getContext("2d");context.scale(.5,.5);draw(context);
        return {name, depth, bitmap, width:640, height:400, scale:1.04, offset_x:0, offset_y:0, opacity:1, background};
    };
    const background=make("Background",12,c=>{c.fillStyle=kind==="design"?"#efe8da":"#182c47";c.fillRect(0,0,640,400);},true);
    if (kind === "design") return [background,
        make("Color field",9,c=>{c.fillStyle="#df7154";c.beginPath();c.arc(360,225,160,0,7);c.fill();}),
        make("Typography",5,c=>{c.fillStyle="#152a3c";c.font="bold 85px sans-serif";c.fillText("MOVE",45,145);c.fillText("IN DEPTH",45,235);}),
        make("Accent",2,c=>{c.fillStyle="#e3bc36";c.fillRect(430,270,145,60);})];
    if (kind === "product") return [background,
        make("Backdrop",9,c=>{c.fillStyle="#426275";c.beginPath();c.ellipse(320,290,230,65,0,0,7);c.fill();}),
        make("Object",4,c=>{c.fillStyle="#efa86b";c.fillRect(220,110,180,180);c.fillStyle="#b97349";c.beginPath();c.moveTo(400,110);c.lineTo(455,70);c.lineTo(455,245);c.lineTo(400,290);c.fill();c.fillStyle="#ffcf8e";c.beginPath();c.moveTo(220,110);c.lineTo(270,70);c.lineTo(455,70);c.lineTo(400,110);c.fill();}),
        make("Foreground",2,c=>{c.fillStyle="#82c6bc";c.beginPath();c.arc(150,285,50,0,7);c.fill();})];
    return [background,
        make("Distant peaks",9,c=>{c.fillStyle="#607d94";c.beginPath();c.moveTo(0,300);c.lineTo(140,120);c.lineTo(265,280);c.lineTo(440,80);c.lineTo(640,310);c.lineTo(640,400);c.lineTo(0,400);c.fill();}),
        make("Middle ground",5,c=>{c.fillStyle="#315d60";c.beginPath();c.moveTo(0,310);c.quadraticCurveTo(280,180,640,300);c.lineTo(640,400);c.lineTo(0,400);c.fill();}),
        make("Near foliage",2,c=>{c.fillStyle="#91b18c";for(const [x,y] of [[20,335],[75,370],[570,350],[635,320]]){c.beginPath();c.ellipse(x,y,80,130,-.3,0,7);c.fill();}})];
}
