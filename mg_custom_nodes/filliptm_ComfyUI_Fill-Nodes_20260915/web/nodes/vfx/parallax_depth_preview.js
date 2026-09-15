// Small GPU inverse-warp preview. Textures belong to one editor and are disposed with it.
export function createDepthPreview() {
    const canvas=document.createElement('canvas');
    const gl=canvas.getContext('webgl',{alpha:true,premultipliedAlpha:true,antialias:false});
    if(!gl)return null;
    const compile=(type,source)=>{const shader=gl.createShader(type);gl.shaderSource(shader,source);gl.compileShader(shader);if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(shader));return shader;};
    const vertex=compile(gl.VERTEX_SHADER,'attribute vec2 p; varying vec2 uv; void main(){uv=vec2(p.x*.5+.5,.5-p.y*.5);gl_Position=vec4(p,0.,1.);}');
    const fragment=compile(gl.FRAGMENT_SHADER,`
        precision highp float; varying vec2 uv; uniform sampler2D colorMap,depthMap;
        uniform vec2 sizeScale,offset,travel; uniform float invDepth,push,strength,anchor,invert,opaque;
        vec2 project(vec2 g,float d){return (g*(1.-push*d)+2.*travel*d-2.*offset)/sizeScale;}
        void main(){vec2 g=uv*2.-1.;vec2 q=project(g,invDepth);
            for(int i=0;i<2;i++){float d=texture2D(depthMap,clamp(q*.5+.5,0.,1.)).r;d=mix(d,1.-d,invert);q=project(g,invDepth*(1.+strength*(d-anchor)));}
            vec2 t=q*.5+.5;vec4 c=texture2D(colorMap,clamp(t,0.,1.));
            if(opaque<.5&&(t.x<0.||t.x>1.||t.y<0.||t.y>1.))c=vec4(0.);
            gl_FragColor=c;
        }`);
    const program=gl.createProgram();gl.attachShader(program,vertex);gl.attachShader(program,fragment);gl.linkProgram(program);
    if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));
    gl.deleteShader(vertex);gl.deleteShader(fragment);gl.useProgram(program);
    const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,1,1]),gl.STATIC_DRAW);
    const position=gl.getAttribLocation(program,'p');gl.enableVertexAttribArray(position);gl.vertexAttribPointer(position,2,gl.FLOAT,false,0,0);
    const uniforms=Object.fromEntries(['colorMap','depthMap','sizeScale','offset','travel','invDepth','push','strength','anchor','invert','opaque'].map(k=>[k,gl.getUniformLocation(program,k)]));
    const textures=new Map();
    function upload(unit,data,width,height){const texture=gl.createTexture();gl.activeTexture(gl.TEXTURE0+unit);gl.bindTexture(gl.TEXTURE_2D,texture);gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL,true);
        if(width)gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,width,height,0,gl.RGBA,gl.UNSIGNED_BYTE,data);else gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,data);
        for(const key of [gl.TEXTURE_MIN_FILTER,gl.TEXTURE_MAG_FILTER])gl.texParameteri(gl.TEXTURE_2D,key,gl.LINEAR);
        for(const key of [gl.TEXTURE_WRAP_S,gl.TEXTURE_WRAP_T])gl.texParameteri(gl.TEXTURE_2D,key,gl.CLAMP_TO_EDGE);return texture;}
    return {
        draw(ctx,p,s,value,width,height){
            if(!p.relief||s.relief_scope==='off'||!s.relief_strength||(!p.background&&(s.relief_scope!=='background + artwork'||p.kind==='text')))return false;
            if(canvas.width!==Math.round(width)||canvas.height!==Math.round(height)){canvas.width=Math.round(width);canvas.height=Math.round(height);}
            gl.viewport(0,0,canvas.width,canvas.height);
            let pair=textures.get(p);
            if(!pair){const data=new Uint8Array(32*32*4);p.relief.forEach((d,i)=>{data[i*4]=data[i*4+1]=data[i*4+2]=Math.round(d*255);data[i*4+3]=255;});pair=[upload(0,p.bitmap),upload(1,data,32,32)];textures.set(p,pair);}
            pair.forEach((texture,i)=>{gl.activeTexture(gl.TEXTURE0+i);gl.bindTexture(gl.TEXTURE_2D,texture);});gl.uniform1i(uniforms.colorMap,0);gl.uniform1i(uniforms.depthMap,1);
            const depth=p.background?s.background_depth:p.depth,scale=p.background?s.overscan:p.scale;
            const fit=!p.background&&s.layer_fit==='contain'?Math.min:Math.max,cover=fit(width/p.width,height/p.height);
            gl.uniform2f(uniforms.sizeScale,p.width*cover/width*scale,p.height*cover/height*scale);gl.uniform2f(uniforms.offset,p.background?0:p.offset_x||0,p.background?0:p.offset_y||0);gl.uniform2f(uniforms.travel,value*s.travel_x,value*s.travel_y);
            for(const [key,v] of Object.entries({invDepth:1/depth,push:value*s.push_in,strength:s.relief_strength,anchor:s.relief_anchor,invert:s.depth_invert?1:0,opaque:p.background?1:0}))gl.uniform1f(uniforms[key],v);
            gl.drawArrays(gl.TRIANGLE_STRIP,0,4);ctx.drawImage(canvas,0,0,width,height);return true;
        },
        clear(){for(const pair of textures.values())for(const texture of pair)gl.deleteTexture(texture);textures.clear();},
        dispose(){this.clear();gl.deleteBuffer(buffer);gl.deleteProgram(program);gl.getExtension('WEBGL_lose_context')?.loseContext();}
    };
}
