import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import test from "node:test";
const source=await readFile(new URL("../web/nodes/vfx/parallax_preview.js",import.meta.url),"utf8");
const {cameraValue,plateRect,viewport,CAMERA_PRESETS}=await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);
const settings={width:640,height:400,motion:"glide",travel_x:.3,travel_y:0,push_in:0,background_depth:12,overscan:1,layer_fit:"cover"};
const plate={width:640,height:400,depth:2,scale:1,offset_x:0,offset_y:0,opacity:1};
test("camera curves match render anchors",()=>{
    assert.equal(cameraValue(.25,"loop"),1);
    assert.equal(cameraValue(.5,"glide"),0);
    assert.equal(cameraValue(0,"glide"),-1);
    assert.equal(cameraValue(1,"glide"),1);
    for(const [t,v] of [[0,-1],[.1,-1],[.28,.4],[.45,.4],[.61,-.3],[.73,-.3],[.92,1],[1,1]])assert.ok(Math.abs(cameraValue(t,"bursts")-v)<1e-12);
    assert.equal(cameraValue(.3,"locked"),0);
});
test("near plates move twice as much at half the depth",()=>{
    const near=plateRect(plate,settings,1,640,400),far=plateRect({...plate,depth:4},settings,1,640,400);
    assert.equal(near.x,2*far.x);
    assert.equal(near.x,-96);
});
test("neutral projection and contain preserve aspect",()=>{
    assert.deepEqual(plateRect(plate,settings,.5,640,400),{x:0,y:0,w:640,h:400});
    const result=plateRect({...plate,width:400},{...settings,layer_fit:"contain"},.5,640,400);
    assert.deepEqual(result,{x:120,y:0,w:400,h:400});
    assert.deepEqual(viewport(480,360,.75),{x:105,y:0,w:270,h:360});
});
test("background uses camera framing, presets only change motion",()=>{
    const p=plateRect({...plate,background:true},{...settings,overscan:1.2},.5,640,400);
    assert.equal(p.w,768);
    assert.equal(p.h,480);
    for(const preset of Object.values(CAMERA_PRESETS))assert.deepEqual(Object.keys(preset),["motion","travel_x","travel_y","push_in"]);
});
