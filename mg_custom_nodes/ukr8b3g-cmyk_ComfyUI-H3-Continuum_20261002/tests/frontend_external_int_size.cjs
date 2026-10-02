// Reuse the existing real-production-JS harness; do not copy UI predicates.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ROOT = process.env.H3_TEST_ROOT || path.resolve(__dirname, '..');
const harness = fs.readFileSync(path.join(__dirname, 'frontend_review_queue.cjs'), 'utf8');
const marker = harness.indexOf('(async()=>{');
assert(marker > 0);
const scope = {require, process, console, __dirname, setTimeout, clearTimeout, setImmediate, structuredClone};
vm.runInNewContext(harness.slice(0, marker) + '\nglobalThis.fixtures={environment,project};', scope);
const {environment, project} = scope.fixtures;
const results = [];
const widget = (e,n,name) => e.w(n,name);
const sources = ['width','height'];
const slot = (name, link=null) => ({name,type:'INT',widget:{name},link});
function readyInputs(e,n) {
  for (const name of ['model','clip','video_vae','sampler','sigmas']) n.inputs.push({name,link:100});
}
async function test(name, body) {
  const e = environment();
  try { await body(e); results.push({name,pass:true}); }
  catch(error) { results.push({name,pass:false,error:error.stack}); }
  finally { e.close(); }
}
(async()=>{
for (const nodeClass of ['H3ContinuumSamplerV38','H3ContinuumSamplerV39']) {
  await test(`${nodeClass}: current slots bind to dimension and duration Facades`, async e=>{
    const n=e.makeNode(312,'fixture',nodeClass);
    for (const [name,label] of [['width','Width'],['height','Height'],['chunks','Chunks'],['chunk_seconds','Seconds per Chunk']]) {
      const s=slot(name); n.inputs.push(s); e.f.configureNode(n);
      assert.equal(s._widget,e.w(n,label));
      assert.equal(n.getWidgetFromSlot(s),e.w(n,label));
      assert.equal(n.getSlotFromWidget(e.w(n,label)),s);
      assert.equal(n.getSlotFromWidget(e.w(n,name)),s);
    }
    const getter=n.getWidgetFromSlot; e.f.configureNode(n); assert.equal(n.getWidgetFromSlot,getter);
    assert.equal(n.inputs.length,4);
  });
  await test(`${nodeClass}: restored replacement slots rebind without duplicate sockets`, async e=>{
    const n=e.makeNode(312,'fixture',nodeClass); n.inputs.push(slot('width'),slot('height'));
    e.f.configureNode(n); const old=n.inputs;
    n.inputs=[slot('width',41),{name:'height',type:'INT',link:42}];
    e.f.configureNode(n); e.f.configureNode(n);
    for (const [i,label] of ['Width','Height'].entries()) {
      assert.notEqual(n.inputs[i],old[i]); assert.equal(n.inputs[i]._widget,e.w(n,label));
      assert.equal(n.getWidgetFromSlot(n.inputs[i]),e.w(n,label));
      assert.equal(n.getSlotFromWidget(e.w(n,label)),n.inputs[i]);
    }
    assert.equal(n.inputs.length,2);
  });
  for (const linkedAxes of [['width'],['height'],sources]) {
    await test(`${nodeClass}: linked ${linkedAxes.join('+')} reserve survives First Image preview`, async e=>{
      const n=e.makeNode(312,'fixture',nodeClass); e.w(n,'width').value=640; e.w(n,'height').value=768;
      n.inputs.push(...sources.map((name,i)=>slot(name,linkedAxes.includes(name)?41+i:null)),{name:'first_frame',link:45});
      e.app.graph._nodes.push({id:900,mode:0,imgs:[{naturalWidth:1920,naturalHeight:1080}]});
      e.app.graph.links[45]={origin_id:900,origin_slot:0};
      e.w(n,'size_source').value='First Image'; e.f.configureNode(n);
      assert.equal(e.w(n,'size_source').value,'First Image');
      for(const name of linkedAxes) assert.equal(e.w(n,name).value,name==='width'?640:768);
      e.w(n,'Size Source').callback('Manual'); e.f.configureNode(n);
      for(const name of sources) {
        const facade=e.w(n,name==='width'?'Width':'Height');
        assert.equal(facade.disabled,linkedAxes.includes(name));
        if(linkedAxes.includes(name)) { facade.callback(1024); assert.equal(e.w(n,name).value,name==='width'?640:768); }
      }
      for(const s of n.inputs) if(linkedAxes.includes(s.name))s.link=null;
      e.f.configureNode(n);
      for(const name of linkedAxes) {
        const facade=e.w(n,name==='width'?'Width':'Height'); assert.equal(facade.disabled,false);
        assert.equal(facade.value,name==='width'?640:768); facade.callback(1024); assert.equal(e.w(n,name).value,1024);
      }
    });
  }
  for (const linkedAxes of [[],['width'],['height'],sources]) {
    await test(`${nodeClass}: Manual summary and API adapter ${linkedAxes.join('+')||'native'}`, async e=>{
      const n=e.makeNode(312,'fixture',nodeClass); readyInputs(e,n);
      e.w(n,'run_storage').value='Off'; e.w(n,'width').value=640; e.w(n,'height').value=768;
      n.inputs.push(...sources.map((name,i)=>slot(name,linkedAxes.includes(name)?41+i:null)));
      e.f.configureNode(n);
      const summary=e.w(n,'Ready to Queue').detail;
      assert.equal(e.w(n,'Ready to Queue').headline,'Ready to Queue');
      if(linkedAxes.length===2) assert.match(summary,/Width.*Height.*connected/i);
      else if(linkedAxes.length===1) {
        assert.match(summary,new RegExp(`${linkedAxes[0]}.*connected`,'i'));
        assert.match(summary,new RegExp(linkedAxes[0]==='width'?'Height 768':'Width 640'));
      } else assert.match(summary,/640.*768/);
      const inputs=await e.inputs(n);
      for(const name of linkedAxes)inputs[name]=[name==='width'?'901':'902',0];
      const response=await e.api.queuePrompt(0,{output:{[n.id]:{class_type:nodeClass,inputs}},workflow:n.serialize()});
      const actual=e.submissions.at(-1).data.output[n.id].inputs;
      for(const name of sources) assert.deepEqual(actual[name],linkedAxes.includes(name)?[name==='width'?'901':'902',0]:name==='width'?640:768);
      assert(!Object.hasOwn(actual,'Width')); assert(!Object.hasOwn(actual,'Height'));
      await e.emit('execution_success',{prompt_id:response.prompt_id});
      e.w(n,'Size Source').callback('First Image'); e.f.configureNode(n);
      const fallback=e.w(n,'Ready to Queue').detail;
      if(linkedAxes.length) assert.match(fallback,/Fallback.*connected/i);
      else assert.match(fallback,/640.*768/);
    });
  }
  await test(`${nodeClass}: native reserve and links survive recreated workflow lifecycle`, async e=>{
    const n=e.makeNode(312,'fixture',nodeClass); e.w(n,'width').value=640; e.w(n,'height').value=768;
    n.inputs.push(slot('width',41),slot('height',42)); e.f.configureNode(n);
    const saved=n.serialize(), savedInputs=JSON.parse(JSON.stringify(n.inputs,(key,value)=>key==='_widget'?undefined:value));
    const nativeCount=saved.widgets_values.length;
    for(let i=0;i<5;i++) {
      e.app.extension.beforeConfigureGraph?.(); e.app.graph._nodes=[];
      const restored=e.makeNode(312,'fixture',nodeClass,false); e.app.extension.nodeCreated(restored);
      restored.configure(saved); restored.inputs=structuredClone(savedInputs);
      e.app.extension.loadedGraphNode(restored); await e.finishRestore();
      assert.equal(e.w(restored,'width').value,640); assert.equal(e.w(restored,'height').value,768);
      assert.equal(restored.serialize().widgets_values.length,nativeCount);
      assert.deepEqual(restored.inputs.map(s=>s.link),[41,42]);
      assert.equal(restored.inputs[0]._widget,e.w(restored,'Width')); assert.equal(restored.inputs[1]._widget,e.w(restored,'Height'));
    }
  });
  await test(`${nodeClass}: external INT edits retain Review safety guard`, async e=>{
    const n=e.makeNode(312,'fixture',nodeClass);
    const upstream={id:901,comfyClass:'PrimitiveInt',mode:0,widgets:[{name:'value',value:640}],inputs:[]};
    e.app.graph._nodes.push(upstream); e.app.graph.links[41]={origin_id:901,origin_slot:0};
    n.inputs.push(slot('width',41)); e.w(n,'generation_mode').value='Review Each Chunk'; e.f.configureNode(n);
    await e.load(n,project(1)); assert(e.visible(n,'Use it and continue'));
    upstream.widgets[0].value=672; n.__h3ContinuumProductionUxRefresh?.();
    assert(e.f.reviewSettingsChanged(n)); assert(!e.visible(n,'Use it and continue'));
    upstream.widgets[0].value=640; n.__h3ContinuumProductionUxRefresh?.();
    assert(!e.f.reviewSettingsChanged(n)); assert(e.visible(n,'Use it and continue'));
  });
}
console.log(JSON.stringify(results,null,2)); if(results.some(r=>!r.pass))process.exitCode=1;
})();
