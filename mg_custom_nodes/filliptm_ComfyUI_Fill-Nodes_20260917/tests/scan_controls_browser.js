export async function testControls() {
    const {createScanControls}=await import(new URL('/extensions/ComfyUI_Fill-Nodes/nodes/vfx/scan_controls.js',location.origin));
    const {createAudioMappingPanel}=await import(new URL('/extensions/ComfyUI_Fill-Nodes/nodes/vfx/scan_audio_mapping.js',location.origin));
    const info=(await (await fetch('/object_info/FL_InteractiveScanFX')).json()).FL_InteractiveScanFX.input.required;
    const settings={value:info.advanced_settings[1].default};
    const node={widgets:Object.entries(info).filter(([k,v])=>v[1]?.default!==undefined&&k!=='advanced_settings').map(([name,v])=>({name,value:v[1].default,options:v[1]})),inputs:[],graph:{setDirtyCanvas(){}}};
    const panel=createAudioMappingPanel({settings,targets:info.advanced_settings[1].scan_mapping_targets,onChange(){},onSeek(){}});
    panel.update({frames:192,envelopes:Array.from({length:3},()=>Array(192).fill(0)),segments:[],curves:{}},0);
    const ui=createScanControls({node,settings,options:{...info.advanced_settings[1],widget_specs:Object.fromEntries(Object.entries(info).map(([k,v])=>[k,v[1]]))},onChange(){},mappingPanel:panel,onSolo(){}});
    const host=document.createElement('div');host.style.display='none';host.append(ui.element,panel.element);document.body.append(host);ui.rebuild();
    const check=(ok,msg)=>{if(!ok)throw Error(msg)};
    const button=(text,root=ui.element)=>[...root.querySelectorAll('button')].find(b=>b.textContent===text);
    const input=(name,value)=>{const e=ui.element.querySelector(`input[aria-label="${name}"]`);e.value=value;e.dispatchEvent(new Event('change'));};
    const results=[];
    try {
        [...ui.element.querySelectorAll('button')].find(b=>b.textContent.startsWith('1 ')).click();ui.element.querySelector('[data-target=animation]').click();
        check(JSON.parse(settings.value).audio_mappings[0].target==='animation','click assignment');results.push('click assignment');
        input('animation mapping 1 End frame (exclusive)','48');
        const transfer=new DataTransfer();transfer.setData('application/x-fl-envelope','2');
        ui.element.querySelector('[data-target=animation]').dispatchEvent(new DragEvent('drop',{dataTransfer:transfer,bubbles:true,cancelable:true}));button('Add time range').click();
        check(JSON.parse(settings.value).audio_mappings[1].start_frame===48,'gap assignment');results.push('drop and non-overlapping range');
        input('animation mapping 2 Start frame','47');
        check(JSON.parse(settings.value).audio_mappings[1].start_frame===48,'overlap rejection');results.push('overlap rejection');
        button('Remove mapping').click();check(JSON.parse(settings.value).audio_mappings.length===1,'remove');
        const slider=ui.element.querySelector('input[aria-label="animation slider"]');slider.value='.5';slider.dispatchEvent(new Event('input'));check(node.widgets.find(w=>w.name==='animation').value===.5,'slider');
        ui.element.querySelector('[aria-label="Reset animation"]').click();check(node.widgets.find(w=>w.name==='animation').value===.18,'reset');check(JSON.parse(settings.value).audio_mappings.length===1,'reset retained mapping');results.push('slider, reset, removal');
        for(const title of ['Camera','Reveals','Overlay','Finish','Layers']){button(title).click();check(ui.element.querySelectorAll('.scan-card').length>0,'tab '+title);}results.push('all tabs');
        const spacing=ui.element.querySelector('[data-target=stack_spacing]');
        spacing.dispatchEvent(new DragEvent('drop',{dataTransfer:transfer,bubbles:true,cancelable:true}));
        check(JSON.parse(settings.value).audio_mappings.some(r=>r.target==='stack_spacing'&&r.source===2),'stack envelope mapping');
        const stack=ui.element.querySelector('input[aria-label="stack count"]');stack.value='6';stack.dispatchEvent(new Event('input'));
        check(JSON.parse(settings.value).stack_count===6,'stack count');
        stack.value='2.5';stack.dispatchEvent(new Event('input'));check(JSON.parse(settings.value).stack_count===6,'integer validation');
        button('random on snare').click();check(JSON.parse(settings.value).window_order==='random_on_snare','window order');
        check(ui.element.querySelector('[aria-label="Help: window_order"]').title.includes('Envelope 2'),'ordering help');results.push('layer controls and drag mapping');
        button('Overlay').click();
        const minimum=ui.element.querySelector('input[aria-label="min cut frames"]');minimum.value='40';minimum.dispatchEvent(new Event('input'));
        check(JSON.parse(settings.value).max_cut_frames===40,'maximum follows minimum');
        check(ui.element.querySelector('input[aria-label="max cut frames"]').value==='40','maximum field synchronized');
        const maximum=ui.element.querySelector('input[aria-label="max_cut_frames slider"]');maximum.value='5';maximum.dispatchEvent(new Event('input'));
        check(JSON.parse(settings.value).min_cut_frames===5&&minimum.value==='5','minimum follows maximum');results.push('paired cut controls');
        return results;
    } finally {host.remove();}
}
