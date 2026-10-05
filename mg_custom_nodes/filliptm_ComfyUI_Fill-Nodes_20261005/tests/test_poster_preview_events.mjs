import test from 'node:test';
import assert from 'node:assert/strict';
import {ownsPosterEvent,posterLayerId,replacePreviewUrl} from '../web/nodes/vfx/poster_preview_events.js';

test('routes expanded nodes without matching unrelated node prefixes',()=>{
    assert.equal(ownsPosterEvent(106,{node:'106.0.0.layer_2_asset'}),true);
    assert.equal(ownsPosterEvent(106,{displayNodeId:'106',realNodeId:'106.0.0.layer_2_sample'}),true);
    assert.equal(ownsPosterEvent(106,{node:'1060.0.layer_2_sample'}),false);
    assert.equal(ownsPosterEvent(106,{node:null}),false);
    assert.equal(ownsPosterEvent(106,{node_id:'106.0.0.layer_2_sample'}),true);
});
test('identifies sampling, decoding and completed layers',()=>{
    for(const stage of ['sample','decode','asset'])assert.equal(posterLayerId({node:'106.0.0.layer_12_'+stage}),'layer_12');
    assert.equal(posterLayerId({node:106}),null);
});
test('releases replaced previews and releases final URL on cleanup',()=>{
    const revoked=[],state={url:null};let i=0;
    const urls={createObjectURL:()=>`blob:${++i}`,revokeObjectURL:url=>revoked.push(url)};
    assert.equal(replacePreviewUrl(state,{},urls),'blob:1');
    assert.equal(replacePreviewUrl(state,{},urls),'blob:2');
    assert.equal(replacePreviewUrl(state,null,urls),null);
    assert.deepEqual(revoked,['blob:1','blob:2']);
});
