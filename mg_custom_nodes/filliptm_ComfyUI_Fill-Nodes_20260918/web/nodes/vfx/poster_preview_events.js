export function ownsPosterEvent(nodeId, event) {
    const id=String(nodeId);
    return [event.node, event.node_id, event.nodeId, event.display_node, event.displayNodeId, event.parentNodeId, event.realNodeId]
        .some(value=>value!=null&&(String(value)===id||String(value).startsWith(id+'.')));
}

export function posterLayerId(event) {
    for(const value of [event.realNodeId,event.nodeId,event.node]){
        const match=String(value??'').match(/(?:^|\.)((?:layer_)\d+)_(?:sample|decode|asset)$/);
        if(match)return match[1];
    }
    return null;
}

export function replacePreviewUrl(state, blob, urls=URL) {
    if(state.url)urls.revokeObjectURL(state.url);
    state.url=blob?urls.createObjectURL(blob):null;
    return state.url;
}
