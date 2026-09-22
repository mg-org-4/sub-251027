import { app } from "../../scripts/app.js";

const NODE_CLASS = "H3ContinuumSamplerV38";
const MODE = "video_reference_mode";
const REPEAT = "Repeat Reference";
const FOLLOW = "Follow Timeline";

export function restoreVideoReferenceMode(node, info) {
    const widget = node.widgets?.find((item) => item.name === MODE);
    if (!widget) return;
    const linked = info?.inputs?.some((input) => (
        (input.name === MODE || input.widget?.name === MODE) && input.link != null
    ));
    if (linked) return; // The connected input, not a hidden widget, owns its value.

    let value;
    const named = info?.widgets_values_named;
    if (named && Object.prototype.hasOwnProperty.call(named, MODE)) {
        value = named[MODE];
    } else if (!named) {
        // The existing V38X2 backend widgets occupy slots 0..29. This new
        // optional scalar is appended; no upload/UI/transient row is inserted.
        const values = info?.widgets_values;
        if (Array.isArray(values) && values.length > 30) value = values[30];
    }
    // In particular A(Follow)->B(old workflow) must not inherit A's mode.
    widget.value = value === FOLLOW ? FOLLOW : REPEAT;
    node.__h3ContinuumIntuitiveUxRefresh?.();
}

app.registerExtension({
    name: "H3Continuum.TimelineVideoModeExperimental",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_CLASS && nodeType?.prototype?.comfyClass !== NODE_CLASS) return;
        const previous = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(info, ...args) {
            const result = previous?.call(this, info, ...args);
            restoreVideoReferenceMode(this, info);
            return result;
        };
    },
});
