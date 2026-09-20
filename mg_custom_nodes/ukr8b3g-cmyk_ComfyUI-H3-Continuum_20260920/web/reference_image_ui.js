import { app } from "../../scripts/app.js";

const BUNDLE_NODE = "H3ContinuumReferenceImages";
const BUNDLE_INPUT = "image_references";
const BUNDLE_TYPE = "H3_CONTINUUM_IMAGE_REFERENCES";

export function migrateReferenceImageInputs(node) {
    if (node.comfyClass !== "H3ContinuumSamplerV38") return;
    const graph = node.graph || app.graph;
    const legacy = (node.inputs || []).filter(
        (input) => ["reference_image_4", "reference_image_5"].includes(input.name),
    );
    if (!node.inputs?.some((input) => input.name === BUNDLE_INPUT)) {
        node.addInput(BUNDLE_INPUT, BUNDLE_TYPE, { label: "Reference Images (Optional)" });
    }
    if (!legacy.length) return;

    const connected = legacy.filter((input) => input.link != null);
    const bundleSlot = node.inputs.findIndex((input) => input.name === BUNDLE_INPUT);
    const bundleLink = graph.links[node.inputs[bundleSlot].link];
    let bundle = bundleLink && graph.getNodeById(bundleLink.origin_id);
    if (connected.length) {
        const sources = connected.map((input) => {
            const link = graph.links[input.link];
            const source = link && graph.getNodeById(link.origin_id);
            return { input, link, source };
        });
        if (sources.some(({ source }) => !source)
            || (bundleLink && bundle?.comfyClass !== BUNDLE_NODE)) {
            console.warn("H3 Continuum: legacy Reference Image links retained; their source or bundle cannot be migrated.");
            return;
        }
        if (bundle && connected.some((input) => {
            const slot = bundle.inputs.find((item) => item.name === input.name);
            return !slot || slot.link != null;
        })) {
            console.warn("H3 Continuum: legacy Reference Image links retained; the matching bundle slots are occupied.");
            return;
        }
        if (!bundle) {
            bundle = globalThis.LiteGraph?.createNode(BUNDLE_NODE);
            if (!bundle) {
                console.warn("H3 Continuum: legacy Reference Image links retained; the Reference Images node is unavailable.");
                return;
            }
            bundle.pos = [node.pos[0] - 360, node.pos[1] + 220];
            graph.add(bundle);
            bundle.connect(0, node, bundleSlot);
            if (node.inputs[bundleSlot].link == null) {
                graph.remove(bundle);
                console.warn("H3 Continuum: legacy Reference Image links retained; bundle connection failed.");
                return;
            }
        }
        for (const { input, link, source } of sources) {
            const slot = bundle.inputs.findIndex((item) => item.name === input.name);
            source.connect(link.origin_slot, bundle, slot);
            const moved = graph.links[bundle.inputs[slot].link];
            if (!moved || moved.origin_id !== link.origin_id || moved.origin_slot !== link.origin_slot) {
                console.warn("H3 Continuum: legacy Reference Image link retained; image connection failed.");
                continue;
            }
            node.removeInput(node.inputs.indexOf(input));
        }
    }
    for (const input of legacy) {
        const slot = node.inputs.indexOf(input);
        if (slot >= 0 && input.link == null) node.removeInput(slot);
    }
    node.setDirtyCanvas?.(true, true);
}
