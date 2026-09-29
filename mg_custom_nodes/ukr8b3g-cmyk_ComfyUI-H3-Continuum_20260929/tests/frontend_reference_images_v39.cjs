const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = process.env.H3_TEST_ROOT || path.resolve(__dirname, '..');
const source = fs.readFileSync(path.join(root, 'web/reference_images_v39.js'), 'utf8')
    .replace('export function configureV39ReferenceImages', 'function configureV39ReferenceImages')
    .replace('export function refreshV39ReferenceImagesForSampler', 'function refreshV39ReferenceImagesForSampler')
    .replace('export function connectedV39LegacyReferenceInputs', 'function connectedV39LegacyReferenceInputs')
    .replace('export function pruneV39LegacyReferenceInputs', 'function pruneV39LegacyReferenceInputs');
function dom(tag) {
    return {
        tag, children: [], style: {}, textContent: '',
        append(...items) { this.children.push(...items); },
        replaceChildren(...items) { this.children = items; },
        addEventListener() {}, setAttribute(name, value) { this[name] = value; },
    };
}
const graph = { links: {}, nodes: new Map(), getNodeById(id) { return this.nodes.get(id); } };
const sandbox = { document: { createElement: dom }, app: { graph }, Set, Array, Number, JSON };
vm.runInNewContext(source + '\nglobalThis.v39={configureV39ReferenceImages,refreshV39ReferenceImagesForSampler,connectedV39LegacyReferenceInputs,pruneV39LegacyReferenceInputs};', sandbox);
const {
    configureV39ReferenceImages, refreshV39ReferenceImagesForSampler,
    connectedV39LegacyReferenceInputs, pruneV39LegacyReferenceInputs,
} = sandbox.v39;
const selectors = Array.from({ length: 9 }, (_, i) => ({
    name: `reference_r${i + 1}_chunks`, value: 'off', type: 'text', options: {},
}));
const mode = { name: 'reference_use', value: 'All chunks', type: 'combo', options: {} };
const helper = {
    id: 1, graph, comfyClass: 'H3ContinuumReferenceImagesV39',
    size: [440, 500],
    widgets: [mode, ...selectors],
    inputs: Array.from({ length: 9 }, (_, i) => ({ name: `reference_image_${i + 1}`, link: i === 0 || i === 3 || i === 8 ? i + 100 : null })),
    outputs: [{ name: 'reference_images', links: [10] }],
    addDOMWidget(_name, _type, host, options) { this.host = host; this.panel = host.children[0]; this.domOptions = options; this.display = { options: {} }; return this.display; },
    computeSize() { return [this.size[0], 310 + this.display.computeSize(this.size[0])[1]]; },
    setSize(size) { this.size = size; },
    setDirtyCanvas() {},
};
const chunks = { name: 'chunks', value: 6 };
const sampler = {
    id: 2, graph, comfyClass: 'H3ContinuumSamplerV39',
    widgets: [chunks], inputs: [{ name: 'reference_images', link: 10 }],
    __h3ContinuumReferencePlan: 'verified', setDirtyCanvas() {},
};
graph.nodes.set(1, helper); graph.nodes.set(2, sampler);
graph.links[10] = { origin_id: 1, target_id: 2 };
configureV39ReferenceImages(helper);
assert.equal(helper.domOptions.serialize, false);
assert.equal(selectors[0].options.serialize, undefined);
assert.equal(selectors[0].type, 'hidden');
assert.equal(selectors[0].hidden, true);
assert.equal(selectors[0].computeSize()[1], 0);
assert.equal(typeof selectors[0].draw, 'function');
assert.match(helper.host.style.cssText, /padding-top:8px/);
assert.match(helper.panel.children[0].textContent, /all chunks/i);
mode.value = 'Per chunk'; mode.callback('Per chunk');
assert.equal(helper.display.computeSize(440)[1], 193);
assert.equal(helper.panel.style.height, '185px');
assert.equal(helper.size[1], 503, 'three-row assignment panel must size the node');
const visit = (item, predicate) => {
    if (predicate(item)) return item;
    for (const child of item.children || []) {
        const found = visit(child, predicate);
        if (found) return found;
    }
    return null;
};
function click(slot, chunk) {
    const box = visit(helper.panel, (item) => item.tag === 'input'
        && item['aria-label'] === `Image ${slot}, Chunk ${chunk}`);
    assert(box, `missing Image ${slot} / Chunk ${chunk}`);
    box.checked = !box.checked; box.onchange();
}
for (const chunk of [1, 3, 6]) click(1, chunk);
assert.equal(selectors[0].value, '1,3,6');
assert.equal(sampler.__h3ContinuumReferencePlan, null);
chunks.value = 3; refreshV39ReferenceImagesForSampler(sampler);
assert.match(helper.panel.children.find((item) => item.textContent.startsWith('Image 1 →')).textContent, /saved beyond range: 6/);
assert.equal(selectors[0].value, '1,3,6');
chunks.value = 6; refreshV39ReferenceImagesForSampler(sampler);
assert.equal(selectors[0].value, '1,3,6');
mode.value = 'All chunks'; mode.callback('All chunks');
assert.equal(helper.display.computeSize(440)[1], 62);
assert.equal(helper.size[1], 372, 'switching to All chunks must remove unused panel space');
mode.value = 'Per chunk'; mode.callback('Per chunk');
assert.equal(selectors[0].value, '1,3,6');
helper.inputs.forEach((input, index) => { input.link = index + 100; });
chunks.value = 16;
helper.__h3ReferenceImagesRefresh();
assert.equal(helper.display.computeSize(440)[1], 368);
assert.equal(helper.panel.style.height, '360px');
const oldSampler = {
    comfyClass: 'H3ContinuumSamplerV39',
    __h3ContinuumReferencePlan: 'old plan',
    inputs: [
        { name: 'reference_image_1', link: null },
        { name: 'reference_image_2', link: 22 },
        { name: 'image_references', link: null },
        { name: 'reference_images', link: 10 },
    ],
    removeInput(index) { this.inputs.splice(index, 1); },
    setDirtyCanvas() { this.dirty = true; },
};
assert.equal(pruneV39LegacyReferenceInputs(oldSampler), 2);
assert.deepEqual(oldSampler.inputs.map((input) => input.name), ['reference_image_2', 'reference_images']);
assert.equal(oldSampler.__h3ContinuumReferencePlan, null);
assert.equal(oldSampler.dirty, true);
assert.equal(connectedV39LegacyReferenceInputs(oldSampler).join(','), 'reference_image_2');
assert.equal(pruneV39LegacyReferenceInputs(oldSampler), 0);
oldSampler.inputs[0].link = null;
assert.equal(pruneV39LegacyReferenceInputs(oldSampler), 1);
assert.deepEqual(oldSampler.inputs.map((input) => input.name), ['reference_images']);
assert.equal(pruneV39LegacyReferenceInputs({ ...oldSampler, comfyClass: 'H3ContinuumSamplerV38' }), 0);
console.log('V3.9 Reference Images UI contract PASS');
