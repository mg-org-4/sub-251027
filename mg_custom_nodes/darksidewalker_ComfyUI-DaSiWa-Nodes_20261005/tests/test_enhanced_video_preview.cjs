const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const file = path.join(__dirname, '..', 'js', 'enhanced_video_combine_preview.js');
const source = fs.readFileSync(file, 'utf8').replace(/^import .*;\s*$/gm, '');
let extension;
vm.runInNewContext(source, {
    app: { registerExtension: (value) => { extension = value; } },
    api: { apiURL: (value) => value },
    URLSearchParams,
    Set,
});
assert.ok(extension);
for (const name of ['DaSiWa_EnhancedVideoCombine']) {
    class Node {}
    extension.beforeRegisterNodeDef(Node, { name, input: { required: {} } });
    assert.equal(typeof Node.prototype.onNodeCreated, 'function', `${name} missing preview`);
    assert.equal(typeof Node.prototype.onExecuted, 'function', `${name} missing video update`);
}
console.log('Preview registered on regular combine node');
