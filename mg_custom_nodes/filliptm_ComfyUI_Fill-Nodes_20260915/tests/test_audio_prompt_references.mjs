import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = await readFile(new URL("../web/nodes/audio/audio_prompt_references.js", import.meta.url), "utf8");
const helpers = await import(`data:text/javascript;base64,${Buffer.from(source.slice(source.indexOf("export function ensureReferenceIds"), source.indexOf("export function mountReferences"))).toString("base64")}`);

const storySource = await readFile(new URL("../web/nodes/audio/audio_prompt_storyboards.js", import.meta.url), "utf8");
const storyHelpers = await import(`data:text/javascript;base64,${Buffer.from(storySource.slice(storySource.indexOf("const PREFIX"))).toString("base64")}`);

test("thumbnails represent assigned images only, grouping storyboard panels into one sheet", () => {
  const assets = { a: { kind: "image", storyboard_id: "job", source: { filename: "sheet.png" } },
    b: { kind: "image", storyboard_id: "job" }, unused: { kind: "image" }, audio: { kind: "audio" } };
  const clip = { references: { mode: "custom", asset_ids: ["a", "b", "audio"] } };
  assert.deepEqual(storyHelpers.timelineReferenceImages(clip, assets), [{key: "job", image: assets.a.source, assetIds: ["a", "b"]}]);
  clip.references.mode = "defaults";
  assert.deepEqual(storyHelpers.timelineReferenceImages(clip, assets), []);
  clip.references = {mode: "custom", asset_ids: []};
  assert.deepEqual(storyHelpers.timelineReferenceImages(clip, assets), []);
});

test("reference wiring reuses the schedule library and enables full visual conditioning", () => {
  const graph = { _nodes: [], links: {1: {origin_id: 10}, 2: {origin_id: 10}}, change() {} };
  const library = {id: 20, type: "FL_Prompt_Reference_Library", inputs: [{name: "prompt_schedule", link: 1}],
    connect(output, planner, slot) { graph.links[3] = {origin_id: this.id}; planner.inputs[slot].link = 3; } };
  const planner = {type: "FL_MiniMaxH3BeatShotPlanner", inputs: [{name: "prompt_schedule", link: 2}, {name: "reference_library", link: null}], widgets: [{name: "visual_reference_mode", value: "qwen only"}]};
  graph._nodes = [library, planner];
  const editor = {node: {id: 10, graph}};
  storyHelpers.ensureReferenceWiring(editor);
  storyHelpers.ensureReferenceWiring(editor);
  assert.equal(planner.inputs[1].link, 3);
  assert.equal(planner.widgets[0].value, "full");
  assert.equal(graph._nodes.length, 2);
  assert.throws(() => storyHelpers.ensureReferenceWiring({node: {id: 99, graph}}), /Connect this scheduler/);
});

test("stable IDs survive moves; duplicated clips get new IDs and independent assignments", () => {
  const a = { references: { mode: "custom", asset_ids: ["image"] } };
  helpers.ensureReferenceIds([a]);
  const id = a.sectionId;
  const duplicate = { ...a };
  helpers.ensureReferenceIds([a, duplicate]);
  assert.equal(a.sectionId, id);
  assert.notEqual(duplicate.sectionId, id);
  duplicate.references.asset_ids.push("other");
  assert.deepEqual(a.references.asset_ids, ["image"]);
  helpers.ensureReferenceIds([duplicate, a]);
  assert.equal(a.sectionId, id);
});

test("reference document survives serialization and reload", () => {
  const editor = { clips: [{ prompt: "A" }, { prompt: "B", references: { mode: "none", asset_ids: [] } }], referenceAssets: {}, widgets: { referenceSchedule: {} } };
  helpers.serializeReferences(editor);
  const ids = editor.clips.map(clip => clip.sectionId);
  const restored = { clips: [{}, {}], widgets: editor.widgets };
  helpers.loadReferences(restored);
  assert.deepEqual(restored.clips.map(clip => clip.sectionId), ids);
  assert.equal(restored.clips[1].references.mode, "none");
  restored.clips.pop();
  assert.throws(() => helpers.loadReferences(restored), /no longer match/);
});

test("Partner submission uses scoped credentials without changing the shared API", async () => {
  const source = await readFile(new URL("../web/nodes/audio/audio_prompt_storyboards.js", import.meta.url), "utf8");
  const body = source.slice(source.indexOf("const PREFIX"));
  const bindings = `
    const api = { clientId: 'client', fetchApi() {} };
    const app = { extensionManager: { _p: { _s: new Map([
      ['auth', {getAuthToken: async () => 'fixture-token'}],
      ['apiKeyAuth', {getApiKey: () => undefined}]
    ]) } } };
    class ComfyApi { async queuePrompt(index, prompt) {
      return { authenticated: !!this.authToken, sharedUntouched: !api.authToken,
        isolated: this !== api, index, prompt };
    } }
  `;
  const module = await import(`data:text/javascript;base64,${Buffer.from(bindings + body).toString("base64")}`);
  const submit = await module.prepareStoryboardQueue();
  const result = await submit({ output: "fixture" }, "job");
  assert.equal(result.authenticated, true);
  assert.equal(result.sharedUntouched, true);
  assert.equal(result.isolated, true);
  assert.equal(result.prompt.workflow.extra.storyboard_id, "job");
  assert.equal(JSON.stringify(result.prompt).includes("fixture-token"), false);
});

test("Missing Partner sign-in stops before any queue call", async () => {
  const source = await readFile(new URL("../web/nodes/audio/audio_prompt_storyboards.js", import.meta.url), "utf8");
  const bindings = `
    const api = {};
    const app = { extensionManager: { _p: { _s: new Map([
      ['auth', {getAuthToken: async () => null}], ['apiKeyAuth', {getApiKey: () => null}]
    ]) } } };
    class ComfyApi { queuePrompt() { throw new Error('Must not queue'); } }
  `;
  const module = await import(`data:text/javascript;base64,${Buffer.from(bindings + source.slice(source.indexOf("const PREFIX"))).toString("base64")}`);
  await assert.rejects(module.prepareStoryboardQueue(), /Sign in/);
});

test("Writer generation queues directly, attaches completed panels, and rerolls without clearing active references", async () => {
  class Element {
    children = [];
    dataset = {};
    style = {};
    selectors = new Map();
    append(...items) { this.children.push(...items); }
    after() {}
    remove() {}
    setAttribute(key,value) { this[key]=value; }
    replaceChildren(...items) { this.children = items; }
    querySelector(selector) {
      if (!this.selectors.has(selector)) this.selectors.set(selector, new Element());
      return this.selectors.get(selector);
    }
  }
  let poll;
  let queued = 0;
  let extracted = 0;
  const jobs = [];
  const parent = new Element();
  const clip = {sectionId: "section", start: 0, end: 121, prompt: "Fast motion", references: {mode: "defaults", asset_ids: []}};
  const graph = {links: {1: {origin_id: 10}, 2: {origin_id: 10}, 3: {origin_id: 20}}, change() {}};
  graph._nodes = [
    {id: 20, type: "FL_Prompt_Reference_Library", inputs: [{name: "prompt_schedule", link: 1}]},
    {type: "FL_MiniMaxH3BeatShotPlanner", inputs: [{name: "prompt_schedule", link: 2}, {name: "reference_library", link: 3}], widgets: []},
  ];
  const editor = {node: {id: 10, graph}, clips: [clip], referenceAssets: {}, canvas: {parentElement: parent},
    clipRects: [{index: 0, x: 10, y: 20, width: 250, height: 200}],
    runEdit(label, fn) { fn(); }, serialize() {}, syncInspector() {},
    scheduleDraw() { this.renderStoryboardThumbnails?.(); }};
  const errors = [];
  const writer = {editor, nodeSettings: {schedulerId: "schedule", moodboards: [null, null, null, null]},
    root: new Element(), saveNodeSettings() {}, showError(message) { errors.push(message); },
    client: {imageUrl(image) { return image.filename; }},
    currentDocument: {allowed_indices: [0], boxes: [{index: 0, start_frame: 0, end_frame: 121, prompt: clip.prompt}],
      sectionIds: {0: "section"}, referenceSelections: {0: structuredClone(clip.references)}}};
  const api = {clientId: "fixture", apiURL:path=>path, async fetchApi(path, options) {
    const body = options?.body ? JSON.parse(options.body) : null;
    let result;
    if (path.includes("?scheduler_id=")) result = jobs;
    else if (path.endsWith("/storyboards")) {
      result = jobs.find(job => job.spec.request_key === body.request_key);
      if (!result) { result = {id: String(jobs.length + 1), state: "proposed", spec: body, result: {}}; jobs.unshift(result); }
    } else {
      const [, id, action] = path.match(/\/([^/]+)\/([^/]+)$/);
      const job = jobs.find(job => job.id === id);
      if (id === "batch" && action === "submit") { result={graph:{}};for(const id of body.job_ids){jobs.find(j=>j.id===id).state="submitted";result.graph[id]={class_type:"GeminiNanoBanana2V2"};} }
      else if (action === "submit") { job.state = "submitted"; result = {graph: {storyboard: {class_type: "GeminiNanoBanana2V2"}}}; }
      else if (action === "receipt") { job.result.prompt_id = body.prompt_id; result = job; }
      else if (action === "refresh") result = job;
      else if (action === "extract") {
        extracted++;
        job.result.assets = Object.fromEntries(Array.from({length: 4}, (_, index) => [`${id}-${index}`, {
          kind: "image", filename: `panel-${id}-${index}.png`, storyboard_id: id, source: job.result.source,
        }]));
        result = job;
      } else throw Error(`Unexpected action ${action}`);
    }
    return {ok: true, json: async () => structuredClone(result)};
  }};
  globalThis.__storyboardTest = {
    api, app: {extensionManager: {_p: {_s: new Map([
      ["auth", {getAuthToken: async () => "fixture"}], ["apiKeyAuth", {getApiKey: () => null}],
    ])}}},
    ComfyApi: class { async queuePrompt() { queued++; return {prompt_id: `queue-${queued}`}; } },
    document: {createElement: () => new Element(), createTextNode: value => value},
    setTimeout: callback => { poll = callback; return 1; }, clearTimeout() {},
  };
  const bindings = "const {api, app, ComfyApi, document, setTimeout, clearTimeout} = globalThis.__storyboardTest;\n";
  const module = await import(`data:text/javascript;base64,${Buffer.from(bindings + storySource.slice(storySource.indexOf("const PREFIX"))).toString("base64")}`);
  const flush = async () => { for (let i = 0; i < 8; i++) await new Promise(resolve => setImmediate(resolve)); };
  const descendants = root => (root.children||[]).flatMap(child=>[child,...descendants(child)]);
  try {
    module.mountStoryboards(writer);
    await flush();
    await writer.generateStoryboards([{index: 0, grid: 2, prompt: "Four running poses"}], [], "message");
    assert.equal(queued, 1);
    assert.equal(clip.references.mode, "defaults");
    jobs[0].state = "complete";
    jobs[0].result.source = {filename: "sheet.png"};
    await poll();
    assert.equal(extracted, 1);
    assert.equal(clip.references.asset_ids.length, 4);
    assert.equal(module.timelineReferenceImages(clip, editor.referenceAssets).length, 1);
    const original = structuredClone(clip.references);
    const row = parent.children[0].children[0];
    await descendants(row).find(child => child.title?.startsWith("Reroll")).onclick();
    assert.equal(queued, 2);
    assert.deepEqual(clip.references, original);
    jobs[0].state = "complete";
    jobs[0].result.source = {filename: "reroll.png"};
    await poll();
    assert.notDeepEqual(clip.references, original);
    assert.equal(module.timelineReferenceImages(clip, editor.referenceAssets)[0].image.filename, "reroll.png");
    descendants(parent.children[0].children[0]).find(child => child.title?.startsWith("Remove")).onclick();
    await poll();
    assert.deepEqual(clip.references.asset_ids, []);
    assert.equal(queued, 2);
    writer.currentDocument.referenceSelections[0] = structuredClone(clip.references);
    await writer.generateStoryboards([{index: 0, grid: 2, prompt: "New storyboard"}], [], "next-message");
    assert.equal(queued, 3);
    clip.prompt = "User edited the section during generation";
    jobs[0].state = "complete";
    jobs[0].result.source = {filename: "stale.png"};
    await poll();
    assert.deepEqual(clip.references.asset_ids, []);
    assert.equal(extracted, 2);
    assert.equal(writer.nodeSettings.storyboardResults.includes(jobs[0].id), false, "edited sections must not consume saved results");
    const savedJob = jobs[0];
    writer.disposeStoryboards();
    parent.replaceChildren();
    module.mountStoryboards(writer);
    await flush();
    await poll();
    const attach = descendants(parent).find(child => child.textContent === "Attach");
    assert.ok(attach, "unattached saved images remain recoverable after reopening");
    await attach.onclick();
    assert.equal(extracted, 3);
    assert.equal(queued, 3, "recovering a saved image never queues a paid generation");
    assert.equal(clip.prompt, "User edited the section during generation");
    assert.equal(clip.references.asset_ids.length, 4);
    assert.ok(writer.nodeSettings.storyboardResults.includes(savedJob.id));
    await poll();
    assert.equal(extracted, 3, "attached results are not repeatedly extracted");
    editor.clips=Array.from({length:4},(_,i)=>({sectionId:`batch-${i}`,start:i*24,end:(i+1)*24,prompt:`Scene ${i}`,references:{mode:"defaults",asset_ids:[]}}));
    writer.currentDocument={allowed_indices:[0,1,2,3],boxes:editor.clips.map((c,index)=>({index,start_frame:c.start,end_frame:c.end,prompt:c.prompt})),sectionIds:Object.fromEntries(editor.clips.map((c,i)=>[i,c.sectionId])),referenceSelections:Object.fromEntries(editor.clips.map((c,i)=>[i,structuredClone(c.references)]))};
    await writer.generateStoryboards(editor.clips.map((c,index)=>({index,grid:2,prompt:c.prompt})),[],"batch-message");
    assert.equal(queued,4,"all four images use one independent async graph");
    const batchJobs=jobs.filter(j=>j.spec.section_id.startsWith('batch-'));
    assert.equal(batchJobs.length,4);
    assert.equal(new Set(batchJobs.map(j=>j.spec.continuity)).size,1);
    assert.match(batchJobs[0].spec.continuity,/Scene 0/);
    assert.match(batchJobs[0].spec.continuity,/Scene 3/);
    assert.deepEqual(errors, []);
  } finally {
    writer.disposeStoryboards();
    delete globalThis.__storyboardTest;
  }
});
