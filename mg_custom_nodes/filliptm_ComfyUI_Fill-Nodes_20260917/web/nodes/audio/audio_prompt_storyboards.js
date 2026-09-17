import { api, ComfyApi } from "../../../../scripts/api.js";
import { app } from "../../../../scripts/app.js";

const PREFIX = "/fl/audio-prompt-timeline/storyboards";

export function storyboardThumbnailUrl(image) {
  return api.apiURL(`${PREFIX}/thumbnail?${new URLSearchParams({filename:image.filename,subfolder:image.subfolder||"",type:image.type||"input"})}`);
}

export function storyboardContinuity(actions) {
  const brief = "These sections belong to ONE continuous production. Keep recurring characters, costumes, visual style and environments coherent. Do not invent a new design for each section.\n"
    + actions.map((action, index) => `Section ${index+1}: ${action.prompt}`).join("\n\n");
  if (brief.length > 32000) throw new Error("The shared storyboard brief is too long. Generate a smaller group of sections.");
  return brief;
}

export async function prepareStoryboardQueue() {
  const stores = app.extensionManager._p._s;
  const auth = stores.get("auth");
  const keyAuth = stores.get("apiKeyAuth");
  if (!auth || !keyAuth) throw new Error("ComfyUI Partner authentication is not ready. Refresh ComfyUI and sign in.");
  const authToken = await auth.getAuthToken();
  const apiKey = keyAuth.getApiKey();
  if (!authToken && !apiKey) throw new Error("Sign in to ComfyUI before generating a Partner Node image.");
  // Match ComfyApp's queue credentials without modifying the shared API or active graph.
  const submission = { clientId: api.clientId, authToken, apiKey, fetchApi: api.fetchApi.bind(api) };
  return (graph, jobId) => ComfyApi.prototype.queuePrompt.call(submission, 0, {
    output: graph, workflow: { nodes: [], links: [], extra: { storyboard_id: jobId } },
  });
}

async function request(path = "", body) {
  const response = await api.fetchApi(PREFIX + path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "Storyboard request failed.");
  return result;
}

export function ensureReferenceWiring(editor) {
  const node = editor.node;
  const graph = node.graph;
  if (!graph) throw new Error("The scheduler must belong to a workflow before using references.");
  const planners = graph._nodes.filter(target => target.type === "FL_MiniMaxH3BeatShotPlanner"
    && target.inputs.some(input => input.name === "prompt_schedule" && graph.links[input.link]?.origin_id === node.id));
  if (!planners.length) throw new Error("Connect this scheduler directly to an H3 Beat Shot Planner before generating references.");
  let library = graph._nodes.find(target => target.type === "FL_Prompt_Reference_Library"
    && target.inputs.some(input => input.name === "prompt_schedule" && graph.links[input.link]?.origin_id === node.id));
  if (!library) {
    library = LiteGraph.createNode("FL_Prompt_Reference_Library");
    if (!library) throw new Error("Restart ComfyUI to load FL Prompt Reference Library.");
    library.pos = [node.pos[0] + node.size[0] + 40, node.pos[1] + 120];
    graph.add(library);
    node.connect(0, library, 0);
  }
  for (const planner of planners) {
    const slot = planner.inputs.findIndex(input => input.name === "reference_library");
    if (slot < 0) throw new Error("Refresh ComfyUI to update the planner's reference input.");
    const input = planner.inputs[slot];
    if (graph.links[input.link]?.origin_id !== library.id) library.connect(0, planner, slot);
    const mode = planner.widgets.find(widget => widget.name === "visual_reference_mode");
    if (mode && mode.value !== "full") { mode.value = "full"; mode.callback?.(mode.value); }
  }
  graph.change();
}

export function timelineReferenceImages(clip, assets) {
  if (clip.references?.mode !== "custom") return [];
  const groups = new Map();
  for (const id of clip.references.asset_ids) {
    const asset = assets[id];
    if (asset?.kind !== "image") continue;
    const key = asset.storyboard_id || id;
    if (!groups.has(key)) groups.set(key, { key, image: asset.source || asset, assetIds: [] });
    groups.get(key).assetIds.push(id);
  }
  return [...groups.values()];
}

export function mountStoryboards(writer) {
  const editor = writer.editor;
  const boards = writer.nodeSettings.moodboards;
  const root = document.createElement("section");
  root.className = "flbps-storyboards";
  root.innerHTML = `<div class="flbps-moodboard-slots"></div>
    <small>Generate sections concurrently with shared character and style references. All requested images may spend ComfyUI credits at once; provider limits still apply. Checked moodboards are sent to Google. Rerolls retain their original references.</small>
    <div data-story="status" role="status"></div>`;
  writer.root.querySelector(".flbps-writer-topbar").after(root);
  const slots = root.querySelector(".flbps-moodboard-slots");
  const status = root.querySelector('[data-story="status"]');
  const error = task => Promise.resolve().then(task).catch(value => { status.textContent = value.message; writer.showError(value.message); });
  const busy = new Set();
  const attachmentErrors = new Map();
  let jobs = [];
  let disposed = false;
  let timer;
  let checking = false;
  let renderKey = "";
  const handled = new Set(writer.nodeSettings.storyboardResults || []);
  const overlay = document.createElement("div");
  overlay.className = "flbps-timeline-references";
  editor.canvas.parentElement.append(overlay);
  overlay.onwheel = event => editor.onWheel(event);
  let viewer = null;
  function openPreview(image, metadata = {}) {
    viewer?.close();
    const focus = document.activeElement;
    const dialog = document.createElement("dialog");viewer=dialog;
    dialog.className="flbps-image-dialog";dialog.setAttribute("aria-label","Storyboard image and metadata");
    const visual=document.createElement("div"),details=document.createElement("aside"),full=document.createElement("img"),close=document.createElement("button");
    visual.className="flbps-image-view loading";full.alt=metadata.title||"Reference image";
    const loading=document.createElement("span");loading.className="flbps-image-loading";loading.textContent="Loading original…";
    full.onload=()=>{visual.className="flbps-image-view";loading.remove();field("Dimensions",`${full.naturalWidth} × ${full.naturalHeight}`);};
    full.onerror=()=>{visual.className="flbps-image-view";loading.textContent="Original image unavailable.";};
    close.textContent="Close ×";close.className="flbps-image-close";close.onclick=()=>dialog.close();
    function field(label,value){if(value===undefined||value===null||value==="")return;const h=document.createElement("h4"),p=document.createElement("p");h.textContent=label;p.textContent=String(value);details.append(h,p);}
    const title=document.createElement("h2");title.textContent=metadata.title||"Reference image";details.append(title,close);
    field("File",image.filename);field("Location",`${image.type||"input"}/${image.subfolder||""}`);
    field("Section",metadata.section);field("State",metadata.job?.state);field("Model",metadata.job?"Nano Banana 2 (Gemini 3.1 Flash Image)":null);
    const spec=metadata.job?.spec;field("Grid",spec?`${spec.grid} × ${spec.grid}`:null);field("Requested size",spec?`${spec.resolution} · ${spec.aspect_ratio}`:null);
    field("Prompt",spec?.prompt);field("Continuity brief",spec?.continuity);field("Reference roles",spec?.moodboards?.map((b,i)=>`${i+1}. ${b.role||b.filename}`).join("\n"));field("Role",metadata.role);
    visual.append(loading,full);dialog.append(visual,details);document.body.append(dialog);
    dialog.onclick=e=>{if(e.target===dialog)dialog.close();};dialog.onclose=()=>{full.removeAttribute("src");dialog.remove();if(viewer===dialog)viewer=null;if(focus?.isConnected)focus.focus();};
    dialog.onkeydown=e=>e.stopPropagation();
    full.src=writer.client.imageUrl(image,false);dialog.showModal();close.focus();
  }
  function thumbnail(image, label, click) {
    const card=document.createElement("div");card.className="flbps-image-card loading";
    const img=document.createElement("img");img.alt=label;img.loading="lazy";img.decoding="async";img.draggable=false;img.tabIndex=0;img.setAttribute("role","button");
    const state=document.createElement("span");state.className="flbps-image-loading";state.textContent="Loading…";
    img.onload=()=>{card.className="flbps-image-card";state.remove();};img.onerror=()=>{card.className="flbps-image-card failed";state.textContent="Preview unavailable";};
    img.onclick=click;img.onkeydown=e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();e.stopPropagation();click();}};
    img.src=storyboardThumbnailUrl(image);card.append(img,state);return card;
  }
  function renderSlots() {
    slots.replaceChildren();
    for (let index = 0; index < 4; index++) {
      const board = boards[index];
      const slot = document.createElement("div");
      slot.className = "flbps-moodboard-slot";
      const choose = document.createElement("button");
      choose.type = "button";
      choose.textContent = board ? "Replace" : `+ Moodboard ${index+1}`;
      const input = document.createElement("input");
      input.type = "file";
      input.accept = "image/png,image/jpeg,image/webp,image/gif";
      input.hidden = true;
      const upload = async file => {
        if (!file || !["image/png", "image/jpeg", "image/webp", "image/gif"].includes(file.type) || file.size > 32*1024*1024) {
          throw new Error("Choose a PNG, JPEG, WebP or GIF under 32 MB.");
        }
        const image = await writer.client.uploadImage(file, `fl-beat-writer/${writer.nodeSettings.schedulerId}`);
        boards[index] = { ...image, role: board?.role || "", selected: true, kind: "image", label: file.name, originalName: file.name, mimeType: file.type };
        writer.saveNodeSettings();
        renderSlots();
      };
      choose.onclick = () => input.click();
      input.onchange = () => error(() => upload(input.files[0]));
      slot.ondragover = event => event.preventDefault();
      slot.ondrop = event => { event.preventDefault(); event.stopPropagation(); error(() => upload(event.dataTransfer.files[0])); };
      slot.tabIndex = 0;
      slot.onpaste = event => {
        const file = [...event.clipboardData.files].find(value => value.type.startsWith("image/"));
        if (file) { event.preventDefault(); event.stopPropagation(); error(() => upload(file)); }
      };
      if (board) {
        const image = thumbnail(board,board.label || `Moodboard ${index+1}`,()=>openPreview(board,{title:`Moodboard ${index+1}`,role:board.role}));
        const role = document.createElement("input");
        role.placeholder = "Role: style, character…";
        role.title = "How this image should guide generation: character identity, costume, palette, or style.";
        role.value = board.role;
        role.maxLength = 500;
        role.onchange = () => { board.role = role.value; writer.saveNodeSettings(); };
        const selected = document.createElement("input");
        selected.type = "checkbox";
        selected.checked = board.selected;
        selected.title = "Include in the next storyboard generation";
        selected.onchange = () => { board.selected = selected.checked; writer.saveNodeSettings(); };
        const clear = document.createElement("button");
        clear.textContent = "Clear";
        clear.type = "button";
        clear.onclick = () => { boards[index] = null; writer.saveNodeSettings(); renderSlots(); };
        const use = document.createElement("label");
        use.append(selected, document.createTextNode("Use"));
        const actions=document.createElement("div");actions.className="flbps-moodboard-actions";actions.append(use,clear,choose);
        slot.append(image, role, actions);
      }
      if(!board)slot.append(choose);
      slot.append(input);
      slots.append(slot);
    }
  }

  function markHandled(job) {
    handled.add(job.id);
    writer.nodeSettings.storyboardResults = [...handled];
    writer.saveNodeSettings();
  }
  async function install(job, explicit = false) {
    if (handled.has(job.id)) return;
    const clip = editor.clips.find(value => value.sectionId === job.spec.section_id);
    if (!clip) return;
    if (clip.references?.asset_ids.some(id => editor.referenceAssets[id]?.storyboard_id === job.id)) {
      markHandled(job);
      return;
    }
    const revision = await promptRevision(clip);
    if (!explicit && revision !== job.spec.revision) {
      attachmentErrors.set(job.id, "Section changed. Saved image is ready to attach; current references were kept.");
      return;
    }
    ensureReferenceWiring(editor);
    if (!job.result.assets) job = await request(`/${job.id}/extract`, {});
    if (disposed) return;
    // Extraction is asynchronous: recheck before applying to a section the user may have edited.
    if (!editor.clips.includes(clip) || await promptRevision(clip) !== revision) {
      attachmentErrors.set(job.id, "Section changed during attachment. Review and attach the saved image again.");
      return;
    }
    editor.runEdit("Use generated storyboard references", () => {
      Object.assign(editor.referenceAssets, job.result.assets);
      clip.references = { mode: "custom", asset_ids: Object.keys(job.result.assets) };
      editor.serialize();
      editor.syncInspector();
    });
    markHandled(job);
    attachmentErrors.delete(job.id);
    status.textContent = "Storyboard attached. Its panels are now this section's video references.";
    editor.scheduleDraw();
  }
  async function refresh() {
    if (disposed || checking) return;
    checking = true;
    try {
      const listed = await request(`?scheduler_id=${encodeURIComponent(writer.nodeSettings.schedulerId)}`);
      jobs = [...listed, ...jobs.filter(job => job.state === "submitted" && !listed.some(value => value.id === job.id))];
      for (let index = 0; index < jobs.length && !disposed; index++) {
        let job = jobs[index];
        if (["submitted", "unknown"].includes(job.state)) job = await request(`/${job.id}/refresh`, {});
        jobs[index] = job;
        if (job.state === "complete") {
          try { await install(job); }
          catch (failure) { attachmentErrors.set(job.id, failure.message); }
        }
      }
      if (["failed", "unknown"].includes(jobs[0]?.state)) status.textContent = jobs[0].result.error;
      editor.scheduleDraw();
    } catch (failure) {
      status.textContent = failure.message;
    } finally {
      checking = false;
      if (!disposed) timer = setTimeout(refresh, 3000);
    }
  }
  async function generate(clip, prompt, grid, requestKey, context = {}) {
    if (busy.has(clip.sectionId) || jobs.some(job => job.spec.section_id === clip.sectionId && job.state === "submitted")) {
      throw new Error("This section already has an active image request. Wait for it to finish before generating again.");
    }
    busy.add(clip.sectionId);
    editor.scheduleDraw();
    try {
      ensureReferenceWiring(editor);
      const revision = await promptRevision(clip);
      const submit = await prepareStoryboardQueue();
      const job = await request("", {
        scheduler_id: writer.nodeSettings.schedulerId, section_id: clip.sectionId,
        revision, prompt, grid, resolution: "2K", aspect_ratio: "16:9",
        moodboards: context.moodboards || boards.filter(board => board?.selected), request_key: requestKey,
        continuity: context.continuity || "",
      });
      if (job.state !== "proposed") return;
      const { graph } = await request(`/${job.id}/submit`, {});
      try {
        const receipt = await submit(graph, job.id);
        await request(`/${job.id}/receipt`, { prompt_id: receipt.prompt_id });
        status.textContent = "Storyboard queued in ComfyUI. It will attach to its section when finished.";
        jobs.unshift({ ...job, state: "submitted" });
      } catch (failure) {
        throw new Error(`Submission outcome may be unknown. Check ComfyUI history before rerolling. ${failure.message}`);
      }
    } finally {
      busy.delete(clip.sectionId);
      editor.scheduleDraw();
    }
  }
  async function generateBatch(items, context) {
    if (items.length === 1) {
      const {clip,action,key} = items[0];
      return generate(clip,action.prompt,action.grid,key,context);
    }
    if (items.some(({clip}) => busy.has(clip.sectionId) || jobs.some(j => j.spec.section_id === clip.sectionId && j.state === "submitted"))) {
      throw new Error("A section already has an active storyboard request. Wait before generating it again.");
    }
    items.forEach(({clip}) => busy.add(clip.sectionId));editor.scheduleDraw();
    try {
      ensureReferenceWiring(editor);
      const revisions = await Promise.all(items.map(({clip}) => promptRevision(clip)));
      const submit = await prepareStoryboardQueue();
      const created = await Promise.all(items.map(async ({clip,action,key},index) => request("", {
        scheduler_id: writer.nodeSettings.schedulerId, section_id: clip.sectionId,
        revision: revisions[index], prompt: action.prompt, grid: action.grid,
        resolution: "2K", aspect_ratio: "16:9", request_key: key, ...context,
      })));
      const proposed = created.filter(job => job.state === "proposed");
      if (!proposed.length) return;
      const {graph} = await request("/batch/submit", {job_ids: proposed.map(job => job.id)});
      try {
        const receipt = await submit(graph, proposed.map(job => job.id).join(","));
        jobs.unshift(...proposed.map(job => ({...job,state:"submitted"})));
        await Promise.all(proposed.map(job => request(`/${job.id}/receipt`, {prompt_id: receipt.prompt_id})));
        status.textContent = `${proposed.length} storyboards queued together for asynchronous generation. Shared references and continuity brief applied.`;
      } catch (failure) {
        throw new Error(`Batch submission outcome may be unknown. Check ComfyUI history before rerolling; requests may be billed. ${failure.message}`);
      }
    } finally {items.forEach(({clip}) => busy.delete(clip.sectionId));editor.scheduleDraw();}
  }
  editor.renderStoryboardThumbnails = () => {
    const rows = (editor.clipRects || []).map(rect => {
      const clip = editor.clips[rect.index];
      return { rect, clip, images: timelineReferenceImages(clip, editor.referenceAssets || {}),
        ready: jobs.filter(job => job.spec.section_id === clip.sectionId && job.state === "complete" && job.result.source && !handled.has(job.id)),
        pending: busy.has(clip.sectionId) || jobs.some(job => job.spec.section_id === clip.sectionId && job.state === "submitted") };
    });
    const key = JSON.stringify(rows.map(row => [row.clip.sectionId, row.images, row.pending, row.ready.map(job => [job.id, attachmentErrors.get(job.id)]), row.images.map(item => jobs.some(job => job.id === item.key))]));
    if (key !== renderKey) {
      renderKey = key;
      overlay.replaceChildren();
      for (const { clip, images, pending, ready } of rows) {
        const row = document.createElement("div");
        row.className = "flbps-timeline-reference-row";
        row.dataset.sectionId = clip.sectionId;
        row.onpointerdown = event => event.stopPropagation();
        row.ondblclick = event => event.stopPropagation();
        for (const item of images) {
          const source = jobs.find(job => job.id === item.key);
          const card=thumbnail(item.image,"Active video reference",()=>openPreview(item.image,{title:"Storyboard reference",section:clip.sectionId,job:source}));
          const actions=document.createElement("div");actions.className="flbps-image-actions";card.append(actions);row.append(card);
          if (source) {
            const reroll = document.createElement("button");
            reroll.type = "button";
            reroll.textContent = "↻";
            reroll.title = "Reroll this storyboard (uses ComfyUI credits)";
            reroll.disabled = pending;
            reroll.onclick = () => error(() => generate(clip, source.spec.prompt, source.spec.grid, crypto.randomUUID(), {
              continuity: source.spec.continuity || "", moodboards: source.spec.moodboards,
            }));
            reroll.setAttribute("aria-label","Reroll storyboard");actions.append(reroll);
          }
          const remove = document.createElement("button");
          remove.type = "button";
          remove.textContent = "×";
          remove.title = "Remove this reference from the section";
          remove.onclick = () => editor.runEdit("Remove timeline reference", () => {
            clip.references = { mode: "custom", asset_ids: clip.references.asset_ids.filter(id => !item.assetIds.includes(id)) };
            editor.serialize();
            editor.syncInspector();
            editor.scheduleDraw();
          });
          remove.setAttribute("aria-label","Remove section reference");actions.append(remove);
        }
        for (const job of ready) {
          const card = thumbnail(job.result.source, "Saved storyboard — not attached", () => openPreview(job.result.source, {title:"Saved storyboard — not attached", section:clip.sectionId, job}));
          const reason = document.createElement("small");
          reason.textContent = attachmentErrors.get(job.id) || "Saved image — not attached";
          reason.title = reason.textContent;
          const actions = document.createElement("div");actions.className = "flbps-image-actions";
          const attach = document.createElement("button");attach.textContent = "Attach";
          attach.title = "Use this saved storyboard as this section's references (no generation credits)";
          attach.onclick = () => error(async () => {
            attach.disabled = true;
            try { await install(job, true); }
            finally { attach.disabled = false; editor.scheduleDraw(); }
          });
          const dismiss = document.createElement("button");dismiss.textContent = "×";
          dismiss.title = "Dismiss this saved result; keep current references";
          dismiss.onclick = () => { markHandled(job); attachmentErrors.delete(job.id); editor.scheduleDraw(); };
          actions.append(attach, dismiss);card.append(reason, actions);row.append(card);
        }
        if (pending) {const pendingCard=document.createElement("div");pendingCard.className="flbps-generation-pending";pendingCard.setAttribute("role","status");pendingCard.textContent="Queued / generating…";row.append(pendingCard);}
        overlay.append(row);
      }
    }
    for (const { rect, clip } of rows) {
      const row = [...overlay.children].find(value => value.dataset.sectionId === clip.sectionId);
      if (!row) continue;
      row.style.left = `${Math.max(0, rect.x + 8)}px`;
      row.style.top = `${rect.y + Math.max(32, Math.min(48, rect.height - 100))}px`;
      row.style.width = `${Math.max(0, rect.width - 16)}px`;
      row.style.maxHeight = `${Math.max(0,rect.height-40)}px`;
    }
  };
  function actionClip(proposal, document, updates) {
    const clip = editor.clips[proposal.index];
    const expected = updates?.find(value => value.index === proposal.index) || document?.boxes?.find(value => value.index === proposal.index);
    if (!document?.allowed_indices?.includes(proposal.index) || !expected || clip?.sectionId !== document.sectionIds?.[proposal.index]
      || clip.start !== expected.start_frame || clip.end !== expected.end_frame || clip.prompt !== expected.prompt
      || JSON.stringify(clip.references) !== JSON.stringify(document.referenceSelections?.[proposal.index])) {
      throw new Error("The section changed while the Writer was working. Ask for a fresh storyboard.");
    }
    return clip;
  }
  writer.generateStoryboards = (actions, updates, messageId) => error(async () => {
    const document = writer.currentDocument;
    const context = {continuity: storyboardContinuity(actions), moodboards: structuredClone(boards.filter(board => board?.selected))};
    // Validate every section before creating any paid submission.
    const sections = actions.map(action => ({action,clip:actionClip(action,document,updates)}));
    if (new Set(sections.map(({clip})=>clip.sectionId)).size !== sections.length) throw new Error("Request each storyboard section only once per batch.");
    if (disposed || !sections.length) return;
      const items=sections.map(({action})=>{
        const clip=actionClip(action,document,updates);return {action,clip,key:`${messageId}:${clip.sectionId}`};
      });
      await generateBatch(items,context);
  });
  writer.applyReferenceAssignments = (actions, updates) => error(async () => {
    for (const action of actions) {
      const clip = actionClip(action, writer.currentDocument, updates);
      ensureReferenceWiring(editor);
      editor.runEdit("Apply Writer references", () => {
        clip.references = { mode: action.mode, asset_ids: [...action.asset_ids] };
        editor.serialize();
        editor.syncInspector();
      });
      editor.scheduleDraw();
    }
  });
  writer.disposeStoryboards = () => {
    disposed = true;
    clearTimeout(timer);
    viewer?.close();
    delete editor.renderStoryboardThumbnails;
    overlay.remove();
  };
  renderSlots();
  refresh();
}

async function promptRevision(clip) {
  const bytes = new TextEncoder().encode(JSON.stringify([clip.start, clip.end, clip.prompt, clip.references]));
  return [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))].map(value => value.toString(16).padStart(2, "0")).join("");
}
