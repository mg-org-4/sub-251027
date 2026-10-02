import { PromptWriterClient } from "./audio_prompt_writer_client.js";
import { ensureReferenceWiring } from "./audio_prompt_storyboards.js";

export function ensureReferenceIds(clips) {
  const seen = new Set();
  for (const clip of clips) {
    if (!clip.sectionId || seen.has(clip.sectionId)) clip.sectionId = crypto.randomUUID();
    seen.add(clip.sectionId);
    clip.references = structuredClone(clip.references || { mode: "defaults", asset_ids: [] });
  }
}

export function loadReferences(editor) {
  const raw = editor.widgets.referenceSchedule?.value;
  const document = raw ? JSON.parse(raw) : { version: 1, sections: [], assets: {} };
  if (document.version !== 1) throw new Error("Unsupported reference schedule version.");
  if (document.sections.length && document.sections.length !== editor.clips.length) {
    throw new Error("Reference sections no longer match the timeline. Restore the matching timeline before editing.");
  }
  editor.referenceAssets = document.assets || {};
  editor.clips.forEach((clip, index) => {
    const saved = document.sections[index];
    if (saved) {
      clip.sectionId = saved.id;
      clip.references = { mode: saved.mode, asset_ids: [...saved.asset_ids] };
    }
  });
  ensureReferenceIds(editor.clips);
}

export function serializeReferences(editor) {
  ensureReferenceIds(editor.clips);
  if (editor.widgets.referenceSchedule) editor.widgets.referenceSchedule.value = JSON.stringify({
    version: 1,
    assets: editor.referenceAssets || {},
    sections: editor.clips.map(clip => ({ id: clip.sectionId, ...clip.references })),
  });
}

export function mountReferences(editor) {
  const client = new PromptWriterClient();
  const panel = document.createElement("details");
  panel.className = "flbps-reference-panel";
  panel.innerHTML = `<summary>Section references</summary><div>
    <select aria-label="Section reference mode"><option value="defaults">Planner defaults</option><option value="custom">Custom references</option><option value="none">No references</option></select>
    <button type="button" data-ref="add">Add media</button><button type="button" data-ref="apply">Apply to selected sections</button>
    <input type="file" accept="image/*,video/*,audio/*" multiple hidden>
    <p>Custom references replace planner defaults. The timeline soundtrack is unchanged. Picture and Video tags each start at 1 within this section.</p>
    <div data-ref="items"></div><select data-ref="library" aria-label="Add existing reference"><option value="">Add from library…</option></select>
  </div>`;
  editor.clipInspector.append(panel);
  const mode = panel.querySelector("select");
  const input = panel.querySelector("input");
  const items = panel.querySelector('[data-ref="items"]');
  const library = panel.querySelector('[data-ref="library"]');
  const change = (label, update) => {
    if (!editor.selectedClip()) return;
    try {
      ensureReferenceWiring(editor);
      editor.runEdit(label, () => {
        update(editor.selectedClip());
        editor.serialize();
        editor.syncInspector();
        editor.scheduleDraw();
      });
    } catch (error) { editor.showError(error.message); }
  };
  mode.onchange = () => change("Change section references", clip => {
    clip.references = { mode: mode.value, asset_ids: mode.value === "custom" ? (clip.references?.asset_ids || []) : [] };
  });
  panel.querySelector('[data-ref="apply"]').onclick = () => change("Apply section references", clip => {
    for (const index of editor.selectedClipIndices()) editor.clips[index].references = structuredClone(clip.references);
  });
  panel.querySelector('[data-ref="add"]').onclick = () => input.click();
  input.onchange = async () => {
    const sectionId = editor.selectedClip()?.sectionId;
    const files = [...input.files];
    input.value = "";
    try {
      ensureReferenceWiring(editor);
      const assets = [];
      for (const file of files) {
        const kind = file.type.split("/")[0];
        if (!["image", "audio", "video"].includes(kind) || file.size > 512 * 1024 * 1024) {
          throw new Error("Choose image, audio, or video media under 512 MB.");
        }
        const media = await client.uploadImage(file, "fl-prompt-references");
        assets.push([crypto.randomUUID(), { ...media, kind, label: file.name }]);
      }
      editor.runEdit("Add section references", () => {
        Object.assign(editor.referenceAssets, Object.fromEntries(assets));
        const clip = editor.clips.find(value => value.sectionId === sectionId);
        if (clip) clip.references = { mode: "custom", asset_ids: [...(clip.references?.asset_ids || []), ...assets.map(([id]) => id)] };
        editor.serialize();
        editor.syncInspector();
        editor.scheduleDraw();
      });
    } catch (error) { editor.showError(error.message); }
  };
  library.onchange = () => {
    if (!library.value) return;
    change("Add library reference", clip => {
      clip.references = { mode: "custom", asset_ids: [...new Set([...(clip.references?.asset_ids || []), library.value])] };
    });
  };
  editor.syncReferences = () => {
    const clip = editor.selectedClip();
    panel.hidden = !clip;
    if (!clip) return;
    const reference = clip.references || { mode: "defaults", asset_ids: [] };
    panel.querySelector("summary").textContent = `Section references · ${reference.mode === "custom" ? `${reference.asset_ids.length} custom` : reference.mode === "none" ? "None" : "Planner defaults"}`;
    mode.value = reference.mode;
    items.replaceChildren();
    let picture = 0;
    let video = 0;
    const ids = [...reference.asset_ids].sort((a, b) => {
      const order = { image: 0, video: 1, audio: 2 };
      return order[editor.referenceAssets?.[a]?.kind] - order[editor.referenceAssets?.[b]?.kind];
    });
    for (const id of ids) {
      const asset = editor.referenceAssets[id];
      if (!asset) continue;
      const row = document.createElement("div");
      row.className = "flbps-reference-item";
      const preview = document.createElement(asset.kind === "image" ? "img" : asset.kind);
      preview.src = client.imageUrl(asset, asset.kind === "image");
      if (asset.kind !== "image") preview.controls = true;
      else preview.onclick = () => window.open(client.imageUrl(asset, false), "_blank", "noopener");
      const label = document.createElement("span");
      label.textContent = `${asset.kind === "image" ? `<Picture ${++picture}> ` : asset.kind === "video" ? `<Video ${++video}> ` : "Audio reference: "}${asset.label || asset.filename}`;
      row.append(preview, label);
      for (const [text, delta] of [["↑", -1], ["↓", 1], ["Remove", 0]]) {
        const button = document.createElement("button");
        button.textContent = text;
        button.type = "button";
        button.onclick = () => change("Edit reference order", target => {
          const values = [...target.references.asset_ids];
          const index = values.indexOf(id);
          if (!delta) values.splice(index, 1);
          else if (index + delta >= 0 && index + delta < values.length) [values[index], values[index + delta]] = [values[index + delta], values[index]];
          target.references = { mode: "custom", asset_ids: values };
        });
        row.append(button);
      }
      items.append(row);
    }
    library.replaceChildren(new Option("Add from library…", ""));
    for (const [id, asset] of Object.entries(editor.referenceAssets || {})) {
      if (!reference.asset_ids.includes(id)) library.append(new Option(asset.label || asset.filename, id));
    }
  };
}
