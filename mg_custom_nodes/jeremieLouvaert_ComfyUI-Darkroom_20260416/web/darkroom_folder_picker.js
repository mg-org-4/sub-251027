import { app } from "../../scripts/app.js";

const STYLE_ID = "darkroom-folder-picker-styles";

const CSS = `
.drfp-overlay {
    position: fixed; inset: 0;
    background: rgba(0, 0, 0, 0.6);
    z-index: 10000;
    display: flex; align-items: center; justify-content: center;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
.drfp-modal {
    background: #1e1e1e;
    border: 1px solid #3a3a3a;
    border-radius: 6px;
    width: 760px; max-width: 92vw;
    height: 560px; max-height: 92vh;
    display: flex; flex-direction: column;
    color: #e0e0e0; font-size: 13px;
    box-shadow: 0 24px 72px rgba(0, 0, 0, 0.6);
}
.drfp-header {
    padding: 14px 18px;
    border-bottom: 1px solid #2a2a2a;
    display: flex; justify-content: space-between; align-items: center;
}
.drfp-title { font-size: 14px; font-weight: 600; color: #fff; }
.drfp-close {
    cursor: pointer; color: #888; font-size: 22px;
    width: 30px; height: 30px;
    display: flex; align-items: center; justify-content: center;
    border-radius: 4px; user-select: none;
}
.drfp-close:hover { background: #2a2a2a; color: #fff; }
.drfp-body { display: flex; flex: 1; min-height: 0; }
.drfp-sidebar {
    width: 210px;
    background: #181818;
    border-right: 1px solid #2a2a2a;
    padding: 10px 0;
    overflow-y: auto;
}
.drfp-sidebar-section {
    font-size: 10px; color: #666;
    text-transform: uppercase;
    padding: 10px 16px 6px;
    letter-spacing: 0.6px;
}
.drfp-quick {
    padding: 7px 16px;
    cursor: pointer;
    color: #ccc;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.drfp-quick:hover { background: #2a2a2a; color: #fff; }
.drfp-main { flex: 1; display: flex; flex-direction: column; min-width: 0; }
.drfp-pathbar {
    padding: 11px 16px;
    background: #252525;
    border-bottom: 1px solid #2a2a2a;
    font-family: Consolas, "Courier New", monospace;
    font-size: 12px;
    color: #9ac;
    word-break: break-all;
}
.drfp-toolbar {
    display: flex; gap: 8px;
    padding: 8px 14px;
    border-bottom: 1px solid #2a2a2a;
}
.drfp-btn {
    background: #2a2a2a; color: #e0e0e0;
    border: 1px solid #3a3a3a;
    padding: 6px 13px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 12px;
    font-family: inherit;
}
.drfp-btn:hover { background: #353535; border-color: #4a4a4a; }
.drfp-btn:disabled {
    opacity: 0.4; cursor: not-allowed;
}
.drfp-btn-primary {
    background: #3a6ea5; border-color: #4a7eb5; color: #fff;
}
.drfp-btn-primary:hover { background: #4a7eb5; border-color: #5a8ec5; }
.drfp-list { flex: 1; overflow-y: auto; padding: 4px 0; }
.drfp-entry {
    padding: 7px 16px;
    cursor: pointer;
    color: #ddd;
    display: flex; align-items: center;
    user-select: none;
}
.drfp-entry:hover { background: #2a2a2a; }
.drfp-entry.selected { background: #3a6ea5; color: #fff; }
.drfp-entry-icon {
    width: 14px; margin-right: 10px;
    color: #888; font-size: 10px;
    flex-shrink: 0;
}
.drfp-entry.drfp-file .drfp-entry-icon { color: #6a9; }
.drfp-entry.selected .drfp-entry-icon { color: #cde; }
.drfp-empty {
    padding: 24px;
    text-align: center;
    color: #666;
    font-style: italic;
}
.drfp-footer {
    padding: 12px 18px;
    border-top: 1px solid #2a2a2a;
    display: flex; justify-content: flex-end; gap: 8px;
    background: #1a1a1a;
}
.drfp-error {
    color: #e09090;
    padding: 8px 16px;
    font-size: 11px;
    background: #2a1818;
    border-bottom: 1px solid #3a1a1a;
    font-family: Consolas, monospace;
}
`;

function injectStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = CSS;
    document.head.appendChild(style);
}

async function fetchList(path, extensions) {
    const params = new URLSearchParams();
    if (path) params.set("path", path);
    if (extensions && extensions.length) params.set("extensions", extensions.join(","));
    const resp = await fetch(`/darkroom/list_dir?${params.toString()}`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return await resp.json();
}

async function createDir(path) {
    const resp = await fetch("/darkroom/mkdir", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path }),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok || !data.ok) {
        throw new Error(data.error || `HTTP ${resp.status}`);
    }
    return data;
}

/**
 * Open the path picker modal.
 *
 * options:
 *   mode:         "folder" (default) | "file"
 *   initialPath:  starting path (file paths resolve to their parent)
 *   extensions:   array of extensions in file mode, e.g. [".cube"]
 *   title:        header text
 *   selectLabel:  confirm button label
 */
function openPathPicker(options, onSelect) {
    const {
        mode = "folder",
        initialPath = "",
        extensions = null,
        title = "Choose folder",
        selectLabel = "Select",
    } = options || {};
    const isFileMode = mode === "file";

    injectStyles();

    const overlay = document.createElement("div");
    overlay.className = "drfp-overlay";
    overlay.innerHTML = `
        <div class="drfp-modal">
            <div class="drfp-header">
                <div class="drfp-title"></div>
                <div class="drfp-close" data-act="close">&times;</div>
            </div>
            <div class="drfp-body">
                <div class="drfp-sidebar">
                    <div class="drfp-sidebar-section">Quick access</div>
                    <div data-slot="quick"></div>
                </div>
                <div class="drfp-main">
                    <div class="drfp-pathbar" data-slot="path">(no path)</div>
                    <div class="drfp-error" data-slot="error" style="display:none;"></div>
                    <div class="drfp-toolbar">
                        <button class="drfp-btn" data-act="up">Up</button>
                        <button class="drfp-btn" data-act="newfolder" data-slot="newfolder">+ New folder</button>
                    </div>
                    <div class="drfp-list" data-slot="list"></div>
                </div>
            </div>
            <div class="drfp-footer">
                <button class="drfp-btn" data-act="cancel">Cancel</button>
                <button class="drfp-btn drfp-btn-primary" data-act="select" data-slot="selectbtn">Select</button>
            </div>
        </div>
    `;
    document.body.appendChild(overlay);

    const state = {
        viewPath: "",
        viewParent: "",
        selectedPath: "",
        selectionIsFile: false,
    };

    const $ = (sel) => overlay.querySelector(sel);
    const $$ = (sel) => overlay.querySelectorAll(sel);

    $(".drfp-title").textContent = title;
    $("[data-slot=selectbtn]").textContent = selectLabel;
    if (isFileMode) {
        $("[data-slot=newfolder]").style.display = "none";
    }

    function close() {
        document.removeEventListener("keydown", onKey, true);
        overlay.remove();
    }

    function onKey(e) {
        if (e.key === "Escape") { e.stopPropagation(); close(); }
        else if (e.key === "Enter") { e.stopPropagation(); submit(); }
    }
    document.addEventListener("keydown", onKey, true);

    function canSubmit() {
        if (!state.selectedPath) return false;
        if (isFileMode && !state.selectionIsFile) return false;
        return true;
    }

    function refreshSubmitState() {
        $("[data-slot=selectbtn]").disabled = !canSubmit();
    }

    function submit() {
        if (!canSubmit()) return;
        onSelect(state.selectedPath);
        close();
    }

    function showError(msg) {
        const el = $("[data-slot=error]");
        el.textContent = msg;
        el.style.display = "block";
    }
    function clearError() {
        $("[data-slot=error]").style.display = "none";
    }

    function updatePathbar() {
        $("[data-slot=path]").textContent =
            state.selectedPath || state.viewPath || "(no path)";
    }

    function renderQuick(roots) {
        const el = $("[data-slot=quick]");
        if (el.dataset.rendered === "1") return;
        el.innerHTML = "";
        for (const r of (roots || [])) {
            const item = document.createElement("div");
            item.className = "drfp-quick";
            item.textContent = r.name;
            item.title = r.path;
            item.addEventListener("click", () => navigate(r.path));
            el.appendChild(item);
        }
        el.dataset.rendered = "1";
    }

    function makeEntry(item, isFile) {
        const entry = document.createElement("div");
        entry.className = "drfp-entry" + (isFile ? " drfp-file" : "");
        const icon = document.createElement("span");
        icon.className = "drfp-entry-icon";
        icon.textContent = isFile ? "\u25CF" : "\u25B8";
        const label = document.createElement("span");
        label.textContent = item.name;
        entry.appendChild(icon);
        entry.appendChild(label);

        entry.addEventListener("click", () => {
            state.selectedPath = item.path;
            state.selectionIsFile = isFile;
            $$(".drfp-entry.selected").forEach(e => e.classList.remove("selected"));
            entry.classList.add("selected");
            updatePathbar();
            refreshSubmitState();
        });
        entry.addEventListener("dblclick", () => {
            if (isFile) {
                state.selectedPath = item.path;
                state.selectionIsFile = true;
                submit();
            } else {
                navigate(item.path);
            }
        });
        return entry;
    }

    function renderList(subdirs, files) {
        const listEl = $("[data-slot=list]");
        listEl.innerHTML = "";

        const hasDirs = subdirs && subdirs.length > 0;
        const hasFiles = files && files.length > 0;

        if (!hasDirs && !hasFiles) {
            const empty = document.createElement("div");
            empty.className = "drfp-empty";
            empty.textContent = isFileMode
                ? "(no subfolders or matching files)"
                : "(no subfolders)";
            listEl.appendChild(empty);
            return;
        }
        if (hasDirs) {
            for (const d of subdirs) listEl.appendChild(makeEntry(d, false));
        }
        if (hasFiles) {
            for (const f of files) listEl.appendChild(makeEntry(f, true));
        }
    }

    async function navigate(path) {
        clearError();
        try {
            const data = await fetchList(path, extensions);
            state.viewPath = data.path || "";
            state.viewParent = data.parent || "";
            // In folder mode, default the selection to the current view. In file
            // mode, the user must actively pick a file — don't preselect.
            state.selectedPath = isFileMode ? "" : (data.path || "");
            state.selectionIsFile = false;
            renderQuick(data.roots);
            renderList(data.subdirs, data.files);
            updatePathbar();
            refreshSubmitState();
            if (data.error) showError(data.error);
        } catch (e) {
            showError(String(e.message || e));
        }
    }

    overlay.addEventListener("click", async (e) => {
        if (e.target === overlay) { close(); return; }
        const actEl = e.target.closest("[data-act]");
        if (!actEl || actEl.disabled) return;
        const act = actEl.dataset.act;
        if (act === "close" || act === "cancel") {
            close();
        } else if (act === "select") {
            submit();
        } else if (act === "up") {
            if (state.viewParent) navigate(state.viewParent);
        } else if (act === "newfolder") {
            if (!state.viewPath) {
                showError("Navigate into a folder before creating a new one");
                return;
            }
            const name = prompt("New folder name:", "");
            if (!name || !name.trim()) return;
            const cleanName = name.trim().replace(/[\\/:*?"<>|]/g, "_");
            const newPath = state.viewPath.replace(/[\\/]+$/, "") + "/" + cleanName;
            try {
                await createDir(newPath);
                await navigate(newPath);
            } catch (err) {
                showError(String(err.message || err));
            }
        }
    });

    navigate(initialPath || "");
}

function attachBrowseButton(node, widgetName, buttonLabel, pickerOptions) {
    const widget = node.widgets?.find(w => w.name === widgetName);
    if (!widget) return;
    node.addWidget("button", buttonLabel, null, () => {
        const start = (widget.value && String(widget.value).trim()) || "";
        openPathPicker({ ...pickerOptions, initialPath: start }, (chosen) => {
            widget.value = chosen;
            if (typeof widget.callback === "function") widget.callback(chosen);
            node.setDirtyCanvas(true, true);
        });
    });
}

const RAW_EXTENSIONS = [
    ".cr3", ".cr2", ".crw",
    ".nef", ".nrw",
    ".arw", ".sr2", ".srf",
    ".raf",
    ".rw2", ".raw",
    ".dng", ".rwl",
    ".3fr", ".fff",
    ".orf",
    ".pef", ".ptx",
    ".x3f",
    ".iiq", ".mos", ".eip",
];

app.registerExtension({
    name: "Darkroom.PathPicker",
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name === "DarkroomLUTExport") {
            const orig = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = orig?.apply(this, arguments);
                attachBrowseButton(this, "output_directory", "Browse for folder...", {
                    mode: "folder",
                    title: "Choose output folder",
                    selectLabel: "Select this folder",
                });
                return r;
            };
        } else if (nodeData.name === "DarkroomLUTApply") {
            const orig = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = orig?.apply(this, arguments);
                attachBrowseButton(this, "lut_file", "Browse for .cube file...", {
                    mode: "file",
                    extensions: [".cube"],
                    title: "Choose .cube LUT file",
                    selectLabel: "Select this file",
                });
                return r;
            };
        } else if (nodeData.name === "DarkroomRAWLoad") {
            const orig = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = orig?.apply(this, arguments);
                attachBrowseButton(this, "raw_file", "Browse for RAW file...", {
                    mode: "file",
                    extensions: RAW_EXTENSIONS,
                    title: "Choose camera RAW file",
                    selectLabel: "Select this file",
                });
                return r;
            };
        }
    },
});
