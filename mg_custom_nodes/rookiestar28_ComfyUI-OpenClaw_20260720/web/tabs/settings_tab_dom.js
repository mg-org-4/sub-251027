/** Stable DOM builders shared by Settings section owners. */

export function createSection(title) {
    const div = document.createElement("div");
    div.className = "openclaw-section openclaw-section moltbot-section";
    const h4 = document.createElement("h4");
    h4.textContent = title;
    div.appendChild(h4);
    return div;
}

export function createCollapsibleSection(title, description, defaultExpanded = false) {
    const container = document.createElement("div");
    container.className = "openclaw-section openclaw-section moltbot-section openclaw-collapsible-section openclaw-collapsible-section moltbot-collapsible-section";

    const header = document.createElement("div");
    header.className = "openclaw-collapsible-header openclaw-collapsible-header moltbot-collapsible-header";
    header.style.cursor = "pointer";
    header.style.display = "flex";
    header.style.justifyContent = "space-between";
    header.style.alignItems = "center";
    header.style.userSelect = "none";

    const titleWrap = document.createElement("div");
    titleWrap.style.display = "flex";
    titleWrap.style.alignItems = "center";
    titleWrap.style.gap = "8px";

    const h4 = document.createElement("h4");
    h4.style.margin = "0";
    h4.innerHTML = title;
    titleWrap.appendChild(h4);

    // Add help button inline
    const helpBtn = createHelpButton(
        "UI Key Store (Security & Usage)",
        `
        <p>This feature lets you paste an LLM provider API key in the UI and save it to the <b>server-side</b> secret store (<code>{STATE_DIR}/secrets.json</code>).</p>
        <p><b>Important</b>:</p>
        <ul>
          <li>Recommended: use environment variables for API keys.</li>
          <li>Only use UI storage on a single-user, localhost-only setup.</li>
          <li>ENV keys always take priority over stored keys.</li>
          <li>Secrets are stored as plaintext JSON on disk (protected by OS permissions).</li>
          <li>Outbound LLM requests are protected by an SSRF policy. Built-in providers are allowlisted by default; custom Base URL hosts must be added via <code>OPENCLAW_LLM_ALLOWED_HOSTS</code> (or use <code>OPENCLAW_ALLOW_ANY_PUBLIC_LLM_HOST=1</code> at your own risk).</li>
        </ul>
        <p><b>PowerShell</b>: <code>$env:OPENCLAW_LLM_API_KEY="&lt;YOUR_API_KEY&gt;"</code></p>
        <p><b>CMD</b>: <code>set OPENCLAW_LLM_API_KEY=&lt;YOUR_API_KEY&gt;</code></p>
        `
    );
    titleWrap.appendChild(helpBtn);



    const toggle = document.createElement("span");
    toggle.textContent = defaultExpanded ? "▼" : "►";
    toggle.style.fontSize = "12px";
    toggle.style.transition = "transform 0.2s";

    header.appendChild(titleWrap);
    header.appendChild(toggle);
    container.appendChild(header);

    const descDiv = document.createElement("div");
    descDiv.className = "openclaw-note openclaw-note moltbot-note";
    descDiv.style.margin = "8px 0";
    descDiv.innerHTML = description;
    container.appendChild(descDiv);

    const content = document.createElement("div");
    content.className = "openclaw-collapsible-content openclaw-collapsible-content moltbot-collapsible-content";
    content.style.display = defaultExpanded ? "block" : "none";
    content.style.marginTop = "8px";
    container.appendChild(content);

    header.onclick = () => {
        const isExpanded = content.style.display !== "none";
        content.style.display = isExpanded ? "none" : "block";
        toggle.textContent = isExpanded ? "►" : "▼";
    };

    return { container, content };
}

export function createFormRow(label, locked = false, helpBtn = null) {
    const row = document.createElement("div");
    row.className = "openclaw-form-row openclaw-form-row moltbot-form-row";
    const header = document.createElement("div");
    header.style.display = "flex";
    header.style.alignItems = "center";
    header.style.justifyContent = "space-between";
    header.style.gap = "8px";

    const lbl = document.createElement("label");
    lbl.className = "openclaw-label openclaw-label moltbot-label";
    lbl.textContent = label + (locked ? " 🔒" : "");
    if (locked) lbl.title = "Locked (env override)";

    header.appendChild(lbl);
    if (helpBtn) header.appendChild(helpBtn);
    row.appendChild(header);
    return row;
}

export function createHelpButton(title, html) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "openclaw-help-btn openclaw-help-btn moltbot-help-btn";
    btn.textContent = "?";
    btn.title = "Help";
    btn.onclick = (e) => {
        e.stopPropagation(); // Prevent collapsible toggle
        showHelpModal(title, html);
    };
    return btn;
}

export function showHelpModal(title, html) {
    // Remove any existing modal overlay
    const existing = document.querySelector(".openclaw-modal-overlay");
    if (existing) existing.remove();

    const overlay = document.createElement("div");
    overlay.className = "openclaw-modal-overlay openclaw-modal-overlay moltbot-modal-overlay";
    overlay.addEventListener("click", (e) => {
        if (e.target === overlay) overlay.remove();
    });

    const modal = document.createElement("div");
    modal.className = "openclaw-modal openclaw-modal moltbot-modal";

    const header = document.createElement("div");
    header.className = "openclaw-modal-header openclaw-modal-header moltbot-modal-header";
    header.textContent = title;

    const closeBtn = document.createElement("button");
    closeBtn.className = "openclaw-btn openclaw-btn moltbot-btn openclaw-btn-secondary openclaw-btn-secondary moltbot-btn-secondary";
    closeBtn.textContent = "Close";
    closeBtn.onclick = () => overlay.remove();
    header.appendChild(closeBtn);

    const body = document.createElement("div");
    body.className = "openclaw-modal-body openclaw-modal-body moltbot-modal-body";
    body.innerHTML = html;

    modal.appendChild(header);
    modal.appendChild(body);
    overlay.appendChild(modal);
    document.body.appendChild(overlay);
}

export function addRow(container, key, val, valClass = "") {
    const row = document.createElement("div");
    row.className = "openclaw-kv-row openclaw-kv-row moltbot-kv-row";

    const k = document.createElement("span");
    k.className = "openclaw-kv-key openclaw-kv-key moltbot-kv-key";
    k.textContent = key;

    const v = document.createElement("span");
    v.className = `openclaw-kv-val openclaw-kv-val moltbot-kv-val ${valClass}`;
    v.textContent = val;

    row.appendChild(k);
    row.appendChild(v);
    container.appendChild(row);
}

export async function detectComfyUiVersion(api) {
    const candidates = [
        () => window?.COMFYUI_VERSION,
        () => window?.comfyui_version,
        () => window?.ComfyUI?.version,
        () => window?.app?.version,
        () => window?.app?.ui?.settings?.getSettingValue?.("ComfyUI.Version", null),
        () => window?.app?.ui?.settings?.getSettingValue?.("comfyui.version", null),
    ];

    for (const get of candidates) {
        try {
            const v = normalizeVersion(get?.());
            if (v) return v;
        } catch { }
    }

    const endpoints = ["/system_stats", "/system_info", "/version"];
    for (const path of endpoints) {
        try {
            const res = await api.fetch(path, { timeout: 1500 });
            if (!res.ok) continue;
            const v = extractComfyVersion(res.data);
            if (v) return v;
        } catch { }
    }

    return null;
}

export function extractComfyVersion(data) {
    if (!data) return null;
    if (typeof data === "string") return normalizeVersion(data);
    if (typeof data !== "object") return null;

    const direct = normalizeVersion(data.comfyui_version || data.comfyuiVersion);
    if (direct) return direct;

    const nested = normalizeVersion(data.comfyui?.version || data.comfyui?.comfyui_version);
    if (nested) return nested;

    const system = normalizeVersion(data.system?.comfyui_version || data.system?.version);
    if (system) return system;

    const name = String(data.name || data.app || "").toLowerCase();
    const namedVersion = normalizeVersion(data.version);
    if (namedVersion && name.includes("comfy")) return namedVersion;

    return null;
}

export function normalizeVersion(value) {
    if (value === null || value === undefined) return null;
    const str = String(value).trim();
    return str ? str : null;
}
