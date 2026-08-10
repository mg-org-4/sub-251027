/** Render backend diagnostics and health without owning lifecycle state. */
import { addRow, createSection, detectComfyUiVersion } from "./settings_tab_dom.js";

export async function renderSettingsStatus({ scroll, healthRes, logRes, configRes, capabilities, api }) {
    // If everything is 404, backend routes not registered

    // -- UI Boot Diagnostics / Backend Warning --
    const all404 = [healthRes, logRes, configRes].every(r => r && r.ok === false && r.status === 404);
    if (all404) {
        const warn = createSection("Backend Not Loaded");
        const hint = document.createElement("div");
        hint.className = "openclaw-note openclaw-note moltbot-note";
        hint.style.borderLeft = "4px solid #ff4444";
        hint.innerHTML = `
            OpenClaw UI loaded, but the server endpoints returned <code>HTTP 404</code>.
            This usually means ComfyUI did not load the Python part of this custom node pack.
            <br/><br/>
            Check ComfyUI startup logs for errors while importing <code>ComfyUI-OpenClaw</code>/<code>Comfyui-OpenClaw</code>,
            then restart ComfyUI.
            <br/><br/>
            Expected endpoints:
            <ul>
              <li><code>/openclaw/health</code> (legacy: <code>/moltbot/health</code>)</li>
              <li><code>/openclaw/config</code></li>
              <li><code>/openclaw/logs/tail</code></li>
            </ul>
        `;
        warn.appendChild(hint);
        scroll.appendChild(warn);
    }

    // F39: Show degraded-state banner when capabilities are missing or partial
    if (!all404 && healthRes.ok && Object.keys(capabilities).length === 0) {
        const degradedWarn = createSection("Limited Mode");
        const degradedHint = document.createElement("div");
        degradedHint.className = "openclaw-note openclaw-note moltbot-note";
        degradedHint.style.borderLeft = "4px solid #ffaa00";
        degradedHint.innerHTML = `
            <b>⚠ Capabilities endpoint unavailable.</b> Some features may be hidden or behave differently.
            This can happen if the backend pack version is older than the frontend UI.
            <br/>Consider updating ComfyUI-OpenClaw to the latest version.
        `;
        degradedWarn.appendChild(degradedHint);
        scroll.appendChild(degradedWarn);
    }

    // -- System Health & Diagnostics --
    const healthSec = createSection("System Health");

    // F26: Diagnostics Block (Shim status + ComfyUI version)
    const diagDetails = document.createElement("details");
    diagDetails.style.marginBottom = "10px";
    diagDetails.style.padding = "8px";
    diagDetails.style.background = "var(--comfy-input-bg)";
    diagDetails.style.borderRadius = "4px";
    diagDetails.style.fontSize = "12px";
    diagDetails.style.color = "var(--input-text)";

    // Detect Shim
    const hasShim = typeof window.comfyAPI?.fetchApi === "function" || typeof window.fetchApi === "function" || !!healthRes.ok;
    // Note: fetchApi is imported in module scope, not global. If request worked, shim worked.
    // Actually best check is if healthRes.ok or we can inspect 'openclawApi.prefix' implicitly.

    const packVer = (healthRes.ok && healthRes.data?.pack?.version) || "Unknown";
    const basePath = (healthRes.ok && healthRes.data?.pack?.base_path) || "/openclaw (inferred)";
    const comfyVersion = await detectComfyUiVersion(api);

    // Collapsed by default; auto-expand if errors
    diagDetails.open = all404 || !hasShim;

    const summary = document.createElement("summary");
    summary.style.display = "flex";
    summary.style.justifyContent = "space-between";
    summary.style.alignItems = "center";
    summary.style.cursor = "pointer";
    summary.innerHTML = `
        <span><b>UI Boot Status</b></span>
        <span>${all404 ? "⚠️ Backend 404" : "✓ Connected"}</span>
    `;

    const body = document.createElement("div");
    body.innerHTML = `
        <div style="margin-top:4px; opacity:0.8;">
            ComfyUI: ${comfyVersion || "Unknown"} | Pack: ${packVer} | Prefix: ${basePath}
        </div>
        <div style="margin-top:4px; font-size:11px; color:${hasShim ? "var(--input-text)" : "#ff6666"}">
            Shim: ${hasShim ? "✓ Detected" : "⚠️ Missing (shim broken)"}
        </div>
    `;

    diagDetails.appendChild(summary);
    diagDetails.appendChild(body);
    healthSec.appendChild(diagDetails);

    if (healthRes.ok) {
        const { pack, config, uptime_sec } = healthRes.data;
        addRow(healthSec, "Uptime", `${Math.floor(uptime_sec)}s`);

        const keyStatus = config.llm_key_configured
            ? "Configured"
            : (config.llm_key_required ? "Missing" : "Not Req");
        const keyClass = (config.llm_key_configured || !config.llm_key_required) ? "ok" : "error";
        addRow(healthSec, "API Key", keyStatus, keyClass);
    } else {
        if (!all404) {
            addRow(healthSec, "Status", "Error", "error");
            const detail = [
                healthRes.status ? `HTTP ${healthRes.status}` : null,
                healthRes.error || "request_failed",
            ].filter(Boolean).join(" — ");
            addRow(healthSec, "Detail", detail);
        }
    }
    scroll.appendChild(healthSec);
}
