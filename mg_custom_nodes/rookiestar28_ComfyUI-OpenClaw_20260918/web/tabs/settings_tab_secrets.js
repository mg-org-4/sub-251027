/** Render server-side secret-store controls without retaining secret values. */
import { createCollapsibleSection, createFormRow } from "./settings_tab_dom.js";

export function renderSettingsSecrets({ scroll, configRes, api, session, container, isCurrent }) {
    // --- S26: Collapsible Secrets Section (always visible) ---
    if (configRes.ok) {
        const { config, sources, providers } = configRes.data;

        const secretsSec = createCollapsibleSection(
            "UI Key Store (Advanced)",
            `Server-side API key storage for portability. <b>Recommended:</b> Use ENV. <b>Acceptable:</b> Localhost-only single-user setups.`,
            false // Default collapsed
        );

        const secretsContent = secretsSec.content;

        const secretProviderRow = createFormRow("Store For");
        const secretProviderSelect = document.createElement("select");
        secretProviderSelect.className = "openclaw-input openclaw-input moltbot-input";
        // Build options from provider catalog + generic fallback
        const providerOptions = [];
        providers.forEach(p => providerOptions.push({ id: p.id, label: p.label, requires_key: p.requires_key }));
        providerOptions.push({ id: "generic", label: "Generic (fallback)", requires_key: true });
        providerOptions.forEach(p => {
            // Skip local providers (no key required) unless "generic"
            if (p.id !== "generic" && p.requires_key === false) return;
            const opt = document.createElement("option");
            opt.value = p.id;
            opt.textContent = p.label;
            secretProviderSelect.appendChild(opt);
        });
        secretProviderRow.appendChild(secretProviderSelect);
        secretsContent.appendChild(secretProviderRow);

        const secretKeyRow = createFormRow("API Key");
        const secretKeyWrap = document.createElement("div");
        secretKeyWrap.style.display = "flex";
        secretKeyWrap.style.gap = "8px";
        secretKeyWrap.style.alignItems = "center";

        const secretKeyInput = document.createElement("input");
        secretKeyInput.type = "password";
        secretKeyInput.className = "openclaw-input openclaw-input moltbot-input";
        secretKeyInput.placeholder = "Paste provider API key (not stored in browser)";
        secretKeyInput.value = "";
        secretKeyInput.autocomplete = "off";
        secretKeyInput.style.flex = "1";

        const secretKeyClearBtn = document.createElement("button");
        secretKeyClearBtn.className = "openclaw-btn openclaw-btn moltbot-btn openclaw-btn-secondary openclaw-btn-secondary moltbot-btn-secondary";
        secretKeyClearBtn.textContent = "Clear";
        secretKeyClearBtn.onclick = () => {
            secretKeyInput.value = "";
        };

        secretKeyWrap.appendChild(secretKeyInput);
        secretKeyWrap.appendChild(secretKeyClearBtn);
        secretKeyRow.appendChild(secretKeyWrap);
        secretsContent.appendChild(secretKeyRow);

        const secretsStatus = document.createElement("div");
        secretsStatus.className = "openclaw-status openclaw-status moltbot-status";
        secretsContent.appendChild(secretsStatus);

        const getAdminToken = () => {
            const tok = (container.querySelector('input[type="password"][placeholder*="OPENCLAW_ADMIN_TOKEN"]')?.value || session.getAdminToken() || "").trim();
            return tok;
        };

        const refreshSecretsStatus = async () => {
            const token = getAdminToken();

            secretsStatus.textContent = "Loading...";
            secretsStatus.className = "openclaw-status openclaw-status moltbot-status";
            const res = await api.getSecretsStatus(token);
            if (!isCurrent()) return;
            if (res.ok) {
                const secrets = res.data?.secrets || {};
                const keys = Object.keys(secrets);
                if (keys.length === 0) {
                    secretsStatus.textContent = "✓ No stored keys.";
                    secretsStatus.className = "openclaw-status openclaw-status moltbot-status ok";
                } else {
                    secretsStatus.textContent = `✓ Stored keys: ${keys.join(", ")}`;
                    secretsStatus.className = "openclaw-status openclaw-status moltbot-status ok";
                }
            } else {
                const detail = [
                    res.status ? `HTTP ${res.status}` : null,
                    res.error || "Failed",
                ].filter(Boolean).join(" — ");
                secretsStatus.textContent = `✗ ${detail}`;
                secretsStatus.className = "openclaw-status openclaw-status moltbot-status error";
            }
        };

        const secretsBtnRow = document.createElement("div");
        secretsBtnRow.className = "openclaw-btn-row openclaw-btn-row moltbot-btn-row";

        const secretsStatusBtn = document.createElement("button");
        secretsStatusBtn.className = "openclaw-btn openclaw-btn moltbot-btn openclaw-btn-secondary openclaw-btn-secondary moltbot-btn-secondary";
        secretsStatusBtn.textContent = "Check Status";
        secretsStatusBtn.onclick = async () => {
            secretsStatusBtn.disabled = true;
            await refreshSecretsStatus();
            if (isCurrent()) secretsStatusBtn.disabled = false;
        };
        secretsBtnRow.appendChild(secretsStatusBtn);

        const secretsSaveBtn = document.createElement("button");
        secretsSaveBtn.className = "openclaw-btn openclaw-btn moltbot-btn";
        secretsSaveBtn.textContent = "Save Key";
        secretsSaveBtn.onclick = async () => {
            const token = getAdminToken();
            const apiKey = (secretKeyInput.value || "").trim();
            if (!apiKey) {
                secretsStatus.textContent = "Please paste an API key first.";
                secretsStatus.className = "openclaw-status openclaw-status moltbot-status error";
                return;
            }
            if (token) session.setAdminToken(token);

            secretsSaveBtn.disabled = true;
            secretsStatus.textContent = "Saving...";
            secretsStatus.className = "openclaw-status openclaw-status moltbot-status";

            const res = await api.saveSecret(secretProviderSelect.value, apiKey, token);
            if (!isCurrent()) return;
            if (res.ok) {
                secretKeyInput.value = "";
                secretsStatus.textContent = "✓ Saved to server store. Restart ComfyUI if needed.";
                secretsStatus.className = "openclaw-status openclaw-status moltbot-status ok";
                await refreshSecretsStatus();
            } else {
                const detail = [
                    res.status ? `HTTP ${res.status}` : null,
                    res.error || "Failed",
                ].filter(Boolean).join(" — ");
                secretsStatus.textContent = `✗ ${detail}`;
                secretsStatus.className = "openclaw-status openclaw-status moltbot-status error";
            }
            if (isCurrent()) secretsSaveBtn.disabled = false;
        };
        secretsBtnRow.appendChild(secretsSaveBtn);

        const secretsClearBtn = document.createElement("button");
        secretsClearBtn.className = "openclaw-btn openclaw-btn moltbot-btn openclaw-btn-danger openclaw-btn-danger moltbot-btn-danger";
        secretsClearBtn.textContent = "Clear Stored Key";
        secretsClearBtn.onclick = async () => {
            const token = getAdminToken();
            if (token) session.setAdminToken(token);

            secretsClearBtn.disabled = true;
            secretsStatus.textContent = "Clearing...";
            secretsStatus.className = "openclaw-status openclaw-status moltbot-status";

            const res = await api.clearSecret(secretProviderSelect.value, token);
            if (!isCurrent()) return;
            if (res.ok) {
                secretsStatus.textContent = "✓ Cleared.";
                secretsStatus.className = "openclaw-status openclaw-status moltbot-status ok";
                await refreshSecretsStatus();
            } else {
                const detail = [
                    res.status ? `HTTP ${res.status}` : null,
                    res.error || "Failed",
                ].filter(Boolean).join(" — ");
                secretsStatus.textContent = `✗ ${detail}`;
                secretsStatus.className = "openclaw-status openclaw-status moltbot-status error";
            }
            if (isCurrent()) secretsClearBtn.disabled = false;
        };
        secretsBtnRow.appendChild(secretsClearBtn);

        secretsContent.appendChild(secretsBtnRow);

        scroll.appendChild(secretsSec.container);
    }
}
