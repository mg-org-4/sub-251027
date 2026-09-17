/** Config, health, history, trace, and secret route-family methods. */
import { apiURL } from "./openclaw_comfy_api.js";
import { normalizeComfyOutputRef } from "./openclaw_asset_refs.js";

export const configApiMethods = {
    async getHealth() {
        return this.fetch(this._path("/health"));
    },

    async getLogs(lines = 200) {
        return this.fetch(`${this._path("/logs/tail")}?lines=${lines}`);
    },

    async validateWebhook(payload) {
        return this.fetch(this._path("/webhook"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
    },

    async submitWebhook(payload) {
        return this.fetch(this._path("/webhook/submit"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
    },

    // R19: Capabilities

    async getCapabilities() {
        const now = Date.now();
        if (this._capabilitiesCache && (now - this._capabilitiesCacheTs) < 5000) {
            return this._capabilitiesCache;
        }
        const res = await this.fetch(this._path("/capabilities"));
        if (res?.ok) {
            this._capabilitiesCache = res;
            this._capabilitiesCacheTs = now;
        }
        return res;
    },

    async supportsAssistStreaming() {
        const caps = await this.getCapabilities();
        return !!caps?.ok && !!caps?.data?.features?.assist_streaming;
    },

    // F17: ComfyUI History

    async getHistory(promptId) {
        // /history is a ComfyUI native endpoint.
        // ComfyUI's shim handles it if we pass "/history/..."?
        // Wait, ComfyUI endpoints are usually /history.
        // fetchApi('/history/...') maps to /api/history/...
        // ComfyUI backend registers /history?
        // Checking ComfyUI source: yes, app.routes.get("/history"...)
        // But usually under /api ?
        // Actually ComfyUI 'fetchApi' prefixes with '/api'.
        // Does 'history' live under '/api/history'? Yes.
        const res = await this.fetch(`/history/${promptId}`);
        if (!res.ok) return res;

        // ComfyUI returns: { "<prompt_id>": { ...historyItem... } }
        const data = res.data;
        const historyItem = (data && typeof data === "object") ? data[promptId] : null;
        return { ...res, data: historyItem };
    },

    async getPromptQueue() {
        return this.fetch("/queue");
    },

    // R25: Trace timeline (optional)

    async getTrace(promptId) {
        return this.fetch(`${this._path("/trace")}/${encodeURIComponent(promptId)}`);
    },

    // Helper: Build ComfyUI /view URL

    buildViewUrl(filename, subfolder = "", type = "output") {
        const params = new URLSearchParams({ filename, type });
        if (subfolder) params.set("subfolder", subfolder);
        // apiURL returns the full path including standard base
        return apiURL(`/view?${params.toString()}`);
    },

    buildViewUrlForRef(imageRef) {
        const normalized = normalizeComfyOutputRef(imageRef);
        if (!normalized || !normalized.viewParams) {
            return "";
        }
        return apiURL(`/view?${new URLSearchParams(normalized.viewParams).toString()}`);
    },

    // R21/F20: Get config

    async getConfig() {
        return this.fetch(this._path("/config"));
    },

    // R21/S13/F20: Update config (requires admin token)

    async putConfig(config, adminToken) {
        return this.fetch(this._path("/config"), {
            method: "PUT",
            headers: {
                "Content-Type": "application/json",
                ...this._adminTokenHeaders(adminToken),
            },
            body: JSON.stringify(config),
        });
    },

    // F20: Test LLM connection (uses effective config, no api_key in frontend)

    async runLLMTest() {
        return this.fetch(this._path("/llm/test"), {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                ...this._adminTokenHeaders(),
            },
            body: JSON.stringify({}), // Empty body = use effective config
            timeout: 30000,
        });
    },

    // Backwards compatibility alias for settings_tab.js

    async testLLM(adminToken, overrides = null) {
        return this.fetch(this._path("/llm/test"), {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                ...this._adminTokenHeaders(adminToken),
            },
            // IMPORTANT: Settings UI uses this to test the currently selected provider/model
            // without requiring a config "Save" first. Backend accepts an empty body too.
            body: JSON.stringify(overrides || {}),
            timeout: 30000,
        });
    },

    // F20+: Fetch remote model list (best-effort; admin boundary)

    async getModelList(providerId, adminToken) {
        const q = providerId ? `?provider=${encodeURIComponent(providerId)}` : "";
        return this.fetch(`${this._path("/llm/models")}${q}`, {
            method: "GET",
            headers: {
                ...this._adminTokenHeaders(adminToken),
            },
            timeout: 30000,
        });
    },

    // --- S25: Secrets Management (Admin-gated) ---

    /**
     * Get secrets status (NO VALUES).
     * Admin boundary (token if configured; otherwise loopback-only).
     */

    async getSecretsStatus(adminToken) {
        return this.fetch(this._path("/secrets/status"), {
            method: "GET",
            headers: {
                ...this._adminTokenHeaders(adminToken),
            },
        });
    },

    /**
     * Save API key to server store.
     * Admin boundary (token if configured; otherwise loopback-only).
     *
     * @param {string} provider - Provider ID ("openai", "anthropic", "generic")
     * @param {string} apiKey - API key value (NEVER logged)
     * @param {string} adminToken - Admin token
     */

    async saveSecret(provider, apiKey, adminToken) {
        return this.fetch(this._path("/secrets"), {
            method: "PUT",
            headers: {
                "Content-Type": "application/json",
                ...this._adminTokenHeaders(adminToken),
            },
            body: JSON.stringify({
                provider: provider,
                api_key: apiKey,
            }),
        });
    },

    /**
     * Clear provider secret.
     * Admin boundary (token if configured; otherwise loopback-only).
     */

    async clearSecret(provider, adminToken) {
        return this.fetch(this._path(`/secrets/${encodeURIComponent(provider)}`), {
            method: "DELETE",
            headers: {
                ...this._adminTokenHeaders(adminToken),
            },
        });
    },

    // --- Assist Endpoints (F8/F21) ---

};
