/** Preset, approval, pack, preflight, and checkpoint route-family methods. */
import { fetchApi, fileURL } from "./openclaw_comfy_api.js";
import { getApiPathCandidates } from "./openclaw_compat.js";

export const resourceApiMethods = {
    async listPresets(category) {
        const query = category ? `?category=${encodeURIComponent(category)}` : "";
        return this.fetch(`${this._path("/presets")}${query}`, {
            headers: {
                ...this._adminTokenHeaders(),
            },
        });
    },

    async getPreset(id) {
        return this.fetch(`${this._path("/presets")}/${encodeURIComponent(id)}`, {
            headers: {
                ...this._adminTokenHeaders(),
            },
        });
    },

    async createPreset(data) {
        return this.fetch(this._path("/presets"), {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                ...this._adminTokenHeaders(),
            },
            body: JSON.stringify(data),
        });
    },

    async updatePreset(id, data) {
        return this.fetch(`${this._path("/presets")}/${encodeURIComponent(id)}`, {
            method: "PUT",
            headers: {
                "Content-Type": "application/json",
                ...this._adminTokenHeaders(),
            },
            body: JSON.stringify(data),
        });
    },

    async deletePreset(id) {
        return this.fetch(`${this._path("/presets")}/${encodeURIComponent(id)}`, {
            method: "DELETE",
            headers: {
                ...this._adminTokenHeaders(),
            },
        });
    },
    // --- S7: Approval Gates ---

    async getApprovals({ status, limit = 100, offset = 0 } = {}) {
        const params = new URLSearchParams({ limit, offset });
        if (status) params.set("status", status);

        return this.fetch(`${this._path("/approvals")}?${params.toString()}`, {
            headers: {
                ...this._adminTokenHeaders(),
            },
        });
    },

    async getApproval(id) {
        return this.fetch(`${this._path("/approvals")}/${encodeURIComponent(id)}`, {
            headers: {
                ...this._adminTokenHeaders(),
            },
        });
    },

    async approveRequest(id, { actor = "web_user", autoExecute = true } = {}) {
        return this.fetch(`${this._path("/approvals")}/${encodeURIComponent(id)}/approve`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                ...this._adminTokenHeaders(),
            },
            body: JSON.stringify({ actor, auto_execute: autoExecute }),
        });
    },

    async rejectRequest(id, { actor = "web_user" } = {}) {
        return this.fetch(`${this._path("/approvals")}/${encodeURIComponent(id)}/reject`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                ...this._adminTokenHeaders(),
            },
            body: JSON.stringify({ actor }),
        });
    },

    // --- S8/F11: Asset Packs ---

    async getPacks() {
        return this.fetch(this._path("/packs"), {
            headers: {
                ...this._adminTokenHeaders(),
            },
        });
    },

    async importPack(file, overwrite = false) {
        const formData = new FormData();
        formData.append("file", file);

        const query = overwrite ? "?overwrite=true" : "";

        return this.fetch(`${this._path("/packs/import")}${query}`, {
            method: "POST",
            headers: {
                ...this._adminTokenHeaders(),
                // Let browser set Content-Type for FormData
            },
            body: formData,
        });
    },

    async exportPack(name, version) {
        // Return URL for download (or blob fetch if needed)
        // Since it requires a token, we might need to fetch blob
        // But for simplicity, we can use a token parameter if supported, or fetch blob and create object URL.

        // Fetch as blob
        // R26: Use fetchApi to ensure base path
        const primaryPath = `${this._path("/packs/export")}/${encodeURIComponent(name)}/${encodeURIComponent(version)}`;
        const legacyPath = getApiPathCandidates(primaryPath)[1];

        const headers = this._adminTokenHeaders();

        let res = await fetchApi(primaryPath, { headers });
        if (res.status === 404) res = await fetchApi(legacyPath, { headers });

        if (res.status === 404) {
            try {
                res = await fetch(fileURL(primaryPath), { headers });
            } catch { }
        }
        if (res.status === 404) {
            try {
                res = await fetch(fileURL(legacyPath), { headers });
            } catch { }
        }

        if (res.ok) {
            const blob = await res.blob();
            return { ok: true, data: blob };
        }

        // If error, try to parse json error
        let error = "Download failed";
        try {
            const json = await res.json();
            error = json.error || error;
        } catch (e) { }

        return { ok: false, error };
    },

    async deletePack(name, version) {
        return this.fetch(`${this._path("/packs")}/${encodeURIComponent(name)}/${encodeURIComponent(version)}`, {
            method: "DELETE",
            headers: {
                ...this._adminTokenHeaders(),
            },
        });
    },

    // --- R42/F28: Preflight & Explorer ---

    async runPreflight(workflow) {
        return this.fetch(this._path("/preflight"), {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                ...this._adminTokenHeaders(),
            },
            body: JSON.stringify(workflow),
        });
    },

    async getInventory() {
        return this.fetch(this._path("/preflight/inventory"), {
            method: "GET",
            headers: {
                ...this._adminTokenHeaders(),
            },
        });
    },

    // --- R47: Checkpoints ---

    async listCheckpoints() {
        return this.fetch(this._path("/checkpoints"), {
            headers: { ...this._adminTokenHeaders() }
        });
    },

    async createCheckpoint(name, workflow, description = "") {
        return this.fetch(this._path("/checkpoints"), {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                ...this._adminTokenHeaders()
            },
            body: JSON.stringify({ name, workflow, description })
        });
    },

    async getCheckpoint(id) {
        return this.fetch(`${this._path("/checkpoints")}/${encodeURIComponent(id)}`, {
            headers: { ...this._adminTokenHeaders() }
        });
    },

    async deleteCheckpoint(id) {
        return this.fetch(`${this._path("/checkpoints")}/${encodeURIComponent(id)}`, {
            method: "DELETE",
            headers: { ...this._adminTokenHeaders() }
        });
    },

    // --- F54: Model Search / Download / Import ---

};
