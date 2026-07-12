/** Model search, download, installation, and PNG-info route-family methods. */

export const modelApiMethods = {
    async searchModels(params = {}) {
        const qs = new URLSearchParams();
        if (params.q) qs.set("q", String(params.q));
        if (params.source) qs.set("source", String(params.source));
        if (params.model_type) qs.set("model_type", String(params.model_type));
        if (typeof params.installed === "boolean") qs.set("installed", params.installed ? "true" : "false");
        if (params.limit != null) qs.set("limit", String(params.limit));
        if (params.offset != null) qs.set("offset", String(params.offset));
        const suffix = qs.toString() ? `?${qs.toString()}` : "";
        return this.fetch(`${this._path("/models/search")}${suffix}`, {
            headers: { ...this._adminTokenHeaders() }
        });
    },

    async createModelDownloadTask(payload) {
        return this.fetch(this._path("/models/downloads"), {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                ...this._adminTokenHeaders()
            },
            body: JSON.stringify(payload || {})
        });
    },

    async listModelDownloadTasks(params = {}) {
        const qs = new URLSearchParams();
        if (params.state) qs.set("state", String(params.state));
        if (params.limit != null) qs.set("limit", String(params.limit));
        if (params.offset != null) qs.set("offset", String(params.offset));
        if (params.since_seq != null) qs.set("since_seq", String(params.since_seq));
        const suffix = qs.toString() ? `?${qs.toString()}` : "";
        return this.fetch(`${this._path("/models/downloads")}${suffix}`, {
            headers: { ...this._adminTokenHeaders() }
        });
    },

    async getModelDownloadTask(taskId) {
        return this.fetch(`${this._path("/models/downloads")}/${encodeURIComponent(taskId)}`, {
            headers: { ...this._adminTokenHeaders() }
        });
    },

    async cancelModelDownloadTask(taskId) {
        return this.fetch(`${this._path("/models/downloads")}/${encodeURIComponent(taskId)}/cancel`, {
            method: "POST",
            headers: { ...this._adminTokenHeaders() }
        });
    },

    async importDownloadedModel(payload) {
        return this.fetch(this._path("/models/import"), {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                ...this._adminTokenHeaders()
            },
            body: JSON.stringify(payload || {})
        });
    },

    async listModelInstallations(params = {}) {
        const qs = new URLSearchParams();
        if (params.model_type) qs.set("model_type", String(params.model_type));
        if (params.limit != null) qs.set("limit", String(params.limit));
        if (params.offset != null) qs.set("offset", String(params.offset));
        const suffix = qs.toString() ? `?${qs.toString()}` : "";
        return this.fetch(`${this._path("/models/installations")}${suffix}`, {
            headers: { ...this._adminTokenHeaders() }
        });
    },

    async parsePngInfo(imageB64) {
        return this.fetch(this._path("/pnginfo"), {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                ...this._adminTokenHeaders()
            },
            body: JSON.stringify({
                image_b64: String(imageB64 || "")
            }),
            timeout: 30000,
        });
    },

    // --- R71: Job Events ---

    /**
     * Poll for recent events (fallback).
     * @param {number} lastSeq - Sequence ID to start from
     */

};
