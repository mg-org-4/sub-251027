/** Planner and refiner route-family methods. */

export const generationApiMethods = {
    async runPlanner(params, signal = null) {
        return this.fetch(this._path("/assist/planner"), {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                ...this._adminTokenHeaders(),
            },
            body: JSON.stringify(params),
            timeout: 60000, // LLM calls may be slow
            signal, // R38-Lite: Pass signal
        });
    },

    async listPlannerProfiles(signal = null) {
        return this.fetch(this._path("/assist/planner/profiles"), {
            headers: {
                ...this._adminTokenHeaders(),
            },
            signal,
        });
    },

    async runPlannerStream(params, { signal = null, onEvent = null } = {}) {
        return this.streamSSEPost(this._path("/assist/planner/stream"), params, {
            signal,
            timeout: 60000,
            onEvent,
        });
    },

    /**
     * Run Prompt Refiner.
     * @param {object} params - { image_b64, orig_positive, orig_negative, issue, params_json, goal }
     * @param {AbortSignal} signal - Optional AbortSignal for cancellation (R38-Lite)
     */

    async runRefiner(params, signal = null) {
        return this.fetch(this._path("/assist/refiner"), {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                ...this._adminTokenHeaders(),
            },
            body: JSON.stringify(params),
            timeout: 60000,
            signal, // R38-Lite: Pass signal
        });
    },

    async runRefinerStream(params, { signal = null, onEvent = null } = {}) {
        return this.streamSSEPost(this._path("/assist/refiner/stream"), params, {
            signal,
            timeout: 60000,
            onEvent,
        });
    },

    // --- F22: Presets ---

};
