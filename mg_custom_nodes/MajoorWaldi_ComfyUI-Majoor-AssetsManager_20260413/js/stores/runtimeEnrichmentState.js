import { getOptionalRuntimeStore } from "./getOptionalRuntimeStore.js";

const RUNTIME_STATE_KEY = Symbol.for("majoor.assets_manager.runtime_state");

function getLegacyRuntimeState() {
    try {
        const root = typeof globalThis !== "undefined" ? globalThis : {};
        if (!root[RUNTIME_STATE_KEY] || typeof root[RUNTIME_STATE_KEY] !== "object") {
            root[RUNTIME_STATE_KEY] = {
                api: null,
                assetsDeletedHandler: null,
                enrichmentActive: false,
                enrichmentQueueLength: 0,
            };
        }
        return root[RUNTIME_STATE_KEY];
    } catch {
        return {
            api: null,
            assetsDeletedHandler: null,
            enrichmentActive: false,
            enrichmentQueueLength: 0,
        };
    }
}

export function setRuntimeEnrichmentState(active, queueLength) {
    const store = getOptionalRuntimeStore();
    if (store) {
        store.setEnrichmentState(active, queueLength);
        return;
    }
    const state = getLegacyRuntimeState();
    state.enrichmentActive = !!active;
    state.enrichmentQueueLength = Math.max(0, Number(queueLength || 0) || 0);
}

export function getRuntimeEnrichmentState() {
    const store = getOptionalRuntimeStore();
    if (store) {
        return store.getEnrichmentState();
    }
    const state = getLegacyRuntimeState();
    return {
        active: !!state.enrichmentActive,
        queueLength: Math.max(0, Number(state.enrichmentQueueLength || 0) || 0),
    };
}
