/**
 * Global state key registry for Majoor Assets Manager.
 *
 * This module is now a **thin shim** over the Pinia store
 * `useRuntimeStore` (js/stores/useRuntimeStore.js).
 *
 * When Pinia is active (Vue mounted), reads and writes are forwarded to the
 * reactive store.  Before Pinia is ready (early bootstrap), the legacy
 * Symbol-keyed globalThis fallback is used so the extension never crashes.
 *
 * Consumers do NOT need to change their imports — this file keeps the same
 * public API: getRuntimeState, setRuntimeStatePatch, setEnrichmentState,
 * getEnrichmentState.
 */

export {
    getRuntimeEnrichmentState as getEnrichmentState,
    setRuntimeEnrichmentState as setEnrichmentState,
} from "../stores/runtimeEnrichmentState.js";

const RUNTIME_STATE_KEY = Symbol.for("majoor.assets_manager.runtime_state");

function _runtimeRoot() {
    try {
        return typeof globalThis !== "undefined" ? globalThis : {};
    } catch {
        return {};
    }
}

export function getRuntimeState() {
    const root = _runtimeRoot();
    try {
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

const _BLOCKED_KEYS = new Set(["__proto__", "constructor", "prototype"]);

export function setRuntimeStatePatch(patch = {}) {
    const state = getRuntimeState();
    try {
        const safePatch = patch || {};
        for (const key of Object.keys(safePatch)) {
            if (!_BLOCKED_KEYS.has(key)) {
                state[key] = safePatch[key];
            }
        }
    } catch (e) {
        console.debug?.(e);
    }
    return state;
}
