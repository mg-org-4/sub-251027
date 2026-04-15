import { getOptionalPanelStore } from "./getOptionalPanelStore.js";

/**
 * Creates a thin read/write facade over the Pinia panel store.
 *
 * The `state` and `keys` parameters are kept for call-site compatibility but
 * are no longer used — migration to Pinia-only (Phase 3) is complete.
 *
 * @param {*} _state - Unused (legacy transition parameter).
 * @param {string[]} _keys - Unused (legacy transition parameter).
 * @returns {{ panelStore, read, write, controllerState, isPiniaOnly: true }}
 */
export function createPanelStateBridge(_state, _keys = []) {
    const panelStore = getOptionalPanelStore();

    const read = (key, fallback = "") => {
        try {
            if (panelStore && key in panelStore) {
                const value = panelStore[key];
                if (value !== undefined) return value;
            }
        } catch (e) {
            console.debug?.(e);
        }
        return fallback;
    };

    const write = (key, value) => {
        try {
            if (panelStore && key in panelStore) {
                panelStore[key] = value;
            }
        } catch (e) {
            console.debug?.(e);
        }
    };

    const controllerState = () => panelStore;

    return {
        panelStore,
        hydrateStoreFromLegacy: () => {},
        read,
        write,
        controllerState,
        isPiniaOnly: true,
    };
}
