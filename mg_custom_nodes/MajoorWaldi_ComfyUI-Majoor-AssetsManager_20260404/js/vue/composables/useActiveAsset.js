/**
 * useActiveAsset.js — Module-level singleton for the asset currently shown in
 * the sidebar panel.
 *
 * The full asset object is transient display state, NOT stored in Pinia (which
 * would create deep watchers on a large, frequently-replaced object). Instead a
 * module-level `shallowRef` is used: Vue tracks only the top-level reference,
 * so replacing the asset triggers re-renders while mutations inside the object
 * don't cause spurious updates.
 *
 * When async metadata enrichment arrives (see SidebarView.js loadMetadataAsync),
 * `patchActiveAsset(patches)` merges the extra fields and calls `triggerRef` to
 * force a re-render with the updated data.
 *
 * Phase 5.
 */

import { shallowRef, triggerRef } from "vue";

// Singleton refs — sidebar is a singleton panel.
const _activeAsset = shallowRef(null);
const _onUpdateCallback = shallowRef(null);

// ─── public composable ───────────────────────────────────────────────────────

export function useActiveAsset() {
    return {
        activeAsset: _activeAsset,
        onUpdateCallback: _onUpdateCallback,
    };
}

// ─── imperative bridge (called from SidebarView.js) ─────────────────────────

/**
 * Set the asset currently shown in the sidebar.
 * Called from `showAssetInSidebar(sidebar, asset, onUpdate)` in SidebarView.js.
 */
export function setActiveAsset(asset, onUpdate) {
    _onUpdateCallback.value = typeof onUpdate === "function" ? onUpdate : null;
    _activeAsset.value = asset ?? null;
}

/**
 * Clear the active asset (called from `closeSidebar`).
 */
export function clearActiveAsset() {
    _activeAsset.value = null;
    _onUpdateCallback.value = null;
}

/**
 * Merge metadata patches into the active asset and trigger a Vue re-render.
 * Called from `loadMetadataAsync` in SidebarView.js when async enrichment data
 * arrives.
 *
 * @param {Object} patches — fields to merge into the current asset object
 * @param {Function|null} onUpdate — optional callback forwarded from showAssetInSidebar
 */
export function patchActiveAsset(patches, onUpdate) {
    if (!_activeAsset.value || !patches) return;
    Object.assign(_activeAsset.value, patches);
    triggerRef(_activeAsset);
    // Forward to sidebarController's onUpdate so card badges stay in sync.
    try {
        const cb = onUpdate ?? _onUpdateCallback.value;
        if (typeof cb === "function") cb(_activeAsset.value);
    } catch {
        /* ignore */
    }
}
