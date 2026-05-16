import { onUnmounted, watch } from "vue";
import { installGridKeyboard } from "../../features/grid/GridKeyboard.js";

/**
 * useGridKeyboard — Vue lifecycle wrapper around the legacy GridKeyboard engine.
 *
 * Phase 4.4 step: the keyboard lifecycle moves under Vue even though the
 * underlying keyboard implementation is still the legacy imperative module.
 *
 * @param {import("vue").Ref<HTMLElement|null>} gridContainerRef
 */
export function bindLegacyGridKeyboard(container) {
    if (!container) return () => {};
    try {
        const keyboard = installGridKeyboard({
            gridContainer: container,
            getState: () => container._mjrGetGridState?.() || {},
            getSelectedAssets: () => container._mjrGetSelectedAssets?.() || [],
            getActiveAsset: () => container._mjrGetActiveAsset?.() || null,
            onAssetChanged: (asset) => {
                container._mjrOnKeyboardAssetChanged?.(asset);
            },
            onOpenDetails: () => {
                container._mjrOpenKeyboardDetails?.();
            },
        });
        keyboard.bind();
        container._mjrGridKeyboard = keyboard;
        return () => {
            try {
                keyboard.dispose?.();
            } catch (e) {
                console.debug?.("[useGridKeyboard] dispose failed", e);
            }
            if (container._mjrGridKeyboard === keyboard) {
                container._mjrGridKeyboard = null;
            }
        };
    } catch (e) {
        console.debug?.("[useGridKeyboard] bind failed", e);
        return () => {};
    }
}

export function useGridKeyboard(gridContainerRef) {
    let disposeKeyboard = null;

    const cleanup = () => {
        try {
            disposeKeyboard?.();
        } catch (e) {
            console.debug?.("[useGridKeyboard] cleanup failed", e);
        }
        disposeKeyboard = null;
    };

    watch(
        gridContainerRef,
        (next, prev) => {
            if (prev) cleanup();
            if (next) {
                disposeKeyboard = bindLegacyGridKeyboard(next);
            }
        },
        { immediate: true },
    );

    onUnmounted(cleanup);
}
