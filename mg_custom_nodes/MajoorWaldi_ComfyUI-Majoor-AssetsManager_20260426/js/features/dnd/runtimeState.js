const assetDragStartCleanupByContainer = new WeakMap();

let internalDropOccurred = false;

export function getAssetDragStartCleanup(containerEl) {
    if (!containerEl) return null;
    return assetDragStartCleanupByContainer.get(containerEl) || null;
}

export function setAssetDragStartCleanup(containerEl, cleanup) {
    if (!containerEl) return null;
    if (typeof cleanup === "function") {
        assetDragStartCleanupByContainer.set(containerEl, cleanup);
        return cleanup;
    }
    assetDragStartCleanupByContainer.delete(containerEl);
    return null;
}

export function clearAssetDragStartCleanup(containerEl) {
    if (!containerEl) return;
    assetDragStartCleanupByContainer.delete(containerEl);
}

export function markInternalDropOccurred() {
    internalDropOccurred = true;
}

export function consumeInternalDropOccurred() {
    if (!internalDropOccurred) return false;
    internalDropOccurred = false;
    return true;
}

export function resetDndRuntimeState() {
    internalDropOccurred = false;
}
