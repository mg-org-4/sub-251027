/** GridView facade preserving public exports. */

export {
    buildGridSnapshotKey,
    createGridContainer,
    hydrateGridFromSnapshot,
    loadAssets,
    prepareGridForScopeSwitch,
    loadAssetsFromList,
    removeAssetsFromGrid,
    disposeGrid,
    upsertAsset,
    scrollGridToTop,
    captureAnchor,
    restoreAnchor,
    refreshGrid,
    bindGridScanListeners,
    disposeGridScanListeners,
} from "./GridView_impl.js";
