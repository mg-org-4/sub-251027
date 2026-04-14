import { ensureViewerMetadataAsset } from "./genInfo.js";
import { getAssetMetadata, getFileMetadataScoped } from "../../api/client.js";
import { MFV_MODES } from "./floatingViewerConstants.js";

export function loadFloatingViewerMediaA(viewer, fileData, { autoMode = false } = {}) {
    viewer._mediaA = fileData || null;
    viewer._resetMfvZoom();
    if (autoMode && viewer._mode !== MFV_MODES.SIMPLE) {
        viewer._mode = MFV_MODES.SIMPLE;
        viewer._updateModeBtnUI();
    }
    if (viewer._mediaA && typeof ensureViewerMetadataAsset === "function") {
        const gen = ++viewer._refreshGen;
        (async () => {
            try {
                const enriched = await ensureViewerMetadataAsset(viewer._mediaA, {
                    getAssetMetadata,
                    getFileMetadataScoped,
                });
                if (viewer._refreshGen !== gen) return;
                if (enriched && typeof enriched === "object") {
                    viewer._mediaA = enriched;
                    viewer._refresh();
                }
            } catch (e) {
                console.debug?.("[MFV] metadata enrich error", e);
            }
        })();
    } else {
        viewer._refresh();
    }
}

export function loadFloatingViewerMediaPair(viewer, a, b) {
    viewer._mediaA = a || null;
    viewer._mediaB = b || null;
    viewer._resetMfvZoom();
    if (viewer._mode === MFV_MODES.SIMPLE) {
        viewer._mode = MFV_MODES.AB;
        viewer._updateModeBtnUI();
    }
    const gen = ++viewer._refreshGen;
    const hydrate = async (asset) => {
        if (!asset) return asset;
        try {
            const enriched = await ensureViewerMetadataAsset(asset, {
                getAssetMetadata,
                getFileMetadataScoped,
            });
            return enriched || asset;
        } catch {
            return asset;
        }
    };
    (async () => {
        const [A, B] = await Promise.all([hydrate(viewer._mediaA), hydrate(viewer._mediaB)]);
        if (viewer._refreshGen !== gen) return;
        viewer._mediaA = A || null;
        viewer._mediaB = B || null;
        viewer._refresh();
    })();
}

export function loadFloatingViewerMediaQuad(viewer, a, b, c, d) {
    viewer._mediaA = a || null;
    viewer._mediaB = b || null;
    viewer._mediaC = c || null;
    viewer._mediaD = d || null;
    viewer._resetMfvZoom();
    if (viewer._mode !== MFV_MODES.GRID) {
        viewer._mode = MFV_MODES.GRID;
        viewer._updateModeBtnUI();
    }
    const gen = ++viewer._refreshGen;
    const hydrate = async (asset) => {
        if (!asset) return asset;
        try {
            const enriched = await ensureViewerMetadataAsset(asset, {
                getAssetMetadata,
                getFileMetadataScoped,
            });
            return enriched || asset;
        } catch {
            return asset;
        }
    };
    (async () => {
        const [A, B, C, D] = await Promise.all([
            hydrate(viewer._mediaA),
            hydrate(viewer._mediaB),
            hydrate(viewer._mediaC),
            hydrate(viewer._mediaD),
        ]);
        if (viewer._refreshGen !== gen) return;
        viewer._mediaA = A || null;
        viewer._mediaB = B || null;
        viewer._mediaC = C || null;
        viewer._mediaD = D || null;
        viewer._refresh();
    })();
}
