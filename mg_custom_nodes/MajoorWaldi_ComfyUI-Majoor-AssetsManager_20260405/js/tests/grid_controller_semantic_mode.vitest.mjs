import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { usePanelStore } from "../stores/usePanelStore.js";

const apiState = vi.hoisted(() => ({
    hybridSearch: vi.fn(),
    vectorSearch: vi.fn(),
    vectorStats: vi.fn(),
}));

vi.mock("../api/client.js", () => ({
    hybridSearch: apiState.hybridSearch,
    vectorSearch: apiState.vectorSearch,
    vectorStats: apiState.vectorStats,
}));

vi.mock("../app/toast.js", () => ({
    comfyToast: vi.fn(),
}));

vi.mock("../app/i18n.js", () => ({
    t: (_key, fallback, vars) => {
        if (!vars) return fallback;
        return String(fallback || "").replace(/\{(\w+)\}/g, (_m, key) => String(vars[key] ?? ""));
    },
}));

vi.mock("../app/settings.js", () => ({
    loadMajoorSettings: () => ({ ai: { vectorSearchEnabled: true } }),
}));

function createGridContainer() {
    return {
        dataset: {},
        dispatchEvent: vi.fn(),
    };
}

describe("gridController semantic mode", () => {
    beforeEach(() => {
        vi.clearAllMocks();
        globalThis.window = {};
        globalThis.CustomEvent = class {
            constructor(type, init = {}) {
                this.type = type;
                this.detail = init.detail;
            }
        };
    });

    it("uses hybrid AI search when the semantic toggle is active", async () => {
        setActivePinia(createPinia());
        const panelStore = usePanelStore();
        panelStore.scope = "output";
        panelStore.customRootId = "";
        panelStore.currentFolderRelativePath = "animals";
        panelStore.kindFilter = "image";
        panelStore.workflowOnly = true;
        panelStore.minRating = 4;
        panelStore.minSizeMB = 2;
        panelStore.maxSizeMB = 5;
        panelStore.resolutionCompare = "gte";
        panelStore.minWidth = 1920;
        panelStore.minHeight = 1080;
        panelStore.maxWidth = 3840;
        panelStore.maxHeight = 2160;
        panelStore.workflowType = "T2I";
        panelStore.dateRangeFilter = "this_month";
        panelStore.dateExactFilter = "";
        panelStore.sort = "mtime_desc";
        panelStore.selectedAssetIds = [];
        panelStore.activeAssetId = "";
        panelStore.collectionId = "";
        panelStore.collectionName = "";
        panelStore.viewScope = "";

        const { createGridController } =
            await import("../features/panel/controllers/gridController.js");

        apiState.hybridSearch.mockResolvedValue({
            ok: true,
            data: [{ id: 3, asset_id: 3, filename: "dino.png" }],
        });
        apiState.vectorSearch.mockResolvedValue({ ok: true, data: [] });
        apiState.vectorStats.mockResolvedValue({ ok: true, data: { total: 33 } });

        const loadAssets = vi.fn(async () => ({ ok: true, count: 0, total: 0 }));
        const loadAssetsFromList = vi.fn(async (_grid, assets) => ({
            ok: true,
            count: assets.length,
            total: assets.length,
        }));

        const controller = createGridController({
            gridContainer: createGridContainer(),
            loadAssets,
            loadAssetsFromList,
            getCollectionAssets: vi.fn(),
            disposeGrid: vi.fn(),
            getQuery: () => "dinosaure",
            searchInputEl: { dataset: { mjrSemanticMode: "1" } },
            state: panelStore,
        });

        await controller.reloadGrid();

        expect(apiState.hybridSearch).toHaveBeenCalledWith(
            "dinosaure",
            expect.objectContaining({
                topK: 100,
                scope: "output",
                subfolder: "animals",
                kind: "image",
                hasWorkflow: true,
                minRating: 4,
                minSizeMB: 2,
                maxSizeMB: 5,
                minWidth: 1920,
                minHeight: 1080,
                maxWidth: 3840,
                maxHeight: 2160,
                workflowType: "T2I",
                dateRange: "this_month",
            }),
        );
        expect(loadAssetsFromList).toHaveBeenCalledWith(
            expect.any(Object),
            [{ id: 3, asset_id: 3, filename: "dino.png" }],
            expect.objectContaining({ reset: true }),
        );
        expect(loadAssets).not.toHaveBeenCalled();
    });

    it("keeps long natural language queries on lexical search when semantic toggle is off", async () => {
        setActivePinia(createPinia());
        const panelStore = usePanelStore();
        panelStore.scope = "output";

        const { createGridController } =
            await import("../features/panel/controllers/gridController.js");

        apiState.hybridSearch.mockResolvedValue({ ok: true, data: [{ id: 9 }] });
        apiState.vectorSearch.mockResolvedValue({ ok: true, data: [{ id: 10 }] });

        const loadAssets = vi.fn(async () => ({ ok: true, count: 1, total: 1 }));
        const loadAssetsFromList = vi.fn(async (_grid, assets) => ({
            ok: true,
            count: assets.length,
            total: assets.length,
        }));

        const controller = createGridController({
            gridContainer: createGridContainer(),
            loadAssets,
            loadAssetsFromList,
            getCollectionAssets: vi.fn(),
            disposeGrid: vi.fn(),
            getQuery: () =>
                "a single organic flower is blooming at the center with delicate asymmetrical petals",
            searchInputEl: { dataset: { mjrSemanticMode: "0" } },
            state: panelStore,
        });

        await controller.reloadGrid();

        expect(loadAssets).toHaveBeenCalledTimes(1);
        expect(apiState.hybridSearch).not.toHaveBeenCalled();
        expect(apiState.vectorSearch).not.toHaveBeenCalled();
        expect(loadAssetsFromList).not.toHaveBeenCalled();
    });
});
