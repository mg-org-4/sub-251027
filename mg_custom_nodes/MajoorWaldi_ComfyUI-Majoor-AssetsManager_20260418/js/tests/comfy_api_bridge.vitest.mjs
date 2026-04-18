import { beforeEach, describe, expect, it, vi } from "vitest";

beforeEach(() => {
    vi.resetModules();
    globalThis.window = {};
    globalThis.globalThis = globalThis;
});

describe("comfyApiBridge", () => {
    it("discovers extension manager native surfaces from the app", async () => {
        const toast = { add: vi.fn() };
        const dialog = { confirm: vi.fn(), prompt: vi.fn() };
        const registerSidebarTab = vi.fn();
        const app = {
            ui: {},
            extensionManager: {
                toast,
                dialog,
                registerSidebarTab,
                activateSidebarTab: vi.fn(),
                registerCommand: vi.fn(),
                registerKeybinding: vi.fn(),
                registerBottomPanelTab: vi.fn(),
            },
        };

        const mod = await import("../app/comfyApiBridge.js");
        mod.setComfyApp(app);

        expect(mod.getExtensionToastApi()).toBe(toast);
        expect(mod.getExtensionDialogApi()).toBe(dialog);
        expect(mod.activateSidebarTabCompat(app, "majoor-assets")).toBe(true);
        expect(app.extensionManager.activateSidebarTab).toHaveBeenCalledWith("majoor-assets");
        expect(mod.registerSidebarTabCompat(app, { id: "majoor-assets" })).toBe(true);
        expect(registerSidebarTab).toHaveBeenCalledWith({ id: "majoor-assets" });
        expect(mod.registerCommandCompat(app, { id: "mjr.test" })).toBe(true);
        expect(mod.registerKeybindingCompat(app, { commandId: "mjr.test" })).toBe(true);
        expect(mod.registerBottomPanelTabCompat(app, { id: "mjr.feed" })).toBe(true);
    });

    it("supports nested sidebar and bottom panel controllers from newer ComfyUI runtimes", async () => {
        const toggleSidebarTab = vi.fn();
        const registerSidebarTab = vi.fn();
        const registerBottomPanelTab = vi.fn();
        const app = {
            ui: {},
            extensionManager: {
                sidebarTab: {
                    activeSidebarTabId: "",
                    toggleSidebarTab,
                    registerSidebarTab,
                },
                bottomPanel: {
                    registerBottomPanelTab,
                },
            },
        };

        const mod = await import("../app/comfyApiBridge.js");
        mod.setComfyApp(app);

        expect(mod.activateSidebarTabCompat(app, "majoor-assets")).toBe(true);
        expect(toggleSidebarTab).toHaveBeenCalledWith("majoor-assets");
        expect(mod.registerSidebarTabCompat(app, { id: "majoor-assets" })).toBe(true);
        expect(registerSidebarTab).toHaveBeenCalledWith({ id: "majoor-assets" });
        expect(mod.registerBottomPanelTabCompat(app, { id: "mjr.feed" })).toBe(true);
        expect(registerBottomPanelTab).toHaveBeenCalledWith({ id: "mjr.feed" });
    });

    it("does not toggle a sidebar tab that is already active", async () => {
        const toggleSidebarTab = vi.fn();
        const app = {
            ui: {},
            extensionManager: {
                sidebarTab: {
                    activeSidebarTabId: "majoor-assets",
                    toggleSidebarTab,
                },
            },
        };

        const mod = await import("../app/comfyApiBridge.js");
        mod.setComfyApp(app);

        expect(mod.activateSidebarTabCompat(app, "majoor-assets")).toBe(true);
        expect(toggleSidebarTab).not.toHaveBeenCalled();
    });

    it("returns false when native manager APIs are unavailable", async () => {
        const app = { ui: {} };
        const mod = await import("../app/comfyApiBridge.js");
        mod.setComfyApp(app);

        expect(mod.getExtensionToastApi()).toBeNull();
        expect(mod.getExtensionDialogApi()).toBeNull();
        expect(mod.activateSidebarTabCompat(app, "missing")).toBe(false);
        expect(mod.registerSidebarTabCompat(app, { id: "missing" })).toBe(false);
        expect(mod.registerCommandCompat(app, { id: "mjr.test" })).toBe(false);
        expect(mod.registerBottomPanelTabCompat(app, { id: "mjr.feed" })).toBe(false);
    });
});
