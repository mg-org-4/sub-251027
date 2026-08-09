// @vitest-environment happy-dom

import { mount } from "@vue/test-utils";
import { nextTick } from "vue";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../vue/components/common/TagsEditor.vue", () => ({
    default: { template: "<div />" },
}));

const MButtonStub = {
    inheritAttrs: false,
    template: '<button v-bind="$attrs"><slot /></button>',
};

function resetGridContextState(state, helpers) {
    helpers.closeAllGridContextMenus();
    state.portalOwnerId = "";
    state.mountedPortalIds.splice(0, state.mountedPortalIds.length);
    state.main.items = [];
    state.submenu.items = [];
}

describe("GridContextMenu tones", () => {
    beforeEach(() => {
        vi.resetModules();
        document.body.innerHTML = "";
        window.innerWidth = 1280;
        window.innerHeight = 720;
    });

    afterEach(async () => {
        const contextMenu = await import("../features/contextmenu/gridContextMenuState.js");
        resetGridContextState(contextMenu.gridContextMenuState, contextMenu);
        document.body.innerHTML = "";
    });

    it("renders action tones for graph map, rename, delete, and floating viewer", async () => {
        const { default: GridContextMenu } = await import("../vue/components/grid/GridContextMenu.vue");
        const contextMenu = await import("../features/contextmenu/gridContextMenuState.js");
        const wrapper = mount(GridContextMenu, {
            attachTo: document.body,
            global: {
                stubs: {
                    MButton: MButtonStub,
                    teleport: true,
                },
            },
        });

        contextMenu.openGridContextMenu({
            x: 20,
            y: 20,
            items: [
                { id: "graph", type: "item", label: "Open Graph Map", tone: "graph-map" },
                { id: "rename", type: "item", label: "Rename...", tone: "rename" },
                { id: "delete", type: "item", label: "Delete", tone: "delete" },
                {
                    id: "viewer",
                    type: "item",
                    label: "Open in Floating Viewer",
                    tone: "floating-viewer",
                },
            ],
        });
        await nextTick();

        expect(document.body.querySelector(".is-tone-graph-map")?.textContent).toContain(
            "Open Graph Map",
        );
        expect(document.body.querySelector(".is-tone-rename")?.textContent).toContain("Rename");
        expect(document.body.querySelector(".is-tone-delete")?.textContent).toContain("Delete");
        expect(document.body.querySelector(".is-tone-floating-viewer")?.textContent).toContain(
            "Open in Floating Viewer",
        );

        wrapper.unmount();
    });
});
