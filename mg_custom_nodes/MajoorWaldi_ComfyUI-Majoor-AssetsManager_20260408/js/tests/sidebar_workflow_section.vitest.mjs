// @vitest-environment happy-dom

import { beforeEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";

const drawWorkflowMinimap = vi.fn();
const synthesizeWorkflowFromPromptGraph = vi.fn(() => null);

let settingsState;

vi.mock("../components/sidebar/utils/minimap.js", () => ({
    drawWorkflowMinimap,
    synthesizeWorkflowFromPromptGraph,
}));

vi.mock("../app/settings.js", () => ({
    loadMajoorSettings: () => settingsState,
    saveMajoorSettings: (next) => {
        settingsState = next;
    },
}));

vi.mock("../app/settingsStore.js", () => ({
    MINIMAP_LEGACY_SETTINGS_KEY: "legacy-minimap",
}));

vi.mock("../app/i18n.js", () => ({
    t: (_key, fallback) => fallback,
}));

describe("SidebarWorkflowSection", () => {
    beforeEach(() => {
        settingsState = {
            workflowMinimap: {
                enabled: true,
                nodeColors: true,
                showLinks: true,
                showGroups: true,
                renderBypassState: true,
                renderErrorState: true,
                showViewport: true,
                showNodeLabels: false,
                size: "comfortable",
            },
        };
        drawWorkflowMinimap.mockReset();
        globalThis.ResizeObserver = class {
            observe() {}
            disconnect() {}
        };
        HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
            setTransform: vi.fn(),
        }));
    });

    it("shows workflow stats and persists size and label toggles", async () => {
        const { default: SidebarWorkflowSection } = await import(
            "../vue/components/panel/sidebar/SidebarWorkflowSection.vue"
        );

        const wrapper = mount(SidebarWorkflowSection, {
            props: {
                asset: {
                    has_generation_data: true,
                    workflow: {
                        nodes: [{ id: 1, pos: [0, 0], size: [180, 80], type: "KSampler" }],
                        links: [[1, 1, 0, 2, 0, "LINK"]],
                        groups: [{ bounding: [0, 0, 220, 120] }],
                        extra: {},
                    },
                },
            },
            attachTo: document.body,
        });

        expect(wrapper.text()).toContain("Nodes");
        expect(wrapper.text()).toContain("Links");
        expect(wrapper.text()).toContain("Groups");
        expect(wrapper.text()).toContain("Embedded");

        await wrapper.find('button[title="Expanded minimap"]').trigger("click");
        expect(settingsState.workflowMinimap.size).toBe("expanded");

        await wrapper.find('button[title="Minimap settings"]').trigger("click");
        await wrapper.findAll("button").find((node) => node.text().includes("Node Labels")).trigger("click");
        expect(settingsState.workflowMinimap.showNodeLabels).toBe(true);

        const canvas = wrapper.find("canvas");
        expect(canvas.attributes("style")).toContain("height: 220px");
        expect(drawWorkflowMinimap).toHaveBeenCalled();

        wrapper.unmount();
    });
});