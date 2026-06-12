import { app } from "../../scripts/app.js";

const RESOLUTION_CHOICES = {
    "2k": [
        "512x512 (1:1)",
        "576x432 (4:3)",
        "432x576 (3:4)",
        "624x416 (3:2)",
        "416x624 (2:3)",
        "672x384 (16:9)",
        "384x672 (9:16)",
        "784x336 (21:9)",
        "336x784 (9:21)",
    ],
    "2kto4k": [
        "1024x1024 (1:1)",
        "1024x768 (4:3)",
        "768x1024 (3:4)",
        "1008x672 (3:2)",
        "672x1008 (2:3)",
        "1024x576 (16:9)",
        "576x1024 (9:16)",
        "1008x432 (21:9)",
        "432x1008 (9:21)",
    ],
};

function ratioFromLabel(label) {
    return String(label ?? "").match(/\(([^)]+)\)/)?.[1] ?? "1:1";
}

function installResolutionSwitcher(node) {
    const modeWidget = node.widgets?.find((widget) => widget.name === "pid_ckpt_type");
    const resolutionWidget = node.widgets?.find((widget) => widget.name === "resolution");
    if (!modeWidget || !resolutionWidget || resolutionWidget.__pidResolutionSwitcherInstalled) {
        return;
    }

    resolutionWidget.__pidResolutionSwitcherInstalled = true;

    const updateResolutionChoices = () => {
        const mode = modeWidget.value === "2kto4k" ? "2kto4k" : "2k";
        const choices = RESOLUTION_CHOICES[mode];
        const previousRatio = ratioFromLabel(resolutionWidget.value);

        if (resolutionWidget.options) {
            resolutionWidget.options.values = choices;
        }

        if (!choices.includes(resolutionWidget.value)) {
            resolutionWidget.value = choices.find((choice) => ratioFromLabel(choice) === previousRatio) ?? choices[0];
        }

        node.setDirtyCanvas(true, true);
    };

    const oldModeCallback = modeWidget.callback;
    modeWidget.callback = function pidModeCallback(value, ...args) {
        const result = oldModeCallback?.apply(this, [value, ...args]);
        updateResolutionChoices();
        return result;
    };

    // Run once after Comfy has restored widget values from workflow JSON.
    requestAnimationFrame(updateResolutionChoices);
}

app.registerExtension({
    name: "ComfyUI-PiD.EmptyLatentImage",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "PiDEmptyLatentImage") {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function pidEmptyLatentOnNodeCreated(...args) {
            const result = onNodeCreated?.apply(this, args);
            installResolutionSwitcher(this);
            return result;
        };
    },
});
