import { app } from "../../scripts/app.js";

const NODE_CLASS = "H3DecodeCacheHelper";
const MAX_TOKEN = 2147483647;

function isJapanese() {
    let locale = "";
    try {
        locale = String(app?.ui?.settings?.getSettingValue?.("Comfy.Locale") || "");
    } catch {}
    if (!locale) {
        locale = String(
            globalThis.document?.documentElement?.lang
            || globalThis.navigator?.language
            || "en",
        );
    }
    return /^ja(?:-|$)/i.test(locale);
}

function labels() {
    if (isJapanese()) {
        return {
            button: "キャッシュをクリア",
            tooltip: "次回QueueでこのHelperのキャッシュを破棄します。次回は通常Decodeとなります。普段は操作不要です。",
        };
    }
    return {
        button: "Clear cache",
        tooltip: "Clear this Helper's cache on the next Queue. The next run uses normal Decode. Usually no action is needed.",
    };
}

// Keep the backend widget in its original position. Only its presentation is
// hidden; Core still serializes and sends reset_token as an INT.
function hideToken(widget) {
    widget.hidden = true;
    widget.options ||= {};
    widget.options.hidden = true;
    widget.type = "converted-widget";
    widget.computeSize = () => [0, -4];
}

function withoutButton(node, button, operation) {
    const index = node.widgets?.indexOf(button) ?? -1;
    if (index >= 0) node.widgets.splice(index, 1);
    try {
        return operation();
    } finally {
        if (index >= 0) node.widgets.splice(index, 0, button);
    }
}

export function configureDecodeCacheHelper(node) {
    if (node?.comfyClass !== NODE_CLASS && node?.type !== NODE_CLASS) return;
    const token = node.widgets?.find((widget) => widget.name === "reset_token");
    if (!token || typeof node.addWidget !== "function") return;
    if (!node.__h3DecodeCacheClearButton) {
        const text = labels();
        // Create the control before hiding the numeric widget, so a failed
        // control installation leaves the traditional input available.
        const button = node.addWidget("button", text.button, null, () => {
            const current = node.widgets?.find((widget) => widget.name === "reset_token");
            if (!current) return;
            const value = Number(current.value);
            if (!Number.isInteger(value) || value < 0 || value > MAX_TOKEN) return;
            // Wrap only at the backend's INT limit; never submit an invalid INT.
            current.value = value === MAX_TOKEN ? 0 : value + 1;
            node.setDirtyCanvas?.(true, true);
        }, { serialize: false });
        button.serialize = false;
        button.options ||= {};
        button.options.serialize = false;
        button.tooltip = text.tooltip;
        node.__h3DecodeCacheClearButton = button;

        // Some frontend versions assign widgets_values by index even for
        // non-serialized controls. Exclude the button from both operations.
        const serialize = node.serialize;
        if (typeof serialize === "function") {
            node.serialize = function (...args) {
                return withoutButton(this, button, () => serialize.apply(this, args));
            };
        }
        const configure = node.configure;
        if (typeof configure === "function") {
            node.configure = function (...args) {
                const result = withoutButton(this, button, () => configure.apply(this, args));
                configureDecodeCacheHelper(this);
                return result;
            };
        }
    }
    hideToken(token);
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "H3Continuum.DecodeCacheHelper.ClearButton",
    nodeCreated: configureDecodeCacheHelper,
    loadedGraphNode: configureDecodeCacheHelper,
    afterConfigureGraph() {
        for (const node of app.graph?._nodes || []) configureDecodeCacheHelper(node);
    },
});
