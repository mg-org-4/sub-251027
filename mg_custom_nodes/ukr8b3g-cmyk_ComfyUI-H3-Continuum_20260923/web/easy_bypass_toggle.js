// Convenience toggles delegate to Core node.mode; Core owns persistence/drawing.
const NODE_MODE_ALWAYS =
    typeof LiteGraph !== "undefined" && Number.isInteger(LiteGraph.ALWAYS)
        ? LiteGraph.ALWAYS : 0;
const NODE_MODE_BYPASS =
    typeof LiteGraph !== "undefined" && Number.isInteger(LiteGraph.BYPASS)
        ? LiteGraph.BYPASS : 4;

function isConfiguredNode(node, nodeClass) {
    return node?.comfyClass === nodeClass || node?.type === nodeClass;
}

export function findEasyBypassWidget(node) {
    const name = node.__h3EasyBypassToggle?.widgetName;
    return name ? node.widgets?.find((widget) => widget.name === name) : undefined;
}

function syncWidgetFromMode(node) {
    const widget = findEasyBypassWidget(node);
    if (widget) widget.value = node.mode === NODE_MODE_ALWAYS;
}

function findModeDescriptor(node) {
    for (let owner = node; owner; owner = Object.getPrototypeOf(owner)) {
        const descriptor = Object.getOwnPropertyDescriptor(owner, "mode");
        if (descriptor) return descriptor;
    }
    return undefined;
}

function installModeSourceOfTruth(node) {
    if (node.__h3EasyBypassModeObserved) return true;
    const own = Object.getOwnPropertyDescriptor(node, "mode");
    if (own?.configurable === false) return false;
    const descriptor = findModeDescriptor(node);
    const accessor = descriptor && ("get" in descriptor || "set" in descriptor);
    // Never replace a read-only property or manufacture a second store value.
    if (accessor && (!descriptor.get || !descriptor.set)) return false;
    if (descriptor && !accessor && descriptor.writable === false) return false;
    let currentMode = Number(node.mode ?? NODE_MODE_ALWAYS);
    Object.defineProperty(node, "mode", {
        configurable: true,
        enumerable: own?.enumerable ?? descriptor?.enumerable ?? true,
        get() {
            return accessor ? descriptor.get.call(this) : currentMode;
        },
        set(value) {
            if (accessor) descriptor.set.call(this, value);
            else currentMode = Number(value);
            syncWidgetFromMode(this);
        },
    });
    node.__h3EasyBypassModeObserved = true;
    return true;
}

function setModeFromToggle(node, enabled) {
    node.mode = enabled ? NODE_MODE_ALWAYS : NODE_MODE_BYPASS;
    syncWidgetFromMode(node);
    node.graph?.change?.();
    node.setDirtyCanvas?.(true, true);
}

export function configureEasyBypassToggleNode(
    node, { nodeClass, widgetName, tooltip },
) {
    if (!isConfiguredNode(node, nodeClass)) return;
    node.__h3EasyBypassToggle = { nodeClass, widgetName };
    let widget = findEasyBypassWidget(node);
    if (!widget) {
        widget = node.addWidget(
            "toggle", widgetName, node.mode === NODE_MODE_ALWAYS,
            (value) => setModeFromToggle(node, Boolean(value)),
            { on: "ON", off: "OFF", serialize: false },
        );
        widget.serialize = false; // Workflow persistence.
        widget.options ||= {};
        widget.options.serialize = false; // API prompt serialization.
        widget.tooltip = tooltip;
        // Append only. Never splice/reassign widgets in draw/serialize/configure:
        // newer Core uses a stable mutation view backed by its widget store.
    }
    if (!installModeSourceOfTruth(node)) {
        console.warn(`${nodeClass}: native mode could not be observed; use Core Bypass.`);
    }
    syncWidgetFromMode(node);
    node.setDirtyCanvas?.(true, true);
}

// Retained for existing callers; new Video Adapter has no Enable widget.
export function configureExistingEasyBypassWidgetNode(
    node, { nodeClass, widgetNames },
) {
    if (!isConfiguredNode(node, nodeClass)) return;
    const names = Array.isArray(widgetNames) ? widgetNames : [widgetNames];
    const widget = node.widgets?.find((item) => names.includes(item.name));
    if (!widget) return;
    node.__h3EasyBypassToggle = { nodeClass, widgetName: widget.name };
    if (!widget.__h3EasyBypassCallbackObserved) {
        const originalCallback = widget.callback;
        widget.callback = function (value, ...args) {
            const result = originalCallback?.call(this, value, ...args);
            setModeFromToggle(node, Boolean(value));
            return result;
        };
        widget.__h3EasyBypassCallbackObserved = true;
    }
    installModeSourceOfTruth(node);
    if (node.mode === NODE_MODE_BYPASS) syncWidgetFromMode(node);
    else setModeFromToggle(node, Boolean(widget.value));
}
