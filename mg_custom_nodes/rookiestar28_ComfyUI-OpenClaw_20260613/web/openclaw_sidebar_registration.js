export function registerOpenClawSidebar(app, tabDefinition) {
    const currentRegister = app?.extensionManager?.sidebarTab?.registerSidebarTab;
    if (typeof currentRegister === "function") {
        currentRegister.call(app.extensionManager.sidebarTab, tabDefinition);
        return { ok: true, api: "sidebarTab.registerSidebarTab" };
    }

    const legacyRegister = app?.extensionManager?.registerSidebarTab;
    if (typeof legacyRegister === "function") {
        legacyRegister.call(app.extensionManager, tabDefinition);
        return { ok: true, api: "extensionManager.registerSidebarTab" };
    }

    return {
        ok: false,
        api: "missing",
        error: new Error("Sidebar API missing"),
    };
}
