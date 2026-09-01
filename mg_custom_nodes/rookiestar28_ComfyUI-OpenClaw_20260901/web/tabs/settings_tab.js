/**
 * S26: Settings Tab (simplified, collapsible secrets)
 */
import { getAdminErrorMessage } from "../admin_errors.js";
import { openclawApi } from "../openclaw_api.js";
import { OpenClawSession } from "../openclaw_session.js";
import { beginSettingsRender, disposeSettingsRender, finishSettingsRender } from "./settings_tab_lifecycle.js";
import { renderSettingsLlm } from "./settings_tab_llm.js";
import { renderSettingsLogs } from "./settings_tab_logs.js";
import { renderSettingsSecrets } from "./settings_tab_secrets.js";
import { renderSettingsStatus } from "./settings_tab_status.js";

export const settingsTab = {
    id: "settings",
    title: "Settings",
    icon: "pi pi-cog",
    render: async (container) => {
        const lifecycle = beginSettingsRender(container);
        // IMPORTANT (UI layout): `.openclaw-content` has `overflow: hidden`.
        // This tab MUST render its own scroll container or lower sections are clipped.
        container.innerHTML = `
            <div class="openclaw-panel openclaw-panel moltbot-panel">
                <div class="openclaw-scroll-area openclaw-scroll-area moltbot-scroll-area" id="openclaw-settings-scroll">
                    <div class="openclaw-loading-gate" style="padding:16px;text-align:center;opacity:0.6;">Initializing…</div>
                </div>
            </div>
        `;
        const scroll = container.querySelector("#openclaw-settings-scroll");

        let capabilities = {};
        try {
            const capRes = await openclawApi.getCapabilities();
            if (!lifecycle.isCurrent()) return;
            if (capRes.ok && capRes.data?.features) capabilities = capRes.data.features;
        } catch { /* non-fatal */ }
        if (!lifecycle.isCurrent()) return;

        const [healthRes, logRes, configRes] = await Promise.all([
            openclawApi.getHealth(),
            openclawApi.getLogs(50),
            openclawApi.getConfig(),
        ]);
        if (!lifecycle.isCurrent()) return;

        scroll.innerHTML = "";
        await renderSettingsStatus({ scroll, healthRes, logRes, configRes, capabilities, api: openclawApi });
        if (!lifecycle.isCurrent()) return;
        renderSettingsLlm({
            scroll, configRes, api: openclawApi, session: OpenClawSession,
            getAdminErrorMessage, isCurrent: lifecycle.isCurrent,
        });
        renderSettingsSecrets({
            scroll, configRes, api: openclawApi, session: OpenClawSession,
            container, isCurrent: lifecycle.isCurrent,
        });
        renderSettingsLogs({ scroll, logRes, schedule: lifecycle.schedule });
        finishSettingsRender(lifecycle);
    },
    dispose: (container) => disposeSettingsRender(container),
};
