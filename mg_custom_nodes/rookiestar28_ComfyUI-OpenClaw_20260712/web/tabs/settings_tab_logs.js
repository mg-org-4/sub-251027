/** Render recent logs and bounded deep-link highlighting. */
import { createSection } from "./settings_tab_dom.js";

export function renderSettingsLogs({ scroll, logRes, schedule }) {
    // -- Logs Section --
    const logsSec = createSection("Recent Logs");
    const logView = document.createElement("div");
    logView.className = "openclaw-log-viewer openclaw-log-viewer moltbot-log-viewer";

    if (logRes.ok) {
        const content = logRes.data?.content;
        logView.textContent = Array.isArray(content) ? content.join("\n") : String(content ?? "");
    } else {
        const detail = [
            logRes.status ? `HTTP ${logRes.status}` : null,
            logRes.error || "request_failed",
        ].filter(Boolean).join(" — ");
        logView.textContent = `Failed to load logs: ${detail}`;
    }

    logsSec.appendChild(logView);
    scroll.appendChild(logsSec);

    // F48: Deep Link Handling
    // Format: #settings/sectionId
    // We need to map known sections or just rely on text content matching if we didn't add IDs?
    // Let's rely on checking hash after render.
    schedule(() => {
        const hash = window.location.hash;
        if (hash && hash.startsWith("#settings/")) {
            const sectionKey = hash.split("/")[1];
            let target = null;

            // Simple mapping based on section titles we created
            // "LLM Settings" -> "llm"
            // "UI Key Store" -> "secrets"
            // "Recent Logs" -> "logs"
            // "System Health" -> "health"

            const sections = Array.from(scroll.querySelectorAll(".openclaw-section"));
            if (sectionKey === "llm") target = sections.find(s => s.textContent.includes("LLM Settings"));
            else if (sectionKey === "secrets") {
                target = sections.find(s => s.textContent.includes("UI Key Store"));
                // Auto-expand if targeted
                if (target) {
                    const content = target.querySelector(".openclaw-collapsible-content");
                    const toggle = target.querySelector(".openclaw-collapsible-header span:last-child");
                    if (content) content.style.display = "block";
                    if (toggle) toggle.textContent = "▼";
                }
            }
            else if (sectionKey === "logs") target = sections.find(s => s.textContent.includes("Recent Logs"));
            else if (sectionKey === "health") target = sections.find(s => s.textContent.includes("System Health"));

            if (target) {
                target.scrollIntoView({ behavior: "smooth", block: "start" });
                target.style.outline = "2px solid var(--primary-color, #2196F3)";
                target.style.transition = "outline 1s";
                schedule(() => target.style.outline = "none", 2000);
            }
        }
    }, 100);
}
