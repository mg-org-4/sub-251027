/**
 * F17: Job Monitor Tab
 * Tracks prompt execution and displays outputs.
 */
import { openclawApi } from "../openclaw_api.js";
import { extractHistoryOutputRefs, isHdrImageOutputRef } from "../openclaw_asset_refs.js";
import {
    loadBoundedTextOutput,
    resolveTextOutputViewUrl,
} from "../openclaw_text_output.js";
import { parseJsonSafe } from "../openclaw_utils.js";

const POLL_INTERVAL_MS = 2000;
const POLL_MAX_ATTEMPTS = 150;
const STORAGE_KEY = "openclaw-job-monitor-jobs";
const LEGACY_STORAGE_KEY = "moltbot-job-monitor-jobs";

let currentJobs = [];
let pollIntervals = {};

function loadJobs() {
    try {
        // Keep one-way fallback so existing users keep their tracked jobs after rename.
        const stored = localStorage.getItem(STORAGE_KEY) || localStorage.getItem(LEGACY_STORAGE_KEY);
        currentJobs = stored ? parseJsonSafe(stored, []).value : [];
        if (stored && !localStorage.getItem(STORAGE_KEY)) {
            localStorage.setItem(STORAGE_KEY, stored);
        }
    } catch {
        currentJobs = [];
    }
}

function saveJobs() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(currentJobs.slice(0, 20)));
}

export function addJob(promptId, traceId = null) {
    if (!promptId || currentJobs.some((j) => j.promptId === promptId)) return;
    currentJobs.unshift({ promptId, traceId, timeline: [], status: "pending", outputs: [], addedAt: Date.now() });
    saveJobs();
}

export const jobMonitorTab = {
    id: "job-monitor",
    title: "Jobs",
    icon: "pi pi-briefcase",
    render: async (container) => {
        loadJobs();
        container.innerHTML = "";

        // Header
        const header = document.createElement("div");
        header.className = "openclaw-section moltbot-section";
        header.innerHTML = `<h4>Job Monitor</h4>`;

        // Add Manual Job
        const addRow = document.createElement("div");
        addRow.style.display = "flex";
        addRow.style.gap = "8px";
        addRow.style.marginBottom = "8px";

        const input = document.createElement("input");
        input.type = "text";
        input.placeholder = "prompt_id";
        input.style.flex = "1";
        input.style.padding = "4px";

        const addBtn = document.createElement("button");
        addBtn.textContent = "Add";
        addBtn.onclick = () => {
            const val = input.value.trim();
            if (val) {
                addJob(val);
                input.value = "";
                renderJobList();
            }
        };

        addRow.appendChild(input);
        addRow.appendChild(addBtn);
        header.appendChild(addRow);
        container.appendChild(header);

        // Job List
        const listContainer = document.createElement("div");
        listContainer.id = "openclaw-job-list";
        container.appendChild(listContainer);

        let textPreviewGeneration = 0;
        const textPreviewControllers = new Set();
        renderJobList();

        function renderJobList() {
            textPreviewGeneration += 1;
            const renderGeneration = textPreviewGeneration;
            for (const controller of textPreviewControllers) {
                controller.abort();
            }
            textPreviewControllers.clear();
            listContainer.innerHTML = "";

            if (currentJobs.length === 0) {
                listContainer.innerHTML = "<div style='opacity: 0.5; padding: 8px;'>No jobs tracked.</div>";
                return;
            }

            currentJobs.forEach((job) => {
                const row = document.createElement("div");
                row.className = "openclaw-job-row moltbot-job-row";
                row.style.borderBottom = "1px solid var(--border-color)";
                row.style.padding = "8px 0";

                // Header
                const jobHeader = document.createElement("div");
                jobHeader.style.display = "flex";
                jobHeader.style.justifyContent = "space-between";
                jobHeader.style.alignItems = "center";

                const idSpan = document.createElement("span");
                idSpan.style.fontFamily = "monospace";
                idSpan.textContent = job.promptId.substring(0, 16) + "...";
                idSpan.title = job.promptId;

                const statusBadge = document.createElement("span");
                statusBadge.className = `openclaw-kv-val moltbot-kv-val ${job.status === "completed" ? "ok" : job.status === "error" ? "error" : ""}`;
                statusBadge.textContent = job.status;

                const removeBtn = document.createElement("button");
                removeBtn.textContent = "×";
                removeBtn.title = "Remove";
                removeBtn.style.marginLeft = "8px";
                removeBtn.onclick = () => {
                    currentJobs = currentJobs.filter((j) => j.promptId !== job.promptId);
                    if (pollIntervals[job.promptId]) {
                        clearInterval(pollIntervals[job.promptId]);
                        delete pollIntervals[job.promptId];
                    }
                    saveJobs();
                    renderJobList();
                };

                jobHeader.appendChild(idSpan);
                jobHeader.appendChild(statusBadge);
                jobHeader.appendChild(removeBtn);
                row.appendChild(jobHeader);

                if (job.traceId) {
                    const traceLine = document.createElement("div");
                    traceLine.style.marginTop = "4px";
                    traceLine.style.opacity = "0.75";
                    traceLine.style.fontSize = "12px";
                    traceLine.style.fontFamily = "monospace";
                    traceLine.textContent = `trace: ${job.traceId}`;
                    row.appendChild(traceLine);

                    // Milestone D: Timeline Visualization
                    if (job.timeline && job.timeline.length > 0) {
                        const timelineDiv = document.createElement("div");
                        timelineDiv.style.marginTop = "4px";
                        timelineDiv.style.fontSize = "11px";
                        timelineDiv.style.display = "flex";
                        timelineDiv.style.alignItems = "center";
                        timelineDiv.style.gap = "4px";
                        timelineDiv.style.flexWrap = "wrap";

                        job.timeline.forEach((evt, idx) => {
                            const evtSpan = document.createElement("span");
                            evtSpan.textContent = evt.event; // e.g. "queued"
                            evtSpan.title = new Date(evt.ts * 1000).toLocaleString();
                            evtSpan.style.padding = "2px 4px";
                            evtSpan.style.background = "var(--bg-color)";
                            evtSpan.style.border = "1px solid var(--border-color)";
                            evtSpan.style.borderRadius = "3px";

                            timelineDiv.appendChild(evtSpan);

                            if (idx < job.timeline.length - 1) {
                                const arrow = document.createElement("span");
                                arrow.textContent = "→";
                                arrow.style.opacity = "0.5";
                                timelineDiv.appendChild(arrow);
                            }
                        });
                        row.appendChild(timelineDiv);
                    }
                }

                // Outputs
                if (job.outputs && job.outputs.length > 0) {
                    const outputGrid = document.createElement("div");
                    outputGrid.style.display = "flex";
                    outputGrid.style.flexWrap = "wrap";
                    outputGrid.style.gap = "4px";
                    outputGrid.style.marginTop = "8px";

                    job.outputs.forEach((out) => {
                        if (out.media_type === "images" && out.view_url && isHdrImageOutputRef(out)) {
                            const hdrFallback = document.createElement("div");
                            hdrFallback.className = "openclaw-job-output-fallback openclaw-job-output-media-fallback openclaw-job-output-hdr-fallback";
                            hdrFallback.style.width = "110px";
                            hdrFallback.style.minHeight = "80px";
                            hdrFallback.style.padding = "6px";
                            hdrFallback.style.display = "flex";
                            hdrFallback.style.alignItems = "center";
                            hdrFallback.style.justifyContent = "center";
                            hdrFallback.style.textAlign = "center";
                            hdrFallback.style.fontSize = "10px";
                            hdrFallback.style.lineHeight = "1.3";
                            hdrFallback.style.border = "1px dashed var(--border-color)";
                            hdrFallback.style.borderRadius = "6px";
                            hdrFallback.style.background = "var(--comfy-menu-bg, rgba(255,255,255,0.04))";
                            hdrFallback.style.cursor = "pointer";
                            hdrFallback.title = out.filename;
                            hdrFallback.textContent = "HDR output available. Open source preview.";
                            hdrFallback.onclick = () => window.open(out.view_url, "_blank");
                            outputGrid.appendChild(hdrFallback);
                            return;
                        }

                        if (out.media_type === "images" && out.view_url) {
                            const img = document.createElement("img");
                            img.src = out.view_url;
                            img.style.maxWidth = "80px";
                            img.style.maxHeight = "80px";
                            img.style.objectFit = "cover";
                            img.style.cursor = "pointer";
                            img.title = out.filename;
                            img.onclick = () => window.open(out.view_url, "_blank");
                            outputGrid.appendChild(img);
                            return;
                        }

                        if (out.media_type === "text" && out.content) {
                            const textOutput = document.createElement("div");
                            textOutput.className = "openclaw-job-output-fallback openclaw-job-output-text";
                            textOutput.style.width = "160px";
                            textOutput.style.minHeight = "80px";
                            textOutput.style.padding = "6px";
                            textOutput.style.fontSize = "10px";
                            textOutput.style.lineHeight = "1.35";
                            textOutput.style.whiteSpace = "pre-wrap";
                            textOutput.style.overflowWrap = "anywhere";
                            textOutput.style.border = "1px dashed var(--border-color)";
                            textOutput.style.borderRadius = "6px";
                            textOutput.style.background = "var(--comfy-menu-bg, rgba(255,255,255,0.04))";
                            textOutput.title = out.text_truncated ? "Text output truncated" : "Text output";
                            textOutput.textContent = out.text_truncated
                                ? `${out.content}\n...`
                                : out.content;
                            outputGrid.appendChild(textOutput);
                            return;
                        }

                        if (out.media_type === "text" && out.view_url) {
                            const safeViewUrl = resolveTextOutputViewUrl(out.view_url);
                            const textFileOutput = document.createElement("div");
                            textFileOutput.className = "openclaw-job-output-fallback openclaw-job-output-text-file";
                            textFileOutput.style.width = "200px";
                            textFileOutput.style.minHeight = "80px";
                            textFileOutput.style.padding = "6px";
                            textFileOutput.style.fontSize = "10px";
                            textFileOutput.style.lineHeight = "1.35";
                            textFileOutput.style.border = "1px dashed var(--border-color)";
                            textFileOutput.style.borderRadius = "6px";
                            textFileOutput.style.background = "var(--comfy-menu-bg, rgba(255,255,255,0.04))";
                            textFileOutput.title = out.filename || "Text output";

                            const status = document.createElement("div");
                            status.className = "openclaw-job-output-text-status";
                            status.textContent = "Loading text preview...";
                            textFileOutput.appendChild(status);

                            const content = document.createElement("div");
                            content.className = "openclaw-job-output-text-content";
                            content.style.marginTop = "4px";
                            content.style.whiteSpace = "pre-wrap";
                            content.style.overflowWrap = "anywhere";
                            content.textContent = "";
                            textFileOutput.appendChild(content);

                            const sourceLink = document.createElement("a");
                            sourceLink.className = "openclaw-job-output-text-source";
                            if (safeViewUrl) {
                                sourceLink.href = safeViewUrl;
                            }
                            sourceLink.target = "_blank";
                            sourceLink.rel = "noopener noreferrer";
                            sourceLink.textContent = safeViewUrl ? "Open source" : "Source unavailable";
                            sourceLink.style.display = safeViewUrl ? "inline-block" : "none";
                            sourceLink.style.marginTop = "6px";
                            textFileOutput.appendChild(sourceLink);
                            outputGrid.appendChild(textFileOutput);

                            if (!safeViewUrl) {
                                status.textContent = "Text preview unavailable.";
                                return;
                            }

                            const controller = new AbortController();
                            textPreviewControllers.add(controller);
                            loadBoundedTextOutput(safeViewUrl, { signal: controller.signal })
                                .then((result) => {
                                    if (
                                        controller.signal.aborted
                                        || renderGeneration !== textPreviewGeneration
                                        || !textFileOutput.isConnected
                                    ) {
                                        return;
                                    }
                                    if (result.status === "success" || result.status === "truncated") {
                                        // SECURITY: fetched bytes must remain literal text. Never
                                        // replace this with HTML or Markdown rendering.
                                        content.textContent = result.content;
                                        status.textContent = result.status === "truncated"
                                            ? "Text preview truncated."
                                            : "Text preview.";
                                        return;
                                    }
                                    content.textContent = "";
                                    status.textContent = result.reason === "oversized"
                                        ? "Text preview too large."
                                        : "Text preview unavailable.";
                                })
                                .finally(() => {
                                    textPreviewControllers.delete(controller);
                                });
                            return;
                        }

                        if (out.asset_api_required) {
                            const fallback = document.createElement("div");
                            fallback.className = "openclaw-job-output-fallback";
                            fallback.style.width = "80px";
                            fallback.style.minHeight = "80px";
                            fallback.style.padding = "6px";
                            fallback.style.display = "flex";
                            fallback.style.alignItems = "center";
                            fallback.style.justifyContent = "center";
                            fallback.style.textAlign = "center";
                            fallback.style.fontSize = "10px";
                            fallback.style.lineHeight = "1.3";
                            fallback.style.border = "1px dashed var(--border-color)";
                            fallback.style.borderRadius = "6px";
                            fallback.style.background = "var(--comfy-menu-bg, rgba(255,255,255,0.04))";
                            fallback.title = out.asset_api_id || out.filename || "Asset API output";
                            fallback.textContent = "Asset API output requires /api/assets. Preview disabled.";
                            outputGrid.appendChild(fallback);
                            return;
                        }

                        if (out.view_url) {
                            const mediaFallback = document.createElement("div");
                            mediaFallback.className = "openclaw-job-output-fallback openclaw-job-output-media-fallback";
                            mediaFallback.style.width = "110px";
                            mediaFallback.style.minHeight = "80px";
                            mediaFallback.style.padding = "6px";
                            mediaFallback.style.display = "flex";
                            mediaFallback.style.alignItems = "center";
                            mediaFallback.style.justifyContent = "center";
                            mediaFallback.style.textAlign = "center";
                            mediaFallback.style.fontSize = "10px";
                            mediaFallback.style.lineHeight = "1.3";
                            mediaFallback.style.border = "1px dashed var(--border-color)";
                            mediaFallback.style.borderRadius = "6px";
                            mediaFallback.style.background = "var(--comfy-menu-bg, rgba(255,255,255,0.04))";
                            mediaFallback.style.cursor = "pointer";
                            mediaFallback.title = out.filename;
                            mediaFallback.textContent = `${out.media_type || "media"} output available. Open preview.`;
                            mediaFallback.onclick = () => window.open(out.view_url, "_blank");
                            outputGrid.appendChild(mediaFallback);
                        }
                    });

                    row.appendChild(outputGrid);
                }

                listContainer.appendChild(row);

                // Start polling if pending/unknown
                if (job.status === "pending" && !pollIntervals[job.promptId]) {
                    startPolling(job.promptId, renderJobList);
                }
            });
        }
    },
};

async function startPolling(promptId, onUpdate) {
    let attempts = 0;
    pollIntervals[promptId] = setInterval(async () => {
        attempts++;
        if (attempts > POLL_MAX_ATTEMPTS) {
            clearInterval(pollIntervals[promptId]);
            delete pollIntervals[promptId];
            return;
        }

        const res = await openclawApi.getHistory(promptId);
        if (!res.ok) return;

        const job = currentJobs.find((j) => j.promptId === promptId);
        if (!job) {
            clearInterval(pollIntervals[promptId]);
            delete pollIntervals[promptId];
            return;
        }

        const historyItem = res.data;
        if (!historyItem) return;

        // R25: Best-effort trace lookup (optional endpoint; ignore 403/404)
        if (!job.traceId && (attempts === 1 || attempts % 5 === 0)) {
            const t = await openclawApi.getTrace(promptId);
            if (t.ok && t.data?.trace?.trace_id) {
                job.traceId = t.data.trace.trace_id;
                job.timeline = t.data.trace.events || [];
                saveJobs();
                onUpdate();
            }
        }

        const statusStr = historyItem?.status?.status_str;
        if (statusStr === "error") {
            job.status = "error";
            saveJobs();
            onUpdate();
            clearInterval(pollIntervals[promptId]);
            delete pollIntervals[promptId];
            return;
        }

        if (statusStr === "success" || historyItem.outputs) {
            job.status = "completed";
            job.outputs = extractOutputs(historyItem);
            saveJobs();
            onUpdate();
            clearInterval(pollIntervals[promptId]);
            delete pollIntervals[promptId];
        }
    }, POLL_INTERVAL_MS);

}

function extractOutputs(historyItem) {
    return extractHistoryOutputRefs(historyItem).map((img) => ({
        filename: img.filename,
        subfolder: img.subfolder,
        type: img.type,
        media_type: img.media_type,
        asset_hash: img.asset_hash,
        asset_api_id: img.asset_api_id,
        asset_api_required: img.asset_api_required,
        resolution: img.resolution,
        content: img.content,
        text_truncated: img.text_truncated,
        unsupported_reason: img.unsupported_reason,
        view_url: openclawApi.buildViewUrlForRef(img),
    }));
}
