import { openclawApi } from "./openclaw_api.js";

function extractQueuePromptId(entry) {
    if (!entry) return "";
    if (typeof entry === "string") return entry;
    if (Array.isArray(entry)) return String(entry[1] || "");
    if (typeof entry === "object") {
        return String(entry.prompt_id || entry.promptId || entry.id || entry.job_id || "");
    }
    return "";
}

function extractActiveQueuePromptIds(queueData = {}) {
    const entries = [
        ...(Array.isArray(queueData.queue_running) ? queueData.queue_running : []),
        ...(Array.isArray(queueData.queue_pending) ? queueData.queue_pending : []),
        ...(Array.isArray(queueData.Running) ? queueData.Running : []),
        ...(Array.isArray(queueData.Pending) ? queueData.Pending : []),
    ];
    return new Set(entries.map(extractQueuePromptId).filter(Boolean));
}

/**
 * F48/F49: Queue Lifecycle Monitor.
 * Consumes R71 events (SSE) with polling fallback to show deduplicated status banners.
 * Handles disconnected state and recovery based on B-Strict/B-Loose contracts.
 */
export class QueueMonitor {
    constructor(ui, deps = {}) {
        this.ui = ui;
        this.api = deps.api || openclawApi;
        this.setIntervalRef = deps.setIntervalRef || window.setInterval.bind(window);
        this.now = deps.now || (() => Date.now());
        this.lastBannerTime = 0;
        this.lastStatusId = null;
        this.bannerTTL = 5000;
        this.startupGraceMs = Number.isFinite(deps.startupGraceMs) ? deps.startupGraceMs : 30000;
        this.disconnectAlertThreshold = Number.isFinite(deps.disconnectAlertThreshold) ? deps.disconnectAlertThreshold : 3;
        this.es = null;
        this.isConnected = true;
        this.startedAt = this.now();
        this.disconnectFailures = 0;
        this.hasObservedHealthyBackend = false;
        this.disconnectAlertActive = false;
        this.activePromptIds = new Set();
    }

    start() {
        this.startedAt = this.now();
        this.connectSSE();
        this.setIntervalRef(() => this.checkHealth(), 10000);
    }

    connectSSE() {
        if (this.es) {
            this.es.close();
        }

        this.es = this.api.subscribeEvents(
            (data) => this.handleEvent(data),
            (err) => this.handleConnectionError(err)
        );
    }

    handleEvent(data) {
        this._markHealthy();
        if (!this.isConnected) {
            this.isConnected = true;
            this.showBanner({
                severity: "success",
                message: "\u2705 OpenClaw Backend Connected",
                id: "connection_restored",
                ttl_ms: 3000,
                source: "queue-monitor",
            });
        }

        const type = data.event_type;
        const promptId = data.prompt_id ? String(data.prompt_id) : "";
        const pid = promptId ? promptId.slice(0, 8) : "???";

        switch (type) {
            case "queued":
                if (promptId) this.activePromptIds.add(promptId);
                this.showBanner({
                    severity: "info",
                    message: `\u23F3 Job ${pid} queued`,
                    id: `job_${type}`,
                    ttl_ms: 2000,
                    source: "queue-monitor",
                });
                break;
            case "running":
                if (promptId) this.activePromptIds.add(promptId);
                this.showBanner({
                    severity: "info",
                    message: `\u25B6 Job ${pid} running...`,
                    id: `job_${type}`,
                    ttl_ms: 5000,
                    source: "queue-monitor",
                });
                break;
            case "failed":
                if (promptId) this.activePromptIds.delete(promptId);
                this.showBanner({
                    severity: "error",
                    message: `\u274C Job ${pid} failed`,
                    id: `job_${type}`,
                    ttl_ms: 10000,
                    source: "queue-monitor",
                    persist: true,
                    action: {
                        label: "Open Jobs",
                        type: "tab",
                        payload: "job-monitor",
                    },
                });
                break;
            case "completed":
                if (promptId) this.activePromptIds.delete(promptId);
                break;
        }
    }

    handleConnectionError(err) {
        this._registerDisconnect("connection_lost", "\u26A0\uFE0F Backend Disconnected. Retrying...");
        return err;
    }

    async checkHealth() {
        try {
            const res = await this.api.getHealth();
            if (res.ok && res.data) {
                const wasDisconnected = !this.isConnected;
                this._markHealthy();
                if (wasDisconnected) {
                    await this._reconcileActiveJobsAfterReconnect();
                    this.isConnected = true;
                    this.showBanner({
                        severity: "success",
                        message: "\u2705 Connection Restored",
                        id: "connection_restored",
                        ttl_ms: 3000,
                        source: "queue-monitor",
                    });
                    if (!this.es || this.es.readyState === 2) {
                        this.connectSSE();
                    }
                }

                const stats = res.data.stats || {};
                const obs = stats.observability || {};
                if (obs.total_dropped > 0) {
                    this.showBanner({
                        severity: "warning",
                        message: `\u26A0\uFE0F High load: ${obs.total_dropped} events dropped.`,
                        id: "backpressure",
                        source: "queue-monitor",
                        persist: true,
                        action: {
                            label: "Open Explorer",
                            type: "tab",
                            payload: "explorer",
                        },
                    });
                }
            } else {
                this._registerDisconnect("health_check_failed", "\u26A0\uFE0F Backend Unreachable");
            }
        } catch (_err) {
            this._registerDisconnect("health_check_exception", "\u26A0\uFE0F Connection Error");
        }
    }

    async _reconcileActiveJobsAfterReconnect() {
        if (!this.activePromptIds.size || typeof this.api.getPromptQueue !== "function") {
            return;
        }

        const res = await this.api.getPromptQueue();
        if (!res?.ok) {
            return;
        }

        const activeQueueIds = extractActiveQueuePromptIds(res.data);
        const staleIds = [...this.activePromptIds].filter((promptId) => !activeQueueIds.has(promptId));
        staleIds.forEach((promptId) => this.activePromptIds.delete(promptId));
        if (!staleIds.length) {
            return;
        }

        const first = staleIds[0].slice(0, 8) || "unknown";
        this.showBanner({
            severity: "info",
            message: `Job ${first} no longer active after reconnect`,
            id: `job_reconnect_cleared_${first}`,
            ttl_ms: 3000,
            source: "queue-monitor",
        });
    }

    _markHealthy() {
        this.hasObservedHealthyBackend = true;
        this.disconnectFailures = 0;
        this.disconnectAlertActive = false;
    }

    _shouldAlertDisconnect() {
        if (this.hasObservedHealthyBackend) {
            return true;
        }

        const elapsed = Math.max(0, this.now() - this.startedAt);
        // IMPORTANT: sidebar bootstrap can legitimately race backend startup; do not persist disconnect
        // alerts until the backend was healthy once or the initial misses are sustained beyond the grace window.
        return (
            elapsed >= this.startupGraceMs &&
            this.disconnectFailures >= this.disconnectAlertThreshold
        );
    }

    _registerDisconnect(id, message) {
        this.disconnectFailures += 1;
        this.isConnected = false;
        if (!this._shouldAlertDisconnect()) {
            return;
        }

        if (!this.disconnectAlertActive) {
            this.disconnectAlertActive = true;
            this.showBanner({
                severity: "error",
                message,
                id,
                source: "queue-monitor",
                persist: true,
            });
        }
    }

    showBanner(type, message, statusId, ttl = this.bannerTTL) {
        const payload = typeof type === "object"
            ? {
                id: type.id || `monitor_${this.now()}`,
                severity: type.severity || "info",
                message: type.message || "",
                source: type.source || "QueueMonitor",
                ttl_ms: type.ttl_ms != null ? type.ttl_ms : this.bannerTTL,
                dismissible: type.dismissible !== false,
                action: type.action,
                persist: type.persist,
            }
            : {
                id: statusId || `monitor_${this.now()}`,
                severity: type,
                message,
                source: "QueueMonitor",
                ttl_ms: ttl,
                dismissible: true,
            };
        const now = this.now();
        if (this.lastStatusId === payload.id && (now - this.lastBannerTime < payload.ttl_ms)) {
            return;
        }

        this.lastStatusId = payload.id;
        this.lastBannerTime = now;
        this.ui.showBanner(payload);
    }
}
