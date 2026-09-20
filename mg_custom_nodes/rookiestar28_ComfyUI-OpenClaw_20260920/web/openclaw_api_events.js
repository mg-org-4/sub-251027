/** Job event polling and SSE subscription route-family methods. */
import { apiURL } from "./openclaw_comfy_api.js";
import { parseJsonSafe } from "./openclaw_utils.js";

export const eventApiMethods = {
    async getEvents(lastSeq = 0) {
        return this.fetch(`${this._path("/events")}?since=${lastSeq}`);
    },

    /**
     * Subscribe to SSE event stream.
     * @param {function} onEvent - Callback for events (eventData) => void
     * @param {function} onError - Callback for errors (error) => void
     * @returns {EventSource} The event source instance (caller must .close() it)
     */

    subscribeEvents(onEvent, onError) {
        // Use apiURL from shim to get full path
        const url = apiURL(this._path("/events/stream"));
        const es = new EventSource(url);

        const handle = (e) => {
            if (!e.data) return;
            const parsed = parseJsonSafe(e.data);
            if (!parsed.ok || !parsed.value || typeof parsed.value !== "object") {
                console.warn("[OpenClaw] Failed to parse SSE event:", parsed.error);
                return;
            }
            const data = parsed.value;
            // Unified event type injection if missing
            if (!data.event_type && e.type !== "message") {
                data.event_type = e.type;
            }
            onEvent(data);
        };

        es.onmessage = handle;
        es.addEventListener("queued", handle);
        es.addEventListener("running", handle);
        es.addEventListener("completed", handle);
        es.addEventListener("failed", handle);

        es.onerror = (err) => {
            if (onError) onError(err);
        };

        return es;
    },

};
