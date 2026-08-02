export const TEXT_OUTPUT_MAX_BYTES = 64 * 1024;
export const TEXT_OUTPUT_MAX_CHARS = 4096;
export const TEXT_OUTPUT_TIMEOUT_MS = 5000;

const ALLOWED_QUERY_KEYS = new Set(["filename", "subfolder", "type"]);
const ALLOWED_FILE_TYPES = new Set(["input", "output", "temp"]);
const ALLOWED_TEXT_EXTENSIONS = new Set([
    "txt", "md", "markdown", "json", "csv", "yaml", "yml", "xml", "log",
]);
const MAX_FILE_FIELD_CHARS = 1024;
const ACTIVE_MIME_TYPES = new Set([
    "text/html",
    "text/javascript",
    "text/css",
    "image/svg+xml",
    "application/xhtml+xml",
]);
const APPLICATION_TEXT_MIME_TYPES = new Set([
    "application/json",
    "application/xml",
    "application/yaml",
    "application/x-yaml",
]);

function fixedResult(status, reason = "", content = "", truncated = false) {
    return { status, content, truncated, reason };
}

function resolveLocationHref() {
    return globalThis.location?.href || "http://127.0.0.1/";
}

function codePointLength(value = "") {
    return Array.from(String(value)).length;
}

function hasUnsafeCharacters(value = "") {
    return Array.from(String(value)).some((char) => {
        const code = char.charCodeAt(0);
        return code < 32 || code === 127;
    });
}

function isSafeFilename(filename) {
    if (
        !filename
        || codePointLength(filename) > MAX_FILE_FIELD_CHARS
        || hasUnsafeCharacters(filename)
        || filename === "."
        || filename === ".."
        || filename.includes("/")
        || filename.includes("\\")
    ) {
        return false;
    }
    const dotIndex = filename.lastIndexOf(".");
    const suffix = dotIndex >= 0 ? filename.slice(dotIndex + 1).toLowerCase() : "";
    return ALLOWED_TEXT_EXTENSIONS.has(suffix);
}

function isSafeSubfolder(subfolder) {
    return codePointLength(subfolder) <= MAX_FILE_FIELD_CHARS
        && !hasUnsafeCharacters(subfolder)
        && !subfolder.includes("\\")
        && !subfolder.startsWith("/")
        && !subfolder.split("/").some((part) => part === "." || part === "..");
}

function validateViewUrl(input, baseHref = resolveLocationHref()) {
    if (typeof input !== "string" || !input) {
        return null;
    }
    try {
        const base = new URL(baseHref);
        const resolved = new URL(input, base);
        if (
            resolved.origin !== base.origin
            || !["http:", "https:"].includes(resolved.protocol)
            || resolved.username
            || resolved.password
            || resolved.hash
            || !resolved.pathname.endsWith("/view")
        ) {
            return null;
        }

        const keys = Array.from(resolved.searchParams.keys());
        if (keys.some((key) => !ALLOWED_QUERY_KEYS.has(key))) {
            return null;
        }
        if (
            resolved.searchParams.getAll("filename").length !== 1
            || !isSafeFilename(resolved.searchParams.get("filename") || "")
            || resolved.searchParams.getAll("type").length !== 1
            || !ALLOWED_FILE_TYPES.has(resolved.searchParams.get("type") || "")
            || resolved.searchParams.getAll("subfolder").length > 1
            || !isSafeSubfolder(resolved.searchParams.get("subfolder") || "")
        ) {
            return null;
        }
        return resolved;
    } catch {
        return null;
    }
}

export function resolveTextOutputViewUrl(input) {
    return validateViewUrl(input)?.href || "";
}

function isAllowedTextMime(value) {
    const parts = String(value || "").split(";");
    const mime = parts.shift()?.trim().toLowerCase() || "";
    if (!mime || ACTIVE_MIME_TYPES.has(mime)) {
        return false;
    }
    for (const parameter of parts) {
        const separator = parameter.indexOf("=");
        if (separator < 0) {
            continue;
        }
        const key = parameter.slice(0, separator).trim().toLowerCase();
        const rawValue = parameter.slice(separator + 1).trim();
        const parameterValue = rawValue.replace(/^"|"$/g, "").toLowerCase();
        if (key === "charset" && !["utf-8", "utf8"].includes(parameterValue)) {
            return false;
        }
    }
    if (mime.startsWith("text/")) {
        return true;
    }
    return APPLICATION_TEXT_MIME_TYPES.has(mime);
}

function parseDeclaredLength(value) {
    const raw = String(value || "").trim();
    if (!/^\d+$/.test(raw)) {
        return null;
    }
    const parsed = Number(raw);
    return Number.isSafeInteger(parsed) ? parsed : null;
}

async function cancelReader(reader) {
    try {
        await reader?.cancel?.();
    } catch {
        // Best-effort release after fail-closed size rejection.
    }
}

function getByteChunkLength(value) {
    const isByteView = ArrayBuffer.isView(value) && value?.BYTES_PER_ELEMENT === 1;
    const hasByteTag = Object.prototype.toString.call(value) === "[object Uint8Array]";
    if (!isByteView && !hasByteTag) {
        return null;
    }
    const byteLength = Number(value?.byteLength);
    return Number.isSafeInteger(byteLength) && byteLength >= 0 ? byteLength : null;
}

function copyByteChunk(value) {
    try {
        return Uint8Array.from(value);
    } catch {
        return null;
    }
}

export async function loadBoundedTextOutput(viewUrl, { fetchFn = globalThis.fetch, signal } = {}) {
    const validatedHref = resolveTextOutputViewUrl(viewUrl);
    const validatedUrl = validatedHref ? new URL(validatedHref) : null;
    if (!validatedUrl || typeof fetchFn !== "function") {
        return fixedResult("unavailable", "invalid_url");
    }

    const controller = new AbortController();
    let timedOut = false;
    let cancelled = false;
    const onCallerAbort = () => {
        cancelled = true;
        controller.abort();
    };
    if (signal?.aborted) {
        return fixedResult("unavailable", "cancelled");
    }
    signal?.addEventListener?.("abort", onCallerAbort, { once: true });
    const timeoutId = setTimeout(() => {
        timedOut = true;
        controller.abort();
    }, TEXT_OUTPUT_TIMEOUT_MS);

    try {
        // SECURITY: do not route through the generic JSON/text wrapper; it uses an
        // unbounded body reader. This loader owns every byte and never adds tokens.
        const response = await fetchFn(validatedUrl.href, {
            method: "GET",
            credentials: "same-origin",
            redirect: "error",
            headers: {
                Accept: "text/plain, text/markdown, text/csv, application/json, application/xml, application/yaml",
            },
            signal: controller.signal,
        });

        if (!response || !response.ok) {
            return fixedResult("unavailable", "http_error");
        }
        if (response.redirected) {
            return fixedResult("unavailable", "redirected");
        }
        if (response.url) {
            const responseUrl = validateViewUrl(response.url);
            if (!responseUrl || responseUrl.href !== validatedUrl.href) {
                return fixedResult("unavailable", "redirected");
            }
        }
        if (!isAllowedTextMime(response.headers?.get?.("content-type"))) {
            return fixedResult("unavailable", "mime_rejected");
        }

        const declaredLength = parseDeclaredLength(response.headers?.get?.("content-length"));
        if (declaredLength !== null && declaredLength > TEXT_OUTPUT_MAX_BYTES) {
            controller.abort();
            return fixedResult("unavailable", "oversized");
        }

        if (!response.body || typeof response.body.getReader !== "function") {
            return fixedResult("link_only", "stream_unavailable");
        }

        const reader = response.body.getReader();
        const chunks = [];
        let totalBytes = 0;
        while (true) {
            const step = await reader.read();
            if (step.done) {
                break;
            }
            const chunkLength = getByteChunkLength(step.value);
            if (chunkLength === null) {
                await cancelReader(reader);
                return fixedResult("link_only", "stream_unavailable");
            }
            if (chunkLength > TEXT_OUTPUT_MAX_BYTES - totalBytes) {
                controller.abort();
                await cancelReader(reader);
                return fixedResult("unavailable", "oversized");
            }
            const chunk = copyByteChunk(step.value);
            if (!chunk || chunk.byteLength !== chunkLength) {
                await cancelReader(reader);
                return fixedResult("link_only", "stream_unavailable");
            }
            totalBytes += chunkLength;
            chunks.push(chunk);
        }

        const bytes = new Uint8Array(totalBytes);
        let offset = 0;
        for (const chunk of chunks) {
            bytes.set(chunk, offset);
            offset += chunk.byteLength;
        }

        let content;
        try {
            content = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
        } catch {
            return fixedResult("unavailable", "invalid_utf8");
        }
        const contentCodePoints = Array.from(content);
        if (contentCodePoints.length > TEXT_OUTPUT_MAX_CHARS) {
            return fixedResult(
                "truncated",
                "display_limit",
                contentCodePoints.slice(0, TEXT_OUTPUT_MAX_CHARS).join(""),
                true
            );
        }
        return fixedResult("success", "", content, false);
    } catch {
        if (cancelled || signal?.aborted) {
            return fixedResult("unavailable", "cancelled");
        }
        if (timedOut) {
            return fixedResult("unavailable", "timeout");
        }
        return fixedResult("unavailable", "network_error");
    } finally {
        clearTimeout(timeoutId);
        signal?.removeEventListener?.("abort", onCallerAbort);
    }
}
