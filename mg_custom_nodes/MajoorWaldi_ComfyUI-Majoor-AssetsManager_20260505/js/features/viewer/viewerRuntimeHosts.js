let viewerOverlayHost = null;
let floatingViewerHost = null;

const VIEWER_OVERLAY_SELECTOR = ".mjr-viewer-overlay";
const FLOATING_VIEWER_SELECTOR = ".mjr-mfv";

function canAppendChildren(host) {
    return !!host && typeof host.appendChild === "function";
}

function getDocumentBody() {
    if (typeof document === "undefined") return null;
    return document?.body || null;
}

function getTopLevelDocumentHost() {
    if (typeof document === "undefined") return null;
    return document?.body || document?.documentElement || null;
}

function isLiveHost(host) {
    if (!canAppendChildren(host)) return false;
    if (host === getDocumentBody()) return true;
    if (typeof host?.isConnected === "boolean") return host.isConnected;
    return true;
}

function normalizeHost(host) {
    return canAppendChildren(host) ? host : null;
}

function resolveHost(preferredHost) {
    if (isLiveHost(preferredHost)) return preferredHost;
    return getDocumentBody();
}

function resolveFloatingViewerHost(preferredHost) {
    const topLevelHost = getTopLevelDocumentHost();
    if (isLiveHost(topLevelHost)) return topLevelHost;
    return resolveHost(preferredHost);
}

function collectNodesAcrossHosts(selector, preferredHost, resolvePreferredHost = resolveHost) {
    const out = [];
    const seen = new Set();
    for (const host of [resolvePreferredHost(preferredHost), getDocumentBody(), preferredHost]) {
        if (!host || seen.has(host)) continue;
        seen.add(host);
        let nodes = [];
        try {
            nodes = Array.from(host.querySelectorAll?.(selector) || []);
        } catch (e) {
            console.debug?.(e);
        }
        for (const node of nodes) {
            if (!node || out.includes(node)) continue;
            out.push(node);
        }
    }
    return out;
}

function rehomeManagedNodes(selector, preferredHost, resolvePreferredHost = resolveHost) {
    const host = resolvePreferredHost(preferredHost);
    if (!host) return;
    const nodes = collectNodesAcrossHosts(selector, preferredHost, resolvePreferredHost);
    for (const node of nodes) {
        if (!node) continue;
        if (node.parentNode === host) continue;
        try {
            host.appendChild(node);
        } catch (e) {
            console.debug?.(e);
        }
    }
}

export function registerViewerOverlayHost(host) {
    viewerOverlayHost = normalizeHost(host);
    rehomeManagedNodes(VIEWER_OVERLAY_SELECTOR, viewerOverlayHost);
    return () => unregisterViewerOverlayHost(host);
}

export function unregisterViewerOverlayHost(host) {
    if (!host || viewerOverlayHost === host) {
        viewerOverlayHost = null;
    }
}

export function registerFloatingViewerHost(host) {
    floatingViewerHost = normalizeHost(host);
    rehomeManagedNodes(FLOATING_VIEWER_SELECTOR, floatingViewerHost, resolveFloatingViewerHost);
    return () => unregisterFloatingViewerHost(host);
}

export function unregisterFloatingViewerHost(host) {
    if (!host || floatingViewerHost === host) {
        floatingViewerHost = null;
    }
}

export function appendViewerOverlayNode(node) {
    const host = resolveHost(viewerOverlayHost);
    try {
        host?.appendChild?.(node);
    } catch (e) {
        console.debug?.(e);
    }
    return host;
}

export function appendFloatingViewerNode(node) {
    const host = resolveFloatingViewerHost(floatingViewerHost);
    try {
        host?.appendChild?.(node);
    } catch (e) {
        console.debug?.(e);
    }
    return host;
}

export function getManagedViewerOverlayNodes() {
    return collectNodesAcrossHosts(VIEWER_OVERLAY_SELECTOR, viewerOverlayHost);
}
