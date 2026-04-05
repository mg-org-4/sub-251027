import { get } from "../../api/client.js";
import { buildStackMembersURL } from "../../api/endpoints.js";
import { EVENTS } from "../../app/events.js";

function _isGroupEnabled(gridContainer) {
    return String(gridContainer?.dataset?.mjrGroupStacks || "") === "1";
}

function _getGroupKey(asset) {
    const stackId = String(asset?.stack_id || "").trim();
    if (stackId) return `stack:${stackId}`;
    return "";
}

async function _fetchGroupMembers(asset) {
    const stackId = String(asset?.stack_id || "").trim();
    if (stackId) {
        const res = await get(buildStackMembersURL(stackId), { timeoutMs: 30_000 });
        if (res?.ok && Array.isArray(res?.data)) return res.data;
        throw new Error(String(res?.error || "Failed to load stack members"));
    }
    return [asset].filter(Boolean);
}

function _sortMembers(items) {
    return (Array.isArray(items) ? items : []).slice().sort((a, b) => {
        const am = Number(a?.mtime || a?.created_at || 0) || 0;
        const bm = Number(b?.mtime || b?.created_at || 0) || 0;
        return bm - am;
    });
}

export function getStackAwareAssetKey(gridContainer, asset, fallbackKey) {
    if (!_isGroupEnabled(gridContainer)) return fallbackKey;
    const groupKey = _getGroupKey(asset);
    return groupKey || fallbackKey;
}

export function ensureStackGroupCard(gridContainer, card, asset) {
    if (!_isGroupEnabled(gridContainer) || !card || !asset) return;
    const groupKey = _getGroupKey(asset);
    if (!groupKey) return;
    card.dataset.mjrStackGroupKey = groupKey;

    const cachedMembers = gridContainer?._mjrStackMembersCache?.get?.(groupKey);
    const knownCount = Array.isArray(cachedMembers)
        ? cachedMembers.length
        : Number(asset?.stack_asset_count || asset?._mjrFeedGroupCount || 0) || 0;
    if (knownCount > 1) {
        card.dataset.mjrStacked = "true";
        card.dataset.mjrStackCount = String(knownCount);
    }

    if (card.querySelector(".mjr-stack-group-button")) return;

    const button = document.createElement("button");
    button.type = "button";
    button.className = "mjr-stack-group-button";
    button.title = "Open generation group in grid";
    button.setAttribute("aria-label", "Open generation group in grid");
    button.innerHTML = `<span class="pi pi-clone"></span>`;

    button.addEventListener("click", async (event) => {
        event.preventDefault();
        event.stopPropagation();

        try {
            if (!gridContainer._mjrStackMembersCache)
                gridContainer._mjrStackMembersCache = new Map();
            let members = gridContainer._mjrStackMembersCache.get(groupKey);
            if (!Array.isArray(members)) {
                members = await _fetchGroupMembers(asset);
                gridContainer._mjrStackMembersCache.set(groupKey, members);
            }
            const ordered = _sortMembers(members);
            const count = ordered.length;
            if (count > 1) {
                card.dataset.mjrStacked = "true";
                card.dataset.mjrStackCount = String(count);
                button.innerHTML = `<span class="pi pi-clone"></span><span class="mjr-stack-group-button-count">${count}</span>`;
            }
            gridContainer.dispatchEvent(
                new CustomEvent(EVENTS.OPEN_STACK_GROUP, {
                    bubbles: true,
                    detail: {
                        asset,
                        members: ordered,
                        title: `Generation group (${count} assets)`,
                        stackId: String(asset?.stack_id || "").trim(),
                    },
                }),
            );
        } catch (err) {
            console.debug?.(err);
        }
    });

    card.appendChild(button);
}

export function disposeStackGroupCards(gridContainer) {
    try {
        gridContainer._mjrStackMembersCache?.clear?.();
    } catch (e) {
        console.debug?.(e);
    }
}

/**
 * Attach or update a "duplicate stack" button on a card.
 * Called when a filename collision is detected (same filename, different paths).
 * The button opens all copies in the similar/stack-group panel.
 */
export function ensureDupStackCard(gridContainer, card, asset) {
    if (!card || !asset) return;
    const isDupStack = !!asset._mjrDupStack;
    const count = Number(asset._mjrDupCount || 0) || 0;

    // Remove the button if no longer a dup stack
    if (!isDupStack || count < 2) {
        const existing = card.querySelector(".mjr-dup-stack-button");
        if (existing) existing.remove();
        delete card.dataset.mjrDupStacked;
        delete card.dataset.mjrDupCount;
        return;
    }

    // Mark the card
    card.dataset.mjrDupStacked = "true";
    card.dataset.mjrDupCount = String(count);

    // Update existing button or create it
    let button = card.querySelector(".mjr-dup-stack-button");
    if (!button) {
        button = document.createElement("button");
        button.type = "button";
        button.className = "mjr-dup-stack-button";
        button.setAttribute("aria-label", "Show all duplicates");

        button.addEventListener("click", (event) => {
            event.preventDefault();
            event.stopPropagation();
            try {
                const members = Array.isArray(asset._mjrDupMembers) ? asset._mjrDupMembers : [asset];
                const n = members.length;
                gridContainer.dispatchEvent(
                    new CustomEvent(EVENTS.OPEN_STACK_GROUP, {
                        bubbles: true,
                        detail: {
                            asset,
                            members,
                            title: `Duplicates — ${n} copies`,
                            isDupGroup: true,
                        },
                    }),
                );
            } catch (err) {
                console.debug?.(err);
            }
        });
        card.appendChild(button);
    }

    const label = `${count} duplicate${count > 1 ? "s" : ""}`;
    button.title = `${label} — click to compare all copies`;
    button.innerHTML = `<span class="pi pi-copy"></span><span class="mjr-dup-stack-count">${count}</span>`;
}
