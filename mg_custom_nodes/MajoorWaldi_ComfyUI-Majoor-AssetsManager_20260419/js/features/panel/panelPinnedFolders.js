import { get, post } from "../../api/client.js";
import { ENDPOINTS } from "../../api/endpoints.js";
import { comfyConfirm } from "../../app/dialogs.js";
import { comfyToast } from "../../app/toast.js";
import { t } from "../../app/i18n.js";

/**
 * Renders and binds the pinned-folders popover.
 *
 * Fetches custom roots from the API, renders them as open/unpin button pairs,
 * and wires the `pinnedFoldersBtn` click to show the menu.
 *
 * Listeners are removed automatically when `panelLifecycleAC` is aborted.
 */
export function bindPinnedFolders({
    pinnedFoldersBtn,
    pinnedFoldersMenu,
    pinnedFoldersPopover,
    popovers,
    closeOtherPopovers,
    state,
    gridContainer,
    customSelect,
    scopeController,
    customRootsController,
    gridController,
    applyWatcherForScope,
    refreshDuplicateAlerts,
    notifyContextChanged,
    panelLifecycleAC,
}) {
    // ── Menu item factory ──────────────────────────────────────────────────

    const createPinnedMenuItem = (label, { right = null, danger = false } = {}) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "mjr-menu-item";
        if (danger) btn.style.color = "var(--error-text, #f44336)";
        btn.style.cssText = `
            display:flex;
            align-items:center;
            justify-content:space-between;
            width:100%;
            gap:10px;
            padding:9px 10px;
            border-radius:9px;
            border:1px solid rgba(120,190,255,0.18);
            background:linear-gradient(135deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
            transition: border-color 120ms ease, background 120ms ease;
        `;
        const leftSpan = document.createElement("span");
        leftSpan.textContent = String(label || "");
        leftSpan.style.cssText = "font-weight:600; color:var(--fg-color, #e6edf7);";
        const rightWrap = document.createElement("span");
        rightWrap.style.cssText = "display:flex; align-items:center; gap:8px;";
        if (right) {
            const hint = document.createElement("span");
            hint.textContent = String(right);
            hint.style.cssText = "opacity:0.68; font-size:11px;";
            rightWrap.appendChild(hint);
        }
        btn.appendChild(leftSpan);
        btn.appendChild(rightWrap);
        btn.addEventListener(
            "mouseenter",
            () => {
                btn.style.borderColor = "rgba(145,205,255,0.4)";
                btn.style.background =
                    "linear-gradient(135deg, rgba(80,140,255,0.18), rgba(32,100,200,0.14))";
            },
            { signal: panelLifecycleAC?.signal },
        );
        btn.addEventListener(
            "mouseleave",
            () => {
                btn.style.borderColor = "rgba(120,190,255,0.18)";
                btn.style.background =
                    "linear-gradient(135deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01))";
            },
            { signal: panelLifecycleAC?.signal },
        );
        return btn;
    };

    // ── Menu renderer ──────────────────────────────────────────────────────

    const renderPinnedFoldersMenu = async () => {
        try {
            pinnedFoldersMenu.replaceChildren();
        } catch (e) {
            console.debug?.(e);
        }

        let roots = [];
        try {
            const res = await get(
                ENDPOINTS.CUSTOM_ROOTS,
                panelLifecycleAC?.signal ? { signal: panelLifecycleAC.signal } : undefined,
            );
            roots = res?.ok && Array.isArray(res.data) ? res.data : [];
        } catch (e) {
            console.debug?.(e);
        }

        if (!roots.length) {
            const empty = document.createElement("div");
            empty.className = "mjr-muted";
            empty.style.cssText = "padding:10px 12px; opacity:0.75;";
            empty.textContent = t("msg.noPinnedFolders", "No pinned folders");
            pinnedFoldersMenu.appendChild(empty);
            return;
        }

        for (const r of roots) {
            const id = String(r?.id || "").trim();
            if (!id) continue;
            const label = String(r?.name || r?.path || id).trim();
            const path = String(r?.path || "").trim();

            const row = document.createElement("div");
            row.style.cssText = "display:flex; align-items:stretch; width:100%;";

            const openBtn = createPinnedMenuItem(label, { right: path || null });
            openBtn.style.flex = "1";
            openBtn.addEventListener(
                "click",
                async () => {
                    state.scope = "custom";
                    try {
                        if (gridContainer) gridContainer.dataset.mjrScope = "custom";
                    } catch (e) {
                        console.debug?.(e);
                    }
                    state.customRootId = id;
                    state.customRootLabel = label;
                    state.currentFolderRelativePath = "";
                    try {
                        if (customSelect) customSelect.value = id;
                    } catch (e) {
                        console.debug?.(e);
                    }
                    try {
                        scopeController?.setActiveTabStyles?.();
                    } catch (e) {
                        console.debug?.(e);
                    }
                    try {
                        await applyWatcherForScope(state.scope);
                    } catch (e) {
                        console.debug?.(e);
                    }
                    try {
                        await refreshDuplicateAlerts();
                    } catch (e) {
                        console.debug?.(e);
                    }
                    try {
                        notifyContextChanged();
                    } catch (e) {
                        console.debug?.(e);
                    }
                    popovers.close(pinnedFoldersPopover);
                    await gridController.reloadGrid();
                },
                { signal: panelLifecycleAC?.signal },
            );

            const unpinBtn = document.createElement("button");
            unpinBtn.type = "button";
            unpinBtn.className = "mjr-menu-item";
            unpinBtn.title = t("ctx.unpinFolder", "Unpin folder");
            unpinBtn.style.cssText =
                "width:42px; justify-content:center; padding:0; border:1px solid rgba(255,95,95,0.35); border-radius:9px; margin-left:6px; background: linear-gradient(135deg, rgba(255,70,70,0.16), rgba(160,20,20,0.12));";
            const trash = document.createElement("i");
            trash.className = "pi pi-times";
            trash.style.cssText = "opacity:0.88; color: rgba(255,130,130,0.96);";
            unpinBtn.appendChild(trash);
            unpinBtn.addEventListener(
                "click",
                async (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    const ok = await comfyConfirm(
                        t("dialog.unpinFolder", 'Unpin folder "{name}"?', { name: label }),
                    );
                    if (!ok) return;
                    const res = await post(ENDPOINTS.CUSTOM_ROOTS_REMOVE, { id });
                    if (!res?.ok) {
                        comfyToast(
                            res?.error || t("toast.unpinFolderFailed", "Failed to unpin folder"),
                            "error",
                        );
                        return;
                    }
                    if (String(state.customRootId || "") === id) {
                        state.customRootId = "";
                        state.customRootLabel = "";
                        state.currentFolderRelativePath = "";
                    }
                    try {
                        await customRootsController.refreshCustomRoots();
                    } catch (e) {
                        console.debug?.(e);
                    }
                    await renderPinnedFoldersMenu();
                    await gridController.reloadGrid();
                },
                { signal: panelLifecycleAC?.signal },
            );

            row.appendChild(openBtn);
            row.appendChild(unpinBtn);
            pinnedFoldersMenu.appendChild(row);
        }
    };

    // ── Button click ───────────────────────────────────────────────────────

    pinnedFoldersBtn?.addEventListener(
        "click",
        async (e) => {
            e.stopPropagation();
            try {
                await renderPinnedFoldersMenu();
            } catch (e) {
                console.debug?.(e);
            }
            closeOtherPopovers();
            popovers.toggle(pinnedFoldersPopover, pinnedFoldersBtn);
        },
        { signal: panelLifecycleAC?.signal },
    );
}
