<script setup>
/**
 * SortPopover.vue — Reactive sort menu, content-only.
 *
 * Vue owns the menu content (rendered from usePanelStore.sort).
 * Visibility is controlled externally by the legacy popoverManager
 * (popovers.toggle/close) so both can coexist during the Phase 2
 * transition without conflicts.
 *
 * On sort selection:
 *   1. Writes new sort to usePanelStore
 *   2. Dispatches `mjr:sort-changed` window event → AssetsManagerPanel
 *      picks it up and triggers gridController.reloadGrid()
 */
import { computed } from "vue";
import { usePanelStore } from "../../../stores/usePanelStore.js";
import { t } from "../../../app/i18n.js";

const panelStore = usePanelStore();

const sortOptions = computed(() => [
    { key: "mtime_desc", label: t("sort.newest"), icon: "pi pi-sort-amount-down" },
    { key: "mtime_asc",  label: t("sort.oldest"), icon: "pi pi-sort-amount-up-alt" },
    { key: "name_asc",   label: t("sort.nameAZ"),  icon: "pi pi-sort-alpha-down" },
    { key: "name_desc",  label: t("sort.nameZA"),  icon: "pi pi-sort-alpha-up" },
    { key: "rating_desc",label: t("sort.ratingHigh"), icon: "pi pi-star-fill" },
    { key: "size_desc",  label: t("sort.sizeDesc"), icon: "pi pi-sort-numeric-down-alt" },
    { key: "size_asc",   label: t("sort.sizeAsc"),  icon: "pi pi-sort-numeric-up-alt" },
]);

const currentSort = computed(() => panelStore.sort || "mtime_desc");

const handleSortChange = (key) => {
    panelStore.sort = key;
    try {
        window.dispatchEvent(new CustomEvent("mjr:sort-changed", { detail: { sort: key } }));
    } catch (e) {
        console.debug?.(e);
    }
};
</script>

<template>
    <!--
        Root element has class "mjr-popover mjr-sort-popover" but NO v-show.
        The legacy popoverManager controls style.display on this element.
        Vue only manages the menu content reactively.
    -->
    <div class="mjr-popover mjr-sort-popover" style="display: none;">
        <div class="mjr-menu" style="display: grid; gap: 6px;">
            <button
                v-for="option in sortOptions"
                :key="option.key"
                type="button"
                class="mjr-menu-item"
                :class="{ 'is-active': currentSort === option.key }"
                @click="handleSortChange(option.key)"
            >
                <span class="mjr-menu-item-label">{{ option.label }}</span>
                <i
                    :class="[option.icon, 'mjr-menu-item-check']"
                    :style="{ opacity: currentSort === option.key ? 1 : 0 }"
                />
            </button>
        </div>
    </div>
</template>
