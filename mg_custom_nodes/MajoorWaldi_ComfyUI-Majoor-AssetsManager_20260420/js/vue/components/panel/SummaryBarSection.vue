<script setup>
/**
 * SummaryBarSection.vue — Phase-2→3 transition component for the summary bar.
 *
 * This replaces the previous renderless wrapper with a real Vue template while
 * keeping the same `defineExpose()` contract expected by `App.vue` and the
 * panel runtime: `{ summaryBar, updateSummaryBar, folderBreadcrumb }`.
 *
 * The controller layer still calls `updateSummaryBar()` imperatively for now.
 * That function synchronizes transient counts into Pinia and updates the Vue
 * view without changing the existing controller wiring.
 */
import { onMounted, ref } from "vue";
import { createContextPillsView } from "../../../features/panel/views/contextPillsView.js";
import { usePanelStore } from "../../../stores/usePanelStore.js";
import { buildSummaryBarState } from "./summaryBarState.js";

const panelStore = usePanelStore();

const summaryBar = ref(null);
const pillsHost = ref(null);
const summaryText = ref("");
const duplicateText = ref("");
const showDuplicateAlert = ref(false);

const pillsView = createContextPillsView();

let duplicateAlertAction = null;

const folderBreadcrumb = (() => {
    const el = document.createElement("div");
    el.className = "mjr-folder-breadcrumb";
    el.style.cssText =
        "display:none; align-items:center; gap:4px; padding:3px 6px; margin:1px 0 3px 0;" +
        "font-size:11px; opacity:0.78; overflow:auto; white-space:nowrap; border-radius:8px;" +
        "background:color-mix(in srgb, var(--mjr-surface-2, rgba(255,255,255,0.05)) 42%, transparent);";
    return el;
})();

function handleDuplicateAlertClick() {
    try {
        duplicateAlertAction?.();
    } catch (e) {
        console.debug?.(e);
    }
}

function updateSummaryBar({ state, gridContainer, context = null, actions = null } = {}) {
    const nextState = buildSummaryBarState({ state, gridContainer, context });

    panelStore.lastGridCount = nextState.shown;
    panelStore.lastGridTotal = nextState.total;
    panelStore.viewScope = nextState.activeScope;

    summaryText.value = nextState.summaryText;
    duplicateText.value = nextState.duplicateText;
    showDuplicateAlert.value = nextState.showDuplicates;
    duplicateAlertAction = nextState.showDuplicates
        ? () => {
              try {
                  actions?.onDuplicateAlertClick?.(context?.duplicatesAlert || null);
              } catch (e) {
                  console.debug?.(e);
              }
          }
        : null;

    try {
        pillsView.update({
            state,
            gridContainer,
            rawQuery: nextState.rawQuery,
            actions,
        });
    } catch (e) {
        console.debug?.(e);
    }
}

onMounted(() => {
    try {
        pillsHost.value?.appendChild?.(pillsView.wrap);
    } catch (e) {
        console.debug?.(e);
    }
});

defineExpose({ summaryBar, updateSummaryBar, folderBreadcrumb });
</script>

<template>
    <div ref="summaryBar" class="mjr-am-summary" aria-live="polite">
        <div class="mjr-am-summary-left">
            <div class="mjr-am-summary-text">{{ summaryText }}</div>
        </div>
        <div class="mjr-am-summary-right">
            <button
                v-if="showDuplicateAlert"
                type="button"
                class="mjr-am-dup-alert"
                :title="duplicateText"
                style="
                    padding: 2px 8px;
                    border: 1px solid var(--border-color, #555);
                    background: var(--comfy-menu-bg, #222);
                    color: var(--input-text, #eee);
                    border-radius: 999px;
                    cursor: pointer;
                "
                @click="handleDuplicateAlertClick"
            >
                {{ duplicateText }}
            </button>
            <div ref="pillsHost" />
        </div>
    </div>
</template>
