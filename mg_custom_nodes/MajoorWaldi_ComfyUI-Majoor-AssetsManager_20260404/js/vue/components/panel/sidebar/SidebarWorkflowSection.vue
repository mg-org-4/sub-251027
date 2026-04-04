<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { drawWorkflowMinimap, synthesizeWorkflowFromPromptGraph } from "../../../../components/sidebar/utils/minimap.js";
import { loadMajoorSettings, saveMajoorSettings } from "../../../../app/settings.js";
import { MINIMAP_LEGACY_SETTINGS_KEY } from "../../../../app/settingsStore.js";
import { t } from "../../../../app/i18n.js";

const props = defineProps({
    asset: { type: Object, required: true },
});

const DEFAULT_SETTINGS = Object.freeze({
    nodeColors: true,
    showLinks: true,
    showGroups: true,
    renderBypassState: true,
    renderErrorState: true,
    showViewport: true,
});

const canvasRef = ref(null);
const showTools = ref(false);
const rawJsonOpen = ref(false);
const minimapSettings = ref(loadWorkflowMinimapSettings());

let resizeObserver = null;

function coerceMetadataRawObject(asset) {
    const raw = asset?.metadata_raw ?? null;
    if (!raw) return null;
    if (typeof raw === "object") return raw;
    if (typeof raw === "string") {
        const text = raw.trim();
        if (!text) return null;
        try {
            const parsed = JSON.parse(text);
            return parsed && typeof parsed === "object" ? parsed : null;
        } catch {
            return null;
        }
    }
    return null;
}

function looksLikePromptGraph(obj) {
    try {
        const entries = Object.entries(obj || {});
        if (!entries.length) return false;
        let hits = 0;
        for (const [, value] of entries.slice(0, 50)) {
            if (!value || typeof value !== "object") continue;
            if (value.inputs && typeof value.inputs === "object") hits += 1;
            if (hits >= 2) return true;
        }
    } catch {
        return false;
    }
    return false;
}

function coerceWorkflow(asset) {
    const metadataRaw = coerceMetadataRawObject(asset);
    const value =
        asset?.workflow ||
        asset?.Workflow ||
        asset?.comfy_workflow ||
        metadataRaw?.workflow ||
        metadataRaw?.Workflow ||
        metadataRaw?.comfy_workflow ||
        null;
    if (!value) return null;
    if (typeof value === "object") return value;
    if (typeof value === "string") {
        const text = value.trim();
        if (!text) return null;
        try {
            return JSON.parse(text);
        } catch {
            return null;
        }
    }
    return null;
}

function coercePromptGraph(asset) {
    const metadataRaw = coerceMetadataRawObject(asset);
    const value =
        asset?.prompt || asset?.Prompt || metadataRaw?.prompt || metadataRaw?.Prompt || null;
    if (!value) return null;
    if (typeof value === "object") return looksLikePromptGraph(value) ? value : null;
    if (typeof value === "string") {
        const text = value.trim();
        if (!text) return null;
        try {
            const parsed = JSON.parse(text);
            return looksLikePromptGraph(parsed) ? parsed : null;
        } catch {
            return null;
        }
    }
    return null;
}

function loadWorkflowMinimapSettings() {
    try {
        const main = loadMajoorSettings?.();
        const stored = main?.workflowMinimap;
        if (stored && typeof stored === "object") {
            return { ...DEFAULT_SETTINGS, ...stored };
        }
    } catch (e) {
        console.debug?.(e);
    }

    try {
        const raw = localStorage?.getItem?.(MINIMAP_LEGACY_SETTINGS_KEY);
        if (!raw) return { ...DEFAULT_SETTINGS };
        const parsed = JSON.parse(raw);
        if (!parsed || typeof parsed !== "object") return { ...DEFAULT_SETTINGS };
        const merged = { ...DEFAULT_SETTINGS, ...parsed };
        try {
            const next = loadMajoorSettings();
            next.workflowMinimap = { ...next.workflowMinimap, ...merged };
            saveMajoorSettings(next);
            localStorage?.removeItem?.(MINIMAP_LEGACY_SETTINGS_KEY);
        } catch (e) {
            console.debug?.(e);
        }
        return merged;
    } catch {
        return { ...DEFAULT_SETTINGS };
    }
}

function persistWorkflowMinimapSettings(nextSettings) {
    try {
        const next = loadMajoorSettings();
        next.workflowMinimap = { ...next.workflowMinimap, ...nextSettings };
        saveMajoorSettings(next);
    } catch (e) {
        console.debug?.(e);
    }
}

const workflow = computed(() => {
    const rawWorkflow = coerceWorkflow(props.asset);
    const promptGraph = coercePromptGraph(props.asset);
    if (!rawWorkflow && !promptGraph) return null;
    return rawWorkflow || synthesizeWorkflowFromPromptGraph(promptGraph);
});

const statusLabel = computed(() => (props.asset?.has_generation_data ? "Complete" : "Partial"));
const rawWorkflowJson = computed(() =>
    workflow.value ? JSON.stringify(workflow.value, null, 2) : "",
);

const toggleOptions = computed(() => [
    { key: "nodeColors", label: "Node Colors", iconClass: "pi pi-palette" },
    { key: "showLinks", label: "Show Links", iconClass: "pi pi-share-alt" },
    { key: "showGroups", label: "Show Frames/Groups", iconClass: "pi pi-th-large" },
    { key: "renderBypassState", label: "Render Bypass State", iconClass: "pi pi-ban" },
    {
        key: "renderErrorState",
        label: "Render Error State",
        iconClass: "pi pi-exclamation-triangle",
    },
    { key: "showViewport", label: "Show Viewport", iconClass: "pi pi-window-maximize" },
]);

function renderCanvas() {
    const canvas = canvasRef.value;
    const currentWorkflow = workflow.value;
    if (!canvas || !currentWorkflow) return;

    const width = Math.max(1, canvas.clientWidth || 320);
    const height = Math.max(1, canvas.clientHeight || 120);
    const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    const ctx = canvas.getContext("2d");
    if (ctx) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawWorkflowMinimap(canvas, currentWorkflow, minimapSettings.value);
}

function toggleSetting(key) {
    minimapSettings.value = {
        ...minimapSettings.value,
        [key]: !minimapSettings.value?.[key],
    };
    persistWorkflowMinimapSettings(minimapSettings.value);
}

onMounted(() => {
    if (canvasRef.value && typeof ResizeObserver === "function") {
        resizeObserver = new ResizeObserver(() => renderCanvas());
        resizeObserver.observe(canvasRef.value);
    }
    renderCanvas();
});

watch(workflow, () => {
    renderCanvas();
}, { flush: "post" });

watch(minimapSettings, () => {
    renderCanvas();
}, { deep: true, flush: "post" });

watch(showTools, () => {
    renderCanvas();
}, { flush: "post" });

onBeforeUnmount(() => {
    try {
        resizeObserver?.disconnect?.();
    } catch (e) {
        console.debug?.(e);
    }
    resizeObserver = null;
});
</script>

<template>
    <div
        v-if="workflow"
        class="mjr-sidebar-section"
        style="background:var(--comfy-menu-bg, rgba(0,0,0,0.2));border:1px solid var(--border-color, rgba(255,255,255,0.14));border-radius:8px;padding:12px;min-width:300px"
    >
        <div
            style="font-size:13px;font-weight:600;color:var(--fg-color, #eaeaea);margin-bottom:12px;text-transform:uppercase;letter-spacing:0.5px"
        >
            ComfyUI Workflow
        </div>

        <div
            style="display:flex;gap:8px;margin-bottom:8px;font-size:12px"
        >
            <div
                style="color:var(--mjr-muted, rgba(255,255,255,0.65));min-width:100px;font-weight:500"
            >
                Status
            </div>
            <div style="color:var(--fg-color, #eaeaea);flex:1">
                {{ statusLabel }}
            </div>
        </div>

        <div
            style="display:flex;align-items:center;justify-content:space-between;gap:10px;margin-top:8px"
        >
            <div style="flex:1" />
            <button
                type="button"
                class="mjr-btn mjr-icon-btn"
                :title="t('tooltip.minimapSettings', 'Minimap settings')"
                style="width:28px;height:28px;border-radius:8px;display:inline-flex;align-items:center;justify-content:center;border:1px solid var(--mjr-border, rgba(255,255,255,0.12));background:rgba(255,255,255,0.06);color:rgba(255,255,255,0.9);cursor:pointer"
                @click="showTools = !showTools"
            >
                <i class="pi pi-sliders-h" />
            </button>
        </div>

        <div
            style="display:flex;gap:10px;align-items:stretch;margin-top:10px"
        >
            <div
                v-if="showTools"
                style="width:190px;flex:0 0 auto;border-radius:10px;padding:10px;background:rgba(0,0,0,0.20);border:1px solid var(--mjr-border, rgba(255,255,255,0.12))"
            >
                <button
                    v-for="option in toggleOptions"
                    :key="option.key"
                    type="button"
                    style="width:100%;display:flex;align-items:center;gap:10px;padding:8px 8px;border-radius:10px;border:1px solid transparent;background:transparent;cursor:pointer;color:rgba(255,255,255,0.92);text-align:left"
                    @click="toggleSetting(option.key)"
                >
                    <span
                        :style="{
                            width: '22px',
                            height: '22px',
                            borderRadius: '6px',
                            display: 'inline-flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            background: minimapSettings?.[option.key] ? 'rgba(76,175,80,0.95)' : 'rgba(255,255,255,0.08)',
                            border: minimapSettings?.[option.key] ? '1px solid rgba(76,175,80,0.35)' : '1px solid rgba(255,255,255,0.12)',
                            flex: '0 0 auto',
                        }"
                    >
                        <i
                            class="pi pi-check"
                            :style="{ fontSize: '12px', opacity: minimapSettings?.[option.key] ? '1' : '0' }"
                        />
                    </span>
                    <i
                        :class="option.iconClass"
                        style="font-size:18px;opacity:0.9;width:18px"
                    />
                    <div style="font-size:13px;font-weight:600">
                        {{ option.label }}
                    </div>
                </button>
            </div>

            <canvas
                ref="canvasRef"
                style="width:100%;height:120px;border-radius:8px;margin-top:0;background:rgba(0,0,0,0.25);border:1px solid var(--mjr-border, rgba(255,255,255,0.12))"
            />
        </div>

        <details
            :open="rawJsonOpen"
            style="margin-top:10px"
            @toggle="rawJsonOpen = $event.target.open"
        >
            <summary
                style="cursor:pointer;color:var(--mjr-muted, rgba(255,255,255,0.65));font-size:12px;user-select:none"
            >
                Show raw JSON
            </summary>
            <pre
                style="background:rgba(0,0,0,0.5);padding:10px;border-radius:6px;font-size:11px;overflow:auto;max-height:180px;margin:10px 0 0 0;color:#90CAF9;font-family:'Consolas', 'Monaco', monospace"
            >{{ rawWorkflowJson }}</pre>
        </details>
    </div>
</template>
