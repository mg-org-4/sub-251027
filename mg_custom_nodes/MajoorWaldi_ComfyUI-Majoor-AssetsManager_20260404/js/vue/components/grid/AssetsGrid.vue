<script setup>
import { computed, onBeforeUnmount, onMounted, provide, ref, watch } from "vue";
import { usePanelStore } from "../../../stores/usePanelStore.js";
import {
    clearActiveGridContainer,
    setActiveGridContainer,
} from "../../../features/panel/panelRuntimeRefs.js";
import { createAssetsQueryController } from "../../composables/useAssetsQuery.js";
import { useGridContextMenu } from "../../composables/useGridContextMenu.js";
import { useGridKeyboard } from "../../composables/useGridKeyboard.js";
import VirtualAssetGridHost from "./VirtualAssetGridHost.vue";
import {
    bindGridHostState as bindGridHostStateRuntime,
    installGridScrollSync,
    restoreGridUiState as restoreGridUiStateRuntime,
} from "./assetsGridHostState.js";

const browseSectionRef = ref(null);
const gridWrapperRef = ref(null);
const gridHostRef = ref(null);

const panelStore = usePanelStore();

const gridContainerRef = computed(() => gridHostRef.value?.gridContainer ?? null);

let disposeScrollSync = null;
let disposeGridHostState = null;
let gridHostStateOptions = null;
let assetsQueryController = null;

provide("mjr-grid-container-ref", gridContainerRef);

useGridContextMenu(
    gridContainerRef,
    () => panelStore,
);
useGridKeyboard(gridContainerRef);

function currentGridContainer() {
    return gridContainerRef.value || null;
}

function bindGridHostState(opts = {}) {
    const container = currentGridContainer();
    if (!container) return;
    gridHostStateOptions = opts;
    try {
        disposeGridHostState?.();
    } catch (e) {
        console.debug?.(e);
    }
    disposeGridHostState = bindGridHostStateRuntime(container, {
        panelStore,
        ...opts,
    });
}

onMounted(() => {
    disposeScrollSync = installGridScrollSync(gridWrapperRef.value, panelStore);
    if (gridWrapperRef.value && Number(panelStore.scrollTop || 0) > 0) {
        gridWrapperRef.value.scrollTop = Number(panelStore.scrollTop || 0);
    }
});

onBeforeUnmount(() => {
    try {
        clearActiveGridContainer(currentGridContainer());
    } catch (e) {
        console.debug?.(e);
    }
    try {
        disposeScrollSync?.();
    } catch (e) {
        console.debug?.(e);
    }
    disposeScrollSync = null;
    try {
        disposeGridHostState?.();
    } catch (e) {
        console.debug?.(e);
    }
    disposeGridHostState = null;
    try {
        assetsQueryController?.dispose?.();
    } catch (e) {
        console.debug?.(e);
    }
    assetsQueryController = null;
});

watch(
    gridContainerRef,
    (container, previous) => {
        try {
            if (previous && previous !== container) {
                clearActiveGridContainer(previous);
            }
        } catch (e) {
            console.debug?.(e);
        }
        try {
            if (container) {
                setActiveGridContainer(container);
            }
        } catch (e) {
            console.debug?.(e);
        }
    },
    { immediate: true },
);

defineExpose({
    get browseSection() {
        return browseSectionRef.value;
    },
    get gridWrapper() {
        return gridWrapperRef.value;
    },
    get gridContainer() {
        return currentGridContainer();
    },
    onGridContainerReady(container) {
        if (container && container !== currentGridContainer()) {
            return currentGridContainer();
        }
        return currentGridContainer();
    },
    bindGridHostState(opts = {}) {
        bindGridHostState(opts);
        return () => {
            try {
                disposeGridHostState?.();
            } catch (e) {
                console.debug?.(e);
            }
            disposeGridHostState = null;
        };
    },
    restoreGridUiState(initialLoadPromise, opts = {}) {
        return restoreGridUiStateRuntime({
            initialLoadPromise,
            gridWrapper: gridWrapperRef.value,
            gridContainer: currentGridContainer(),
            panelStore,
            ...opts,
        });
    },
    initAssetsQueryController(options = {}) {
        const gridContainer = currentGridContainer();
        if (!gridContainer || !gridWrapperRef.value) return null;
        try {
            assetsQueryController?.dispose?.();
        } catch (e) {
            console.debug?.(e);
        }
        assetsQueryController = createAssetsQueryController({
            gridContainer,
            gridWrapper: gridWrapperRef.value,
            ...options,
        });
        return assetsQueryController;
    },
    loadAssets(...args) {
        return gridHostRef.value?.loadAssets?.(...args);
    },
    loadAssetsFromList(...args) {
        return gridHostRef.value?.loadAssetsFromList?.(...args);
    },
    prepareGridForScopeSwitch(...args) {
        return gridHostRef.value?.prepareGridForScopeSwitch?.(...args);
    },
    refreshGrid(...args) {
        return gridHostRef.value?.refreshGrid?.(...args);
    },
    captureAnchor(...args) {
        return gridHostRef.value?.captureAnchor?.(...args);
    },
    restoreAnchor(...args) {
        return gridHostRef.value?.restoreAnchor?.(...args);
    },
    hydrateGridFromSnapshot(...args) {
        return gridHostRef.value?.hydrateFromSnapshot?.(...args);
    },
    upsertAsset(...args) {
        return gridHostRef.value?.upsertAsset?.(...args);
    },
    removeAssets(...args) {
        return gridHostRef.value?.removeAssets?.(...args);
    },
    disposeGrid(...args) {
        return gridHostRef.value?.dispose?.(...args);
    },
});
</script>

<template>
    <div
        ref="browseSectionRef"
        class="mjr-am-browse"
        style="flex:1;display:flex;flex-direction:row;overflow:hidden"
    >
        <div
            ref="gridWrapperRef"
            class="mjr-am-grid-scroll"
            style="flex:1;overflow:auto;position:relative"
        >
            <VirtualAssetGridHost
                ref="gridHostRef"
                :scroll-element="gridWrapperRef"
            />
        </div>
    </div>
</template>
