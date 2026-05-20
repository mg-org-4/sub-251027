<script setup>
/**
 * CustomRootsPopover.vue — Custom browser root folder selector.
 *
 * The customRootsController drives this popover imperatively:
 *   - populates customSelect via innerHTML / appendChild
 *   - toggles customRemoveBtn.disabled
 *   - listens to customAddBtn click
 *
 * Exposes the same DOM refs as createCustomPopoverView() so the controller
 * requires no changes. Visibility is controlled by the legacy popoverManager.
 */
import { ref } from "vue";
import { t } from "../../../app/i18n.js";

const customSelectRef  = ref(null);
const customAddBtnRef  = ref(null);
const customRemoveBtnRef = ref(null);

defineExpose({
    get customSelect()    { return customSelectRef.value; },
    get customAddBtn()    { return customAddBtnRef.value; },
    get customRemoveBtn() { return customRemoveBtnRef.value; },
});
</script>

<template>
    <div class="mjr-popover mjr-custom-popover" style="display: none;">
        <div class="mjr-popover-row">
            <div class="mjr-popover-label">{{ t("label.folder") }}</div>
            <select ref="customSelectRef" class="mjr-select" />
        </div>
        <div class="mjr-popover-row mjr-popover-row--actions">
            <button ref="customAddBtnRef" type="button" class="mjr-btn">
                {{ t("btn.add") }}
            </button>
            <button ref="customRemoveBtnRef" type="button" class="mjr-btn" disabled>
                {{ t("btn.remove") }}
            </button>
        </div>
    </div>
</template>
