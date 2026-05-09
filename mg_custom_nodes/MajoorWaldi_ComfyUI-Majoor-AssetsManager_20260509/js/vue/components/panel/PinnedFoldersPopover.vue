<script setup>
/**
 * PinnedFoldersPopover.vue — Container shell for the pinned folders menu.
 *
 * The panel runtime injects pinned folder items into the exposed
 * `pinnedFoldersMenu` div via replaceChildren() / appendChild().
 * Vue does not manage the menu items — only the shell structure.
 *
 * Visibility is controlled by the legacy popoverManager (popovers.toggle).
 */
import { ref } from "vue";

const menuRef = ref(null);

defineExpose({
    get pinnedFoldersMenu() { return menuRef.value; },
});
</script>

<template>
    <div class="mjr-popover mjr-pinned-folders-popover" style="display: none;">
        <div ref="menuRef" class="mjr-menu" style="display: grid; gap: 6px;" />
    </div>
</template>
