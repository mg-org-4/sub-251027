/**
 * Vue application factory — Majoor Assets Manager
 *
 * Creates an isolated Vue 3 + Pinia application instance for each mounting
 * point (sidebar panel, bottom panel, etc.).  Each call returns a fresh app
 * so that multiple tabs can coexist without sharing reactive state.
 *
 * Usage:
 *   import { createMjrApp } from './vue/createVueApp.js'
 *   import AssetsManagerApp from './vue/App.vue'
 *
 *   const { app, pinia } = createMjrApp(AssetsManagerApp)
 *   app.mount(container)
 *   // later:
 *   app.unmount()
 */

import { createApp } from "vue";
import { createPinia } from "pinia";

/**
 * Creates a self-contained Vue + Pinia application.
 *
 * @param {object} rootComponent  - Vue component to use as the root.
 * @param {object} [props]        - Optional props passed to the root component.
 * @returns {{ app: import('vue').App, pinia: import('pinia').Pinia }}
 */
export function createMjrApp(rootComponent, props = undefined) {
    const pinia = createPinia();
    const app = createApp(rootComponent, props);
    app.use(pinia);
    return { app, pinia };
}

// WeakMap<HTMLElement, Map<mountKey, Vue app>> — avoids polluting DOM elements
// with expando properties and works correctly even on frozen/sealed elements.
const _registry = new WeakMap();

function _getKeyMap(container) {
    let map = _registry.get(container);
    if (!map) {
        map = new Map();
        _registry.set(container, map);
    }
    return map;
}

/**
 * Mount helper that enforces keep-alive semantics.
 *
 * ComfyUI calls `render()` each time the sidebar tab becomes active, and
 * `destroy()` each time it's hidden.  We mount the Vue app only once and
 * keep it alive so that grid state, scroll position and selections survive
 * tab-switch cycles.
 *
 * @param {HTMLElement}  container     - DOM element provided by ComfyUI.
 * @param {object}       component     - Root Vue component.
 * @param {string}       [mountKey]    - Unique key used to identify the app.
 * @returns {boolean} true if a new app was mounted, false if already alive.
 */
export function mountKeepAlive(container, component, mountKey = "_mjrVueApp") {
    if (!container) return false;
    const map = _getKeyMap(container);
    if (map.has(mountKey)) return false; // already mounted — keep-alive

    const { app } = createMjrApp(component);
    app.mount(container);
    map.set(mountKey, app);
    return true;
}

/**
 * Unmount and clean up a keep-alive Vue app.
 * Call this only when the container is being permanently removed (e.g. during
 * full extension teardown), not on every sidebar tab-switch.
 *
 * @param {HTMLElement} container
 * @param {string}      [mountKey]
 */
export function unmountKeepAlive(container, mountKey = "_mjrVueApp") {
    if (!container) return;
    const map = _registry.get(container);
    if (!map) return;
    const app = map.get(mountKey);
    if (!app) return;
    try {
        app.unmount();
    } catch {
        /* ignore */
    }
    map.delete(mountKey);
}
