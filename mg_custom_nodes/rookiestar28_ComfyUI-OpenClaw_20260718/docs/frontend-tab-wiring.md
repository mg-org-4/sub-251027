# Frontend Tab Wiring

OpenClaw's frontend uses modular vanilla ES modules loaded by the ComfyUI extension host. Keep tab work inside this architecture unless a future migration decision explicitly changes the runtime model.

## Tab Registration

- Register the OpenClaw host sidebar entry through `registerOpenClawSidebar(app, tabDefinition)` from `web/openclaw_sidebar_registration.js`; it prefers ComfyUI's current sidebar store API and falls back to the deprecated frontend facade for older host bundles.
- Register tabs through `tabManager.registerTab({ id, title, icon, render, dispose? })`.
- Keep `id` stable; it is used for pane ids and active-tab storage.
- Treat `render(pane)` as the only place that mutates a tab pane.
- Return a promise from `render` only when the tab genuinely performs async work; async failures are routed through the tab error boundary.
- Use optional `dispose(pane)` for pending-render cleanup. Returning `true` requests a fresh render
  when the user revisits the tab; completed panes should remain reusable.

## DOM Helpers

- Prefer shared helpers from `web/openclaw_utils.js` for new shell/tab wiring:
  - `createDomElement(...)` for text-safe element construction.
  - `appendChildren(...)` for optional child nodes.
  - `queryRequired(...)` when a selector is mandatory for the tab to function.
- Use `textContent` semantics for user-visible text. Do not add raw HTML helper paths for convenience.
- Keep legacy class aliasing centralized through existing normalization and alias helpers.
- Keep Settings-specific status, LLM, secrets, logs, DOM, and lifecycle behavior in the focused
  `web/tabs/settings_tab_*.js` owners rather than rebuilding a monolithic renderer.

## API Contracts

- Use `OpenClawAPI.fetch(...)` normalized results instead of direct `fetch` from tabs.
- Check `result.ok` before reading `result.data`.
- Preserve admin-token handling inside `OpenClawAPI` and shared session helpers.
- Add endpoint methods to the matching config, generation, resource, model, or event owner module;
  preserve `web/openclaw_api.js` as the transport/session facade and keep one shared singleton.

## Verification

- Add Vitest coverage for new shared helpers or tab wiring behavior.
- Use Playwright harness specs for user-visible tab behavior such as active panes, rendered content, and action outcomes.
- For async tabs, cover switching/disposal and prove stale completion cannot mutate the new pane.
