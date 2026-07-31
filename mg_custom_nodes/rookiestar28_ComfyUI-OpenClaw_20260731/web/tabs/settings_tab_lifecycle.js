/** Generation-owned lifecycle for async Settings renders. */
let activeRender = null;
let nextGeneration = 0;

function cancel(record) {
    record.controller.abort();
    for (const cleanup of record.cleanups) cleanup();
    record.cleanups.clear();
}

export function beginSettingsRender(container) {
    if (activeRender) cancel(activeRender);
    const record = {
        generation: ++nextGeneration,
        container,
        controller: new AbortController(),
        cleanups: new Set(),
        finished: false,
    };
    const context = Object.freeze({
        container,
        signal: record.controller.signal,
        isCurrent: () => activeRender === record && !record.controller.signal.aborted,
        schedule: (callback, delay) => {
            const timer = setTimeout(() => {
                if (activeRender === record && !record.controller.signal.aborted) callback();
            }, delay);
            record.cleanups.add(() => clearTimeout(timer));
            return timer;
        },
    });
    record.context = context;
    activeRender = record;
    return context;
}

export function finishSettingsRender(context) {
    if (activeRender?.context === context) activeRender.finished = true;
}

export function disposeSettingsRender(container) {
    if (!activeRender || activeRender.container !== container) return false;
    const requiresRerender = !activeRender.finished;
    cancel(activeRender);
    activeRender = null;
    return requiresRerender;
}
