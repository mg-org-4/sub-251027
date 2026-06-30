import { reactive } from "vue";

export const workflowSaveInfoState = reactive({
    open: false,
    initial: null as any,
    resolve: null as (((payload: any | null) => void) | null),
});

export function openWorkflowSaveInfoDialog(initial: any = {}): Promise<any | null> {
    workflowSaveInfoState.open = true;
    workflowSaveInfoState.initial = initial || {};
    return new Promise((resolve) => {
        workflowSaveInfoState.resolve = resolve;
    });
}

export function closeWorkflowSaveInfoDialog(payload: any | null = null): void {
    const resolver = workflowSaveInfoState.resolve;
    workflowSaveInfoState.open = false;
    workflowSaveInfoState.initial = null;
    workflowSaveInfoState.resolve = null;
    if (resolver) resolver(payload);
}
