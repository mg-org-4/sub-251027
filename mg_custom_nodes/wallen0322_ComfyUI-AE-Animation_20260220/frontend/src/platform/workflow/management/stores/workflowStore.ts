// Simplified stub for workflow store
export function useWorkflowStore() {
  return {
    nodeIdToNodeLocatorId: (id: number) => String(id),
    nodeToNodeLocatorId: (node: any) => String(node.id)
  }
}
