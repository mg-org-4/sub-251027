/**
 * File    : common/timer.js
 * Purpose : Helper functions to execute periodic low-priority tasks.
 * Author  : Martin Rizzo | <martinrizzo@gmail.com>
 * Date    : Feb 1, 2026
 * Repo    : https://github.com/martin-rizzo/ComfyUI-ZImagePowerNodes
 * License : MIT
 *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *                        ComfyUI-ZImagePowerNodes
 *       ComfyUI nodes designed specifically for the "Z-Image" model.
 *_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
*/
export { executeLowPriorityTasks, scheduleIntervalCalls }


let nodes = new Set();


/**
 * Registers a controller to receive periodic interval calls.
 *
 * This function sets up automatic, repeated calls to the specified controller's
 * `onInterval` method. It attempts to use the node associated with the given
 * controller, unless a specific node is provided as an argument.
 *
 * Key features:
 * - Automatically detects the controller's node if no explicit node is supplied.
 * - Ensures that interval calls stop when the associated node is removed from ComfyUI.
 *
 * @param {Object} controller - The controller object, which must have an `onInterval` method.
 * @param {Object} [node]     - An optional node to associate with the interval calls.
 *                              If not provided, the controller's own node will be used.
 */
function scheduleIntervalCalls(controller, node) {
    if( !node ) { node = controller?.node; }
    if( !node ) {
        console.error("No node or controller found to call the onInterval method.");
    }

    // add the node to the set to call the `onInterval()` method
    nodes.add(node);

    // intercept the `onRemoved` method to remove the node from the set
    const originalOnRemoved = node.onRemoved;
    node.onRemoved = function() {
        nodes.delete(this);
        return originalOnRemoved?.apply(this, arguments);
    };
}


/**
 * Executes low-priority tasks for all registered nodes.
 *
 * This function is designed to be called periodically, such as every 5 seconds.
 * It handles lower priority background operations across all registered nodes.
 *
 * Note: The "low priority timer extension" will be responsible for
 *       calling this function.
 */
function executeLowPriorityTasks() {
    // iterate over all registered nodes and call `onInterval()` from the controller
    for( const node of nodes ) {
        node.zzController?.onInterval?.();
    }
}
