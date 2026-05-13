/**
 * File    : comfyui_bridge.js
 * Purpose : Provides a safe bridge to the LiteGraph environment within ComfyUI.
 * Author  : Martin Rizzo | <martinrizzo@gmail.com>
 * Date    : May 12, 2026
 * Repo    : https://github.com/martin-rizzo/ComfyUI-ZImagePowerNodes
 * License : MIT
 *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *                        ComfyUI-ZImagePowerNodes
 *       ComfyUI nodes designed specifically for the "Z-Image" model.
 *_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
*/

/**
 * Reference to the global LiteGraph instance.
 * @type {Object}
 */
export const LiteGraph = globalThis.LiteGraph;

/**
 * Reference to the base LGraphNode class.
 * @type {Function}
 */
export const LGraphNode = globalThis.LGraphNode;


/**
 * Validates the presence of LiteGraph dependencies.
 *
 * Logs specific warnings if the environment is not ready,
 * helping in debugging initialization order issues.
 */
export function validateBridge()
{
    if( !LiteGraph ) {
        console.warn(
            "[Z-ImagePowerNodes ComfyUI-Bridge] LiteGraph module not found in global scope. " +
            "This may indicate that Z-Image Power Nodes are out of date due to recent internal changes in ComfyUI."
        );
    }

    if( !LGraphNode ) {
        console.error(
            "[Z-ImagePowerNodes ComfyUI-Bridge] LGraphNode prototype missing. " +
            "The LiteGraph engine might not have initialized properly or is inaccessible for Z-Image Power Nodes. " +
            "Please ensure your environment meets the latest requirements and try updating Z-Image Power Nodes."
        );
    }
}


// Run validation upon module import!
validateBridge();
