/**
 * File    : custom_widgets.js
 * Purpose : Custom widgets used in this project.
 * Author  : Martin Rizzo | <martinrizzo@gmail.com>
 * Date    : Feb 3, 2026
 * Repo    : https://github.com/martin-rizzo/ComfyUI-ZImagePowerNodes
 * License : MIT
 *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *                        ComfyUI-ZImagePowerNodes
 *       ComfyUI nodes designed specifically for the "Z-Image" model.
 *_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

 The ComfyUI native WEBCAM widget code can be found here:
  - https://github.com/Comfy-Org/ComfyUI_frontend/blob/v1.39.5/src/extensions/core/webcamCapture.ts

*/
import { app }       from "../../../scripts/app.js";
import { LiteGraph } from "./comfyui_bridge.js";
const ENABLED = true;
const DEFAULT_WIDGET_HEIGHT = 20;


/*=========================== SEPARATOR WIDGET ============================*/

/**
 * Creates a configurable socketless widget used as visual separator in the Node Graph UI.
 *
 * The widget supports different visual modes: solid line, dotted line, bold line,
 * or just spacing. This widget does not create input/output ports (it is socketless).
 * 
 * @param {LGraphNode} node      - The node instance where the widget is being created
 * @param {string}     inputName - Unique identifier for the widget (not used for value serialization)
 * @param {Array}      inputData - Configuration array where:
 *                                  - [0] = Type name
 *                                  - [1] = Configuration object with these optional properties:
 *                                      - mode: "spacer" | "divider" | "dotted" | "bold"
 *                                      - color: string (CSS color)
 *                                      - height: number
 *                                      - thickness: number
 * @param {object} _app - The ComfyApp instance (not used in this implementation)
 * @returns {{ widget: object }}
 *     Object containing the created widget instance with:
 */
function createSeparatorWidget(node, inputName, inputData, _app) {

    const type    = inputData[0];
    const options = inputData[1] || {};

    const w = node.addCustomWidget({
        type     : type,
        name     : inputName,
        serialize: true,
        options  : {
            socketless: true,
            ...options
        },

        /**
         * Handles the visual rendering of the separator based on the selected mode.
         * 
         * @param {CanvasRenderingContext2D} ctx - The canvas 2D rendering context.
         * @param {Object} node          - The parent node instance.
         * @param {number} widgetWidth   - The current width of the widget.
         * @param {number} y             - The vertical offset.
         * @param {number} _widgetHeight - The calculated height of the widget.
         */
        draw: function(ctx, node, widgetWidth, y, _widgetHeight) {
            const mode      = this?.options?.mode   || "spacer";
            const color     = this?.options?.color  || LiteGraph.NODE_DEFAULT_BOXCOLOR; //WIDGET_BGCOLOR; //WIDGET_OUTLINE_COLOR;
            const height    = this?.options?.height || DEFAULT_WIDGET_HEIGHT;
            const thickness = this?.options?.thickness;

            // the "spacer" mode renders nothing, providing only vertical padding
            if( mode === "spacer") { return; }

            // center the separator vertically
            const left    = Math.trunc(10);
            const right   = Math.trunc(widgetWidth - 10);
            const centerY = Math.trunc(y + height/2 + 0.5);

            // draw the separator
            ctx.save();
            ctx.strokeStyle = color;
            ctx.beginPath();
            switch( mode )
            {
                // standard thin horizontal line
                case "divider":
                default:
                    ctx.lineWidth = thickness || 1;
                    ctx.setLineDash([]);
                    break;

                // dashed line for visual grouping
                case "dotted":
                    ctx.lineWidth = thickness || 2;
                    ctx.setLineDash([2, 2]);
                    break;

                // thick line for strong section separation
                case "bold":
                    ctx.lineWidth = thickness || 6;
                    ctx.setLineDash([]);
                    break;
            }
            ctx.moveTo(left, centerY);
            ctx.lineTo(right, centerY);
            ctx.stroke();
            ctx.restore();
        },

        computeSize: function(widgetWidth) {
            const widgetHeight = this?.options?.height || 20;
            return [widgetWidth, widgetHeight];
        },

    });

    w.serializeValue = () => null;
    return { widget: w };
}


//#=========================================================================#
//#////////////////////////// REGISTER EXTENSION ///////////////////////////#
//#=========================================================================#

app.registerExtension({
    name: "ZImagePowerNodes.CustomWidgets",

    /** Called when the extension is loaded. */
    init() {
        if (!ENABLED) return;
        console.log(`[${this.name}]: Extension loaded.`);
    },

    /** Called to register custom widgets. */
    getCustomWidgets() {
        if (!ENABLED) return {};
        return {
            "ZIPN_SEPARATOR": createSeparatorWidget,
        };
    },

});
