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
import { app } from "../../../scripts/app.js";
const ENABLED = true;


/**
 * Creates a divider widget to visually separate UI elements.
 * @param {LGraphNode}  node - The node where the widget is added.
 * @param {string} inputName - The name of the input associated with this widget.
 * @returns {{widget: object}} An object containing the created widget.
 */
function createDividerWidget( node, inputName ) {

    const w = node.addCustomWidget({
        type      : "ui_spacer",
        name      : inputName,
        serialize : false,

        draw: function(ctx, node, widget_width, y, widget_height) {
            ctx.save();
            ctx.strokeStyle = "#555"; 
            ctx.beginPath();
            ctx.moveTo(10, y + widget_height / 2);
            ctx.lineTo(widget_width - 10, y + widget_height / 2);
            ctx.stroke();
            ctx.restore();
        },

        // computeSize: function(widgetWidth) {
        // }

    });
    w.serialize      = true;
    w.serializeValue = () => null;
    return { widget: w };
}


/**
 * Creates a spacer widget, which can be used for spacing UI elements.
 * @param {LGraphNode}  node - The node where the widget is added.
 * @param {string} inputName - The name of the input associated with this widget.
 * @returns {{widget: object}} An object containing the created widget.
 */
function createSpacerWidget( node, inputName ) {

    const w = node.addCustomWidget({
        type      : "ui_spacer",
        name      : inputName,
        serialize : false,

        draw: function(ctx, node, widget_width, y, widget_height) {
        },

        // computeSize: function(widgetWidth) {
        // }

    });
    w.serialize      = true;
    w.serializeValue = () => null;
    return { widget: w };

}


//#=========================================================================#
//#////////////////////////// REGISTER EXTENSION ///////////////////////////#
//#=========================================================================#

app.registerExtension({
    name: "ZImagePowerNodes.CustomNodes",

    /** Called when the extension is loaded. */
    init() {
        if (!ENABLED) return;
        console.log(`[Extension] ${this.name} loaded.`);
    },

    /** Called to register custom widgets. */
    getCustomWidgets() {
        if (!ENABLED) return {};
        return {
            "ZIPOWER_DIVIDER": createDividerWidget,
            "ZIPOWER_SPACER" : createSpacerWidget,
        };
    },

});
