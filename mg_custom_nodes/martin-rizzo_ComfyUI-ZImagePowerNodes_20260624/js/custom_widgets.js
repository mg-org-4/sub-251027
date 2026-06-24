/**
 * File    : custom_widgets.js
 * Purpose : Register all custom widgets used in this project.
 * Author  : Martin Rizzo | <martinrizzo@gmail.com>
 * Date    : Feb 3, 2026
 * Repo    : https://github.com/martin-rizzo/ComfyUI-ZImagePowerNodes
 * License : MIT
 *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *                        ComfyUI-ZImagePowerNodes
 *       ComfyUI nodes designed specifically for the "Z-Image" model.
 *_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
*/
import { app }                          from "../../../scripts/app.js";
import { addSeparatorWidget }           from "./custom_widgets/separator_widget.js";
import { addVisualStyleGalleryWidget }  from "./custom_widgets/ui_styles.js";
import { addColorPaletteGalleryWidget } from "./custom_widgets/ui_palettes.js";
import { addStyleGalleryButton }        from "./custom_widgets/style_gallery_button.js";
const ENABLED = true;


//#=========================================================================#
//#////////////////////////// REGISTER EXTENSION ///////////////////////////#
//#=========================================================================#

app.registerExtension({
    name: "ZImagePowerNodes.CustomWidgets",

    /** Called when the extension is loaded. */
    init() {
        if( !ENABLED ) return;
        console.log(`[${this.name}]: Extension loaded.`);
    },

    /** Called to register custom widgets. */
    getCustomWidgets() {
        if( !ENABLED ) return {};
        return {
            "ZIPN_SEPARATOR"       : addSeparatorWidget,
            "ZIPN_STYLE_SELECTOR"  : addVisualStyleGalleryWidget,
            "ZIPN_PALETTE_SELECTOR": addColorPaletteGalleryWidget,

            // [DEPRECATED]
            "ZIPN_STYLE_GALLERY_BUTTON": addStyleGalleryButton,
        };
    },

});
