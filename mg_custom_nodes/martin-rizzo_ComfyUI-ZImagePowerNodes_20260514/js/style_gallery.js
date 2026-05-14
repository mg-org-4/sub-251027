/**
 * File    : style_gallery.js
 * Purpose : Extension that presents a gallery of available visual styles to select from.
 * Author  : Martin Rizzo | <martinrizzo@gmail.com>
 * Date    : Feb 7, 2026
 * Repo    : https://github.com/martin-rizzo/ComfyUI-ZImagePowerNodes
 * License : MIT
 *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *                        ComfyUI-ZImagePowerNodes
 *       ComfyUI nodes designed specifically for the "Z-Image" model.
 *_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
*/
import { app }           from "../../../scripts/app.js";
import { api }           from "../../../scripts/api.js";
import { GalleryDialog } from "./gallery_dialog.js";
const ENABLED = true;


class StyleDataProvider extends GalleryDialog.DataProvider {

    constructor() {
        super();
        this._cachedStyles = null; //< List of all styles
    }

    /**
     * Fetches a array with data about each element in the gallery.
     *
     * @param {function(Array<Object>)} callback
     *     A callback function that receives an array containing all elements in the gallery.
     *     Each element is an object with the following properties:
     *       - id         : Unique identifier for the style (the index in the list)
     *       - name       : The name of the style (string)
     *       - lowerName  : The name of the style, converted to lowercase (string)
     *       - category   : The category of the style (string)
     *       - description: Description of the style (string)
     *       - tags       : Array of tags associated with the style (Array<string>)
     *       - thumbnail  : URL for the style's thumbnail image (string)
     */
    async fetchDataArray(callback) {
        if( typeof callback !== "function" )
        { console.error("The provided argument is not a valid function."); return; }

        // if it has already been queried before, returns the stored result
        if( self._cachedStyles ) { callback( self._cachedStyles ); return; }

        try {
            const response = await api.fetchApi("/zi_power/styles/last_version");
            const styles   = await response.json();
            if( typeof styles !== "object" )
            { console.error("The fetching of last version style failed."); return; }

            // prefix used when requesting thumbnails
            const thumbnailRequestPrefix = "/zi_power/styles/samples?file=";

            self._cachedStyles = styles.map((style, index) => {
                return {
                    id         : index,
                    name       : style[0],
                    lowerName  : style[0].toLowerCase(),
                    category   : style[1],
                    description: style[2],
                    tags       : style[3].split(","),
                    thumbnail  : thumbnailRequestPrefix + style[4]
                };
            });
            callback( self._cachedStyles );
        } catch (error) {
            console.error("Failed to fetch styles:", error);
        }
    }

    getCategories() {
        return [
        /*   CATEGORY        DISPLAY_NAME    DESCRIPTION                      */
            [""            , "All"         , "Search all styles"               ],
            ["photo"       , "Photo"       , "Search only photographic styles" ],
            ["illustration", "Illustration", "Search only illustration styles" ],
            ["wild"        , "Wild"        , "Search only wild styles"         ],
            ["custom"      , "Custom"      , "Search only custom styles"       ]
        ];
    }


}


class StyleItemRenderer extends GalleryDialog.ItemRenderer {

}


const StyleGalleryDialog = new GalleryDialog(new StyleDataProvider(),
                                             new StyleItemRenderer()
                                            );




//#========================= Style Gallery Button ==========================#

/**
 * Creates a button widget that when pressed opens the style gallery.
 * @param {LGraphNode}  node - The node where the widget is added.
 * @param {string} inputName - The name of the input associated with this widget.
 * @returns {{widget: object}}
 *     An object containing the created widget.
 */
function createStyleGalleryButton( node, inputName ) {

    // split the inputName to extract any title variant
    // (title variant is the part of the input-name after the last underscore)
    const parts   = inputName.split('_');
    let   variant = parts.length>1 ? parts[ parts.length - 1 ] : "";

    // ensure the variant is a number
    if( variant.match(/^[0-9]+$/) === null ) { variant = ""; }
    const title = `Select Style${variant ? ' ' + variant : ""}`;

    // find the previous widget,
    // which should be a combo widget to receive the style selection
    const prevIndex  = node.widgets.length - 1;
    const prevWidget = prevIndex>=0 ? node.widgets[prevIndex] : null;
    if( !prevWidget || prevWidget.type !== "combo" ) {
        console.error("Style Gallery Button must follow a Combo Widget!");
        return;
    }

    // add a custom button widget to the node
    const button = node.addCustomWidget({
            type     : "button",
            name     : inputName,
            label    : `🖼️  ${title} ...`,
            serialize: true,
    });

    // the serialized value is always null, as it doesn't store a value itself.
    // (disable serialization may cause problems when saving and retrieving nodes)
    button.serializeValue = () => null;

    // when the button is pressed, launch the dialog
    button.callback = () =>
    {
        // const shiftGlobal = window.event?.shiftKey;
        // if (shiftGlobal) {
        //     console.log("Alternative action with shift key pressed");
        //     return;
        // }

        const styleName = prevWidget.value?.replace(/^"|"$/g, '');
        StyleGalleryDialog.launch( title, styleName, (styleName) =>
        {
            // ensure the style name is properly quoted
            if( styleName!="" && styleName!="-" && styleName!="none" ) {
                if( !styleName.startsWith('"') ) { styleName = `"${styleName}"`; }
            }
            // apply the new style name on the previous widget
            prevWidget.value = styleName;
            prevWidget.callback(styleName);
            node?.setDirtyCanvas?.(true);
        });
    };
    return { widget: button };
}


//#=========================================================================#
//#////////////////////////// REGISTER EXTENSION ///////////////////////////#
//#=========================================================================#

app.registerExtension({
    name: "ZImagePowerNodes.StyleGallery",

    /** Called when the extension is loaded */
    init() {
        if( !ENABLED ) return;
        console.log(`[${this.name}]: Extension loaded.`);
    },

    /** Called to register custom widgets */
    getCustomWidgets() {
        if( !ENABLED ) return {};
        return {
            "ZIPN_STYLE_GALLERY": createStyleGalleryButton,
        };
    },

});
