/**
 * File    : ui_palettes.js
 * Purpose : Implements Dialog, Widget, and DataProvider for color palettes,
 *           ensuring compatibility with previous Power Nodes versions.
 *
 * Author  : Martin Rizzo | <martinrizzo@gmail.com>
 * Date    : May 21, 2026
 * Repo    : https://github.com/martin-rizzo/ComfyUI-ZImagePowerNodes
 * License : MIT
 *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *                        ComfyUI-ZImagePowerNodes
 *       ComfyUI nodes designed specifically for the "Z-Image" model.
 *_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
 */
export {
    fetchColorPaletteArray,
    requireColorPaletteGalleryDialog,
    addColorPaletteGalleryWidget,
};
import { api } from "../../../scripts/api.js";
import { GalleryDialog, GalleryDialogDelegate } from "./gallery_dialog.js";
import { GalleryWidget, GalleryWidgetDelegate } from "./gallery_widget.js";

// Cache of promises to avoid duplicate requests for the same endpoint.
const _fetchesByEndpoint = new Map();

// Registry of dialogs for each color-palete database endpoint.
const _dialogsByEndpoint = new Map();


//#==========================================================================#
//#                           FETCH COLOR PALETTES                           #
//# The FIRST generation of UI fetched items directly using this function    #
//# and loaded them into a native ComfyUI combobox. Currently, this function #
//# is internally invoked from GALLERY DIALOG and the GALLERY WIDGET.        #

/**
 * Fetches an array with data about each color palette from the server.
 *
 * Note: The implementation looks a bit complex because of the caching system.
 * It uses an immediately-invoked function (IIFE) to cache the ongoing promise
 * right away, ensuring we don't trigger duplicate network requests for the
 * same endpoint URL.
 *
 * @param {string} endpoint - The full endpoint URL to fetch the palettes from.
 * @returns {Promise<Array<Object>>}
 *     Resolves to the array of formatted color palettes.
 *     Each element in the array is an object with the following properties:
 *       - idx        : Unique identifier for the palette (the index in the list)
 *       - name       : The name of the color palette (string)
 *       - description: Description of the color palette (string)
 *       - tags       : An string of comma-separated tags associated with the palette (string)
 *       - colors     : Array of color objects, each containing a name and a hex code (Array<{name: string, hex: string}>)
 * @example
 *   // Using async/await
 *   const palettes = await fetchColorPaletteArray('/zi_power/palettes/by_version?v=1.0');
 *   console.log(`Loaded ${palettes.length} palettes.`);
 * @example
 *   // Using promises (.then)
 *   fetchColorPaletteArray('/zi_power/styles/by_version?v=1.0').then(palettes => {
 *       console.log(`Loaded ${palettes.length} palettes.`);
 *   });
 */
async function fetchColorPaletteArray(endpoint)
{
    if( typeof endpoint !== 'string' || !endpoint.trim() ) {
        console.error(`Invalid endpoint parameter: "${endpoint}". Expected a non-empty string.`);
        return [];
    }

    // if the endpoint already exists in the "fetch cache"
    // (either the ongoing promise or resolved result),
    // RETURN IT!
    if( _fetchesByEndpoint.has(endpoint) ) {
        return _fetchesByEndpoint.get(endpoint);
    }
    // encapsulate the fetch process in a promise
    const fetchPromise = (async () => {
        try {
            // fetch the palettes for the given endpoint
            const response = await api.fetchApi(endpoint);
            if( !response.ok ) { throw new Error(`HTTP ${response.status}`); }

            // validate that the response is an actual array
            const palettes = await response.json();
            if( !Array.isArray(palettes) ) { throw new Error(`Expected an array but received ${typeof palettes}`); }

            return palettes.map((paldata, index) =>
            {
                const name        = paldata[0] || "Unknown";
                const description = paldata[1] || "";
                const tags        = paldata[2] || "";
                // build the colors array from the data received from the API
                const colors = [];
                for( let i=3; (i+1) < paldata.length; i+=2 ) {
                    colors.push({ hex: paldata[i], color: paldata[i+1] });
                }
                return {
                    idx        : index,
                    name       : name,
                    category   : "",
                    description: description,
                    tags       : tags,
                    colors     : colors
                };
            });

        } catch (error) {
            // if failed, delete the cache for this endpoint to allow future retries
            console.error(`Failed to fetch palettes from "${endpoint}": ${error.message}`);
            _fetchesByEndpoint.delete(endpoint);
            return [];
        }
    })();

    // store the promise in cache for future use
    _fetchesByEndpoint.set(endpoint, fetchPromise);
    return fetchPromise;
}


//#=========================================================================#
//#                         PALETTE GALLERY DIALOG                          #
//# The SECOND generation of UI added a node button that launched a         #
//# GALLERY DIALOG, which in turn modified a native combo-box in ComfyUI.   #
//#                                                                         #

class PaletteGalleryDialogDelegate extends GalleryDialogDelegate {

    constructor(endpoint) {
        super();
        this.endpoint = endpoint;
    }

    /**
     * Fetches an array with data about each item to be displayed in the gallery.
     * @returns {Promise<Array<Object>>}
     *   A promise that resolves to an array of objects with the following properties:
     *       - idx        : Unique identifier for the item (the index in the list)
     *       - name       : The display name of the item (string)
     *       - category   : The category the item belongs to (string)
     *       - description: A detailed description of the item (string)
     *       - tags       : An string of comma-separated tags associated with the palette (string)
     *       - thumbnail  : URL for the item's thumbnail image (string)
     */
    async fetchItemArray() {
        return fetchColorPaletteArray(this.endpoint);
    }

    /**
     * Renders the image HTML element for the selected item.
     * @param {Object|null} item        - The data object representing the item,
     *                                    or `null` if no item is selected.
     * @param {string}      htmlClass   - CSS class to be applied to the img tag
     * @returns {string}
     *    The HTML string representing the image element
     *    or an empty string if the item or thumbnail is missing.
     */
    htmlItemImage(item, _value, _options, htmlClass) {
        if( !item ) { return ""; }
        const colorBars = item.colors.map(color => `
            <div style="
                background-color: ${color.hex};
                flex: 1;
                height: 100%;">
            </div>
        `).join('');

        return `
            <div class="${htmlClass}" style="
                            display   : flex;
                            overflow  : hidden;
                            border    : 1px solid rgba(0,0,0,0.15);
                            box-sizing: border-box;">
                ${colorBars}
            </div>
        `;
    }
}

/**
 * Returns a palette selection dialog containing the palettes loaded from the specified endpoint.
 * @param {string} endpoint - The full endpoint URL to fetch the palettes from.
 * @returns {GalleryDialog}
 *   The gallery dialog instance for the specified endpoint
 * @example
 *   const paletteDialog  = requireColorPaletteGalleryDialog("/api/palettes/v1/list");
 *   const currentPalette = "Volcano";
 *   paletteDialog.launch({}, currentPalette, (selectedPalette) => {
 *       console.log("Selected Palette: " + selectedPalette);
 *   });
 */
function requireColorPaletteGalleryDialog(endpoint) {

    // check if a dialog is already registered for the specified endpoint
    const dialog = _dialogsByEndpoint.get(endpoint);
    if( dialog ) { return dialog; }

    // if no dialog exists for this endpoint, create a new one
    const newDelegate = new PaletteGalleryDialogDelegate(endpoint);
    const newDialog   = new GalleryDialog(newDelegate);
    _dialogsByEndpoint.set(endpoint, newDialog);
    return newDialog;
}


//#=========================================================================#
//#                         PALETTE GALLERY WIDGET                          #
//#   The THIRD generation of UI uses a "GALLERY WIDGET" to launch a        #
//#   "GALLERY DIALOG", both of them customized by 'delegate' objects.      #
//#                                                                         #

class PaletteWidgetDelegate extends GalleryWidgetDelegate {

    constructor(endpoint) {
        super();
        this.endpoint = endpoint;
    }

    /**
     * Fetches an array with data about each item to be displayed in the gallery.
     * @returns {Promise<Array<Object>>}
     *   A promise that resolves to an array of objects with the following properties:
     *       - idx        : Unique identifier for the item (the index in the list)
     *       - name       : The display name of the item (string)
     *       - category   : The category the item belongs to (string)
     *       - description: A detailed description of the item (string)
     *       - tags       : An string of comma-separated tags associated with the palette (string)
     *       - thumbnail  : URL for the item's thumbnail image (string)
     */
    async fetchItemArray() {
        return fetchColorPaletteArray(this.endpoint);
    }

    getItemText(item) {
        if( !item ) { return "Undefined"; }
        return `${item.name}\n${item.category}`;
    }

    /**
     * Called when a thumbnail needs to be drawn for a specific item.
     * @param {CanvasRenderingContext2D} ctx  - The canvas 2D rendering context
     * @param {Object}                   rect - The rectangle object defining the drawing area (left, top, width, height)
     */
    drawItemThumbnail(ctx, rect, item, _value, _options) {
        const numberOfColors = item?.colors?.length || 0;
        if( numberOfColors == 0 ) { return 0; }
        const thumbSize      = 32;
        const rect_right     = rect.left + rect.width;
        const barCount       = Math.min(5, numberOfColors);
        const barSpacing     = 2;
        const barWidth       = Math.floor((thumbSize - 1) / barCount) + 1  -  barSpacing;
        const barHeight      = rect.height;
        const totalBarsWidth = (barWidth * barCount) + (barSpacing * (barCount - 1));

        // draw the bars
        const x      = rect_right - totalBarsWidth;
        const y      = rect.top;
        const colors = item.colors;
        for( let i = 0 ; i < barCount && i < colors.length ; i++ ) {
            ctx.fillStyle = colors[i].hex;
            ctx.fillRect(x + (i * (barWidth + barSpacing)), y, barWidth, barHeight);
        }
        return thumbSize;
    }

    /* Using the default implementation of drawItemText() */
    // drawItemText(ctx, rect, line1, line2, item, value) { }
}

function addColorPaletteGalleryWidget(node, name, data) {
    const type           = data[0];
    const options        = data[1] || {};
    const endpoint       = options.endpoint || "";
    const dialog_options = options.dialog || {};
    let widget = new GalleryWidget(type, node, name, options, new PaletteWidgetDelegate(endpoint), (widget) =>
    {
        // launch dialog and update widget value
        const paletteDialog  = requireColorPaletteGalleryDialog(endpoint);
        const currentPalette = widget.value;
        paletteDialog.launch( dialog_options, currentPalette, (selectedPalette) => {
            widget.forceUpdate( selectedPalette );
        });
    });
    widget = node.addCustomWidget( widget );
    return { widget: widget };
}

