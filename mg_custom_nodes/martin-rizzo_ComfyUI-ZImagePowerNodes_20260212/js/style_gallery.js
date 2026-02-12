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
import { app }                      from "../../../scripts/app.js";
import { ComfyDialog, $el as html } from "../../../scripts/ui.js"; //< deprrecated ?
import { loadCSS }                  from "./common.js";
import { makeCustomDialog, setupCardHoverListeners } from "./common_dialog.js";
import { fetchLastVersionStyles }   from "./common_server.js";
const ENABLED = true;
const DIALOG_ID = 'zipn-style-gallery-dialog';
const TITLE_ID  = 'zipn-style-gallery-title';
loadCSS("style_gallery.css");


//#========================= Style Gallery Dialog ==========================#

class StyleGalleryDialog extends ComfyDialog {

    /**
     * Constructor.
     */
    constructor() {
        super();

        // icons can be taken from PrimeIcons or Pictogrammers MDT
        // PrimeIcons       : e.g. "i.pi.pi-image"   (https://primevue.org/icons)
        // Pictogrammers MDI: e.g. "i.mdi.mdi-image" (https://pictogrammers.com/library/mdi)
        this.element = makeCustomDialog(
            DIALOG_ID                 , //< ID of the DOM element where the dialog is located
            TITLE_ID                  , //< ID of the DOM element where the title is located
            'Select Style'            , //< title
            'i.mdi.mdi-image-multiple', //< icon
            this.createDialogContent(), //< dialog content
            () => this.close() //< close callback
        );

        this.textFilter       = "";
        this.categoryFilter   = "";     // "", "photo", "illustration", "wild", "custom"
        this.viewMode         = "grid"; // "grid" or "list"
        this.allStyles        = {};
        this.gridEl           = this.element.querySelector('#zipn-style-grid');
        this.statusEl         = this.element.querySelector('#zipn-status-text');
        this.c_allButtonEl    = this.element.querySelector('#zipn-all-btn');
        this.c_photoButtonEl  = this.element.querySelector('#zipn-photo-btn');
        this.c_illusButtonEl  = this.element.querySelector('#zipn-illus-btn');
        this.c_wildButtonEl   = this.element.querySelector('#zipn-wild-btn');
        this.c_customButtonEl = this.element.querySelector('#zipn-custom-btn');
        this.gridButtonEl     = this.element.querySelector('#zipn-grid-btn');
        this.listButtonEl     = this.element.querySelector('#zipn-list-btn');
        this.onSelectStyle    = null;

        const CARD_SELECTOR = '.zipn-style-grid-card, .zipn-style-list-card';
        setupCardHoverListeners( this.gridEl, CARD_SELECTOR,
            (card) => { this.onCardEnter(card); },
            (card) => { this.onCardLeave(card); },
            (card) => { this.onCardClick(card); }
        );

        this.updateButtons();
    }


    /**
     * Launches the style gallery dialog.
     */
    static launch(title, onSelectStyle) {
        // create the first time and use the same instance the next time
        if( !this._instance ) { this._instance = new StyleGalleryDialog(); }
        const dialog  = this._instance;
        const titleEl = dialog.element.querySelector(`#${TITLE_ID}`);

        if( titleEl ) { titleEl.textContent = title; }
        dialog.onSelectStyle = onSelectStyle;
        dialog.show();
        fetchLastVersionStyles( (styles) => {
            dialog.allStyles = styles;
            dialog.updateSearch("!refresh");
        });
    }


    /**
     * Updates the button states based on current view mode and category filter.
     */
    updateButtons() {
        this.listButtonEl.classList.toggle('p-highlight', this.viewMode == "list" );
        this.gridButtonEl.classList.toggle('p-highlight', this.viewMode == "grid" );
        this.c_allButtonEl.classList.toggle('p-highlight', this.categoryFilter == "" );
        this.c_photoButtonEl.classList.toggle('p-highlight', this.categoryFilter == "photo" );
        this.c_illusButtonEl.classList.toggle('p-highlight', this.categoryFilter == "illustration" );
        this.c_wildButtonEl.classList.toggle('p-highlight', this.categoryFilter == "wild" );
        this.c_customButtonEl.classList.toggle('p-highlight', this.categoryFilter == "custom" );
    }


    /**
     * Updates the search filters and visualization modes based on a command.
     *
     * This method processes a given command to modify the current view mode
     * or category/text filters, then updates the displayed styles in the
     * gallery according to these changes.
     *
     * @param {string} command - A string representing the update operation,
     *   starting with:
     *     '$' followed by 'grid' or 'list' to switch view modes
     *     '@' followed by a category name to filter by category (empty string for no filtering)
     *     '>' followed by a text to filter styles by name (empty string for no filtering)
     */
    updateSearch(command) {

        // if the command starts with "$", change the view mode
        if( command.startsWith('$') ) {
            const viewMode = command.substring(1);
            if( viewMode == this.viewMode ) { return; }
            this.viewMode = viewMode;
            this.updateButtons();
        }

        // if the command starts with "@", change the category filter
        else if( command.startsWith('@') ) {
            const categoryFilter = command.substring(1);
            if( categoryFilter == this.categoryFilter ) { return; }
            this.categoryFilter = categoryFilter;
            this.updateButtons();
        }

        // if the command starts with ">", change the text filter
        else if( command.startsWith(">") ) {
            const textFilter = command.substring(1);
            if( textFilter == this.textFilter ) { return; }
            this.textFilter = textFilter;
        }

        // apply filters and re-render gallery
        const categoryFilter = this.categoryFilter;
        const filteredStyles = this.allStyles.filter( style => {
            return this.categoryFilter == "" || style.category === categoryFilter;
        });
        StyleGalleryDialog.renderGrid( this.gridEl, this.viewMode, filteredStyles );
        this.statusEl.innerText = `${filteredStyles.length} style(s) found ${categoryFilter ? `in category '${categoryFilter}'` : ''}`;
    }


    /**
     * Renders the gallery grid with the provided visual styles.
     *
     * This static method generates HTML content for displaying a
     * list/grid of styles based on the specified view mode.
     *
     * @param {HTMLElement}   gridEl - The container element where the grid will be rendered.
     * @param {string}      viewMode - The current view mode ('grid' or 'list') that determines
     *                                 the layout of each item.
     * @param {Array<Object>} styles - An array of objects representing the visual styles
     *                                 to display.
     */
    static renderGrid(gridEl, viewMode, styles) {

        gridEl.className = `zipn-style-${viewMode}`;
        gridEl.innerHTML = styles.map(item => `
        <div class="zipn-style-${viewMode}-card" data-id="${item.id}">
            <img src="${item.thumbnail}" loading="lazy" alt="${item.name}">
            <p>${item.name}</p>
        </div>
        `).join('');
    }

   //-- EVENTS -----------------------------------------------------------

    onCardEnter(cardEl) {
        cardEl.classList.add('p-highlight');
    }

    onCardLeave(cardEl) {
        cardEl.classList.remove('p-highlight');
    }

    onCardClick(cardEl) {
        const styleIdx  = cardEl.dataset.id;
        const allStyles = this.allStyles || [];
        const style     = 0<=styleIdx && styleIdx<allStyles.length ? allStyles[styleIdx] : null;
        if( style ) { this.onSelectStyle?.(style.name); }
        this.close();
    }


    //-- DIALOG COMPONENTS ------------------------------------------------

    /** A spacer element in the toolbar. */
    static get SPACER() {
        return html("div.zipn-spacer");
    }

    /** A divider element (vertical line) in the toolbar. */
    static get DIVIDER() {
        return html("div.zipn-divider");
    }

    /**
     * A container for displaying the gallery results in grid format.
     * @returns {HTMLElement} An HTML structure representing the grid container.
     */
    static get RESULT_GRID() {
        return html("main.zipn-result-container", { id: "result-container" }, [
            html("div.zipn-style-grid", { id: "zipn-style-grid" })
        ]);
    }

    /**
     * A status bar to show current status or messages.
     * @returns {HTMLElement} An HTML structure representing the status bar.
     */
    static get STATUS_BAR() {
        return html("footer.zipn-status-bar", {}, [
            html("span", { id: "zipn-status-text", textContent: "Showing 0 elements"})
        ]);
    }


    /**
     * Creates a button for the toolbar.
     * @param {string}   id      - The unique identifier for the button.
     * @param {string}   icon    - The icon to be used in the button. e.g., "i.pi.pi-image"
     * @param {string}   text    - The text content of the button.
     * @param {string}   tooltip - The title attribute value for the button, representing its tooltip.
     * @param {function} onClick - Function to be executed when the button is clicked.
     * @returns {HTMLElement} A button element with the specified icon, text, tooltip and onclick function.
     */
    createToolButton(id, icon, text, tooltip, onClick) {
        // if no ID is provided, generate a random one
        if( !id ) {
            do { id = 'zipn-btn-' + Math.random().toString(36).slice(2, 9); }
            while( document.getElementById(id) );
        }
        // if the icon ID is provided with spaces between words, convert them to dots
        if( icon ) {
            icon = icon.replace(' ', '.');
        }

        // generate the 3 possible types of buttons:
        //   - button with icon only (no text)
        //   - button with text only
        //   - button with icon and text
        if( icon && !text ) {
            return html( "button.p-button-text", { id: id, title: tooltip, onclick: onClick }, [ html(`i.${icon}`) ] );
        }
        if( !icon && text ) {
            return html( "button.p-button-text", { id: id, title: tooltip, textContent: text, onclick: onClick }, [] );
        }
        return html("button.p-button-text", { id: id, title: tooltip, onclick: onClick }, [
            html(`i.${icon}`, { textContent: text})
        ]);
    }


    /**
     * Creates an input search bar for searching within the gallery dialog.
     * @returns {HTMLElement} An HTML structure representing a search bar.
     */
    createSearchBar() {
        return html("div", {}, [
            html("input.p-inputtext.p-component", { type: "search", placeholder: "Search" }),
            StyleGalleryDialog.DIVIDER,
            this.createToolButton("zipn-all-btn"   , '', "All"         , "Search all styles"              , () => { this.updateSearch("@"); }),
            this.createToolButton("zipn-photo-btn" , '', "Photo"       , "Search only photographic styles", () => { this.updateSearch("@photo");}),
            this.createToolButton("zipn-illus-btn" , '', "Illustration", "Search only illustration styles", () => { this.updateSearch("@illustration"); }),
            this.createToolButton("zipn-wild-btn"  , '', "Wild"        , "Search only wild styles"        , () => { this.updateSearch("@wild"); }),
            this.createToolButton("zipn-custom-btn", '', "Custom"      , "Search only custom styles"      , () => { this.updateSearch("@custom"); }),
            StyleGalleryDialog.DIVIDER,
            this.createToolButton("zipn-grid-btn", 'pi pi-image', "", "Grid View", () => { this.updateSearch("$grid"); }),
            this.createToolButton("zipn-list-btn", 'pi pi-list' , "", "List View", () => { this.updateSearch("$list"); }),
            StyleGalleryDialog.DIVIDER,
        ]);
    }


    createDialogContent() {
        return html("div.zipn-dialog", {}, [
            this.createSearchBar(),
            StyleGalleryDialog.RESULT_GRID,
            StyleGalleryDialog.STATUS_BAR
        ]);
    }

}


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
    button.callback = () => {
        StyleGalleryDialog.launch(title, (style) =>
        {
            // ensure the style name is properly quoted
            // before setting the combo widget's value
            if( !style.startsWith('"') ) { style = `"${style}"`; }
            prevWidget.value = style;
            prevWidget.callback(style);
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
