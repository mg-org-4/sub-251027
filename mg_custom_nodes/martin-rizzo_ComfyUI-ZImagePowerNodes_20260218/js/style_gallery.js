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

        // create the custom dialog.
        //   icons can be taken from PrimeIcons or Pictogrammers MDT
        //   PrimeIcons       : e.g. "i.pi.pi-image"   (https://primevue.org/icons)
        //   Pictogrammers MDI: e.g. "i.mdi.mdi-image" (https://pictogrammers.com/library/mdi)
        this.element = makeCustomDialog(
            DIALOG_ID                 , //< ID of the DOM element where the dialog is located
            TITLE_ID                  , //< ID of the DOM element where the title is located
            'Select Style'            , //< title
            'i.mdi.mdi-image-multiple', //< icon

            // DIALOG CONTENT
            html("div.zipn-dialog", {}, [
                this.createSearchBar(),
                html("div.zipn-two-columns", {}, [
                    StyleGalleryDialog.DETAILS_PANE,
                    StyleGalleryDialog.SEARCH_RESULTS_PANE,
                ]),
            ]),

            () => this.close() //< close callback
        );

        // variables used to store dialog states,
        // which should be re-initialized every time the dialog is launched
        this.isOpen            = false;
        this.initialStyleName  = "";     //< the initial style name (before applying the selected one)
        this.initialStyleID    = null;   //< the initial style ID (before applying the selected one)
        this.searchStyleID     = null;   //< ID of the style that matches the search text (null = not matching anything)
        this.pointedStyleID    = null;   //< ID of the style being pointed by the mouse (null = not pointing to anything)
        this.oldSelectionID    = null;   //< ID of the previously selected style (null = no previous selection)
        this.textFilter        = "";     //< text entered by the user to filter styles (case-insensitive)
        this.categoryFilter    = "";     //< "photo", "illustration", "wild", "custom" (empty means all categories)
        this.viewMode          = "grid"; //< "grid" or "list"

        // internal variables:
        this.stylesByID          = [];    //< an array to store styles in ID order (for fast access)
        this.styleIDsByLowerName = {};    //< map lowercase style names to their IDs.
        this.inputChangeTimer1   = null;  //< timer used by the 'onInputChange' event
        this.inputChangeTimer2   = null;  //< timer used by the 'onInputChange' event
        this.isPointerLocked     = false; //< flag used by 'onInputChange' to block mouse events

        // dialog elements:
        this.searchInputEl     = this.element.querySelector('#zipn-search-input');
        this.searchResultsEl   = this.element.querySelector('#zipn-search-results');
        this.detailsHeaderEl   = this.element.querySelector('.zipn-details-pane h1');
        this.detailsImageEl    = this.element.querySelector('.zipn-details-pane img');
        this.detailsTextEl     = this.element.querySelector('.zipn-details-pane p');
        this.onSelectStyle     = null;
        // (toolbar buttons)
        this.tb_allButtonEl    = this.element.querySelector('#zipn-all-btn');
        this.tb_photoButtonEl  = this.element.querySelector('#zipn-photo-btn');
        this.tb_illusButtonEl  = this.element.querySelector('#zipn-illus-btn');
        this.tb_wildButtonEl   = this.element.querySelector('#zipn-wild-btn');
        this.tb_customButtonEl = this.element.querySelector('#zipn-custom-btn');
        this.tb_gridButtonEl   = this.element.querySelector('#zipn-grid-btn');
        this.tb_listButtonEl   = this.element.querySelector('#zipn-list-btn');


        // event listeners
        const CARD_SELECTOR = '.zipn-style-grid-card, .zipn-style-list-card';
        setupCardHoverListeners( this.searchResultsEl, CARD_SELECTOR,
            (card) => { this.onCardEnter(card); },
            (card) => { this.onCardLeave(card); },
            (card) => { this.onCardClick(card); }
        );
        this.searchInputEl.addEventListener('input'  , (e) => { this.onInputChange(e.target); });
        this.searchInputEl.addEventListener('keydown', (e) => { this.onInputKeyDown(e.key); });
        this.searchInputEl.addEventListener('blur'   , ()  => { this.onInputLostFocus(); } );
    }


    /**
     * Launches the style gallery dialog.
     * @param {string}   title         - The title of the dialog.
     * @param {string}   styleName     - The name of the selected style.
     * @param {Function} onSelectStyle - A callback function that gets called
     *                                   when a style is selected by the user.
     */
    static launch(title, styleName, onSelectStyle) {

        // styleName can be wrapped in quotes, remove them
        if( styleName && styleName.startsWith('"') && styleName.endsWith('"') ) {
            styleName = styleName.slice(1, -1);
        }

        // create the first time and use the same instance the next time
        if( !this._instance ) { this._instance = new StyleGalleryDialog(); }
        const dialog  = this._instance;
        const titleEl = dialog.element.querySelector(`#${TITLE_ID}`);

        if( titleEl ) { titleEl.textContent = title; }
        dialog.onSelectStyle = onSelectStyle;
        dialog.onOpen(styleName);
    }


    /**
     * Closes the dialog.
     */
    close()
    { this.onClose(); super.close();  }


   /**
    * Handles the user's choice of a style and closes the dialog.
    */
    userHasChosen() {
        const selectionID = this.pointedStyleID ? this.pointedStyleID : this.searchStyleID;
        const style       = selectionID != null ? this.stylesByID[selectionID] : null;
        if( style ) {
            this.onSelectStyle?.(style.name);
            this.close();
        }
    }


    /**
     * Updates toolbar buttons state based on current view mode and category filter.
     */
    updateToolbarButtons() {
        // view mode buttons
        this.tb_listButtonEl.classList.toggle('p-highlight', this.viewMode == "list" );
        this.tb_gridButtonEl.classList.toggle('p-highlight', this.viewMode == "grid" );
        // category buttons
        this.tb_allButtonEl.classList.toggle('p-highlight', this.categoryFilter == "" );
        this.tb_photoButtonEl.classList.toggle('p-highlight', this.categoryFilter == "photo" );
        this.tb_illusButtonEl.classList.toggle('p-highlight', this.categoryFilter == "illustration" );
        this.tb_wildButtonEl.classList.toggle('p-highlight', this.categoryFilter == "wild" );
        this.tb_customButtonEl.classList.toggle('p-highlight', this.categoryFilter == "custom" );
    }


    /**
     * Updates the selected style and displays its details in the dialog.
     * @param {boolean} force - If true, updates the selection even if
     *                          no change occurred. Defaults to false.
     */
    updateSelection(force=false) {
        const newSelectionID = this.pointedStyleID ? this.pointedStyleID : this.searchStyleID;
        const detailsID      = newSelectionID     ? newSelectionID     : this.initialStyleID;
        if( !force && newSelectionID === this.oldSelectionID ) { return; }

        // deactivate the card with the old style
        const oldCardEl = this.oldSelectionID != null ? this.element.querySelector(`#zipn-style-${this.oldSelectionID}`) : null;
        if( oldCardEl ) { oldCardEl.classList.remove('active'); }

        this.oldSelectionID = newSelectionID;

        // activate the card with the new style
        const newCardEl = newSelectionID != null ? this.element.querySelector(`#zipn-style-${newSelectionID}`) : null;
        if( newCardEl ) { newCardEl.classList.add('active'); }

        // update details pane
        const style       = detailsID != null ? this.stylesByID[ detailsID ] : null;
        const cacheBuster = this.cacheBuster ? '&cache=' + this.cacheBuster : '';

        this.detailsHeaderEl.textContent = style?.name        || "";
        this.detailsTextEl.textContent   = style?.description || "";
        if( style?.thumbnail ) {
            this.detailsImageEl.src              = style.thumbnail + cacheBuster;
            this.detailsImageEl.style.visibility = 'visible';
        } else {
            this.detailsImageEl.style.visibility = 'hidden';
        }
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
    updateSearchResults(command) {

        // if the command starts with "$", change the view mode
        if( command.startsWith('$') ) {
            const viewMode = command.substring(1);
            if( viewMode == this.viewMode ) { return; }
            this.viewMode = viewMode;
            this.updateToolbarButtons();
        }

        // if the command starts with "@", change the category filter
        else if( command.startsWith('@') ) {
            const categoryFilter = command.substring(1);
            if( categoryFilter == this.categoryFilter ) { return; }
            this.categoryFilter = categoryFilter;
            this.updateToolbarButtons();
        }

        // if the command starts with ">", change the text filter
        else if( command.startsWith(">") ) {
            const textFilter = command.substring(1);
            if( textFilter == this.textFilter ) { return; }
            this.textFilter = textFilter;
        }

        // cache buster used to force re-fetching of images from cache each hour
        this.cacheBuster = Math.floor(Date.now() / 3600000);

        // apply filters and re-render gallery
        const filteredStyles = StyleGalleryDialog.applyFilter( this.stylesByID, this.textFilter, this.categoryFilter );
        StyleGalleryDialog.renderResults( this.searchResultsEl, this.viewMode, filteredStyles, this.initialStyleID, this.cacheBuster );

        if( this.textFilter ) { this.searchStyleID = filteredStyles[0]?.id; }
        else                  { this.searchStyleID = null; }
        this.updateSelection(true);
    }


    /**
     * Renders the gallery grid with the provided visual styles.
     *
     * This static method generates HTML content for displaying a list or grid
     * of styles based on the specified view mode. Each style is represented as
     * an object in the 'styles' array and includes properties such as
     * 'id', 'name', and 'thumbnail'.
     *
     * @param {HTMLElement} containerEl    - The container element where the gallery will be rendered.
     * @param {string}      viewMode       - The current view mode ('grid' or 'list') that determines
     *                                       the layout of each item in the gallery. This parameter
     *                                       is used to apply appropriate CSS classes.
     * @param {Array<Object>} styles       - An array of objects representing the visual styles to display.
     * @param {string|null} initialStyleID - The ID of the initially selected style, which will receive an
     *                                       additional CSS class ('initial') for highlighting.
     * @param {string|null} cacheBuster_   - A string used as a cache buster appended to each thumbnail image URL
     *                                       to ensure that the browser fetches the latest version of images.
     * @example
     * const styles = [
     *   { id: 'style-1', name: 'Modern Look', thumbnail: '/images/modern.jpg' },
     *   { id: 'style-2', name: 'Retro Feel', thumbnail: '/images/retro.jpg' }
     * ];
     * renderResults(document.getElementById('gallery-container'), 'grid', styles, 'style-1', Date.now());
     */
    static renderResults(containerEl, viewMode, styles, initialStyleID = null, cacheBuster_ = null) {
        const baseClass   = `zipn-style-${viewMode}`;
        const cacheBuster = cacheBuster_ ? '&cache=' + cacheBuster_ : '';
        containerEl.className = baseClass;
        containerEl.innerHTML = styles.map( style => {
            const extraClass = style.id === initialStyleID ? ' initial' : '';
            const imageSrc   = style.thumbnail + cacheBuster;
            return `
                <div class="${baseClass}-card${extraClass}" id="zipn-style-${style.id}" data-id="${style.id}">
                    <img src="${imageSrc}" loading="lazy" alt="${style.name}">
                    <p>${style.name}</p>
                </div>`;
        }).join('');
    }


    /**
     * Applies a filter to an array of styles based on text and category criteria.
     * @param {Object[]} allStyles - An array of style objects, each containing properties
     *                               such as id, name, description, category, etc.
     * @param {string}  textFilter - A string that filters styles by matching style names
     *                               against search terms.
     * @param {string}    category - The selected category for filtering the styles
     *                               (e.g., "photo", "illustration", etc.). An empty
     *                               string indicates no specific category filter.
     * @returns {Object[]}
     *   Returns an array of style objects that match the given text and category filters.
     */
     static applyFilter(allStyles, textFilter, category) {
        const terms = textFilter.toLowerCase().split(' ');
        //const tags  = terms.filter(t => t.startsWith('#'));
        const words = terms.filter(t => !t.startsWith('#'));
        return allStyles.filter(style => {
            const matchesCategory = category === "" || style.category === category;
            const matchesWords    = words.every(word => style.lowerName.includes(word));
            const matchesTags     = true; //tags.length === 0 || tags.some(tag => style.tags.includes(tag));
            return matchesCategory && matchesWords && matchesTags;
        });
    }


    //-- EVENTS -----------------------------------------------------------

    /**
     * Called when the dialog is launched or re-opened.
     *
     * Initializes variables and loads style data from the server.
     * @param {string} styleName - The name of the initial style to be selected.
     */
    onOpen(styleName) {

        // initialize variables as if the dialog had just been created
        this.isOpen              = true;
        this.initialStyleName    = styleName;
        this.initialStyleID      = null;
        this.searchStyleID       = null;
        this.pointedStyleID      = null;
        this.oldSelectionID      = null;
        this.textFilter          = '';
        this.categoryFilter      = "";
        this.isPointerLocked     = false;
        this.searchInputEl.value = '';
        // `this.viewMode` isn't set here becouse is kept between dialog reopens

        // load style data from server
        fetchLastVersionStyles( (styles) => {
            this.onReceivedStyles(styles);
        });

        // 
        requestAnimationFrame( () => {
            this.show();
            this.updateToolbarButtons();
            this.searchInputEl.focus();
        });
        // trigger enter animation
        //requestAnimationFrame(() => { this.element.classList.add('fade-in'); });
    }


    /**
     * Called when the dialog is closed. Updates the open state flag.
     */
    onClose() {
        this.isOpen = false;
    }


    /**
     * Called when style data is received from the server.
     * Initializes internal arrays and maps with the received data.
     * @param {Array} styles - An array of style objects received from the server.
     */
    onReceivedStyles(styles) {
        this.stylesByID          = styles;
        this.styleIDsByLowerName = Object.fromEntries(styles.map(style => [style.name.toLowerCase(), style.id]));
        this.initialStyleID      = this.styleIDsByLowerName[this.initialStyleName.toLowerCase()];
        this.updateSearchResults("!refresh");
        this.updateSelection();
    }


    /**
     * Called when the search input loses focus.
     * Try to keep the search input focused always.
     */
    onInputLostFocus() {
        if( !this.isOpen ) { return; }
        setTimeout(() => { if (this.isOpen) this.searchInputEl.focus(); }, 0);
    }


    /**
     * Called when the text in the search input changes.
     * Updates the search results and the active item, locking the pointer
     * movement events temporarily to prevent they interfering with the
     * search result autoselection.
     * @param {HTMLInputElement} inputEl - The search input element.
     * @param {boolean} isEnterPressed   - Indicates whether the Enter key was pressed.
     */
    onInputChange(inputEl, isEnterPressed = false) {

        // temporarily lock pointer movement events
        this.isPointerLocked = true;
        clearTimeout(this.inputChangeTimer1);
        this.inputChangeTimer1 = setTimeout(() => { this.isPointerLocked = false; }, 800);

        // debounce the search results update
        clearTimeout(this.inputChangeTimer2);
        this.inputChangeTimer2 = setTimeout(() =>
        {
            // always update the search results first so that when user
            // presses enter it will accept the most updated result
            this.updateSearchResults(`>${inputEl.value}`);
            if( isEnterPressed ) {
                this.userHasChosen();
            }
            this.pointedStyleID = null;
            this.updateSelection();

        }, isEnterPressed ? 100 : 300);

    }


    /**
     * Called when a key is pressed in the search input.
     * @param {string} key - The key that was pressed.
     */
    onInputKeyDown(key) {
        if     ( key == 'Escape' ) { this.close(); }
        else if( key == 'Enter'  ) { this.onInputChange(this.searchInputEl, true); }
    }


    /**
     * Called when the mouse enters a style card.
     * @param {HTMLElement} cardEl - The card element that was entered.
     */
    onCardEnter(cardEl) {
        if( this.isPointerLocked ) { return; }
        // updates the currently pointed style ID and triggers selection updates
        this.pointedStyleID = cardEl.dataset?.id;
        this.updateSelection();
    }


    /**
     * Called when the mouse leaves a style card. 
     * @param {HTMLElement} cardEl - The card element that was left.
     */
    onCardLeave(cardEl) {
        if( this.isPointerLocked ) { return; }
        // resets the currently pointed style ID and triggers selection updates
        if( this.pointedStyleID === cardEl.dataset?.id ) {
            this.pointedStyleID = null;
            this.updateSelection();
        }
    }


    /**
     * Called when a style card is clicked.
     * @param {HTMLElement} cardEl - The card element that was clicked.
     */
    onCardClick(cardEl) {
        // sets the currently pointed style ID and triggers user selection handling
        this.pointedStyleID = cardEl?.dataset?.id;
        this.userHasChosen();
    }


    //-- DIALOG COMPONENTS ------------------------------------------------

    /** A spacer element in the toolbar. */
    static get SPACER() { return html("div.zipn-spacer"); }

    /** A divider element (vertical line) in the toolbar. */
    static get DIVIDER() { return html("div.zipn-divider"); }

    /**
     * A container for displaying detailed information about the hovered style.
     * @returns {HTMLElement} An HTML structure representing the details pane.
     */
    static get DETAILS_PANE() {
        return html(
        "div.zipn-details-pane", {}, [
            html("h1.zipn-details-header"),
            html("img"),
            html("p.zipn-details-description"),
        ]);
    }

    /**
     * A container for displaying the styles resulting from a search query.
     * @returns {HTMLElement} An HTML structure representing the search results pane.
     */
    static get SEARCH_RESULTS_PANE() {
        return html(
        "div.zipn-search-results-pane", {}, [
            html("div.zipn-style-grid", { id: "zipn-search-results" })
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
            html("input.p-inputtext.p-component", { id: "zipn-search-input", type: "search", placeholder: "Search" }),
            StyleGalleryDialog.DIVIDER,
            this.createToolButton("zipn-all-btn"   , '', "All"         , "Search all styles"              , () => { this.updateSearchResults("@"); }),
            this.createToolButton("zipn-photo-btn" , '', "Photo"       , "Search only photographic styles", () => { this.updateSearchResults("@photo");}),
            this.createToolButton("zipn-illus-btn" , '', "Illustration", "Search only illustration styles", () => { this.updateSearchResults("@illustration"); }),
            this.createToolButton("zipn-wild-btn"  , '', "Wild"        , "Search only wild styles"        , () => { this.updateSearchResults("@wild"); }),
            this.createToolButton("zipn-custom-btn", '', "Custom"      , "Search only custom styles"      , () => { this.updateSearchResults("@custom"); }),
            StyleGalleryDialog.DIVIDER,
            this.createToolButton("zipn-grid-btn", 'pi pi-image', "", "Grid View", () => { this.updateSearchResults("$grid"); }),
            this.createToolButton("zipn-list-btn", 'pi pi-list' , "", "List View", () => { this.updateSearchResults("$list"); }),
            StyleGalleryDialog.DIVIDER,
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
        StyleGalleryDialog.launch(title, prevWidget.value, (style) =>
        {
            // ensure the style name is properly quoted
            // before setting the combo widget's value
            if( !style.startsWith('"') ) { style = `"${style}"`; }
            prevWidget.value = style;
            prevWidget.callback(style);
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
