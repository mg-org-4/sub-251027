/**
 * File    : gallery_dialog.js
 * Purpose : Generic class for ComfyUI dialogs that display a thumbnail/image for each selectable option.
 * Author  : Martin Rizzo | <martinrizzo@gmail.com>
 * Date    : May 8, 2026
 * Repo    : https://github.com/martin-rizzo/ComfyUI-ZImagePowerNodes
 * License : MIT
 *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *
 * While this class is part of the `ComfyUI-ZImagePowerNodes` suite, it was
 * designed to be as generic and modular as possible. The goal is for it to
 * be easily plug-and-play in any other ComfyUI Nodes project you might be
 * working on.
 *
 * Feel free to use it, modify it, or integrate it however you like! If you
 * find it useful, a quick shout-out to this project or a mention of my name
 * would be greatly appreciated. :)
 *
 *_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
 */
import { ComfyDialog, $el as html } from "../../../scripts/ui.js"; //< deprecated ??
export { GalleryDialog };
const DIALOG_ID = 'zipn-style-gallery-dialog'; //< ID of the DOM element where the dialog is located
const TITLE_ID  = 'zipn-style-gallery-title';  //< ID of the DOM element where the title is located



//#====================== GalleryDialog.DataProvider =======================#

/**
 * Base provider class for GalleryDialog.
 *
 * Users should extend this class and implement the fetchDataArray method
 * to provide the data source for the gallery.
 *
 * @example
 * class MyDataProvider extends GalleryDialog.DataProvider {
 *     async fetchDataArray(callback) {
 *         const data = [{ id: 0, name: "Example", thumbnail: "url/to/img.png", ... }];
 *         callback(data);
 *     }
 * }
 */
class GalleryDialogDataProvider {

    /**
     * Fetches an array containing the data for each item to be displayed in the gallery.
     *
     * This method must be overridden by the inheriting class to implement
     * the actual data retrieval logic.
     *
     * @param {function(Array<Object>)} callback
     *   A callback function that receives an array of items.
     *    Each item object should contain the following properties:
     *     @property {number|string} id          - Unique identifier for the item (the index in the list).
     *     @property {string}        name        - The display name of the item.
     *     @property {string}        lowerName   - (optional) The name of the item converted to lowercase (used for searching/filtering).
     *     @property {string}        category    - The category the item belongs to.
     *     @property {string}        description - A detailed description of the item.
     *     @property {Array<string>} tags        - A list of strings containing the tags associated with the item.
     *     @property {string}        thumbnail   - The URL path to the item's thumbnail image.
     *
     * @returns {Promise<void>}
     */
    async fetchDataArray(callback) {
        throw new Error(`Method fetchDataArray(callback: ${typeof callback}) must be implemented.`);
    }

    /**
     * Returns an array of category definitions used for filtering the gallery items.
     *
     * When implemented, this method provides the categories used in the
     * gallery's filtering UI. Each category definition is an array of
     * three strings: [category, displayName, description].
     *
     * If not overridden, there will be no filtering available in the gallery.
     *
     * @returns {Array<Array<string>>}
     *   An array of category definitions where each item contains:
     *     - category    (string): The value used for matching gallery items (must match the "category" property in items)
     *     - displayName (string): The visible name shown in the UI
     *     - description (string): Tooltip description for screen readers/accessibility
     *
     * @example
     * class MyDataProvider extends GalleryDialog.DataProvider {
     *   getCategories() {
     *     return [
     *       ['images'   , 'Images'   , 'View all image files'   ],
     *       ['videos'   , 'Videos'   , 'View all video files'   ],
     *       ['documents', 'Documents', 'View all document files']
     *     ];
     *   }
     * }
     */
    getCategories() {
        return null;
    }

}


//#====================== GalleryDialog.ItemRenderer =======================#

/**
 * Base renderer class for GalleryDialog.
 *
 * Provides default HTML templates for rendering items.
 * Users can extend this class to override specific methods
 * and customize the visual presentation.
 *
 * @example
 * class MyItemRenderer extends GalleryDialogItemRenderer {
 *     renderThumbnail(item, className, cacheBuster) {
 *         return `<div class="${className}">Custom Thumbnail: ${item.name}</div>`;
 *     }
 * }
 */
class GalleryDialogItemRenderer {

    /**
     * Renders the detailed HTML view for a selected item.
     *
     * @param {Object|null} item   - The data object representing the item, or `null` if no item is selected.
     * @param {string} cacheBuster - A string appended to the URL to prevent browser caching.
     * @returns {string}
     *     HTML string representing the detailed description view.
     */
    renderDescription(item, cacheBuster) {
        const imageHTML = this.renderImage(item, 'zipn-image', cacheBuster);
        return `
           <h1>${item?.name || ""}</h1>
           ${imageHTML}
           <p>${item?.description || ""}</p>
           `;
    }

    /**
     * Renders the image HTML element for the selected item.
     *
     * @param {Object|null} item   - The data object representing the item, or `null` if no item is selected.
     * @param {string} className   - CSS class to be applied to the img tag.
     * @param {string} cacheBuster - A string appended to the URL to prevent browser caching.
     * @returns {string}
     *    HTML string representing the image of the selected item.
     */
    renderImage(item, className, cacheBuster) {
        if( !item?.thumbnail ) {
            return "";
        }
        const imageURL = item.thumbnail + cacheBuster;
        return `<img class="${className}" src="${imageURL}" loading="lazy" alt="${item.name | ""}"/>`;
    }

    /**
     * Renders the thumbnail HTML element for the gallery grid.
     *
     * @param {Object|null} item   - The data object representing the item, or `null` if no item is selected.
     * @param {string} className   - CSS class to be applied to the img tag.
     * @param {string} cacheBuster - A string appended to the URL to prevent browser caching.
     * @returns {string}
     *     HTML string representing the thumbnail of the selected item.
     */
    renderThumbnail(item, className, cacheBuster) {
        if( !item?.thumbnail ) {
            return "";
        }
        const imageURL = item.thumbnail + cacheBuster;
        return `<img class="${className}" src="${imageURL}" loading="lazy" alt="${item.name | ""}"/>`;
    }

}


//#============================= GalleryDialog =============================#

class GalleryDialog {

    static get ItemRenderer() { return GalleryDialogItemRenderer; }
    static get DataProvider() { return GalleryDialogDataProvider; }


    constructor(dataProvider, itemRenderer) {

        // ensure parameters are of the correct type
        if (!(dataProvider instanceof GalleryDialogDataProvider)) {
            throw new TypeError('dataProvider must be an instance of GalleryDialogDataProvider');
        }
        if (!(itemRenderer instanceof GalleryDialogItemRenderer)) {
            throw new TypeError('itemRenderer must be an instance of GalleryDialogItemRenderer');
        }

        this.dataProvider = dataProvider;
        this.itemRenderer = itemRenderer;
        this._instance    = null;
    }


    /**
     * Launches the gallery dialog with the provided configuration and dependencies.
     *
     * @param {string}                    title - Title of the dialog.
     * @param {string}                    selectedName - Default selected element name.
     * @param {Function}                  onSelect - Callback executed when an element is selected.
     * @returns {void}
     * @example
     * const MyGalleryDialog = new GalleryDialog(
     *                               new MyGalleryDialogDataProvider(),
     *                               new MyGalleryDialogItemRenderer()
     *                               );
     * 
     * MyGalleryDialog.launch("Select an Image", "default", (item) => {
     *   console.log("Selected:", item);
     * });
     */
    launch(title, selectedName, onSelect) {
        // create the instance only once (singleton instance logic)
        if( !this._instance ) {
             this._instance = new _GalleryDialog(this.dataProvider, this.itemRenderer);
        }

        // relaunch the dialog
        const dialog  = this._instance;
        const titleEl = dialog.element?.querySelector(`#${TITLE_ID}`);
        if (titleEl) { titleEl.textContent = title; }
        dialog.onSelectElement = onSelect;
        dialog.onOpen(selectedName);
    }

}


//#=========================================================================#
//#//////////////////////////// GALLERY DIALOG /////////////////////////////#
//#=========================================================================#

class _GalleryDialog extends ComfyDialog {

    /**
     * Creates a new instance, registering a data provider and card renderer.
     * @param {GalleryDialogDataProvider}  dataProvider - The object that provides the data for each element in the gallery.
     * @param {GalleryDialogItemRenderer} itemRenderer - The object that renders each item shown in the gallery.
     */
    constructor(dataProvider, itemRenderer) {
        super();
        ensureCSSLoaded();

        //---- INTERNAL STATE VARIABLES -------------------
        // - should be re-initialized every time the dialog is launched

        /** @type {boolean} Indica si el diálogo está abierto. */
        this.isOpen = false;

        /** @type {string} The initial element name (before applying the selected one). */
        this.initialElementName = "";

        /** @type {number|null} ID of the initial card/item (which will be highlighted). */
        this.initialCardID = null;

        /** @type {number|null} ID of the card/item being pointed by the mouse. */
        this.hoveredCardID = null;

        /** @type {number|null} Index of the selected item in 'resultStyles'. */
        // focusedCardIndex
        this.resultIndex = null;

        /** @type {Array<object>} An array of elements that match the search text. */
        this.resultElements = [];

        /** @type {number} Number of columns used in the search results grid. */
        this.resultColumns = 4;

        /** @type {number|null} ID of the previously selected item. */
        this.oldSelectionID = null;

        /** @type {string} Text entered by the user to filter styles (case-insensitive). */
        this.textFilter = "";

        /** @type {string} Active category filter ("photo", "illustration", "wild", "custom"). Empty means all categories. */
        this.categoryFilter = "";

        /** @type {"grid"|"list"} View mode of the dialog, either "grid" or "list". */
        this.viewMode = "grid";

        //---- INTERNAL VARIABLES -------------------------

        /** @type {GalleryDialogDataProvider} The object that provides the data for each element in the gallery. */
        this.dataProvider = dataProvider;

        /** @type {GalleryDialogItemRenderer} The object that renders each card shown in the gallery.*/
        this.itemRenderer = itemRenderer;

        /** @type {Array<object>} An array to store elements in ID order for fast access. */
        this.elementsByID = [];

        /** @type {Object<string, number>} Map of lowercase element names to their IDs. */
        this.elementIDsByLowerName = {};

        /** @type {number|null} Timer used by the lockPointer method. */
        this.pointerLockedTimer = null;

        /** @type {number|null} Timer used by the 'onInputChange' event. */
        this.inputChangeTimer2 = null;

        /** @type {boolean} Flag used by 'onInputChange' to block mouse events. */
        this.isPointerLocked = false;

        /** @type {Array<[string, HTMLElement]>}
         *     The buttons in the toolbar that allow filtering by category.
         *     The array contains pairs of category name and button element. */
        this.categoryButtons = [];

        /** @type {Function|null} Callback function for when an element is selected. */
        this.onSelectElement = null;

        // create the custom dialog.
        //   icons can be taken from PrimeIcons or Pictogrammers MDT
        //   PrimeIcons       : e.g. "i.pi.pi-image"   (https://primevue.org/icons)
        //   Pictogrammers MDI: e.g. "i.mdi.mdi-image" (https://pictogrammers.com/library/mdi)
        this.element = makeCustomDialog(
            DIALOG_ID                 , //< ID of the DOM element where the dialog is located
            TITLE_ID                  , //< ID of the DOM element where the title is located
            ''                        , //< title
            'i.mdi.mdi-image-multiple', //< icon

            // DIALOG CONTENT
            html("div.zipn-dialog", {}, [
                this.createToolbar(),
                html("div.zipn-two-columns", {}, [
                    html("div.zipn-details-pane"),
                    html("div.zipn-search-results-pane", { id: "zipn-search-results-pane" }, [
                        html("div.zipn-grid", { id: "zipn-search-results" })
                    ])
                ]),
            ]),

            () => this.close() //< close callback
        );

        //---- DIALOG ELEMENTS ----------------------------

        /** @type {HTMLElement|null} Search input element in the dialog. */
        this.searchInputEl = this.element.querySelector('#zipn-search-input');

        /** @type {HTMLElement|null} Element containing search results. */
        this.searchResultsEl = this.element.querySelector('#zipn-search-results');

        /** @type {HTMLElement|null} Element containing the details pane. */
        this.detailsPaneEL = this.element.querySelector('.zipn-details-pane');

        //---- EVENT LISTENERS ----------------------------

        const CARD_SELECTOR = '.zipn-grid-card, .zipn-list-card';
        setupCardHoverListeners( this.searchResultsEl, CARD_SELECTOR,
            (card     ) => { this.onCardEnter(card); },
            (_card    ) => { },
            (card     ) => { this.onCardClick(card); },
            (container) => { this.onCardContainerLeave(container); }
        );
        this.searchInputEl.addEventListener('input'  , (e) => { this.onInputChange(e.target); });
        this.searchInputEl.addEventListener('keydown', (e) => { if (this.onInputKeyDown(e.key)) { event.preventDefault(); } });
        this.searchInputEl.addEventListener('blur'   , ()  => { this.onInputLostFocus(); } );
    }

    /**
     * Closes the dialog.
     */
    close() {
        this.onClose();
        super.close();
    }

    /**
     * Returns the ID of the currently selected element.
     * @returns {number|null} The selected element's ID or null if no selection exists.
     */
    getSelectionID() {
        const resultID = (this.resultIndex != null) ? this.resultElements[this.resultIndex]?.id : null;
        return (this.hoveredCardID != null) ? this.hoveredCardID : resultID;
    }


    /**
     * Updates the selected element and displays its details in the dialog.
     * @param {boolean|Object} shouldScroll - If true, scrolls to the selected element. Defaults to false.
     *                                        Can also be an object containing the scrollIntoView options.
     * @param {boolean}        force        - If true, updates the selection even if no change occurred. Defaults to false.
     */
    updateSelection(shouldScroll=false, force=false) {
        const newSelectionID = this.getSelectionID();
        const detailsID      = newSelectionID != null ? newSelectionID : this.initialCardID;
        if( !force && newSelectionID === this.oldSelectionID ) { return; }

        // deactivate the card with the old element
        const oldCardEl = this.oldSelectionID != null ? this.element.querySelector(`#zipn-element-${this.oldSelectionID}`) : null;
        if( oldCardEl ) { oldCardEl.classList.remove('active'); }

        this.oldSelectionID = newSelectionID;

        // activate the card with the new element and optionally scroll to it
        const newCardEl = newSelectionID != null ? this.element.querySelector(`#zipn-element-${newSelectionID}`) : null;
        if( newCardEl ) { newCardEl.classList.add('active'); }
        if( newCardEl && shouldScroll ) {
            let options = typeof shouldScroll === "object" ? shouldScroll : { behavior: 'smooth', block: 'nearest' };
            newCardEl.scrollIntoView(options);
        }

        // update details pane
        const element     = detailsID != null ? this.elementsByID[ detailsID ] : null;
        const cacheBuster = this.cacheBuster ? '&cache=' + this.cacheBuster : '';
        this.detailsPaneEL.innerHTML = this.itemRenderer.renderDescription(element, cacheBuster);
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
        let shouldScroll = false;

        // if the command starts with "$", change the view mode
        if( command.startsWith('$') ) {
            const viewMode = command.substring(1);
            if( viewMode == this.viewMode ) { return; }
            this.viewMode = viewMode;
            this.resultColumns = (viewMode=="grid" ? 4 : 1);
            this.updateToolbar();
            shouldScroll = { behavior: 'instant', block: 'center' };
        }

        // if the command starts with "@", change the category filter
        else if( command.startsWith('@') ) {
            const categoryFilter = command.substring(1);
            if( categoryFilter == this.categoryFilter ) { return; }
            this.categoryFilter = categoryFilter;
            this.updateToolbar();
            this.resultIndex = null;
        }

        // if the command starts with ">", change the text filter
        else if( command.startsWith(">") ) {
            const textFilter = command.substring(1);
            if( textFilter == this.textFilter ) { return; }
            this.textFilter = textFilter;
            if( this.textFilter  ) { this.resultIndex = 0;    }
            else                   { this.resultIndex = null; }
        }

        // cache buster used to force re-fetching of images from cache each hour
        this.cacheBuster = Math.floor(Date.now() / 3600000);

        // apply filters and re-render gallery
        this.resultElements = _GalleryDialog.applyFilter( this.elementsByID, this.textFilter, this.categoryFilter );
        _GalleryDialog.renderResults( this.searchResultsEl, this.viewMode, this.resultElements, this.itemRenderer, this.initialCardID, this.cacheBuster );

        // disable focus if there are no results
        if( this.resultElements.length == 0 ) { this.resultIndex = null; }

        // update the visual aspect of the card shown as active
        this.updateSelection(shouldScroll,true);
    }

    /**
     * Renders the gallery grid with the provided elements.
     *
     * This static method generates HTML content for displaying a list or grid
     * of elements based on the specified view mode. Each element is represented
     * as an object in the 'elements' array and includes properties such as
     * 'id', 'name', and 'thumbnail'.
     *
     * @param {HTMLElement}               containerEl  - The container element where the gallery will be rendered.
     * @param {string}                    viewMode     - The current view mode ('grid' or 'list') that determines
     *                                                   the layout of each item in the gallery. This parameter
     *                                                   is used to apply appropriate CSS classes.
     * @param {Array<Object>}             elements     - An array of objects representing the elements to display.
     * @param {GalleryDialogItemRenderer} itemRenderer - The object responsible for rendering each item.
     * @param {string|null}           initialElementID - The ID of the initially selected element, which will receive
     *                                                   an additional CSS class ('initial') for highlighting.
     * @param {string|null}               cacheBuster_ - A string used as a cache buster appended to each thumbnail
     *                                                   image URL to ensure that the browser fetches the latest
     *                                                   version of images.
     * @example
     * const elements = [
     *   { id: 'element-1', name: 'Modern Look', thumbnail: '/images/modern.jpg' },
     *   { id: 'element-2', name: 'Retro Feel', thumbnail: '/images/retro.jpg' }
     * ];
     * renderResults(document.getElementById('gallery-container'), 'grid', elements, 'element-1', Date.now());
     */
    static renderResults(containerEl, viewMode, elements, itemRenderer, initialElementID = null, cacheBuster_ = null) {
        const baseClass   = `zipn-${viewMode}`;
        const cacheBuster = cacheBuster_ ? '&cache=' + cacheBuster_ : '';
        containerEl.className = baseClass;

        containerEl.innerHTML = elements.map( element => {
            const extraClass = element.id === initialElementID ? ' initial' : '';
            const thumbnailHTML = itemRenderer.renderThumbnail(element, `zipn-thumb`, cacheBuster);
            return `
                <div class="${baseClass}-card${extraClass}" id="zipn-element-${element.id}" data-id="${element.id}">
                    ${thumbnailHTML}
                    <p class="zipn-text">${element.name}</p>
                </div>`;

        }).join('');
    }


    /**
     * Applies a filter to an array of elements based on text and category criteria.
     * @param {Object[]} allElements - An array of element objects, each containing properties
     *                                such as id, name, description, category, etc.
     * @param {string}   textFilter - A string that filters elements by matching element names
     *                                against search terms.
     * @param {string}     category - The selected category for filtering the elements
     *                                (e.g., "photo", "illustration", etc.). An empty
     *                                string indicates no specific category filter.
     * @returns {Object[]}
     *   Returns an array of element objects that match the given text and category filters.
     */
    static applyFilter(allElements, textFilter, category) {
        const terms = textFilter.toLowerCase().split(' ');
        //const tags  = terms.filter(t => t.startsWith('#'));
        const words = terms.filter(t => !t.startsWith('#'));
        return allElements.filter(element => {
            const matchesCategory = category === "" || element.category === category;
            const matchesWords    = words.every(word => element.lowerName.includes(word));
            const matchesTags     = true; //tags.length === 0 || tags.some(tag => element.tags.includes(tag));
            return matchesCategory && matchesWords && matchesTags;
        });
    }

    //-- EVENTS -----------------------------------------------------------

    /**
     * Called when the dialog is launched or re-opened.
     *
     * Initializes variables and loads elements data from the server.
     * @param {string} initialElementName - The name of the initial element to be selected.
     */
    onOpen(initialElementName) {

        // initialize variables as if the dialog had just been created
        this.isOpen              = true;
        this.initialElementName  = initialElementName;
        this.initialCardID       = null;
        this.resultElements      = [];
        this.resultIndex         = null;
        this.hoveredCardID       = null;
        this.oldSelectionID      = null;
        this.textFilter          = '';
        this.categoryFilter      = "";
        this.isPointerLocked     = false;
        this.searchInputEl.value = '';
        // `this.viewMode` is not set here because it persists between dialog reopenings

        // load style data from server and focus on the initial style
        this.dataProvider.fetchDataArray( (items) =>
        {
            // check each item if it doesn't have the `lowerName` property
            // generate it in base to the `name` property
            for (let item of items) {
                if (!item.lowerName) {
                    item.lowerName = (item.name || '').toLowerCase();
                }
            }

            // process the received data
            this.onReceivedStyles(items);

            // if the initial card is in the list of results,
            // focus on that initial card !
            const initialCardIndex = this.findIndexFromCardID(this.initialCardID);
            if( initialCardIndex >= 0 ) {
                this.resultIndex = initialCardIndex;
                this.updateSelection();
                requestAnimationFrame( () => {
                //requestAnimationFrame( () => {
                    const focusedCardID = this.resultIndex != null ? this.resultElements[this.resultIndex]?.id : null;
                    const focusedCardEl = this.elementFromCardID(focusedCardID);
                    if( focusedCardEl ) { focusedCardEl.scrollIntoView({ block: 'start' }); }
                //});
                });
            }
        });

        this.show();
        this.updateToolbar();
        this.searchInputEl.focus();

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
        this.elementsByID          = styles;
        this.elementIDsByLowerName = Object.fromEntries(styles.map(style => [style.name.toLowerCase(), style.id]));
        this.initialCardID       = this.elementIDsByLowerName[this.initialElementName.toLowerCase()];
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
        this.lockPointer();

        // debounce the search results update
        clearTimeout(this.inputChangeTimer2);
        this.inputChangeTimer2 = setTimeout(() =>
        {
            // always update the search results first so that when user
            // presses enter it will accept the most updated result
            this.updateSearchResults(`>${inputEl.value}`);
            if( isEnterPressed ) {
                this.onElementChosen();
            }
            this.hoveredCardID = null;
            this.updateSelection();

        }, isEnterPressed ? 100 : 300);

    }


    /**
     * Called when a key is pressed in the search input.
     * @param {string} key - The key that was pressed.
     * @return {boolean}
     *   True if the key if handled by the method and should not be processed by the input field.
     */
    onInputKeyDown(key) {
        let resultIndex = this.resultIndex;

        if     ( key === 'Escape' ) { this.close(); }
        else if( key === 'Enter'  ) { this.onInputChange(this.searchInputEl, true); }
        else if( this.resultIndex != null || this.hoveredCardID != null )
        {
            // if the current selection is determined by the mouse pointer,
            // capture that selection!
            if( resultIndex == null ) {
                const hoveredIndex = this.findIndexFromCardID(this.hoveredCardID);
                resultIndex = hoveredIndex >= 0 ? hoveredIndex : 0;
            }

            // cursor key movement
            const oldResultIndex = this.resultIndex;
            if     ( key === 'ArrowUp'    ) { resultIndex-=this.resultColumns; }
            else if( key === 'ArrowDown'  ) { resultIndex+=this.resultColumns; }
            else if( key === 'ArrowLeft'  ) { resultIndex--; }
            else if( key === 'ArrowRight' ) { resultIndex++; }
            if( resultIndex >= this.resultElements.length ) { resultIndex = oldResultIndex; }
            if( resultIndex <  0                        ) { resultIndex = oldResultIndex; }

        }
        // if there is no selection (e.g. just opened the dialog) and user presses down,
        // first search result gets selected
        else if( this.resultIndex == null && key === 'ArrowDown' ) {
            if( this.resultElements ) { resultIndex = 0; }
        }

        // if the selected search result index is modified, update its on-screen representation
        if( this.resultIndex !== resultIndex ) {
            this.resultIndex   = resultIndex;
            this.hoveredCardID = null;
            this.lockPointer();
            this.updateSelection(true);
        }
        return ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(key);
    }


    /**
     * Called when the mouse enters a card.
     * @param {HTMLElement} cardEl - The card element that was entered.
     */
    onCardEnter(cardEl) {
        if( this.isPointerLocked ) { return; }
        // updates the currently pointed card ID and triggers selection updates
        this.hoveredCardID = Number(cardEl.dataset?.id);
        this.updateSelection();
    }

    /**
     * Called when a card is clicked.
     * @param {HTMLElement} cardEl - The card element that was clicked.
     */
    onCardClick(cardEl) {
        // sets the currently pointed card ID and triggers user selection handling
        this.hoveredCardID = Number(cardEl?.dataset?.id);
        this.onElementChosen();
    }

    /**
     * Called when the mouse leaves the card container.
     * This function is responsible for resetting the currently pointed card ID
     * when the user moves the mouse out of the area containing the cards.
     */
    onCardContainerLeave() {
        if( this.isPointerLocked ) { return; }
        this.hoveredCardID = null;
        this.updateSelection();
    }

   /**
    * Called when the user selects an element of the main list.
    * This function calls the `onSelectElement(..)` callback with the selected
    * element's name and closes the dialog.
    */
    onElementChosen() {
        const selectionID = this.getSelectionID();
        const element     = selectionID != null ? this.elementsByID[selectionID] : null;
        if( element ) {
            this.onSelectElement?.(element.name);
            this.close();
        }
    }


    //-- HELPERS ----------------------------------------------------------

    /**
     * Temporarily locks the pointer movement events.
     *
     * This method sets a flag to prevent pointer movement events from being
     * processed, the flag is reset after a short delay (800 milliseconds).
     */
    lockPointer() {
        this.isPointerLocked = true;
        clearTimeout(this.pointerLockedTimer);
        this.pointerLockedTimer = setTimeout(() => { this.isPointerLocked = false; }, 800);
    }

    /**
     * Finds the index of a card in the `resultElements` array based on its ID.
     * @param {number|null} elementID - The ID of the element/card to find.
     * @returns {number} The index of the card with the specified ID, or -1 if not found.
     */
    findIndexFromCardID(elementID) {
        return elementID != null ? this.resultElements.findIndex(card => card.id == elementID) : -1;
    }

    /**
     * Returns the HTML element corresponding to the card with the given ID.
     * @param {number|null} elementID - The ID of the element/card whose corresponding HTML element is sought.
     * @returns {Element|null} The HTML element associated with the given card ID, or null if no match is found.
     */
    elementFromCardID(elementID) {
        return elementID != null ? this.element.querySelector(`#zipn-element-${elementID}`) : null;
    }


    //-- DIALOG COMPONENTS ------------------------------------------------

    /** A spacer element in the toolbar. */
    static get SPACER() { return html("div.zipn-spacer"); }

    /** A divider element (vertical line) in the toolbar. */
    static get DIVIDER() { return html("div.zipn-divider"); }

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
    createToolbar() {
        const categories = this.dataProvider.getCategories() || [];

        // create each category button from the `categories` array
        this.categoryButtons = [];
        for (const [category, displayName, description] of categories) {
            const id       = `zipn-${category}-btn`;
            const command  = `@${category}`;
            const buttonEl = this.createToolButton(id, '', displayName, description, () => { this.updateSearchResults(command); });
            this.categoryButtons.push( [category, buttonEl] );
        }

        // create viewmode buttons
        this.listButtonEl = this.createToolButton("zipn-list-btn", 'pi pi-list' , "", "List View", () => { this.updateSearchResults("$list"); });
        this.gridButtonEl = this.createToolButton("zipn-grid-btn", 'pi pi-image', "", "Grid View", () => { this.updateSearchResults("$grid"); });


        // add search input
        const toolbarElementList = [];
        toolbarElementList.push(
            html("input", { id: "zipn-search-input", type: "search", placeholder: "Search", autocomplete: "off" } )
        );

        // add category buttons
        if( this.categoryButtons.length > 0 ) {
            toolbarElementList.push(_GalleryDialog.DIVIDER);
            for( const [_, buttonEl] of this.categoryButtons ) {
                toolbarElementList.push( buttonEl );
            }
        }

        // add viewmode buttons
        toolbarElementList.push(_GalleryDialog.DIVIDER);
        toolbarElementList.push( this.gridButtonEl );
        toolbarElementList.push( this.listButtonEl );

        return html("div", toolbarElementList );
    }

    /**
     * Updates toolbar buttons state based on current view mode and category filter.
     */
    updateToolbar() {
        // update category buttons
        for( const [category, buttonEl] of this.categoryButtons ) {
            buttonEl.classList.toggle('p-highlight', category == this.categoryFilter );
        }
        // update view mode buttons
        this.listButtonEl.classList.toggle('p-highlight', this.viewMode == "list" );
        this.gridButtonEl.classList.toggle('p-highlight', this.viewMode == "grid" );
    }


}

//#=========================== HELPER FUNCTIONS ============================#

/**
 * Flag to track if the CSS has been loaded to prevent duplicate injections.
 * @type {boolean}
 */
let _cssLoaded = false;

/**
 * Ensures that the gallery dialog stylesheet is loaded and injected into the document.
 */
function ensureCSSLoaded() {
    // check if CSS is already loaded to avoid duplicates
    if( _cssLoaded ) { return; }
    try {
        const cssFullPath = import.meta.resolve('./gallery_dialog.css');

        // create a new <link> element and append it to the head of the document
        const link = document.createElement("link");
        link.rel = "stylesheet";
        link.href = cssFullPath;
        document.head.appendChild(link);

        // mark the CSS as loaded
        _cssLoaded = true;
    }
    catch( error ) { console.error("Failed to load GalleryDialog CSS:", error); }
}

/**
 * Creates and returns a custom dialog HTML element for ComfyUI.
 *
 * This implementation utilizes deprecated ComfyUI functions since there is
 * currently no documented method to create custom dialogs. Some ideas and
 * concepts for this implementation were inspired by ComfYUI-Manager project.
 *
 * @param {string} dialogId  - The ID for the dialog element.
 * @param {string} titleId   - The ID for the title element within the dialog.
 * @param {string} title     - The title to be displayed in the dialog header.
 * @param {string} iconClass - The CSS class for the icon that will appear next to the title.
 * @param {string|HTMLElement[]} content - The content of the dialog, which can be a string or an array of HTML elements.
 * @param {Function} onClose - A callback function to be executed when the dialog is closed.
 * @returns {HTMLElement} 
 *     Returns the main container element for the dialog mask.
 */
function makeCustomDialog(dialogId, titleId, title, iconClass, content, onClose) {

    const dialogOutsideArea = html("div.p-dialog-mask.p-overlay-mask.p-overlay-mask-enter", {
        parent: document.body,
        style : {
            position      : "fixed",
            height        : "100%",
            width         : "100%",
            left          : "0px",
            top           : "0px",
            display       : "flex",
            justifyContent: "center",
            alignItems    : "center",
            pointerEvents : "auto",
            zIndex        : "1000"
        },
        onclick: (e) => {
            // execute `onClose` only if click outside of the dialog
            if( e.target === dialogOutsideArea ) { onClose(); }
        }
    });
    const headerActions = html("div.p-dialog-header-actions");
    //const _closeButton  = html("button.p-button.p-component.p-button-icon-only.p-button-secondary.p-button-rounded.p-button-text.p-dialog-close-button", {
    const _closeButton  = html("button.zipn-close-button", {
        parent    : headerActions,
        type      : "button",
        ariaLabel : "Close",
        onclick   : onClose, //< execute `onClose` when close button is clicked
        innerHTML : '<i class="pi pi-times"></i>'
    });
    const dialogHeader = html("div.p-dialog-header",
    [
        html("div", [
            html("div", { id: "frame-title-container" }, [
                html("h1", [
                    html(iconClass, {
                        style: { "font-size": "1.25rem", "margin-right": ".5rem" }
                    }),
                    html("span", { id: titleId, textContent: title })
                ])
            ])
        ]),
        headerActions
    ]);
    const contentFrame = html("div.p-dialog-content", {}, normalizeDOMnodes(content));
    const dialogFrame  = html("div.p-dialog.p-component.global-dialog", {
            id    : dialogId,
            parent: dialogOutsideArea,
            style : {
                'display'       : 'flex',
                'flex-direction': 'column',
                'pointer-events': 'auto',
                'margin'        : '0px',
            },
            role     : 'dialog',
            ariaModal: 'true',
        },
        [ dialogHeader, contentFrame ]
    );
    const _hiddenAccessible = html("span.p-hidden-accessible.p-hidden-focusable", {
        parent    : dialogFrame,
        tabindex  : "0",
        role      : "presentation",
        ariaHidden: "true",
        "data-p-hidden-accessible": "true",
        "data-p-hidden-focusable" : "true",
        "data-pc-section"         : "firstfocusableelement"
    });
    return dialogOutsideArea;
}


/**
 * Sets up hover and click event listeners for cards within a container.
 *
 * This function is particularly useful for managing gallery images or item
 * listings where interactive behavior such as highlighting on hover and
 * triggering actions on click are required.
 *
 * @param {HTMLElement} containerEl  - The container element where the cards reside.
 * @param {string}      cardSelector - A CSS selector used to identify card elements.
 * @param {(card: HTMLElement) => void} onCardEnter - Callback function when hovering over a card.
 * @param {(card: HTMLElement) => void} onCardLeave - Callback function when moving the mouse away from a card.
 * @param {(card: HTMLElement) => void} onCardClick - Callback function when clicking on a card.
 * @param {(container: HTMLElement) => void} [onContainerLeave] - Optional callback function when moving the mouse away from the container.
 */
function setupCardHoverListeners(containerEl, cardSelector, onCardEnter, onCardLeave, onCardClick, onContainerLeave) {

    const hasPointer = true; //!!window.PointerEvent;
    const events = {
        over : hasPointer ? 'pointerover'  : 'mouseover',
        out  : hasPointer ? 'pointerout'   : 'mouseout',
        leave: hasPointer ? 'pointerleave' : 'mouseleave'
    };

    // ensure parameters are of the correct type
    if (!(containerEl instanceof HTMLElement)) {
        throw new TypeError('`containerEl` must be an instance of `HTMLElement`.');
    }

    containerEl.addEventListener(events.over, (e) => {
        const card = e.target.closest(cardSelector);
        if( card && !card.contains(e.relatedTarget) ) { onCardEnter?.(card, e); }
    });

    containerEl.addEventListener(events.out, (e) => {
        const card = e.target.closest(cardSelector);
        if( card && !card.contains(e.relatedTarget) ) { onCardLeave?.(card, e); }
    });

    containerEl.addEventListener('click', (e) => {
        const card = e.target.closest(cardSelector);
        if( card ) { onCardClick?.(card, e); }
    });

    containerEl.addEventListener(events.leave, (e) => {
        onContainerLeave?.(containerEl, e);
    });
}

/**
 * Normalizes content input into an array of DOM nodes
 * @param {*} content - Content to normalize (string|Node|Node[]|NodeList)
 * @returns {Node[]}
 *    An array of DOM nodes
 */
function normalizeDOMnodes(content) {
    if( !content ) { return []; }

    if (typeof content === 'string') {
        const template = document.createElement('template');
        template.innerHTML = content.trim();
        return Array.from( template.content.childNodes );
    }
    if (content instanceof Node) {
        return [content];
    }
    if (content instanceof NodeList || Array.isArray(content)) {
        return Array.from(content);
    }
    throw new TypeError(`Content must be a string, Node, NodeList or array. Got: ${typeof content}`);
}
