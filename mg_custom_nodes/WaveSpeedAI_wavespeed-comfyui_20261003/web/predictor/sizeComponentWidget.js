/**
 * Size Component Widget - Creates width/height inputs with ratio buttons
 * Ratio buttons are in a separate widget to avoid slot positioning issues
 */

import { updateRequestJson } from './widgets.js';

// Size ratio presets
const SIZE_RATIO_PRESETS = [
    { label: '1:1', width: 1024, height: 1024, icon: '□' },
    { label: '16:9', width: 1024, height: 576, icon: '▭' },
    { label: '9:16', width: 576, height: 1024, icon: '▯' },
    { label: '4:3', width: 1024, height: 768, icon: '▭' },
    { label: '3:4', width: 768, height: 1024, icon: '▯' },
    { label: '3:2', width: 1024, height: 683, icon: '▭' },
    { label: '2:3', width: 683, height: 1024, icon: '▯' }
];

/**
 * Show toast notification near an element
 */
function showSizeToast(message, anchorEl) {
    if (!anchorEl || !anchorEl.ownerDocument) return;
    const doc = anchorEl.ownerDocument;
    const rect = anchorEl.getBoundingClientRect();
    const toast = doc.createElement('div');
    toast.textContent = message;
    toast.style.position = 'fixed';
    toast.style.left = `${Math.max(8, rect.left)}px`;
    toast.style.top = `${Math.max(8, rect.top - 28)}px`;
    toast.style.padding = '4px 8px';
    toast.style.background = '#1f1f1f';
    toast.style.color = '#f1f1f1';
    toast.style.border = '1px solid #b04a4a';
    toast.style.borderRadius = '4px';
    toast.style.fontSize = '11px';
    toast.style.zIndex = '9999';
    toast.style.pointerEvents = 'none';
    doc.body.appendChild(toast);
    
    setTimeout(() => {
        if (toast.parentNode) {
            toast.parentNode.removeChild(toast);
        }
    }, 1800);
}

/**
 * Create ratio buttons widget (separate widget, no input slot)
 */
export function createRatioButtonsWidget(node, param, sharedState) {
    const container = document.createElement('div');
    container.className = 'wavespeed-size-ratios';
    container.style.display = 'flex';
    container.style.flexWrap = 'wrap';
    container.style.gap = '4px';
    container.style.marginBottom = '4px';
    container.style.marginLeft = '12px';  // Indent like array items
    container.style.alignItems = 'center';
    container.style.justifyContent = 'space-between';
    
    // Left side: ratio buttons
    const buttonsContainer = document.createElement('div');
    buttonsContainer.style.display = 'flex';
    buttonsContainer.style.flexWrap = 'wrap';
    buttonsContainer.style.gap = '4px';
    
    sharedState.ratioButtons = [];
    
    SIZE_RATIO_PRESETS.forEach(preset => {
        const btn = document.createElement('button');
        btn.className = 'wavespeed-ratio-btn';
        btn.dataset.ratio = preset.label;
        btn.innerHTML = `<span style="font-size:10px;opacity:0.7;">${preset.icon}</span><span style="font-weight:500;margin-left:2px;">${preset.label}</span>`;
        btn.title = `${preset.width} × ${preset.height}`;
        btn.style.display = 'flex';
        btn.style.alignItems = 'center';
        btn.style.gap = '2px';
        btn.style.padding = '4px 8px';
        btn.style.backgroundColor = '#2a2a2a';
        btn.style.color = '#e0e0e0';
        btn.style.border = '1px solid #444';
        btn.style.borderRadius = '4px';
        btn.style.cursor = 'pointer';
        btn.style.fontSize = '11px';
        btn.style.transition = 'all 0.2s ease';
        
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (btn.disabled) return;
            
            // Update both width and height
            if (sharedState.widthWidget) {
                if (sharedState.widthWidget.setValue) {
                    sharedState.widthWidget.setValue(preset.width);
                } else {
                    sharedState.widthWidget.value = preset.width;
                }
            }
            if (sharedState.heightWidget) {
                if (sharedState.heightWidget.setValue) {
                    sharedState.heightWidget.setValue(preset.height);
                } else {
                    sharedState.heightWidget.value = preset.height;
                }
            }
            
            updateRatioButtons(sharedState);
        });
        
        btn.addEventListener('mousedown', (e) => e.stopPropagation());
        
        sharedState.ratioButtons.push(btn);
        buttonsContainer.appendChild(btn);
    });
    
    container.appendChild(buttonsContainer);
    
    // Right side: range display
    const rangeDisplay = document.createElement('span');
    rangeDisplay.className = 'wavespeed-size-range';
    rangeDisplay.style.color = '#666';
    rangeDisplay.style.fontSize = '10px';
    rangeDisplay.style.whiteSpace = 'nowrap';
    rangeDisplay.style.marginLeft = 'auto';
    rangeDisplay.style.paddingLeft = '8px';
    
    // Get min/max from param (should be passed from parent)
    const minSize = param.min !== undefined ? param.min : 256;
    const maxSize = param.max !== undefined ? param.max : 1536;
    rangeDisplay.textContent = `Range: ${minSize}-${maxSize}`;
    rangeDisplay.title = `Supported size range: ${minSize}px to ${maxSize}px`;
    
    container.appendChild(rangeDisplay);
    
    const widget = node.addDOMWidget(`${param.parentSizeName}_ratios`, 'div', container, { serialize: false });
    
    widget._wavespeed_dynamic = true;
    widget._wavespeed_no_input = true;  // No input slot for ratio buttons
    widget._wavespeed_base = false;
    widget._wavespeed_hidden = false;
    
    widget.computeSize = function() {
        return [node.size[0] - 20, 36];  // Ratio buttons height
    };
    
    sharedState.ratioContainer = container;
    
    return widget;
}

// Update ratio button states
function updateRatioButtons(sharedState) {
    if (!sharedState.ratioButtons) return;
    
    const width = sharedState.widthWidget?.getValue?.() || sharedState.widthWidget?.value;
    const height = sharedState.heightWidget?.getValue?.() || sharedState.heightWidget?.value;
    
    // If either is empty, no ratio is active
    if (width === '' || width === null || width === undefined ||
        height === '' || height === null || height === undefined) {
        sharedState.ratioButtons.forEach(btn => {
            btn.style.backgroundColor = '#2a2a2a';
            btn.style.color = '#e0e0e0';
            btn.style.borderColor = '#444';
        });
        return;
    }
    
    const ratio = width / height;
    
    let matchedRatio = null;
    for (const preset of SIZE_RATIO_PRESETS) {
        const presetRatio = preset.width / preset.height;
        if (Math.abs(ratio - presetRatio) < 0.01) {
            matchedRatio = preset.label;
            break;
        }
    }
    
    sharedState.ratioButtons.forEach(btn => {
        const isActive = btn.dataset.ratio === matchedRatio;
        btn.style.backgroundColor = isActive ? '#4a9eff' : '#2a2a2a';
        btn.style.color = isActive ? 'white' : '#e0e0e0';
        btn.style.borderColor = isActive ? '#4a9eff' : '#444';
    });
}

/**
 * Create size component widget (width or height)
 * Simple input widget like array items
 */
export function createSizeComponentWidget(node, param, sharedState) {
    const isWidth = param.sizeComponent === 'width';
    const parentName = param.parentSizeName;
    
    // Create container for this component (simple row like array items)
    const container = document.createElement('div');
    container.className = 'wavespeed-size-component';
    container.style.display = 'flex';
    container.style.alignItems = 'center';
    container.style.gap = '8px';
    container.style.marginLeft = '12px';  // Indent like array items
    container.style.marginBottom = '2px';
    
    // Label
    const label = document.createElement('label');
    const displayText = param.displayName || (isWidth ? 'Width' : 'Height');
    label.textContent = displayText;  // "Width" or "Height"
    label.className = 'wavespeed-array-item-label';
    label.style.color = '#888';
    label.style.fontSize = '11px';
    label.style.minWidth = '50px';
    label.style.fontFamily = 'monospace';
    
    // Input
    const input = document.createElement('input');
    input.type = 'number';
    input.value = param.default !== undefined && param.default !== null && param.default !== '' ? param.default : '';
    input.min = param.min || 256;
    input.max = param.max || 1536;
    input.step = 8;
    input.placeholder = 'Optional';  // Show it's optional
    input.style.flex = '1';
    input.style.padding = '4px 8px';
    input.style.backgroundColor = '#2a2a2a';
    input.style.color = '#e0e0e0';
    input.style.border = '1px solid #444';
    input.style.borderRadius = '4px';
    input.style.fontSize = '12px';
    
    container.appendChild(label);
    container.appendChild(input);
    
    // Create widget
    const widget = node.addDOMWidget(param.name, 'div', container, { serialize: false });
    
    widget._wavespeed_dynamic = true;
    widget._wavespeed_param = param.name;
    widget._wavespeed_size_component = param.sizeComponent;
    widget._wavespeed_parent_size = param.parentSizeName;
    widget._wavespeed_no_input = false;
    widget._wavespeed_base = false;
    widget._wavespeed_hidden = false;
    
    // Store references
    widget.inputElement = input;
    if (isWidth) {
        sharedState.widthWidget = widget;
        sharedState.widthInput = input;
    } else {
        sharedState.heightWidget = widget;
        sharedState.heightInput = input;
    }
    
    // Define value property
    const descriptor = Object.getOwnPropertyDescriptor(widget, 'value');
    if (!descriptor || descriptor.configurable) {
        if (descriptor) {
            delete widget.value;
        }
        
        Object.defineProperty(widget, 'value', {
            get() {
                const val = input.value;
                return val === '' ? '' : (parseInt(val) || '');
            },
            set(val) {
                input.value = val === '' || val === null || val === undefined ? '' : val;
                updateParentValue();
                updateRatioButtons(sharedState);
            },
            enumerable: true,
            configurable: true
        });
    } else {
        // Fallback: widget already has a non-configurable value property (e.g., from ComfyUI during workflow restore)
        // Use getValue/setValue methods instead - this is expected behavior and works correctly
        widget.getValue = function() {
            const val = input.value;
            return val === '' ? '' : (parseInt(val) || '');
        };
        widget.setValue = function(val) {
            input.value = val === '' || val === null || val === undefined ? '' : val;
            updateParentValue();
            updateRatioButtons(sharedState);
        };
    }
    
    // Update parent value (width*height format)
    function updateParentValue() {
        const widthVal = sharedState.widthWidget?.getValue?.() || sharedState.widthWidget?.value;
        const heightVal = sharedState.heightWidget?.getValue?.() || sharedState.heightWidget?.value;
        
        // If either is empty, store empty values for components
        if (widthVal === '' || widthVal === null || widthVal === undefined) {
            node.wavespeedState.parameterValues[`${parentName}_width`] = '';
        } else {
            node.wavespeedState.parameterValues[`${parentName}_width`] = widthVal;
        }
        
        if (heightVal === '' || heightVal === null || heightVal === undefined) {
            node.wavespeedState.parameterValues[`${parentName}_height`] = '';
        } else {
            node.wavespeedState.parameterValues[`${parentName}_height`] = heightVal;
        }
        
        // Parent value is built in updateRequestJson
        updateRequestJson(node);
    }
    
    // Input event handlers
    input.addEventListener('input', () => {
        // Just update values on input, don't validate yet
        updateParentValue();
        updateRatioButtons(sharedState);
    });
    
    // Validate and format on blur (when user leaves the input)
    input.addEventListener('blur', () => {
        // Allow empty value
        if (input.value === '') {
            return;
        }
        
        let val = parseInt(input.value);
        
        // If invalid number, clear it
        if (isNaN(val)) {
            input.value = '';
            updateParentValue();
            updateRatioButtons(sharedState);
            return;
        }
        
        const minSize = param.min || 256;
        const maxSize = param.max || 1536;
        const originalVal = val;
        
        // Clamp to min/max
        val = Math.max(minSize, Math.min(maxSize, val));
        
        // Show toast if value was out of range
        if (originalVal < minSize || originalVal > maxSize) {
            const componentName = isWidth ? 'Width' : 'Height';
            showSizeToast(`${componentName} must be between ${minSize} and ${maxSize}`, input);
        }
        
        // Align to multiples of 8
        val = Math.round(val / 8) * 8;
        input.value = val;
        
        updateParentValue();
        updateRatioButtons(sharedState);
    });
    
    input.addEventListener('click', (e) => e.stopPropagation());
    input.addEventListener('mousedown', (e) => e.stopPropagation());
    
    // Compute size - simple row like array items
    widget.computeSize = function() {
        return [node.size[0] - 20, 30];  // Same height as array items
    };
    
    // Initialize value - CRITICAL: Don't read from parameterValues for size components
    // Size components should always use param.default on creation to avoid stale values
    // The parameterValues will be updated by updateParentValue() after both width and height are created
    const currentValue = widget.getValue ? widget.getValue() : widget.value;
    node.wavespeedState.parameterValues[param.name] = currentValue;
    
    // Update connection state handler
    widget.updateConnectionState = function() {
        const widthInput = node.inputs?.find(inp => inp.name === `${parentName}_width`);
        const heightInput = node.inputs?.find(inp => inp.name === `${parentName}_height`);
        
        const widthConnected = widthInput && widthInput.link != null;
        const heightConnected = heightInput && heightInput.link != null;
        const anyConnected = widthConnected || heightConnected;
        
        // Disable ratio buttons if any input is connected
        if (sharedState.ratioButtons) {
            sharedState.ratioButtons.forEach(btn => {
                btn.disabled = anyConnected;
                btn.style.opacity = anyConnected ? '0.5' : '1';
                btn.style.cursor = anyConnected ? 'not-allowed' : 'pointer';
            });
        }
        
        // Disable the input if this specific component is connected
        if (isWidth && widthConnected) {
            input.disabled = true;
            input.style.opacity = '0.5';
            input.style.cursor = 'not-allowed';
        } else if (!isWidth && heightConnected) {
            input.disabled = true;
            input.style.opacity = '0.5';
            input.style.cursor = 'not-allowed';
        } else {
            input.disabled = false;
            input.style.opacity = '1';
            input.style.cursor = 'text';
        }
    };
    
    return widget;
}
