// ==========================================================================
//  Illustrious UI Enhanced - Canvas-Based Controls
// ==========================================================================
//
// Adapts sophisticated UI patterns from ResolutionMaster for Illustrious nodes
// Version: 1.0.0
//
// ==========================================================================

import { app } from "/scripts/app.js";

// Runtime feature flag to enable/disable this module entirely
try {
    const v = localStorage.getItem('illustrious_enhanced_ui');
    if (!v || v.toLowerCase() !== 'on') {
        console.log('[Illustrious Enhanced] Disabled by flag');
        // Don’t register anything if the feature is off
        // Early return from module body by throwing a no-op error catched by dynamic import
        // Note: top-level return not allowed; rely on guard conditions below
    }
} catch {}
import { globalLayerManager } from "./ui_components/layer_manager.js";

// --- LOGGER UTILITY ---
class IllustriousLogger {
    constructor(moduleName) {
        this.module = moduleName;
        this.level = localStorage.getItem('illustrious_log_level') || 'NONE';
    }
    
    debug(...args) {
        if (this.level === 'DEBUG') console.log(`[${this.module}]`, ...args);
    }
    
    info(...args) {
        if (['DEBUG', 'INFO'].includes(this.level)) console.info(`[${this.module}]`, ...args);
    }
    
    error(...args) {
        if (this.level !== 'NONE') console.error(`[${this.module}]`, ...args);
    }
}

// --- TOOLTIP SYSTEM ---
class TooltipManager {
    constructor() {
        this.tooltips = {};
        this.currentTooltip = null;
        this.tooltipTimer = null;
        this.delay = 400; // Faster than ResolutionMaster's 500ms
        this.mousePos = { x: 0, y: 0 };
    }
    
    register(elementId, text) {
        this.tooltips[elementId] = text;
    }
    
    showTooltip(ctx, elementId) {
        const text = this.tooltips[elementId];
        if (!text) return;
        
        // Calculate dimensions
        ctx.font = "12px Arial";
        const lines = this.wrapText(ctx, text, 250);
        const lineHeight = 16;
        const padding = 8;
        const width = Math.max(...lines.map(l => ctx.measureText(l).width)) + padding * 2;
        const height = lines.length * lineHeight + padding * 2;
        
        // Position near mouse
        let x = this.mousePos.x + 15;
        let y = this.mousePos.y - height - 10;
        
        // Draw shadow
        ctx.fillStyle = "rgba(0, 0, 0, 0.3)";
        ctx.beginPath();
        ctx.roundRect(x + 2, y + 2, width, height, 6);
        ctx.fill();
        
        // Draw background with gradient
        const grad = ctx.createLinearGradient(x, y, x, y + height);
        grad.addColorStop(0, "rgba(45, 45, 45, 0.95)");
        grad.addColorStop(1, "rgba(35, 35, 35, 0.95)");
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.roundRect(x, y, width, height, 6);
        ctx.fill();
        
        // Draw border
        ctx.strokeStyle = "rgba(200, 200, 200, 0.3)";
        ctx.lineWidth = 1;
        ctx.stroke();
        
        // Draw text
        ctx.fillStyle = "#ffffff";
        ctx.font = "12px Arial";
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        lines.forEach((line, i) => {
            ctx.fillText(line, x + padding, y + padding + i * lineHeight);
        });
    }
    
    wrapText(ctx, text, maxWidth) {
        const words = text.split(' ');
        const lines = [];
        let currentLine = '';
        
        for (const word of words) {
            const testLine = currentLine + (currentLine ? ' ' : '') + word;
            const metrics = ctx.measureText(testLine);
            
            if (metrics.width > maxWidth && currentLine) {
                lines.push(currentLine);
                currentLine = word;
            } else {
                currentLine = testLine;
            }
        }
        if (currentLine) lines.push(currentLine);
        return lines;
    }
    
    handleHover(elementId, mouseX, mouseY) {
        this.mousePos = { x: mouseX, y: mouseY };
        
        if (this.tooltipTimer) {
            clearTimeout(this.tooltipTimer);
        }
        
        if (elementId && this.tooltips[elementId]) {
            this.tooltipTimer = setTimeout(() => {
                this.currentTooltip = elementId;
                app.graph.setDirtyCanvas(true);
            }, this.delay);
        } else {
            this.currentTooltip = null;
        }
    }
}

// --- CUSTOM INPUT DIALOG ---
class CustomInputDialog {
    static show(title, currentValue, callback, options = {}) {
        const overlay = document.createElement('div');
        overlay.style.cssText = `
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.6); z-index: 9999;
            animation: fadeIn 0.2s ease-out;
        `;
        
        const dialog = document.createElement('div');
        dialog.style.cssText = `
            position: fixed; left: 50%; top: 50%; transform: translate(-50%, -50%);
            background: linear-gradient(135deg, #2a2a2a 0%, #1e1e1e 100%);
            border: 2px solid #555; border-radius: 12px; padding: 24px;
            box-shadow: 0 12px 48px rgba(0,0,0,0.9); z-index: 10000;
            min-width: 320px; animation: slideIn 0.3s ease-out;
        `;
        
        dialog.innerHTML = `
            <style>
                @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
                @keyframes slideIn { 
                    from { transform: translate(-50%, -60%); opacity: 0; }
                    to { transform: translate(-50%, -50%); opacity: 1; }
                }
                .illustrious-dialog-input {
                    width: 100%; padding: 10px; margin: 10px 0;
                    border: 2px solid #555; border-radius: 6px;
                    background: #333; color: #fff; font-size: 14px;
                    transition: border-color 0.2s;
                }
                .illustrious-dialog-input:focus {
                    border-color: #5af; outline: none;
                    box-shadow: 0 0 0 3px rgba(85, 170, 255, 0.2);
                }
                .illustrious-dialog-btn {
                    padding: 10px 20px; margin: 5px; border-radius: 6px;
                    font-size: 13px; cursor: pointer; transition: all 0.2s;
                    border: none; font-weight: 500;
                }
                .illustrious-dialog-btn-primary {
                    background: linear-gradient(135deg, #5af, #4090ff);
                    color: white;
                }
                .illustrious-dialog-btn-primary:hover {
                    transform: translateY(-1px);
                    box-shadow: 0 4px 12px rgba(85, 170, 255, 0.4);
                }
                .illustrious-dialog-btn-cancel {
                    background: #444; color: #ccc;
                }
                .illustrious-dialog-btn-cancel:hover {
                    background: #555;
                }
            </style>
            <div style="color: #fff; font-size: 18px; font-weight: bold; margin-bottom: 16px;">
                ${title}
            </div>
            <input type="${options.type || 'text'}" 
                   class="illustrious-dialog-input" 
                   value="${currentValue}"
                   placeholder="${options.placeholder || ''}"
                   ${options.pattern ? `pattern="${options.pattern}"` : ''}>
            ${options.hint ? `<div style="color: #999; font-size: 11px; margin-bottom: 10px;">${options.hint}</div>` : ''}
            <div style="text-align: right; margin-top: 16px;">
                <button class="illustrious-dialog-btn illustrious-dialog-btn-cancel">Cancel</button>
                <button class="illustrious-dialog-btn illustrious-dialog-btn-primary">Apply</button>
            </div>
        `;
        
        document.body.appendChild(overlay);
        document.body.appendChild(dialog);
        
        const input = dialog.querySelector('input');
        const applyBtn = dialog.querySelector('.illustrious-dialog-btn-primary');
        const cancelBtn = dialog.querySelector('.illustrious-dialog-btn-cancel');
        
        input.focus();
        input.select();
        
        const close = () => {
            dialog.style.animation = 'slideIn 0.2s ease-out reverse';
            overlay.style.animation = 'fadeIn 0.2s ease-out reverse';
            setTimeout(() => {
                document.body.removeChild(dialog);
                document.body.removeChild(overlay);
            }, 200);
        };
        
        const apply = () => {
            if (callback(input.value)) close();
        };
        
        applyBtn.onclick = apply;
        cancelBtn.onclick = close;
        overlay.onclick = close;
        dialog.onclick = (e) => e.stopPropagation();
        
        input.onkeydown = (e) => {
            if (e.key === 'Enter') apply();
            else if (e.key === 'Escape') close();
        };
    }
}

// --- CANVAS CONTROL RENDERER ---
class CanvasControlRenderer {
    constructor(node) {
        this.node = node;
        this.hoverElement = null;
        this.controls = {};
        this.logger = new IllustriousLogger('CanvasRenderer');
    }
    
    drawGradientButton(ctx, x, y, w, h, text, hover = false, disabled = false) {
        const grad = ctx.createLinearGradient(x, y, x, y + h);
        if (disabled) {
            grad.addColorStop(0, "#4a4a4a");
            grad.addColorStop(1, "#404040");
        } else if (hover) {
            grad.addColorStop(0, "#6a8aff");
            grad.addColorStop(1, "#5070ee");
        } else {
            grad.addColorStop(0, "#5a7adf");
            grad.addColorStop(1, "#4060cf");
        }
        
        ctx.fillStyle = grad;
        ctx.strokeStyle = disabled ? "#333" : hover ? "#88aaff" : "#4060cf";
        ctx.lineWidth = 1;
        
        ctx.beginPath();
        ctx.roundRect(x, y, w, h, 6);
        ctx.fill();
        ctx.stroke();
        
        // Add subtle inner shadow
        if (!disabled) {
            ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
            ctx.beginPath();
            ctx.roundRect(x + 1, y + 1, w - 2, h - 2, 5);
            ctx.stroke();
        }
        
        // Draw text with shadow
        ctx.fillStyle = disabled ? "#888" : "#fff";
        ctx.font = "13px Arial";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        
        if (!disabled) {
            ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
            ctx.fillText(text, x + w/2 + 1, y + h/2 + 2);
            ctx.fillStyle = "#fff";
        }
        ctx.fillText(text, x + w/2, y + h/2 + 1);
    }
    
    drawSection(ctx, title, x, y, w, h) {
        // Background
        ctx.fillStyle = "rgba(0, 0, 0, 0.3)";
        ctx.strokeStyle = "rgba(100, 150, 255, 0.2)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(x, y, w, h, 8);
        ctx.fill();
        ctx.stroke();
        
        // Title background
        const titleHeight = 24;
        const titleGrad = ctx.createLinearGradient(x, y, x, y + titleHeight);
        titleGrad.addColorStop(0, "rgba(85, 170, 255, 0.15)");
        titleGrad.addColorStop(1, "rgba(85, 170, 255, 0.05)");
        ctx.fillStyle = titleGrad;
        ctx.beginPath();
        ctx.roundRect(x, y, w, titleHeight, 8);
        ctx.fill();
        
        // Title text
        ctx.fillStyle = "#5af";
        ctx.font = "bold 12px Arial";
        ctx.textAlign = "left";
        ctx.fillText(title, x + 10, y + 15);
    }
    
    drawCustomSlider(ctx, x, y, w, h, value, min, max, label = "") {
        // Track
        const trackY = y + h/2;
        const trackHeight = 6;
        ctx.fillStyle = "#222";
        ctx.beginPath();
        ctx.roundRect(x, trackY - trackHeight/2, w, trackHeight, trackHeight/2);
        ctx.fill();
        
        // Fill
        const fillWidth = w * ((value - min) / (max - min));
        const fillGrad = ctx.createLinearGradient(x, trackY, x + fillWidth, trackY);
        fillGrad.addColorStop(0, "#4090ff");
        fillGrad.addColorStop(1, "#5af");
        ctx.fillStyle = fillGrad;
        ctx.beginPath();
        ctx.roundRect(x, trackY - trackHeight/2, fillWidth, trackHeight, trackHeight/2);
        ctx.fill();
        
        // Knob
        const knobX = x + fillWidth;
        const knobRadius = 10;
        
        // Knob shadow
        ctx.fillStyle = "rgba(0, 0, 0, 0.3)";
        ctx.beginPath();
        ctx.arc(knobX, trackY + 2, knobRadius, 0, Math.PI * 2);
        ctx.fill();
        
        // Knob gradient
        const knobGrad = ctx.createRadialGradient(knobX, trackY - 3, 0, knobX, trackY, knobRadius);
        knobGrad.addColorStop(0, "#fff");
        knobGrad.addColorStop(0.7, "#e0e0e0");
        knobGrad.addColorStop(1, "#c0c0c0");
        ctx.fillStyle = knobGrad;
        ctx.strokeStyle = "#999";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(knobX, trackY, knobRadius, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        
        // Label
        if (label) {
            ctx.fillStyle = "#ccc";
            ctx.font = "11px Arial";
            ctx.textAlign = "left";
            ctx.fillText(label, x, y - 2);
        }
        
        // Value
        ctx.fillStyle = "#5af";
        ctx.font = "bold 11px Arial";
        ctx.textAlign = "right";
        ctx.fillText(value.toFixed(1), x + w, y - 2);
    }
    
    drawToggle(ctx, x, y, w, h, isOn, text) {
        const radius = h / 2;
        
        // Track
        ctx.beginPath();
        ctx.roundRect(x, y, w, h, radius);
        
        const trackGrad = ctx.createLinearGradient(x, y, x + w, y);
        if (isOn) {
            trackGrad.addColorStop(0, "#3a76d6");
            trackGrad.addColorStop(1, "#5a96f6");
        } else {
            trackGrad.addColorStop(0, "#555");
            trackGrad.addColorStop(1, "#666");
        }
        ctx.fillStyle = trackGrad;
        ctx.fill();
        
        ctx.strokeStyle = "#222";
        ctx.lineWidth = 1;
        ctx.stroke();
        
        // Knob
        const knobX = isOn ? x + w - radius : x + radius;
        const knobGrad = ctx.createRadialGradient(knobX, y + radius, 0, knobX, y + radius, radius - 3);
        knobGrad.addColorStop(0, "#fff");
        knobGrad.addColorStop(1, "#d0d0d0");
        ctx.fillStyle = knobGrad;
        ctx.beginPath();
        ctx.arc(knobX, y + radius, radius - 3, 0, Math.PI * 2);
        ctx.fill();
        
        // Text
        if (text) {
            ctx.fillStyle = isOn ? "#fff" : "#aaa";
            ctx.font = "bold 11px Arial";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(text, x + w/2, y + h/2);
        }
    }
}

// --- ENHANCED NODE CLASS ---
class IllustriousEnhancedNode {
    constructor(node) {
        this.node = node;
        this.renderer = new CanvasControlRenderer(node);
        this.tooltips = new TooltipManager();
        this.overlayControl = {
            draw: (ctx) => {
                // Draw custom controls and tooltip within overlay so they render on top
                this.drawCustomControls(ctx);
                if (this.tooltips.currentTooltip) {
                    this.tooltips.showTooltip(ctx, this.tooltips.currentTooltip);
                }
            }
        };
        this.setupEnhancements();
    }
    
    setupEnhancements() {
        const node = this.node;
        const self = this;
        
        // Register tooltips for common parameters
        this.tooltips.register('weight_slider', 'Controls the influence strength (0.0-2.0)');
        this.tooltips.register('toggle_advanced', 'Show/hide advanced options');
        this.tooltips.register('quality_boost', 'Apply automatic quality enhancements');
        
        // Register with global overlay so these controls render above all nodes
        try { globalLayerManager.addEnhancedControl(node, this.overlayControl); } catch {}

        // Override draw method to add custom controls (fallback only)
        const originalDraw = node.onDrawForeground;
        node.onDrawForeground = function(ctx) {
            if (originalDraw) originalDraw.apply(this, arguments);
            if (this.flags.collapsed) return;
            // If the overlay patch is not active, draw here as a fallback
            const overlayPatched = !!(window.LGraphCanvas && LGraphCanvas.prototype.__illustriousLayerPatched);
            if (!overlayPatched) {
                self.drawCustomControls(ctx);
                if (self.tooltips.currentTooltip) {
                    self.tooltips.showTooltip(ctx, self.tooltips.currentTooltip);
                }
            }
        };
        
        // Handle mouse events
        const originalMouseMove = node.onMouseMove;
        node.onMouseMove = function(e, pos) {
            if (originalMouseMove) originalMouseMove.apply(this, arguments);
            
            const relX = e.canvasX - node.pos[0];
            const relY = e.canvasY - node.pos[1];
            
            // Check hover over controls
            let hoveredElement = null;
            for (const [id, bounds] of Object.entries(self.renderer.controls)) {
                if (relX >= bounds.x && relX <= bounds.x + bounds.w &&
                    relY >= bounds.y && relY <= bounds.y + bounds.h) {
                    hoveredElement = id;
                    break;
                }
            }
            
            self.tooltips.handleHover(hoveredElement, e.canvasX, e.canvasY);
            
            if (hoveredElement !== self.renderer.hoverElement) {
                self.renderer.hoverElement = hoveredElement;
                app.graph.setDirtyCanvas(true);
            }
        };
        
        // Handle clicks
        const originalMouseDown = node.onMouseDown;
        node.onMouseDown = function(e, pos) {
            if (originalMouseDown) originalMouseDown.apply(this, arguments);
            
            const relX = e.canvasX - node.pos[0];
            const relY = e.canvasY - node.pos[1];
            
            // Check clicks on controls
            for (const [id, bounds] of Object.entries(self.renderer.controls)) {
                if (relX >= bounds.x && relX <= bounds.x + bounds.w &&
                    relY >= bounds.y && relY <= bounds.y + bounds.h) {
                    self.handleControlClick(id);
                    return true;
                }
            }
        };

        // Clean up overlay control when node is removed
        const originalOnRemoved = node.onRemoved;
        node.onRemoved = function() {
            try { globalLayerManager.removeEnhancedControl(node, self.overlayControl); } catch {}
            if (originalOnRemoved) return originalOnRemoved.apply(this, arguments);
        };
    }
    
    drawCustomControls(ctx) {
        const node = this.node;
        const x = 10;
        let y = 30;
        const width = node.size[0] - 20;
        
        // Example: Draw a custom section with controls
        this.renderer.drawSection(ctx, "✨ Enhanced Controls", x, y, width, 120);
        y += 30;
        
        // Weight slider
        this.renderer.controls.weight_slider = { x, y, w: width - 20, h: 30 };
        this.renderer.drawCustomSlider(ctx, x + 10, y, width - 20, 30, 1.0, 0.0, 2.0, "Weight");
        y += 40;
        
        // Toggle
        this.renderer.controls.toggle_advanced = { x: x + 10, y, w: 100, h: 24 };
        this.renderer.drawToggle(ctx, x + 10, y, 100, 24, false, "Advanced");
        
        // Button
        this.renderer.controls.quality_boost = { x: x + 120, y, w: 80, h: 24 };
        this.renderer.drawGradientButton(ctx, x + 120, y, 80, 24, "✨ Boost", 
                                        this.renderer.hoverElement === 'quality_boost');
    }
    
    handleControlClick(controlId) {
        if (controlId === 'quality_boost') {
            CustomInputDialog.show(
                "Quality Boost Settings",
                "high",
                (value) => {
                    console.log("Quality set to:", value);
                    return true;
                },
                {
                    placeholder: "Enter quality level",
                    hint: "Options: low, medium, high, ultra"
                }
            );
        } else if (controlId === 'toggle_advanced') {
            console.log("Toggled advanced options");
            app.graph.setDirtyCanvas(true);
        }
    }
}

// --- REGISTRATION ---
// Only register if the flag is ON
if (localStorage.getItem('illustrious_enhanced_ui')?.toLowerCase() === 'on') app.registerExtension({
    name: "itsjustregi.IllustriousEnhanced",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Apply to specific Illustrious nodes
        const enhancedNodes = [
            "IllustriousPrompt",
            "IllustriousCharacters",
            "IllustriousArtists"
        ];
        
        if (enhancedNodes.includes(nodeData.name)) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                
                // Attach enhanced UI
                this.enhancedUI = new IllustriousEnhancedNode(this);
                
                return result;
            };
        }
    }
});

if (localStorage.getItem('illustrious_enhanced_ui')?.toLowerCase() === 'on') {
    console.log("[Illustrious Enhanced] Canvas-based UI loaded");
}