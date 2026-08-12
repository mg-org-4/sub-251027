// ==========================================================================
//  Weight Control Component for Illustrious Nodes
// ==========================================================================
//
// Visual weight editor with sliders, presets, and validation
// Replaces text inputs with intuitive visual controls
//
// ==========================================================================

import { CanvasRenderer } from './canvas_renderer.js';
import { TooltipManager, TOOLTIP_PRESETS } from './tooltip_manager.js';
import { globalLayerManager } from './layer_manager.js';

export class WeightControl {
    constructor(node, widget) {
        this.node = node;
        this.widget = widget;
        this.renderer = new CanvasRenderer(node);
        this.tooltips = new TooltipManager();
        
        this.value = parseFloat(widget.value) || 1.0;
        this.min = 0.0;
        this.max = 2.0;
        this.step = 0.1;
        
        this.isEnabled = true;
        this.isDragging = false;
        this.customHeight = 60; // Height for the weight control area
        
        this.setupTooltips();
        this.attachToWidget();
        
        // Register with layer manager for proper z-depth
        globalLayerManager.addEnhancedControl(node, this);
    }
    
    setupTooltips() {
        this.tooltips.register('weight_slider', TOOLTIP_PRESETS.weight);
        this.tooltips.register('weight_presets', {
            title: "Weight Presets",
            text: "Quick weight values:\n• 0.5: Subtle influence\n• 1.0: Normal strength\n• 1.5: Strong influence\n• 2.0: Maximum strength"
        });
        this.tooltips.register('weight_input', {
            title: "Custom Weight",
            text: "Click to enter a precise weight value.\nSupports decimal values and mathematical expressions."
        });
    }
    
    attachToWidget() {
        const widget = this.widget;
        const node = this.node;
        const self = this;
        
        // ResolutionMaster approach: COMPLETELY override node drawing
        // Don't hide widgets - just override the entire node interface
        
        // Store original callback to maintain data flow
        this.originalCallback = widget.callback;
        
        // Override the ENTIRE node's onDrawForeground like ResolutionMaster does
        const originalDrawForeground = node.onDrawForeground;
        node.onDrawForeground = function(ctx) {
            // Mark that this node handles its own enhanced foreground to avoid double drawing
            this.__illustriousHandlesForeground = true;
            if (this.flags.collapsed) {
                // If collapsed, use original drawing
                if (originalDrawForeground) originalDrawForeground.call(this, ctx);
                return;
            }
            
            // COMPLETELY REPLACE the node's interface - ResolutionMaster style
            self.drawCompleteInterface(ctx);
        };
        
        // Override mouse handlers completely like ResolutionMaster
        const originalMouseDown = node.onMouseDown;
        node.onMouseDown = function(e, pos, canvas) {
            if (e.canvasY - this.pos[1] < 0) return false; // Above title bar
            if (this.flags.collapsed) {
                return originalMouseDown ? originalMouseDown.call(this, e, pos, canvas) : false;
            }
            return self.handleMouseDown(e, pos, canvas);
        };
        
        const originalMouseMove = node.onMouseMove;  
        node.onMouseMove = function(e, pos, canvas) {
            if (this.flags.collapsed) {
                return originalMouseMove ? originalMouseMove.call(this, e, pos, canvas) : false;
            }
            self.handleMouseMove(e, pos, canvas);
            return false;
        };
        
        const originalMouseUp = node.onMouseUp;
        node.onMouseUp = function(e) {
            if (this.flags.collapsed) {
                return originalMouseUp ? originalMouseUp.call(this, e) : false;
            }
            return self.handleMouseUp(e);
        };
        
        // Ensure minimum node size like ResolutionMaster
        this.ensureMinimumSize();
    }
    
    ensureMinimumSize() {
        const minWidth = 280;
        const minHeight = 120;
        
        if (this.node.size[0] < minWidth) this.node.size[0] = minWidth;
        if (this.node.size[1] < minHeight) this.node.size[1] = minHeight;
    }
    
    drawCompleteInterface(ctx) {
        // Draw the ENTIRE interface like ResolutionMaster - no widgets shown
        const node = this.node;
        const margin = 10;
        
        // Ensure minimum size
        this.ensureMinimumSize();
        
        // Start after title
        let currentY = LiteGraph.NODE_TITLE_HEIGHT + 10;
        
        // Draw the weight control section
        const sectionHeight = this.renderer.drawSection(ctx, 
            `💎 ${this.widget.name}`, margin, currentY, 
            node.size[0] - margin * 2, this.customHeight, {
                highlight: this.renderer.hoverElement?.startsWith('weight_')
            });
        
        // Content area
        const contentY = currentY + sectionHeight + 8;
        const contentHeight = this.customHeight - sectionHeight - 8;
        const width = node.size[0] - margin * 2;
        
        // Register control areas
        this.renderer.registerControl('weight_slider', {
            x: margin + 10, y: contentY, w: width - 120, h: 30
        });
        
        this.renderer.registerControl('weight_presets', {
            x: margin + width - 100, y: contentY, w: 90, h: 15
        });
        
        this.renderer.registerControl('weight_input', {
            x: margin + width - 100, y: contentY + 20, w: 90, h: 15
        });
        
        // Draw main slider
        this.renderer.drawSlider(ctx, 
            margin + 10, contentY, width - 120, 30,
            this.value, this.min, this.max, {
                label: "",
                showValue: false,
                color: this.getWeightColor(),
                hover: this.renderer.hoverElement === 'weight_slider',
                disabled: !this.isEnabled
            });
        
        // Draw preset buttons
        this.drawPresetButtons(ctx, margin + width - 100, contentY);
        
        // Draw value display
        this.drawValueDisplay(ctx, margin + width - 100, contentY + 20);
        
        // Draw tooltips
        this.tooltips.draw(ctx, node.pos);
        
        // Ensure node is tall enough
        const requiredHeight = currentY + this.customHeight + 20;
        if (node.size[1] < requiredHeight) {
            node.size[1] = requiredHeight;
        }
    }
    
    draw(ctx) {
        // This method is now unused - drawCompleteInterface handles everything
        // Keeping for compatibility
        this.drawCompleteInterface(ctx);
    }
    
    drawOriginal_draw(ctx) {
        const node = this.node;
        
        // FORCE top-level drawing with highest z-index equivalent
        ctx.save();
        
        // Clear any existing composite operations and use source-over to ensure we draw on top
        ctx.globalCompositeOperation = 'source-over';
        ctx.globalAlpha = 1.0;
        
        // Find widget position in node
        let y = 30; // Start after title
        const widgets = node.widgets || [];
        const widgetIndex = widgets.indexOf(this.widget);
        
        // Calculate position based on widget order - but draw over the original widget area
        for (let i = 0; i < widgetIndex; i++) {
            const w = widgets[i];
            if (!w.computeSize || w === this.widget) continue;
            const size = w.computeSize ? w.computeSize(node.size[0]) : [200, 20];
            y += size[1] + 4;
        }
        
        const x = 10;
        const width = node.size[0] - 20;
        const height = this.customHeight;
        
        // Draw section background
        const sectionHeight = this.renderer.drawSection(ctx, 
            `💎 ${this.widget.name}`, x, y, width, height, {
                highlight: this.renderer.hoverElement?.startsWith('weight_')
            });
        
        // Content area
        const contentY = y + sectionHeight + 8;
        const contentHeight = height - sectionHeight - 8;
        
        // Register control areas
        this.renderer.registerControl('weight_slider', {
            x: x + 10, y: contentY, w: width - 120, h: 30
        });
        
        this.renderer.registerControl('weight_presets', {
            x: x + width - 100, y: contentY, w: 90, h: 15
        });
        
        this.renderer.registerControl('weight_input', {
            x: x + width - 100, y: contentY + 20, w: 90, h: 15
        });
        
        // Draw main slider
        this.renderer.drawSlider(ctx, 
            x + 10, contentY, width - 120, 30,
            this.value, this.min, this.max, {
                label: "",
                showValue: false,
                color: this.getWeightColor(),
                hover: this.renderer.hoverElement === 'weight_slider',
                disabled: !this.isEnabled
            });
        
        // Draw preset buttons
        this.drawPresetButtons(ctx, x + width - 100, contentY);
        
        // Draw value display
        this.drawValueDisplay(ctx, x + width - 100, contentY + 20);
        
        // Draw tooltips
        this.tooltips.draw(ctx, node.pos);
        
        // Ensure node is tall enough
        const requiredHeight = y + height + 10;
        if (node.size[1] < requiredHeight) {
            node.size[1] = requiredHeight;
        }
        
        // IMPORTANT: Restore canvas context
        ctx.restore();
    }
    
    drawPresetButtons(ctx, x, y) {
        const presets = [
            { value: 0.5, label: "0.5" },
            { value: 1.0, label: "1.0" },
            { value: 1.5, label: "1.5" },
            { value: 2.0, label: "2.0" }
        ];
        
        const buttonWidth = 20;
        const gap = 2;
        
        presets.forEach((preset, i) => {
            const buttonX = x + i * (buttonWidth + gap);
            const isActive = Math.abs(this.value - preset.value) < 0.05;
            const isHovered = this.renderer.hoverElement === `preset_${preset.value}`;
            
            this.renderer.registerControl(`preset_${preset.value}`, {
                x: buttonX, y, w: buttonWidth, h: 15
            });
            
            this.renderer.drawButton(ctx, buttonX, y, buttonWidth, 15, preset.label, {
                hover: isHovered,
                active: isActive,
                style: isActive ? 'primary' : 'secondary'
            });
        });
    }
    
    drawValueDisplay(ctx, x, y) {
        const displayValue = this.value.toFixed(2);
        
        // Background
        ctx.fillStyle = this.renderer.hoverElement === 'weight_input' ? 
            "rgba(85, 170, 255, 0.2)" : "rgba(0, 0, 0, 0.5)";
        ctx.strokeStyle = this.renderer.hoverElement === 'weight_input' ? 
            "rgba(85, 170, 255, 0.5)" : "rgba(100, 100, 100, 0.3)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(x, y, 90, 15, 3);
        ctx.fill();
        ctx.stroke();
        
        // Value text
        ctx.fillStyle = this.getWeightColor();
        ctx.font = "bold 11px Arial";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(displayValue, x + 45, y + 7.5);
        
        // Unit label
        ctx.fillStyle = "#888";
        ctx.font = "9px Arial";
        ctx.fillText("weight", x + 70, y + 7.5);
    }
    
    getWeightColor() {
        if (this.value < 0.5) return "#ff9040"; // Orange for low
        if (this.value < 1.0) return "#40ff90"; // Green for normal-low
        if (this.value < 1.5) return "#4090ff"; // Blue for normal-high
        return "#ff4040"; // Red for high
    }
    
    handleMouseDown(e, pos, canvas) {
        const relX = e.canvasX - this.node.pos[0];
        const relY = e.canvasY - this.node.pos[1];
        
        const hitControl = this.renderer.hitTest(relX, relY);
        if (!hitControl) return false;
        
        if (hitControl === 'weight_slider') {
            this.isDragging = true;
            this.node.capture = true; // ResolutionMaster style capture
            this.updateFromSlider(relX, relY);
            return true;
        }
        
        if (hitControl.startsWith('preset_')) {
            const presetValue = parseFloat(hitControl.replace('preset_', ''));
            this.setValue(presetValue);
            return true;
        }
        
        if (hitControl === 'weight_input') {
            this.showValueDialog();
            return true;
        }
        
        return false;
    }
    
    handleMouseMove(e, pos, canvas) {
        const relX = e.canvasX - this.node.pos[0];
        const relY = e.canvasY - this.node.pos[1];
        
        if (this.isDragging && this.node.capture) {
            this.updateFromSlider(relX, relY);
        } else {
            const hitControl = this.renderer.hitTest(relX, relY);
            const needsRedraw = this.renderer.setHover(hitControl);
            
            this.tooltips.handleHover(hitControl, e.canvasX, e.canvasY);
            
            if (needsRedraw && window.app?.graph) {
                window.app.graph.setDirtyCanvas(true);
            }
        }
    }
    
    handleMouseUp(e) {
        if (this.isDragging) {
            this.isDragging = false;
            this.node.capture = false; // ResolutionMaster style release
            this.renderer.setActive(null);
            return true;
        }
        return false;
    }
    
    updateFromSlider(mouseX, mouseY) {
        const control = this.renderer.controls.get('weight_slider');
        if (!control) return;
        
        const relativeX = mouseX - control.x;
        const normalizedValue = Math.max(0, Math.min(1, relativeX / control.w));
        const newValue = this.min + normalizedValue * (this.max - this.min);
        
        // Snap to step
        const steppedValue = Math.round(newValue / this.step) * this.step;
        this.setValue(steppedValue);
    }
    
    setValue(value) {
        const clampedValue = Math.max(this.min, Math.min(this.max, value));
        
        if (Math.abs(this.value - clampedValue) > 0.001) {
            this.value = clampedValue;
            this.widget.value = clampedValue;
            
            // Trigger widget callback if it exists (maintains data flow to Python)
            if (this.originalCallback) {
                this.originalCallback.call(this.widget, clampedValue, window.app?.graph, this.node);
            } else if (this.widget.callback) {
                this.widget.callback.call(this.widget, clampedValue, window.app?.graph, this.node);
            }
            
            if (window.app?.graph) {
                window.app.graph.setDirtyCanvas(true);
            }
        }
    }
    
    showValueDialog() {
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
            border: 2px solid #555; border-radius: 12px; padding: 20px;
            box-shadow: 0 12px 48px rgba(0,0,0,0.9); z-index: 10000;
            min-width: 280px; animation: slideIn 0.3s ease-out;
        `;
        
        dialog.innerHTML = `
            <style>
                @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
                @keyframes slideIn { 
                    from { transform: translate(-50%, -60%); opacity: 0; }
                    to { transform: translate(-50%, -50%); opacity: 1; }
                }
                .weight-input {
                    width: 100%; padding: 10px; margin: 10px 0;
                    border: 2px solid #555; border-radius: 6px;
                    background: #333; color: #fff; font-size: 14px;
                    text-align: center; transition: border-color 0.2s;
                }
                .weight-input:focus {
                    border-color: #5af; outline: none;
                    box-shadow: 0 0 0 3px rgba(85, 170, 255, 0.2);
                }
                .weight-btn {
                    padding: 8px 16px; margin: 5px; border-radius: 6px;
                    font-size: 13px; cursor: pointer; transition: all 0.2s;
                    border: none; font-weight: 500;
                }
                .weight-btn-primary {
                    background: linear-gradient(135deg, #5af, #4090ff);
                    color: white;
                }
                .weight-btn-primary:hover {
                    transform: translateY(-1px);
                    box-shadow: 0 4px 12px rgba(85, 170, 255, 0.4);
                }
                .weight-btn-cancel {
                    background: #444; color: #ccc;
                }
                .weight-btn-cancel:hover {
                    background: #555;
                }
            </style>
            <div style="color: #fff; font-size: 16px; font-weight: bold; margin-bottom: 10px; text-align: center;">
                💎 Set Weight Value
            </div>
            <div style="color: #aaa; font-size: 11px; margin-bottom: 10px; text-align: center;">
                Range: ${this.min} - ${this.max}
            </div>
            <input type="number" class="weight-input" value="${this.value.toFixed(2)}" 
                   step="${this.step}" min="${this.min}" max="${this.max}"
                   placeholder="Enter weight value">
            <div style="color: #999; font-size: 11px; margin-bottom: 15px; text-align: center;">
                Tip: Try 0.8 for subtle, 1.2 for emphasis, 1.8 for strong
            </div>
            <div style="text-align: right;">
                <button class="weight-btn weight-btn-cancel">Cancel</button>
                <button class="weight-btn weight-btn-primary">Apply</button>
            </div>
        `;
        
        document.body.appendChild(overlay);
        document.body.appendChild(dialog);
        
        const input = dialog.querySelector('input');
        const applyBtn = dialog.querySelector('.weight-btn-primary');
        const cancelBtn = dialog.querySelector('.weight-btn-cancel');
        
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
            const value = parseFloat(input.value);
            if (!isNaN(value)) {
                this.setValue(value);
            }
            close();
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
    
    dispose() {
        this.tooltips.dispose();
        this.renderer.clearControls();
        
        // Unregister from layer manager
        globalLayerManager.removeEnhancedControl(this.node, this);

        // If no other enhanced controls remain, clear the foreground-handled flag
        if (!this.node.enhancedControls || this.node.enhancedControls.length === 0) {
            delete this.node.__illustriousHandlesForeground;
        }
    }
}

// Factory function to create weight controls for widgets
export function createWeightControl(node, widget) {
    // Check if widget is suitable for weight control
    const weightWidgetNames = [
        'weight', 'strength', 'influence', 'factor', 'scale',
        'Character Weight', 'Artist Weight', 'Style Weight'
    ];
    
    const isWeightWidget = weightWidgetNames.some(name => 
        widget.name.toLowerCase().includes(name.toLowerCase())
    );
    
    if (isWeightWidget && (widget.type === 'number' || widget.type === 'slider')) {
        return new WeightControl(node, widget);
    }
    
    return null;
}