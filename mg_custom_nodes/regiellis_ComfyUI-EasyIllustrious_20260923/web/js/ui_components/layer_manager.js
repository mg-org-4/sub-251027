// ==========================================================================
//  Layer Manager for Enhanced UI Controls
// ==========================================================================
//
// Ensures enhanced UI controls render on top of standard ComfyUI widgets
// Fixes z-depth issues with canvas-based controls
//
// ==========================================================================

export class LayerManager {
    constructor() {
        this.enhancedNodes = new Set();
        this.originalNodeMethods = new WeakMap();
        // Only enable layer overrides when explicitly turned on
        try {
            const v = localStorage.getItem('illustrious_enhanced_ui');
            if (v && v.toLowerCase() === 'on') {
                this.setupGlobalLayerOverrides();
            }
        } catch {
            // default off
        }
    }
    
    /**
     * Register a node for enhanced rendering
     * @param {Object} node - ComfyUI node
     */
    registerEnhancedNode(node) {
        if (this.enhancedNodes.has(node)) return;
        
        this.enhancedNodes.add(node);
        this.overrideNodeRendering(node);
    }
    
    /**
     * Override node rendering to ensure enhanced controls draw last
     * @param {Object} node - ComfyUI node
     */
    overrideNodeRendering(node) {
        // Store original methods
        const original = {
            onDrawForeground: node.onDrawForeground,
            onDrawBackground: node.onDrawBackground,
            drawWidgets: node.drawWidgets
        };
        this.originalNodeMethods.set(node, original);
        
        // Override onDrawForeground to ensure our controls draw on top
        node.onDrawForeground = function(ctx) {
            // First, draw the original foreground
            if (original.onDrawForeground) {
                original.onDrawForeground.call(this, ctx);
            }
            
            // Then draw enhanced controls on top with forced layering
            if (this.enhancedControls && this.enhancedControls.length > 0) {
                ctx.save();
                
                // Force highest drawing priority
                ctx.globalCompositeOperation = 'source-over';
                ctx.globalAlpha = 1.0;
                
                // Draw all enhanced controls
                this.enhancedControls.forEach(control => {
                    if (control.draw && typeof control.draw === 'function') {
                        control.draw(ctx);
                    }
                });
                
                ctx.restore();
            }
        };
        
        // Override widget drawing to ensure our controls don't get covered
        if (node.drawWidgets) {
            node.drawWidgets = function(...args) {
                // Only draw standard widgets if they're not replaced by enhanced controls
                if (!this._hasEnhancedControls) {
                    return original.drawWidgets ? original.drawWidgets.apply(this, args) : false;
                }
                
                // Filter out widgets that have enhanced replacements
                const originalWidgets = this.widgets || [];
                const filteredWidgets = originalWidgets.filter(widget => {
                    return !this.enhancedControlMap || !this.enhancedControlMap.has(widget);
                });
                
                // Temporarily replace widgets array
                const savedWidgets = this.widgets;
                this.widgets = filteredWidgets;
                
                const result = original.drawWidgets ? original.drawWidgets.apply(this, args) : false;
                
                // Restore widgets array
                this.widgets = savedWidgets;
                
                return result;
            };
        }
    }
    
    /**
     * Setup global layer management
     */
    setupGlobalLayerOverrides() {
    // Allow runtime disable via localStorage: set 'illustrious_overlay' to 'off'
        try {
            const flag = localStorage.getItem('illustrious_overlay');
            if (flag && flag.toLowerCase() === 'off') {
                return; // skip patching entirely
            }
        } catch (_) {}

        // Prefer patching the prototype so it works for all canvases and late app loads
        if (typeof window !== 'undefined' && window.LGraphCanvas && LGraphCanvas.prototype && !LGraphCanvas.prototype.__illustriousLayerPatched) {
            const proto = LGraphCanvas.prototype;
            const originalDraw = proto.draw;
            const drawEnhancedControlsLayer = function() {
                // 'this' is an instance of LGraphCanvas
                const ctx = this.ctx;
                const graph = this.graph;
                if (!ctx || !graph) return;

                ctx.save();

                // Re-apply world transform so overlays match zoom/pan
                const ds = this.ds || { scale: 1, offset: [0, 0] };
                // setTransform(a, b, c, d, e, f) -> scale + translate
                // Reset first to identity to avoid compounding
                ctx.setTransform(1, 0, 0, 1, 0, 0);
                ctx.scale(ds.scale || 1, ds.scale || 1);
                const offx = Array.isArray(ds.offset) ? (ds.offset[0] || 0) : (ds.offset || 0);
                const offy = Array.isArray(ds.offset) ? (ds.offset[1] || 0) : 0;
                ctx.translate(offx, offy);

                // Draw all nodes that have enhanced controls on top, honoring canvas z-order
                const ordered = this._nodes_in_order || graph._nodes_in_order || graph.nodes || graph._nodes || [];
                for (const node of ordered) {
                    if (!node || !node.enhancedControls || node.enhancedControls.length === 0) continue;
                    // Skip nodes that fully handle their own enhanced foreground to avoid double draw
                    if (node.__illustriousHandlesForeground) continue;

                    ctx.save();
                    ctx.translate(node.pos[0], node.pos[1]);
                    ctx.globalCompositeOperation = 'source-over';
                    ctx.globalAlpha = 1.0;

                    for (const control of node.enhancedControls) {
                        if (control && typeof control.draw === 'function') {
                            try { control.draw(ctx); } catch (e) { /* ignore faulty control draw */ }
                        }
                    }

                    ctx.restore();
                }

                ctx.restore();
            };

            proto.draw = function(force_canvas, force_bgcolor) {
                const result = originalDraw ? originalDraw.call(this, force_canvas, force_bgcolor) : undefined;
                try { drawEnhancedControlsLayer.call(this); } catch (_) {}
                return result;
            };

            // Mark as patched to avoid double wrapping
            Object.defineProperty(proto, '__illustriousLayerPatched', { value: true, enumerable: false });
        } else {
            // If prototype not ready yet, retry shortly
            setTimeout(() => this.setupGlobalLayerOverrides(), 100);
        }
    }
    
    /**
     * Add an enhanced control to a node
     * @param {Object} node - ComfyUI node
     * @param {Object} control - Enhanced control instance
     */
    addEnhancedControl(node, control) {
        if (!node.enhancedControls) {
            node.enhancedControls = [];
        }
        
        if (!node.enhancedControlMap) {
            node.enhancedControlMap = new Map();
        }
        
        node.enhancedControls.push(control);
        
        // Map the control to its widget if applicable
        if (control.widget) {
            node.enhancedControlMap.set(control.widget, control);
        }
        
        // Mark node as having enhanced controls
        node._hasEnhancedControls = true;
        
        // Register the node for enhanced rendering
        this.registerEnhancedNode(node);
        
        console.log(`[LayerManager] Enhanced control added to node: ${node.title || node.type}`);
    }
    
    /**
     * Remove enhanced control from a node
     * @param {Object} node - ComfyUI node
     * @param {Object} control - Enhanced control instance
     */
    removeEnhancedControl(node, control) {
        if (!node.enhancedControls) return;
        
        const index = node.enhancedControls.indexOf(control);
        if (index >= 0) {
            node.enhancedControls.splice(index, 1);
        }
        
        if (control.widget && node.enhancedControlMap) {
            node.enhancedControlMap.delete(control.widget);
        }
        
        // Clean up if no more enhanced controls
        if (node.enhancedControls.length === 0) {
            delete node._hasEnhancedControls;
            this.restoreNodeRendering(node);
            this.enhancedNodes.delete(node);
        }
    }
    
    /**
     * Restore original node rendering
     * @param {Object} node - ComfyUI node
     */
    restoreNodeRendering(node) {
        const original = this.originalNodeMethods.get(node);
        if (!original) return;
        
        // Restore original methods
        node.onDrawForeground = original.onDrawForeground;
        node.onDrawBackground = original.onDrawBackground;
        if (original.drawWidgets) {
            node.drawWidgets = original.drawWidgets;
        }
        
        this.originalNodeMethods.delete(node);
    }
    
    /**
     * Force redraw of all enhanced controls
     */
    forceRedraw() {
        if (window.app && window.app.graph) {
            window.app.graph.setDirtyCanvas(true);
        }
    }
    
    /**
     * Clean up all enhanced nodes
     */
    dispose() {
        for (const node of this.enhancedNodes) {
            this.restoreNodeRendering(node);
        }
        this.enhancedNodes.clear();
        this.originalNodeMethods = new WeakMap();
    }
}

// Global layer manager instance
export const globalLayerManager = new LayerManager();

// Auto-cleanup on page unload
if (typeof window !== 'undefined') {
    window.addEventListener('beforeunload', () => {
        globalLayerManager.dispose();
    });
}