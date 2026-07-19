// ==========================================================================
//  Canvas Renderer for Illustrious Nodes
// ==========================================================================
//
// Provides custom canvas-based UI controls with gradients and animations
// Adapted from ResolutionMaster's rendering techniques
//
// ==========================================================================

export class CanvasRenderer {
    constructor(node) {
        this.node = node;
        this.controls = new Map();
        this.hoverElement = null;
        this.activeElement = null;
        this.animations = new Map();
    }
    
    /**
     * Register a control area for interaction
     * @param {string} id - Control identifier
     * @param {Object} bounds - {x, y, w, h}
     */
    registerControl(id, bounds) {
        this.controls.set(id, bounds);
    }
    
    /**
     * Check if a point is within a control
     * @param {number} x - X coordinate (relative to node)
     * @param {number} y - Y coordinate (relative to node)
     * @returns {string|null} Control ID if hit, null otherwise
     */
    hitTest(x, y) {
        for (const [id, bounds] of this.controls) {
            if (x >= bounds.x && x <= bounds.x + bounds.w &&
                y >= bounds.y && y <= bounds.y + bounds.h) {
                return id;
            }
        }
        return null;
    }
    
    /**
     * Draw a gradient button
     * @param {CanvasRenderingContext2D} ctx
     * @param {number} x - X position
     * @param {number} y - Y position
     * @param {number} w - Width
     * @param {number} h - Height
     * @param {string} text - Button text
     * @param {Object} options - Styling options
     */
    drawButton(ctx, x, y, w, h, text, options = {}) {
        const {
            hover = false,
            active = false,
            disabled = false,
            style = 'primary', // 'primary', 'secondary', 'danger'
            icon = null
        } = options;
        
        // Create gradient based on state and style
        const grad = ctx.createLinearGradient(x, y, x, y + h);
        
        if (disabled) {
            grad.addColorStop(0, "#4a4a4a");
            grad.addColorStop(1, "#3a3a3a");
        } else if (active) {
            grad.addColorStop(0, "#4060cf");
            grad.addColorStop(1, "#3050bf");
        } else if (hover) {
            if (style === 'primary') {
                grad.addColorStop(0, "#6a8aff");
                grad.addColorStop(1, "#5070ee");
            } else if (style === 'secondary') {
                grad.addColorStop(0, "#6a6a6a");
                grad.addColorStop(1, "#505050");
            } else if (style === 'danger') {
                grad.addColorStop(0, "#ff6a6a");
                grad.addColorStop(1, "#ee5050");
            }
        } else {
            if (style === 'primary') {
                grad.addColorStop(0, "#5a7adf");
                grad.addColorStop(1, "#4060cf");
            } else if (style === 'secondary') {
                grad.addColorStop(0, "#5a5a5a");
                grad.addColorStop(1, "#404040");
            } else if (style === 'danger') {
                grad.addColorStop(0, "#df5a5a");
                grad.addColorStop(1, "#cf4040");
            }
        }
        
        // Draw button background
        ctx.fillStyle = grad;
        ctx.strokeStyle = disabled ? "#333" : (hover ? "#88aaff" : "#4060cf");
        ctx.lineWidth = 1;
        
        ctx.beginPath();
        ctx.roundRect(x, y, w, h, 6);
        ctx.fill();
        ctx.stroke();
        
        // Inner highlight
        if (!disabled && !active) {
            ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
            ctx.beginPath();
            ctx.roundRect(x + 1, y + 1, w - 2, h - 2, 5);
            ctx.stroke();
        }
        
        // Draw icon if provided
        if (icon) {
            ctx.fillStyle = disabled ? "#666" : "#fff";
            ctx.font = "14px Arial";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(icon, x + 15, y + h/2);
            
            // Adjust text position
            ctx.fillText(text, x + w/2 + 10, y + h/2);
        } else {
            // Draw text with shadow
            if (!disabled) {
                ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
                ctx.font = "12px Arial";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(text, x + w/2 + 1, y + h/2 + 1);
            }
            
            ctx.fillStyle = disabled ? "#888" : "#fff";
            ctx.font = "12px Arial";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(text, x + w/2, y + h/2);
        }
    }
    
    /**
     * Draw a visual slider with gradient fill
     * @param {CanvasRenderingContext2D} ctx
     * @param {number} x - X position
     * @param {number} y - Y position
     * @param {number} w - Width
     * @param {number} h - Height
     * @param {number} value - Current value
     * @param {number} min - Minimum value
     * @param {number} max - Maximum value
     * @param {Object} options - Styling options
     */
    drawSlider(ctx, x, y, w, h, value, min, max, options = {}) {
        const {
            label = "",
            showValue = true,
            color = 'blue', // 'blue', 'green', 'orange', 'red'
            hover = false,
            disabled = false
        } = options;
        
        // Draw label if provided
        if (label) {
            ctx.fillStyle = disabled ? "#666" : "#aaa";
            ctx.font = "11px Arial";
            ctx.textAlign = "left";
            ctx.textBaseline = "bottom";
            ctx.fillText(label, x, y - 2);
        }
        
        // Track
        const trackY = y + h/2;
        const trackHeight = 6;
        
        // Track shadow
        ctx.fillStyle = "rgba(0, 0, 0, 0.3)";
        ctx.beginPath();
        ctx.roundRect(x, trackY - trackHeight/2 + 1, w, trackHeight, trackHeight/2);
        ctx.fill();
        
        // Track background
        ctx.fillStyle = "#222";
        ctx.beginPath();
        ctx.roundRect(x, trackY - trackHeight/2, w, trackHeight, trackHeight/2);
        ctx.fill();
        
        // Calculate fill width
        const normalizedValue = Math.max(0, Math.min(1, (value - min) / (max - min)));
        const fillWidth = w * normalizedValue;
        
        // Fill gradient
        if (fillWidth > 0) {
            const fillGrad = ctx.createLinearGradient(x, trackY, x + fillWidth, trackY);
            
            if (disabled) {
                fillGrad.addColorStop(0, "#555");
                fillGrad.addColorStop(1, "#666");
            } else if (color === 'blue') {
                fillGrad.addColorStop(0, "#4090ff");
                fillGrad.addColorStop(1, "#5af");
            } else if (color === 'green') {
                fillGrad.addColorStop(0, "#40ff90");
                fillGrad.addColorStop(1, "#5fa");
            } else if (color === 'orange') {
                fillGrad.addColorStop(0, "#ff9040");
                fillGrad.addColorStop(1, "#fa5");
            } else if (color === 'red') {
                fillGrad.addColorStop(0, "#ff4040");
                fillGrad.addColorStop(1, "#f55");
            }
            
            ctx.fillStyle = fillGrad;
            ctx.beginPath();
            ctx.roundRect(x, trackY - trackHeight/2, fillWidth, trackHeight, trackHeight/2);
            ctx.fill();
        }
        
        // Knob
        const knobX = x + fillWidth;
        const knobRadius = hover ? 11 : 10;
        
        // Knob shadow
        ctx.fillStyle = "rgba(0, 0, 0, 0.4)";
        ctx.beginPath();
        ctx.arc(knobX, trackY + 2, knobRadius, 0, Math.PI * 2);
        ctx.fill();
        
        // Knob gradient
        const knobGrad = ctx.createRadialGradient(
            knobX, trackY - 2, 0,
            knobX, trackY, knobRadius
        );
        
        if (disabled) {
            knobGrad.addColorStop(0, "#888");
            knobGrad.addColorStop(0.7, "#666");
            knobGrad.addColorStop(1, "#555");
        } else {
            knobGrad.addColorStop(0, "#fff");
            knobGrad.addColorStop(0.7, "#e0e0e0");
            knobGrad.addColorStop(1, "#c0c0c0");
        }
        
        ctx.fillStyle = knobGrad;
        ctx.strokeStyle = disabled ? "#444" : (hover ? "#5af" : "#999");
        ctx.lineWidth = hover ? 2 : 1;
        ctx.beginPath();
        ctx.arc(knobX, trackY, knobRadius, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        
        // Value display
        if (showValue) {
            // Value background
            const valueText = typeof value === 'number' ? value.toFixed(1) : value.toString();
            const valueWidth = ctx.measureText(valueText).width + 8;
            
            ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
            ctx.beginPath();
            ctx.roundRect(x + w - valueWidth - 5, y - 2, valueWidth, 16, 3);
            ctx.fill();
            
            // Value text
            ctx.fillStyle = disabled ? "#888" : "#5af";
            ctx.font = "bold 11px Arial";
            ctx.textAlign = "right";
            ctx.textBaseline = "middle";
            ctx.fillText(valueText, x + w - 7, y + 6);
        }
    }
    
    /**
     * Draw a toggle switch
     * @param {CanvasRenderingContext2D} ctx
     * @param {number} x - X position
     * @param {number} y - Y position
     * @param {number} w - Width
     * @param {number} h - Height
     * @param {boolean} isOn - Toggle state
     * @param {Object} options - Styling options
     */
    drawToggle(ctx, x, y, w, h, isOn, options = {}) {
        const {
            label = "",
            hover = false,
            disabled = false
        } = options;
        
        const radius = h / 2;
        
        // Track
        ctx.beginPath();
        ctx.roundRect(x, y, w, h, radius);
        
        const trackGrad = ctx.createLinearGradient(x, y, x + w, y);
        if (disabled) {
            trackGrad.addColorStop(0, "#444");
            trackGrad.addColorStop(1, "#555");
        } else if (isOn) {
            trackGrad.addColorStop(0, "#3a76d6");
            trackGrad.addColorStop(1, "#5a96f6");
        } else {
            trackGrad.addColorStop(0, "#555");
            trackGrad.addColorStop(1, "#666");
        }
        
        ctx.fillStyle = trackGrad;
        ctx.fill();
        
        ctx.strokeStyle = hover ? "#88aaff" : "#333";
        ctx.lineWidth = 1;
        ctx.stroke();
        
        // Knob
        const knobX = isOn ? x + w - radius : x + radius;
        const knobRadius = radius - 3;
        
        // Knob shadow
        ctx.fillStyle = "rgba(0, 0, 0, 0.3)";
        ctx.beginPath();
        ctx.arc(knobX, y + radius + 1, knobRadius, 0, Math.PI * 2);
        ctx.fill();
        
        // Knob gradient
        const knobGrad = ctx.createRadialGradient(
            knobX, y + radius - 2, 0,
            knobX, y + radius, knobRadius
        );
        
        if (disabled) {
            knobGrad.addColorStop(0, "#888");
            knobGrad.addColorStop(1, "#666");
        } else {
            knobGrad.addColorStop(0, "#fff");
            knobGrad.addColorStop(1, "#d0d0d0");
        }
        
        ctx.fillStyle = knobGrad;
        ctx.beginPath();
        ctx.arc(knobX, y + radius, knobRadius, 0, Math.PI * 2);
        ctx.fill();
        
        // Label
        if (label) {
            ctx.fillStyle = disabled ? "#666" : (isOn ? "#fff" : "#aaa");
            ctx.font = "bold 11px Arial";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(label, x + w/2, y + h/2);
        }
    }
    
    /**
     * Draw a section container
     * @param {CanvasRenderingContext2D} ctx
     * @param {string} title - Section title
     * @param {number} x - X position
     * @param {number} y - Y position
     * @param {number} w - Width
     * @param {number} h - Height
     * @param {Object} options - Styling options
     */
    drawSection(ctx, title, x, y, w, h, options = {}) {
        const {
            collapsed = false,
            highlight = false
        } = options;
        
        // Background
        ctx.fillStyle = "rgba(0, 0, 0, 0.3)";
        ctx.strokeStyle = highlight ? "rgba(85, 170, 255, 0.4)" : "rgba(100, 150, 255, 0.2)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(x, y, w, h, 8);
        ctx.fill();
        ctx.stroke();
        
        // Title bar
        const titleHeight = 24;
        const titleGrad = ctx.createLinearGradient(x, y, x, y + titleHeight);
        
        if (highlight) {
            titleGrad.addColorStop(0, "rgba(85, 170, 255, 0.25)");
            titleGrad.addColorStop(1, "rgba(85, 170, 255, 0.1)");
        } else {
            titleGrad.addColorStop(0, "rgba(85, 170, 255, 0.15)");
            titleGrad.addColorStop(1, "rgba(85, 170, 255, 0.05)");
        }
        
        ctx.fillStyle = titleGrad;
        ctx.beginPath();
        ctx.roundRect(x, y, w, titleHeight, 8);
        ctx.fill();
        
        // Collapse indicator
        if (!collapsed) {
            ctx.fillStyle = "#5af";
            ctx.font = "12px Arial";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText("▼", x + 12, y + titleHeight/2);
        } else {
            ctx.fillStyle = "#5af";
            ctx.font = "12px Arial";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText("▶", x + 12, y + titleHeight/2);
        }
        
        // Title text
        ctx.fillStyle = highlight ? "#fff" : "#5af";
        ctx.font = "bold 12px Arial";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(title, x + 25, y + titleHeight/2);
        
        return titleHeight; // Return header height for content positioning
    }
    
    /**
     * Clear all registered controls
     */
    clearControls() {
        this.controls.clear();
    }
    
    /**
     * Set hover state for an element
     * @param {string|null} elementId
     */
    setHover(elementId) {
        if (this.hoverElement !== elementId) {
            this.hoverElement = elementId;
            return true; // Needs redraw
        }
        return false;
    }
    
    /**
     * Set active (pressed) state for an element
     * @param {string|null} elementId
     */
    setActive(elementId) {
        if (this.activeElement !== elementId) {
            this.activeElement = elementId;
            return true; // Needs redraw
        }
        return false;
    }
}