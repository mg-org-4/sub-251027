// ==========================================================================
//  Tooltip Manager for Illustrious Nodes
// ==========================================================================
//
// Provides rich tooltips with smart positioning and formatting
// Adapted from ResolutionMaster's tooltip system
//
// ==========================================================================

export class TooltipManager {
    constructor() {
        this.tooltips = new Map();
        this.activeTooltip = null;
        this.tooltipTimer = null;
        this.delay = 400; // ms before showing tooltip
        this.mousePos = { x: 0, y: 0 };
        this.isShowing = false;
        this.fixedPos = null;
    }
    
    /**
     * Register a tooltip for an element
     * @param {string} elementId - Unique identifier for the element
     * @param {string|Object} content - Tooltip content (string or rich object)
     */
    register(elementId, content) {
        this.tooltips.set(elementId, content);
    }
    
    /**
     * Register multiple tooltips at once
     * @param {Object} tooltipMap - Object mapping element IDs to content
     */
    registerBatch(tooltipMap) {
        Object.entries(tooltipMap).forEach(([id, content]) => {
            this.register(id, content);
        });
    }
    
    /**
     * Draw the active tooltip on canvas
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {Object} nodePos - Node position {x, y}
     */
    draw(ctx, nodePos = { x: 0, y: 0 }) {
        if (!this.isShowing || !this.activeTooltip) return;
        
        const content = this.tooltips.get(this.activeTooltip);
        if (!content) return;
        
        // Ensure tooltip draws on top of everything
        ctx.save();
        ctx.globalCompositeOperation = 'source-over';
        ctx.globalAlpha = 1.0;
        
        // Prepare tooltip text
        const text = typeof content === 'object' ? content.text : content;
        const title = typeof content === 'object' ? content.title : null;
        
        // Setup drawing parameters
        const padding = { x: 10, y: 8 };
        const maxWidth = 280;
        const lineHeight = 16;
        const titleHeight = title ? 20 : 0;
        
        // Set font for measuring
        ctx.font = "12px Arial";
        
        // Word wrap the text
        const lines = this.wrapText(ctx, text, maxWidth);
        
        // Calculate tooltip dimensions
        const textWidth = Math.max(...lines.map(line => ctx.measureText(line).width));
        const tooltipWidth = Math.min(textWidth + padding.x * 2, maxWidth + padding.x * 2);
        const tooltipHeight = lines.length * lineHeight + padding.y * 2 + titleHeight;
        
        // Calculate position (relative to node, but can extend beyond)
        const pos = this.fixedPos || this.mousePos;
        let tooltipX = pos.x - nodePos.x + 15;
        let tooltipY = pos.y - nodePos.y - tooltipHeight - 10;
        
        // Adjust if tooltip would go off top
        if (tooltipY < -50) {
            tooltipY = pos.y - nodePos.y + 25;
        }
        
        // Draw tooltip
        ctx.save();
        
        // Shadow effect
        ctx.shadowColor = "rgba(0, 0, 0, 0.4)";
        ctx.shadowBlur = 8;
        ctx.shadowOffsetX = 2;
        ctx.shadowOffsetY = 2;
        
        // Background gradient
        const bgGrad = ctx.createLinearGradient(
            tooltipX, tooltipY, 
            tooltipX, tooltipY + tooltipHeight
        );
        bgGrad.addColorStop(0, "rgba(50, 50, 50, 0.98)");
        bgGrad.addColorStop(1, "rgba(35, 35, 35, 0.98)");
        ctx.fillStyle = bgGrad;
        
        ctx.beginPath();
        ctx.roundRect(tooltipX, tooltipY, tooltipWidth, tooltipHeight, 6);
        ctx.fill();
        
        // Reset shadow for border
        ctx.shadowColor = "transparent";
        
        // Border
        ctx.strokeStyle = "rgba(100, 150, 255, 0.3)";
        ctx.lineWidth = 1;
        ctx.stroke();
        
        // Title if present
        if (title) {
            // Title background
            const titleGrad = ctx.createLinearGradient(
                tooltipX, tooltipY,
                tooltipX, tooltipY + titleHeight
            );
            titleGrad.addColorStop(0, "rgba(85, 170, 255, 0.15)");
            titleGrad.addColorStop(1, "rgba(85, 170, 255, 0.05)");
            ctx.fillStyle = titleGrad;
            ctx.beginPath();
            ctx.roundRect(tooltipX, tooltipY, tooltipWidth, titleHeight, 6);
            ctx.fill();
            
            // Title text
            ctx.fillStyle = "#5af";
            ctx.font = "bold 12px Arial";
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            ctx.fillText(title, tooltipX + padding.x, tooltipY + titleHeight / 2);
        }
        
        // Content text
        ctx.fillStyle = "#ffffff";
        ctx.font = "12px Arial";
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        
        const textStartY = tooltipY + padding.y + titleHeight;
        lines.forEach((line, index) => {
            // Check for special formatting
            if (line.startsWith('•')) {
                // Bullet point - indent slightly
                ctx.fillStyle = "#5af";
                ctx.fillText('•', tooltipX + padding.x, textStartY + index * lineHeight);
                ctx.fillStyle = "#ffffff";
                ctx.fillText(line.substring(1), tooltipX + padding.x + 10, textStartY + index * lineHeight);
            } else if (line.includes(':') && index === 0 && !title) {
                // First line with colon - might be a header
                const parts = line.split(':');
                if (parts.length === 2) {
                    ctx.fillStyle = "#5af";
                    ctx.fillText(parts[0] + ':', tooltipX + padding.x, textStartY + index * lineHeight);
                    ctx.fillStyle = "#ffffff";
                    const headerWidth = ctx.measureText(parts[0] + ':').width;
                    ctx.fillText(parts[1], tooltipX + padding.x + headerWidth, textStartY + index * lineHeight);
                } else {
                    ctx.fillText(line, tooltipX + padding.x, textStartY + index * lineHeight);
                }
            } else {
                ctx.fillText(line, tooltipX + padding.x, textStartY + index * lineHeight);
            }
        });
        
        ctx.restore();
    }
    
    /**
     * Wrap text to fit within maxWidth
     * @private
     */
    wrapText(ctx, text, maxWidth) {
        // Handle newlines in the input text
        const paragraphs = text.split('\n');
        const allLines = [];
        
        paragraphs.forEach(paragraph => {
            const words = paragraph.split(' ');
            let currentLine = '';
            
            for (const word of words) {
                const testLine = currentLine + (currentLine ? ' ' : '') + word;
                const metrics = ctx.measureText(testLine);
                
                if (metrics.width > maxWidth && currentLine) {
                    allLines.push(currentLine);
                    currentLine = word;
                } else {
                    currentLine = testLine;
                }
            }
            
            if (currentLine) {
                allLines.push(currentLine);
            }
        });
        
        return allLines;
    }
    
    /**
     * Handle mouse hover over elements
     * @param {string|null} elementId - Element being hovered (null if none)
     * @param {number} mouseX - Mouse X position
     * @param {number} mouseY - Mouse Y position
     */
    handleHover(elementId, mouseX, mouseY) {
        this.mousePos = { x: mouseX, y: mouseY };
        
        // Clear existing timer
        if (this.tooltipTimer) {
            clearTimeout(this.tooltipTimer);
            this.tooltipTimer = null;
        }
        
        // Hide tooltip if hover ended
        if (!elementId || !this.tooltips.has(elementId)) {
            this.hide();
            return;
        }
        
        // If already showing tooltip for this element, keep it visible
        if (this.isShowing && this.activeTooltip === elementId) {
            return;
        }
        
        // Hide current tooltip if switching elements
        if (this.activeTooltip !== elementId) {
            this.hide();
        }
        
        // Start timer to show new tooltip
        const initialPos = { x: mouseX, y: mouseY };
        this.tooltipTimer = setTimeout(() => {
            this.activeTooltip = elementId;
            this.isShowing = true;
            this.fixedPos = initialPos; // Lock position when tooltip appears
            
            // Request canvas redraw
            if (window.app?.graph) {
                window.app.graph.setDirtyCanvas(true);
            }
        }, this.delay);
    }
    
    /**
     * Hide the current tooltip
     */
    hide() {
        if (this.isShowing) {
            this.isShowing = false;
            this.activeTooltip = null;
            this.fixedPos = null;
            
            // Request canvas redraw
            if (window.app?.graph) {
                window.app.graph.setDirtyCanvas(true);
            }
        }
    }
    
    /**
     * Clean up resources
     */
    dispose() {
        if (this.tooltipTimer) {
            clearTimeout(this.tooltipTimer);
            this.tooltipTimer = null;
        }
        this.hide();
        this.tooltips.clear();
    }
}

// Widget-specific tooltip presets
export const TOOLTIP_PRESETS = {
    weight: {
        title: "Weight Control",
        text: "Controls the influence strength of this element.\n• 0.0: No effect\n• 1.0: Normal strength\n• 2.0: Double strength\n• Negative: Inverse effect"
    },
    
    character: {
        title: "Character Selection",
        text: "Choose a character from the database.\n• Type to search\n• Use wildcards (*)\n• Combine with + for multiple"
    },
    
    artist: {
        title: "Artist Style",
        text: "Apply an artist's style to the generation.\n• Mix multiple artists with +\n• Adjust influence with weights\n• Use 'style of' prefix for emphasis"
    },
    
    quality: {
        title: "Quality Settings",
        text: "Enhance output quality.\n• Low: Fast, lower quality\n• Medium: Balanced\n• High: Best quality, slower\n• Ultra: Maximum quality"
    },
    
    cfg: {
        title: "CFG Scale",
        text: "Classifier-Free Guidance strength.\n• 4-6: Optimal for Illustrious\n• Higher: Stricter prompt following\n• Lower: More creative freedom"
    },
    
    steps: {
        title: "Sampling Steps",
        text: "Number of denoising iterations.\n• 20-30: Standard quality\n• 30-50: High quality\n• 50+: Diminishing returns"
    }
};