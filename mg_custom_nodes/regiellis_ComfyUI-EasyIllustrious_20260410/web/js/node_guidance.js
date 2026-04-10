/**
 * Node Guidance System - Toast notifications for user-friendly parameter guidance
 * Provides contextual help and warnings for Illustrious nodes
 */

// Note: don't capture app/api at module load; defer to window.* at runtime
const app = window.app; // retained for backward compatibility but avoid using directly
const api = window.api; // retained for backward compatibility but avoid using directly

class NodeGuidanceSystem {
    constructor() {
        this.lastToastTime = {};  // Track toast timing to avoid spam
        this.toastCooldown = 5000; // 5 second cooldown between same toasts
        this.setupNodeCallbacks();
        this.setupServerMessageHandlers();
    }

    setupNodeCallbacks() {
        // Hook into node value changes
        const self = this;
        const originalOnWidgetChange = window.app?.canvas?.onWidgetChange;
        if (!window.app || !window.app.canvas) return; // guard if app not ready
        window.app.canvas.onWidgetChange = function(widget, value, node) {
            if (originalOnWidgetChange) {
                originalOnWidgetChange.call(this, widget, value, node);
            }
            
            // Check for guidance on specific nodes
            self.handleWidgetChange(widget, value, node);
        };
    }

    setupServerMessageHandlers() {
    // Listen for server-side guidance messages
        if (!window.api || !window.api.addEventListener) {
            console.warn("[NodeGuidance] window.api not available; skipping server guidance listener");
            return;
        }
        window.api.addEventListener("illustrious.guidance", (event) => {
            const data = event.detail;
            this.showToast(data.severity, data.summary, data.detail, data.life || 4000);
        });
        
        console.log("[NodeGuidance] Server message handlers registered");
    }

    showToast(severity, summary, detail, life = 4000) {
        const toastKey = `${severity}-${summary}`;
        const now = Date.now();
        
        // Avoid spam by checking cooldown
        if (this.lastToastTime[toastKey] && (now - this.lastToastTime[toastKey]) < this.toastCooldown) {
            return;
        }
        
        this.lastToastTime[toastKey] = now;
        
        if (window.app && window.app.extensionManager && window.app.extensionManager.toast) {
            window.app.extensionManager.toast.add({
                severity,
                summary,
                detail,
                life,
                closable: true
            });
        }
    }

    handleWidgetChange(widget, value, node) {
        const nodeType = node.comfyClass;
        const widgetName = widget.name;

        switch (nodeType) {
            case "IllustriousKSamplerPro":
            case "IllustriousKSamplerPresets":
                this.handleKSamplerGuidance(widget, value, node, widgetName);
                break;
                
            case "IllustriousVAEDecode":
            case "IllustriousVAEEncode":
                this.handleVAEGuidance(widget, value, node, widgetName);
                break;
                
            case "IllustriousMultiPassSampler":
            case "IllustriousTriplePassSampler":
                this.handleMultiPassGuidance(widget, value, node, widgetName);
                break;
        }
    }

    handleKSamplerGuidance(widget, value, node, widgetName) {
        switch (widgetName) {
            case "cfg":
                if (value > 7.0) {
                    this.showToast("warn", "High CFG Warning", 
                        `CFG ${value} is quite high for Illustrious models. Values above 7.0 may cause over-saturation or artifacts. Try 4.0-6.5 for best results.`);
                } else if (value < 3.0) {
                    this.showToast("info", "Low CFG Notice", 
                        `CFG ${value} is low. While this can work for artistic effects, try 4.0-5.5 for more controlled generation with Illustrious models.`);
                } else if (value >= 4.0 && value <= 6.5) {
                    this.showToast("success", "Optimal CFG", 
                        `CFG ${value} is in the sweet spot for Illustrious models! 🎯`);
                }
                break;
                
            case "model_version":
                if (value.includes("Illustrious v2.0")) {
                    this.showToast("info", "Illustrious v2.0", 
                        "Latest version with best quality and control. Use CFG 4.5-6.5 and 25-35 steps for optimal results.");
                } else if (value.includes("Illustrious v1.1")) {
                    this.showToast("info", "Illustrious v1.1", 
                        "Stable version with excellent anime quality. Works great with CFG 4.0-6.0 and 20-30 steps.");
                }
                break;
                
            case "steps":
                if (value > 40) {
                    this.showToast("warn", "High Step Count", 
                        `${value} steps is quite high. Illustrious models usually converge well at 25-35 steps. Higher values may waste time without better quality.`);
                } else if (value < 15) {
                    this.showToast("warn", "Low Step Count", 
                        `${value} steps might be too few for good quality. Try 20-30 steps for Illustrious models.`);
                }
                break;
                
            case "denoise":
                if (value > 0.8) {
                    this.showToast("info", "High Denoise", 
                        `Denoise ${value} will create significant changes from the input. Good for major variations.`);
                } else if (value < 0.3) {
                    this.showToast("info", "Low Denoise", 
                        `Denoise ${value} will stay very close to input. Good for subtle refinements.`);
                }
                break;
        }
    }

    handleVAEGuidance(widget, value, node, widgetName) {
        switch (widgetName) {
            case "illustrious_version":
                if (value === "auto") {
                    this.showToast("info", "Auto Detection", 
                        "VAE will detect the best settings for your image automatically based on resolution and content.");
                } else if (value.includes("v2.0")) {
                    this.showToast("success", "Latest Version", 
                        "Using v2.0 optimizations - best quality with advanced processing for ultra-high resolution.");
                }
                break;
                
            case "tile_size":
                if (value > 768) {
                    this.showToast("warn", "Large Tile Size", 
                        `${value}px tiles use more VRAM. Reduce if you get memory errors. 512-640px is usually optimal.`);
                } else if (value < 384) {
                    this.showToast("warn", "Small Tile Size", 
                        `${value}px tiles may create more seams. Try 512px+ for better quality with tiled processing.`);
                }
                break;
                
            case "correction_strength":
                if (value > 1.5) {
                    this.showToast("warn", "High Strength", 
                        `Strength ${value} will apply heavy corrections. May over-process the image. Try 0.8-1.2 for subtle improvements.`);
                } else if (value > 0.8 && value <= 1.2) {
                    this.showToast("success", "Good Strength", 
                        `Strength ${value} provides balanced enhancement without over-processing. 👍`);
                }
                break;
                
            case "optimization_mode":
                switch (value) {
                    case "quality":
                        this.showToast("info", "Quality Mode", 
                            "Prioritizing image quality over speed. Best for final outputs and high-resolution images.");
                        break;
                    case "speed":
                        this.showToast("info", "Speed Mode", 
                            "Prioritizing processing speed. Good for quick previews and iterations.");
                        break;
                    case "memory":
                        this.showToast("info", "Memory Mode", 
                            "Optimized for low VRAM. Uses smaller tiles and conservative processing.");
                        break;
                }
                break;
        }
    }

    handleMultiPassGuidance(widget, value, node, widgetName) {
        switch (widgetName) {
            case "structure_cfg":
                if (value > 7.0) {
                    this.showToast("warn", "High Structure CFG", 
                        `Structure CFG ${value} is high. This pass builds composition - try 5.5-6.5 for balanced results.`);
                }
                break;
                
            case "detail_cfg":
                if (value > 5.0) {
                    this.showToast("warn", "High Detail CFG", 
                        `Detail CFG ${value} might over-sharpen. Detail pass should be gentler - try 3.5-4.5.`);
                }
                break;
                
            case "structure_denoise":
                if (value < 0.7) {
                    this.showToast("warn", "Low Structure Denoise", 
                        `Structure denoise ${value} might not build enough composition. Try 0.75-0.85 for good structure building.`);
                }
                break;
                
            case "detail_denoise":
                if (value > 0.7) {
                    this.showToast("warn", "High Detail Denoise", 
                        `Detail denoise ${value} might rebuild too much. Detail pass should refine, not rebuild - try 0.4-0.6.`);
                }
                break;
                
            case "use_different_samplers":
                if (value) {
                    this.showToast("info", "Multi-Sampler Mode", 
                        "Using different samplers for each pass. Structure: DPM 2S Ancestral, Detail: Euler Ancestral - optimized for Illustrious!");
                }
                break;
        }
    }

    // Method for nodes to call directly for specific guidance
    showParameterWarning(nodeType, paramName, paramValue, customMessage = null) {
        let message = customMessage;
        
        if (!message) {
            // Generate default messages based on common parameters
            if (paramName.includes("cfg") && paramValue > 7.0) {
                message = `${paramName.toUpperCase()} value ${paramValue} is high for anime models. Consider reducing to 4.0-6.5 range.`;
            } else if (paramName.includes("denoise") && paramValue > 0.8) {
                message = `Denoise ${paramValue} will make significant changes from the input image.`;
            } else if (paramName.includes("strength") && paramValue > 1.5) {
                message = `Strength ${paramValue} may over-process the image. Try values between 0.8-1.2.`;
            }
        }
        
        if (message) {
            this.showToast("warn", "Parameter Notice", message);
        }
    }

    // Show helpful tips when nodes are added
    showNodeAddedTip(nodeType) {
        switch (nodeType) {
            case "IllustriousKSamplerPro":
                this.showToast("info", "KSampler Added", 
                    "💡 Tip: For Illustrious models, try CFG 4.5-6.0, 25-30 steps, and use Euler Ancestral or DPM++ 2M Karras samplers.");
                break;
                
            case "IllustriousVAEDecode":
                this.showToast("info", "Illustrious VAE Added", 
                    "💡 Tip: This VAE includes anime-optimized enhancements. Enable tiling for high-res images and adjust strength based on your needs.");
                break;
                
            case "IllustriousMultiPassSampler":
                this.showToast("info", "Multi-Pass Sampler", 
                    "💡 Tip: Structure pass builds composition (high denoise), Detail pass refines (low denoise). Great for complex anime scenes!");
                break;
        }
    }
}

// Defer initialization until ComfyUI app is ready with a one-time guard
(function registerIllustriousNodeGuidance(){
    if (window.__illustrious_node_guidance_registered) return;
    if (!window.app || !window.app.registerExtension) { setTimeout(registerIllustriousNodeGuidance, 50); return; }
    window.__illustrious_node_guidance_registered = true;
    window.app.registerExtension({
        name: "Illustrious.NodeGuidance",
        setup() {
            try {
                const nodeGuidance = new NodeGuidanceSystem();
                // Hook into node creation to show tips
                const originalAddNodeOnGraph = window.app?.canvas?.onNodeAdded;
                if (!window.app || !window.app.canvas) return;
                window.app.canvas.onNodeAdded = function(node) {
                    if (originalAddNodeOnGraph) {
                        originalAddNodeOnGraph.call(this, node);
                    }
                    setTimeout(() => nodeGuidance.showNodeAddedTip(node.comfyClass), 500);
                };
                window.nodeGuidance = nodeGuidance;
            } catch (e) {
                console.error("[NodeGuidance] Failed to initialize:", e);
            }
        }
    });
})();