// ==========================================================================
//  Illustrious Smart Cache Color Corrector - ComfyUI Lifecycle Integration  
// ==========================================================================
//
// Integrates with ComfyUI's lifecycle hooks to provide intelligent caching
// and real-time adjustment without workflow regeneration for Illustrious models
//
// ==========================================================================

const app = window.app;
const api = window.api;

// --- GLOBAL CACHE STATE ---
const CACHE_STATE = {
    activeNodes: new Map(),        // Node instances with active cache
    workflowState: 'idle',         // 'idle', 'executing', 'completed'
    lastExecutionId: null,         // Track workflow executions
    previewCache: new Map(),       // Real-time preview cache
    settingsHistory: new Map()     // Settings change history
};

// --- LIFECYCLE INTEGRATION ---
// Defer registration until app is ready with a one-time guard
(function registerIllustriousSmartCache(){
if (window.__illustrious_smart_cache_registered) return;
if (!window.app || !window.app.registerExtension) { setTimeout(registerIllustriousSmartCache, 50); return; }
window.__illustrious_smart_cache_registered = true;
const app = window.app;
const api = window.api;
app.registerExtension({
    name: "Illustrious.SmartCacheCorrector",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "IllustriousSmartCacheCorrector") {
            // Hijack the original execution method to handle caching logic
            const originalComputeSize = nodeType.prototype.computeSize;
            nodeType.prototype.computeSize = function(out) {
                const result = originalComputeSize?.apply(this, arguments);
                
                // Add cache status indicator to node size
                if (this.cacheStatus && this.cacheStatus.isActive) {
                    this.size[1] = Math.max(this.size[1], this.size[1] + 20);
                }
                
                return result;
            };

            // Hijack the execution to detect workflow state
            const originalOnExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                const result = originalOnExecuted?.apply(this, arguments);
                
                // Mark this node as having fresh cache
                this.cacheStatus = {
                    isActive: true,
                    timestamp: Date.now(),
                    canAdjust: true,
                    executionId: CACHE_STATE.lastExecutionId,
                    modelVersion: this.widgets.find(w => w.name === "model_version")?.value || "auto"
                };
                
                this.setDirtyCanvas(true, true);
                return result;
            };

            // Hijack onConnectionsChange to detect workflow modifications
            const originalOnConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function(side, slot, connect, link_info, output) {
                const result = originalOnConnectionsChange?.apply(this, arguments);
                
                // If upstream connections change, invalidate cache appropriately
                if (side === 0 && connect) { // Input connected
                    this.handleUpstreamChange();
                } else if (side === 0 && !connect) { // Input disconnected
                    this.clearCacheState();
                }
                
                return result;
            };
        }
    },

    async nodeCreated(node) {
        if (node.comfyClass === "IllustriousSmartCacheCorrector") {
            // Initialize node-specific cache state
            node.cacheStatus = {
                isActive: false,
                timestamp: null,
                canAdjust: false,
                executionId: null,
                lastSettings: null,
                previewMode: false,
                modelVersion: "auto"
            };

            // Track this node in global state
            CACHE_STATE.activeNodes.set(node.id, node);

            // Add cache-specific methods
            Object.assign(node, {
                handleUpstreamChange: function() {
                    console.log(`[Illustrious Smart Cache] Upstream change detected for node ${this.id}`);
                    
                    // Determine if cache should be invalidated
                    const shouldInvalidate = this.shouldInvalidateCache();
                    
                    if (shouldInvalidate) {
                        this.clearCacheState();
                    } else {
                        // Mark as potentially needing refresh
                        this.cacheStatus.needsRefresh = true;
                    }
                    
                    this.setDirtyCanvas(true, true);
                },

                shouldInvalidateCache: function() {
                    // Check if upstream nodes have changed significantly
                    // This is where we implement smart cache invalidation logic
                    
                    // If we're in preview mode, don't invalidate for minor changes
                    if (this.cacheStatus.previewMode) {
                        return false;
                    }
                    
                    // Check if connected input nodes have been modified
                    for (let i = 0; i < this.inputs?.length; i++) {
                        const input = this.inputs[i];
                        if (input.link) {
                            const link = app.graph.links[input.link];
                            if (link) {
                                const sourceNode = app.graph.getNodeById(link.origin_id);
                                if (sourceNode && this.hasNodeChanged(sourceNode)) {
                                    return true;
                                }
                            }
                        }
                    }
                    
                    return false;
                },

                hasNodeChanged: function(sourceNode) {
                    // Implement logic to detect if a source node has meaningfully changed
                    // This could check node settings, execution state, etc.
                    
                    // For now, assume any execution means change
                    const nodeLastExecution = sourceNode.lastExecutionTime || 0;
                    const ourCacheTime = this.cacheStatus.timestamp || 0;
                    
                    return nodeLastExecution > ourCacheTime;
                },

                clearCacheState: function() {
                    console.log(`[Illustrious Smart Cache] Clearing cache state for node ${this.id}`);
                    this.cacheStatus = {
                        isActive: false,
                        timestamp: null,
                        canAdjust: false,
                        executionId: null,
                        lastSettings: null,
                        previewMode: false,
                        modelVersion: "auto"
                    };
                    this.setDirtyCanvas(true, true);
                },

                enablePreviewMode: function() {
                    console.log(`[Illustrious Smart Cache] Enabling preview mode for node ${this.id}`);
                    this.cacheStatus.previewMode = true;
                    
                    // Change adjustment_mode to cached_only
                    const adjustmentWidget = this.widgets.find(w => w.name === "adjustment_mode");
                    if (adjustmentWidget) {
                        adjustmentWidget.value = "cached_only";
                    }
                    
                    this.setDirtyCanvas(true, true);
                },

                disablePreviewMode: function() {
                    console.log(`[Illustrious Smart Cache] Disabling preview mode for node ${this.id}`);
                    this.cacheStatus.previewMode = false;
                    
                    // Reset adjustment_mode to hybrid
                    const adjustmentWidget = this.widgets.find(w => w.name === "adjustment_mode");
                    if (adjustmentWidget) {
                        adjustmentWidget.value = "hybrid";
                    }
                    
                    this.setDirtyCanvas(true, true);
                },

                onDrawForeground: function(ctx) {
                    // Call parent draw method if it exists
                    if (this.constructor.prototype.onDrawForeground !== this.onDrawForeground) {
                        this.constructor.prototype.onDrawForeground?.call(this, ctx);
                    }

                    // Draw cache status indicator
                    if (this.cacheStatus.isActive) {
                        this.drawIllustriousCacheStatusIndicator(ctx);
                    }
                },

                drawIllustriousCacheStatusIndicator: function(ctx) {
                    const padding = 5;
                    const indicatorHeight = 20;
                    const y = this.size[1] - indicatorHeight - padding;
                    
                    // Background styling
                    ctx.fillStyle = this.cacheStatus.previewMode ? "rgba(255, 107, 157, 0.8)" : "rgba(34, 197, 94, 0.8)";
                    ctx.fillRect(padding, y, this.size[0] - padding * 2, indicatorHeight);
                    
                    // Text
                    ctx.fillStyle = "#ffffff";
                    ctx.font = "10px Arial";
                    ctx.textAlign = "center";
                    ctx.textBaseline = "middle";
                    
                    const statusText = this.cacheStatus.previewMode ? 
                        "🎨 PREVIEW - Adjusting cached image" : 
                        "🧠 CACHED - Illustrious adjustments ready";
                    
                    ctx.fillText(statusText, this.size[0] / 2, y + indicatorHeight / 2);
                    
                    // Add model version and timestamp
                    if (this.cacheStatus.timestamp) {
                        const age = Math.round((Date.now() - this.cacheStatus.timestamp) / 1000);
                        const version = this.cacheStatus.modelVersion || "auto";
                        ctx.font = "8px Arial";
                        ctx.fillStyle = "rgba(255,255,255,0.7)";
                        ctx.fillText(`${version} | ${age}s ago`, this.size[0] - 45, y - 5);
                    }
                },

                // Override widget callbacks to detect setting changes
                setupIllustriousSmartCacheWidgets: function() {
                    const cacheRelevantWidgets = [
                        "correction_strength", "fix_oversaturation", "enhance_details",
                        "balance_colors", "adjust_contrast", "custom_preset", "model_version"
                    ];

                    cacheRelevantWidgets.forEach(widgetName => {
                        const widget = this.widgets.find(w => w.name === widgetName);
                        if (widget) {
                            const originalCallback = widget.callback;
                            widget.callback = (value) => {
                                // Call original callback if it exists
                                if (originalCallback) {
                                    originalCallback.call(this, value);
                                }

                                // Handle cache-aware adjustment for Illustrious
                                this.handleIllustriousSettingChange(widgetName, value);
                            };
                        }
                    });

                    // Add Illustrious cache control buttons
                    this.addIllustriousCacheControlWidgets();
                },

                handleIllustriousSettingChange: function(settingName, value) {
                    console.log(`[Illustrious Smart Cache] Setting changed: ${settingName} = ${value}`);
                    
                    // Update model version in cache status
                    if (settingName === "model_version") {
                        this.cacheStatus.modelVersion = value;
                    }
                    
                    // If we have active cache, enable preview mode for real-time adjustment
                    if (this.cacheStatus.isActive && !this.cacheStatus.previewMode) {
                        // Ask user if they want to adjust cached image or regenerate
            this.showIllustriousAdjustmentDialog(settingName, value);
                    } else if (this.cacheStatus.previewMode) {
                        // We're already in preview mode, apply change immediately
            this.applyIllustriousPreviewAdjustment(settingName, value);
                    }
                },

        showIllustriousAdjustmentDialog: function(settingName, value) {
                    const dialog = document.createElement('div');
                    dialog.style.cssText = `
                        position: fixed;
                        top: 50%;
                        left: 50%;
                        transform: translate(-50%, -50%);
                        background: #2a2a2a;
                        border: 2px solid #ff6b9d;
                        border-radius: 8px;
                        padding: 20px;
                        z-index: 10000;
                        color: white;
                        font-family: Arial, sans-serif;
                        max-width: 450px;
                    `;

                    const versionText = this.cacheStatus.modelVersion !== "auto" ? ` (${this.cacheStatus.modelVersion})` : "";

                    dialog.innerHTML = `
                        <h3>🎨 Illustrious Color Correction Adjustment</h3>
                        <p>You have a cached Illustrious${versionText} result. How would you like to apply the ${settingName} change?</p>
                        <div style="margin-top: 15px;">
                            <button id="adjust-cached" style="background: #ff6b9d; color: white; border: none; padding: 8px 16px; margin-right: 10px; border-radius: 4px; cursor: pointer;">
                                🎯 Adjust Cached Image (Fast)
                            </button>
                            <button id="regenerate" style="background: #ef4444; color: white; border: none; padding: 8px 16px; margin-right: 10px; border-radius: 4px; cursor: pointer;">
                                🔄 Regenerate (Slow)
                            </button>
                            <button id="cancel" style="background: #6b7280; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer;">
                                ❌ Cancel
                            </button>
                        </div>
                        <div style="margin-top: 10px; font-size: 12px; color: #aaa;">
                            💡 Fast adjustments work best for strength, saturation, and detail changes
                        </div>
                    `;

                    document.body.appendChild(dialog);

                    // Handle button clicks
                    dialog.querySelector('#adjust-cached').onclick = () => {
                        this.enablePreviewMode();
                        this.applyIllustriousPreviewAdjustment(settingName, value);
                        document.body.removeChild(dialog);
                    };

                    dialog.querySelector('#regenerate').onclick = () => {
                        this.clearCacheState();
                        // Force regeneration
                        const forceWidget = this.widgets.find(w => w.name === "force_recalculate");
                        if (forceWidget) {
                            forceWidget.value = true;
                        }
                        document.body.removeChild(dialog);
                    };

                    dialog.querySelector('#cancel').onclick = () => {
                        // Revert the setting change
                        const widget = this.widgets.find(w => w.name === settingName);
                        if (widget && this.cacheStatus.lastSettings) {
                            widget.value = this.cacheStatus.lastSettings[settingName];
                        }
                        document.body.removeChild(dialog);
                    };
                },

                applyIllustriousPreviewAdjustment: async function(settingName, value) {
                    console.log(`[Illustrious Smart Cache] Applying preview adjustment: ${settingName} = ${value}`);
                    
                    // Send real-time adjustment request to Illustrious backend
                    try {
                        const response = await api.fetchApi("/illustrious/smart_cache/adjust_preview", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({
                                node_id: this.id,
                                setting_name: settingName,
                                setting_value: value,
                                current_settings: this.getCurrentSettings()
                            })
                        });

                        const result = await response.json();
                        
                        if (result.preview_image) {
                            // Update preview without triggering workflow
                            this.updateIllustriousPreviewImage(result.preview_image);
                        }

                    } catch (error) {
                        console.error("[Illustrious Smart Cache] Preview adjustment failed:", error);
                    }
                },

                getCurrentSettings: function() {
                    const settings = {};
                    this.widgets.forEach(widget => {
                        if (widget.name && widget.value !== undefined) {
                            settings[widget.name] = widget.value;
                        }
                    });
                    return settings;
                },

                updateIllustriousPreviewImage: function(imageData) {
                    // This would update any connected preview nodes or UI
                    // without triggering the full workflow
                    console.log(`[Illustrious Smart Cache] Updated preview image for node ${this.id}`);
                    
                    // Trigger canvas redraw to show update indicator
                    this.setDirtyCanvas(true, true);
                },

                addIllustriousCacheControlWidgets: function() {
                    // Add "Apply & Commit" button for preview mode
                    if (!this.widgets.find(w => w.name === "Apply & Commit")) {
                        this.addWidget("button", "✅ Apply & Commit", null, () => {
                            this.commitIllustriousPreviewChanges();
                        });
                    }

                    // Add "Revert to Original" button
                    if (!this.widgets.find(w => w.name === "Revert to Original")) {
                        this.addWidget("button", "↩️ Revert to Original", null, () => {
                            this.revertToIllustriousOriginal();
                        });
                    }

                    // Add "Exit Preview Mode" button
                    if (!this.widgets.find(w => w.name === "Exit Preview Mode")) {
                        this.addWidget("button", "🚪 Exit Preview Mode", null, () => {
                            this.disablePreviewMode();
                        });
                    }

                    // Add model version quick-switch
                    const versions = ["auto", "v0.5", "v0.75", "v1.0"];
                    versions.forEach(version => {
                        const buttonName = `⚡ ${version === "auto" ? "Auto" : version}`;
                        if (!this.widgets.find(w => w.name === buttonName)) {
                            this.addWidget("button", buttonName, null, () => {
                                const versionWidget = this.widgets.find(w => w.name === "model_version");
                                if (versionWidget) {
                                    versionWidget.value = version;
                                    this.cacheStatus.modelVersion = version;
                                    if (this.cacheStatus.previewMode) {
                                        this.applyIllustriousPreviewAdjustment("model_version", version);
                                    }
                                }
                            });
                        }
                    });
                },

                commitIllustriousPreviewChanges: async function() {
                    console.log(`[Illustrious Smart Cache] Committing preview changes for node ${this.id}`);
                    
                    // Send commit request to backend
                    try {
                        await api.fetchApi("/illustrious/smart_cache/commit_preview", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({
                                node_id: this.id,
                                settings: this.getCurrentSettings()
                            })
                        });

                        // Update cache status
                        this.cacheStatus.lastSettings = this.getCurrentSettings();
                        this.disablePreviewMode();

                    } catch (error) {
                        console.error("[Illustrious Smart Cache] Commit failed:", error);
                    }
                },

                revertToIllustriousOriginal: async function() {
                    console.log(`[Illustrious Smart Cache] Reverting to original for node ${this.id}`);
                    
                    if (this.cacheStatus.lastSettings) {
                        // Restore previous settings
                        Object.entries(this.cacheStatus.lastSettings).forEach(([key, value]) => {
                            const widget = this.widgets.find(w => w.name === key);
                            if (widget) {
                                widget.value = value;
                                if (key === "model_version") {
                                    this.cacheStatus.modelVersion = value;
                                }
                            }
                        });
                    }
                    
                    this.disablePreviewMode();
                    this.setDirtyCanvas(true, true);
                }
            });

            // Setup the Illustrious smart cache widgets
            node.setupIllustriousSmartCacheWidgets();
        }
    },

    async setup() {
        // Listen for workflow execution events
    api.addEventListener("execution_start", ({ detail }) => {
            CACHE_STATE.workflowState = 'executing';
            CACHE_STATE.lastExecutionId = detail.execution_id || Date.now();
            console.log("[Illustrious Smart Cache] Workflow execution started");
        });

        api.addEventListener("execution_success", ({ detail }) => {
            CACHE_STATE.workflowState = 'completed';
            console.log("[Illustrious Smart Cache] Workflow execution completed successfully");
        });

        api.addEventListener("execution_error", ({ detail }) => {
            CACHE_STATE.workflowState = 'idle';
            console.log("[Illustrious Smart Cache] Workflow execution failed");
        });

    // Add global cache management to menu
        app.ui.settings.addSetting({
            id: "illustrious.cache_settings",
            name: "Illustrious Cache Settings",
            type: "combo",
            options: ["Auto", "Always Cache", "Never Cache"],
            defaultValue: "Auto",
            onChange: (value) => {
                console.log(`[Illustrious Smart Cache] Global cache setting changed to: ${value}`);
                // Update all active Illustrious nodes
                CACHE_STATE.activeNodes.forEach(node => {
                    const cacheWidget = node.widgets.find(w => w.name === "cache_mode");
                    if (cacheWidget) {
                        cacheWidget.value = value.toLowerCase().replace(" ", "_");
                    }
                });
            }
        });
    },

    async afterConfigureGraph() {
        // Reset cache state when loading new workflow
    console.log("[Illustrious Smart Cache] Workflow loaded, resetting cache state");
        CACHE_STATE.workflowState = 'idle';
        CACHE_STATE.lastExecutionId = null;
        
        // Clear preview caches
        CACHE_STATE.previewCache.clear();
        CACHE_STATE.settingsHistory.clear();
    }
});
})();

// --- Event listeners for Illustrious cache events ---
window.api?.addEventListener && window.api.addEventListener("illustrious.smart_cache_preview_update", ({ detail }) => {
    const node = app.graph.getNodeById(detail.node_id);
    if (node) {
    console.log(`[Illustrious Smart Cache] Received preview update for node ${detail.node_id}`);
        node.updateIllustriousPreviewImage(detail.preview_image);
        
        // Update cache status indicator
        if (node.cacheStatus) {
            node.cacheStatus.timestamp = detail.timestamp;
            node.cacheStatus.modelVersion = detail.model_version || node.cacheStatus.modelVersion;
        }
        
        node.setDirtyCanvas(true, true);
    }
});

window.api?.addEventListener && window.api.addEventListener("illustrious.smart_cache_commit", ({ detail }) => {
    const node = app.graph.getNodeById(detail.node_id);
    if (node && detail.success) {
    console.log(`[Illustrious Smart Cache] Preview committed successfully for node ${detail.node_id}`);
        
        // Show success indicator
    node.title = (node.title || "Illustrious Smart Cache Corrector") + " ✅";
        setTimeout(() => {
            node.title = node.title.replace(" ✅", "");
            node.setDirtyCanvas(true, true);
        }, 2000);
        
        node.setDirtyCanvas(true, true);
    }
});

// --- UTILITY FUNCTIONS ---
function isIllustriousNodeInExecutionPath(node, executionNodes) {
    // Check if this Illustrious node is part of the current execution
    return executionNodes && executionNodes.includes(node.id);
}

function getIllustriousUpstreamNodes(node) {
    // Get all nodes that feed into this Illustrious node
    const upstream = new Set();
    
    function traverse(currentNode) {
        if (currentNode.inputs) {
            currentNode.inputs.forEach(input => {
                if (input.link) {
                    const link = app.graph.links[input.link];
                    if (link) {
                        const sourceNode = app.graph.getNodeById(link.origin_id);
                        if (sourceNode && !upstream.has(sourceNode.id)) {
                            upstream.add(sourceNode.id);
                            traverse(sourceNode);
                        }
                    }
                }
            });
        }
    }
    
    traverse(node);
    return Array.from(upstream);
}

console.log("🧠 Illustrious Smart Cache extension loaded - intelligent caching active!");