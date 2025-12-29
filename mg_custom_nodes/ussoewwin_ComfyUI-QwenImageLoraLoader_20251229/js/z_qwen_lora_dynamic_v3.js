import { app } from "../../scripts/app.js";

console.log("★★★ z_qwen_lora_dynamic.js: Qwen Image LoRA Stack V3 ★★★");

const HIDDEN_TAG = "tschide";

app.registerExtension({
    name: "nunchaku.qwen_lora_dynamic_v3",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "NunchakuQwenImageLoraStackV3") {
            nodeType["@visibleLoraCount"] = { type: "number", default: 1, min: 1, max: 10, step: 1 };
        }
    },

        nodeCreated(node) {
        if (node.comfyClass !== "NunchakuQwenImageLoraStackV3") return;

        if (!node.properties) node.properties = {};
        if (node.properties["visibleLoraCount"] === undefined) node.properties["visibleLoraCount"] = 1;

        // Immediately hide lora_count widget if it exists
        const initialLoraCountWidget = node.widgets?.find(w => w.name === "lora_count");
        if (initialLoraCountWidget) {
            if (!initialLoraCountWidget.origType) {
                initialLoraCountWidget.origType = initialLoraCountWidget.type;
                initialLoraCountWidget.origComputeSize = initialLoraCountWidget.computeSize;
            }
            initialLoraCountWidget.type = HIDDEN_TAG;
            initialLoraCountWidget.computeSize = () => [0, -4];
        }

        node.cachedWidgets = {};
        let cacheReady = false;

        const initCache = () => {
            if (cacheReady) return;
            const all = [...node.widgets];
            
            // Cache lora_count widget (required for Python backend, but hidden in UI)
            const loraCountWidget = all.find(w => w.name === "lora_count");
            if (loraCountWidget) {
                node.cachedLoraCount = loraCountWidget;
                // Store original properties for restoration if needed
                if (!loraCountWidget.origType) {
                    loraCountWidget.origType = loraCountWidget.type;
                    loraCountWidget.origComputeSize = loraCountWidget.computeSize;
                }
                // Hide V1's lora_count widget using HIDDEN_TAG and computeSize
                loraCountWidget.type = HIDDEN_TAG;
                loraCountWidget.computeSize = () => [0, -4];
            }
            
            // Cache cpu_offload widget
            const cpuOffloadWidget = all.find(w => w.name === "cpu_offload");
            if (cpuOffloadWidget) {
                node.cachedCpuOffload = cpuOffloadWidget;
            }
            
            // Cache toggle_all widget
            const toggleAllWidget = all.find(w => w.name === "toggle_all");
            if (toggleAllWidget) {
                node.cachedToggleAll = toggleAllWidget;
            }
            
            for (let i = 1; i <= 10; i++) {
                const wEnabled = all.find(w => w.name === `enabled_${i}`);
                const wName = all.find(w => w.name === `lora_name_${i}`);
                const wStrength = all.find(w => w.name === `lora_strength_${i}`);
                if (wEnabled && wName && wStrength) {
                    node.cachedWidgets[i] = [wEnabled, wName, wStrength];
                    wEnabled.type = "toggle";
                    wName.type = "combo";
                    wStrength.type = "number";
                    if (wEnabled.computeSize) delete wEnabled.computeSize;
                    if (wName.computeSize) delete wName.computeSize;
                    if (wStrength.computeSize) delete wStrength.computeSize;
                }
            }
            cacheReady = true;
        };

        const ensureControlWidget = () => {
            const name = "🔢 LoRA Count";
            
            // Remove old button widgets
            for (let i = node.widgets.length - 1; i >= 0; i--) {
                const w = node.widgets[i];
                if (w.name === "🔢 Set LoRA Count" || w.type === "button") {
                    node.widgets.splice(i, 1);
                }
            }

            let w = node.widgets.find(x => x.name === name);
            if (!w) {
                const values = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"];
                w = node.addWidget("combo", name, "1", (v) => {
                    const num = parseInt(v);
                    if (!isNaN(num)) {
                        node.properties["visibleLoraCount"] = num;
                        // Sync with lora_count widget for Python backend
                        if (node.cachedLoraCount) {
                            node.cachedLoraCount.value = num;
                        }
                        node.updateLoraSlots();
                    }
                }, { values });
            }
            w.value = node.properties["visibleLoraCount"].toString();
            // Sync lora_count widget value
            if (node.cachedLoraCount) {
                node.cachedLoraCount.value = node.properties["visibleLoraCount"];
            }
            return w;
        };
        
        const ensureToggleAllWidget = () => {
            if (!node.cachedToggleAll) return null;
            
            // Store original callback if not already stored
            if (!node.cachedToggleAll.origCallback) {
                node.cachedToggleAll.origCallback = node.cachedToggleAll.callback;
            }
            
            // Override callback to sync all individual toggles
            node.cachedToggleAll.callback = (value) => {
                if (node.cachedToggleAll.origCallback) {
                    node.cachedToggleAll.origCallback(value);
                }
                // Sync all individual enabled toggles
                const count = parseInt(node.properties["visibleLoraCount"] || 1);
                for (let i = 1; i <= count; i++) {
                    const pair = node.cachedWidgets[i];
                    if (pair && pair[0]) {
                        pair[0].value = value;
                    }
                }
            };
            
            return node.cachedToggleAll;
        };
        
        node.updateLoraSlots = function() {
            if (!cacheReady) initCache();

            const count = parseInt(this.properties["visibleLoraCount"] || 1);
            const controlWidget = ensureControlWidget();
        
            // Physical widget reconstruction for clean layout (like Flux V2)
            this.widgets = [controlWidget];

            // Add lora_count widget (required for Python backend, but hidden in UI using HIDDEN_TAG)
            if (node.cachedLoraCount) {
                // Ensure it's hidden
                node.cachedLoraCount.type = HIDDEN_TAG;
                node.cachedLoraCount.computeSize = () => [0, -4];
                // Sync value for Python backend
                node.cachedLoraCount.value = count;
                // Add to widgets array (for Python backend, but hidden in UI)
                this.widgets.push(node.cachedLoraCount);
            }

            // Add toggle_all widget (if exists)
            const toggleAllWidget = ensureToggleAllWidget();
            if (toggleAllWidget) {
                this.widgets.push(toggleAllWidget);
            }

            // Add cpu_offload widget from cache (required for Python backend)
            if (node.cachedCpuOffload) {
                this.widgets.push(node.cachedCpuOffload);
            }

            // Add only visible LoRA slots (non-visible widgets are removed from array)
            // Each slot: [enabled_toggle, lora_name, lora_strength]
            for (let i = 1; i <= count; i++) {
                const pair = this.cachedWidgets[i];
                if (pair && pair.length >= 3) {
                    this.widgets.push(pair[0]); // enabled toggle
                    this.widgets.push(pair[1]); // lora_name
                    this.widgets.push(pair[2]); // lora_strength
                }
            }

            // Height calculation
            const HEADER_H = 60;
            const SLOT_H = 54;
            const TOGGLE_ALL_H = toggleAllWidget ? 40 : 0;
            const CPU_OFFLOAD_H = node.cachedCpuOffload ? 40 : 0;
            const PADDING = 20;
            const targetH = HEADER_H + TOGGLE_ALL_H + CPU_OFFLOAD_H + (count * SLOT_H) + PADDING;
            
            this.setSize([this.size[0], targetH]);
            
            if (app.canvas) app.canvas.setDirty(true, true);
        };

        node.onPropertyChanged = function(property, value) {
            if (property === "visibleLoraCount") {
                const w = this.widgets.find(x => x.name === "🔢 LoRA Count");
                if (w) w.value = value.toString();
                this.updateLoraSlots();
            }
        };
        
        // Restore UI on configure
        const origOnConfigure = node.onConfigure;
        node.onConfigure = function() {
             if (origOnConfigure) origOnConfigure.apply(this, arguments);
             setTimeout(() => node.updateLoraSlots(), 100);
        };

        setTimeout(() => {
            initCache();
            node.updateLoraSlots();
        }, 100);
    }
});

