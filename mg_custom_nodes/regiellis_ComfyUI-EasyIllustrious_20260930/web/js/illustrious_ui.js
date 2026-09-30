// ==========================================================================
//  Illustrious Enhancement Suite
// ==========================================================================
//
// Description:
// Comprehensive enhancement suite for Illustrious nodes including:
// - Real-time search suggestions for characters/artists
// - KSampler optimizations and tooltips
// - Interactive help documentation panel
//
// Key Features:
// - Real-time search suggestions as you type
// - Auto-completion from database
// - Comprehensive parameter tooltips
// - Interactive help documentation
// - Visual parameter validation
//
// Version: 5.0.0 - Enhanced UI with Canvas Controls & Tooltips
//
// ==========================================================================
// Note: do not cache app/api at module load; always reference window.app/window.api dynamically
try { console.log("[Illustrious] Enhanced UI script evaluating…"); } catch {}

// --- ENHANCED UI IMPORTS ---
// Load enhanced UI components dynamically
let TooltipManager, CanvasRenderer, WeightControl, createWeightControl, globalLayerManager;

function __illustriousEnhancementsEnabled() {
    try {
        const v = localStorage.getItem('illustrious_enhanced_ui');
        return v && v.toLowerCase() === 'on';
    } catch {
        return false;
    }
}

async function loadEnhancedComponents() {
    try {
        const tooltipModule = await import('./ui_components/tooltip_manager.js');
        const rendererModule = await import('./ui_components/canvas_renderer.js');
        const weightModule = await import('./ui_components/weight_control.js');
        const layerModule = await import('./ui_components/layer_manager.js');
        
        TooltipManager = tooltipModule.TooltipManager;
        CanvasRenderer = rendererModule.CanvasRenderer;
        WeightControl = weightModule.WeightControl;
        createWeightControl = weightModule.createWeightControl;
        globalLayerManager = layerModule.globalLayerManager;
        
        console.log("[Illustrious] Enhanced UI components loaded with layer management");
        return true;
    } catch (e) {
        console.warn("[Illustrious] Enhanced UI components not available, using fallback mode:", e.message);
        return false;
    }
}

// --- SEARCH CACHE ---
const searchCache = new Map();
const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

// Track the last interacted widget per data type so global actions (like load_more) can target it
const __illustrious_lastWidgetByType = Object.create(null);

// Helper: resolve the proper registerExtension function across frontend versions
function __getRegisterExtension() {
    const app = window.app;
    if (!app) return null;
    if (typeof app.registerExtension === 'function') {
        return { fn: app.registerExtension, ctx: app };
    }
    const mgr = app.extensionManager;
    if (mgr && typeof mgr.registerExtension === 'function') {
        return { fn: mgr.registerExtension, ctx: mgr };
    }
    return null;
}

// --- NODE REGISTRATION ---
// Defer registration until app is available
(function registerIllustriousMain(){
    if (window.__illustrious_ui_main_registered) return;
    const reg = __getRegisterExtension();
    if (!window.app || !reg) {
        if (!window.__illustrious_ui_main_wait_logged) {
            window.__illustrious_ui_main_wait_logged = true;
        try { console.log("[Illustrious] Waiting for ComfyUI extension registry (main)…"); } catch {}
        }
        setTimeout(registerIllustriousMain, 50); return; }
    window.__illustrious_ui_main_registered = true;
    const __reg = __getRegisterExtension();
    __reg.fn.call(__reg.ctx, {
    name: "itsjustregi.Illustrious",
    
    async setup() {
        console.log("[Illustrious] Enhancement Suite v5.0 - initializing...");

        let enhancedUILoaded = false;
        if (__illustriousEnhancementsEnabled()) {
            console.log('[Illustrious] Enhanced UI flag ON — loading modules');
            enhancedUILoaded = await loadEnhancedComponents();
            try {
                await import('./illustrious_ui_enhanced.js');
                console.log('[Illustrious] Canvas-based enhanced UI module loaded');
            } catch (e) {
                console.log('[Illustrious] Canvas-based enhanced UI module not available:', e?.message || e);
            }
        } else {
            console.log('[Illustrious] Enhanced UI flag OFF — running standard UI only');
        }

        if (enhancedUILoaded) {
            console.log("[Illustrious] ✨ Enhanced UI with canvas controls and tooltips enabled");
        }
        
        // Always load lightweight prompt canvas UI (non-invasive)
        try {
            await import('./prompt_canvas_ui.js');
            console.log('[Illustrious] PromptCanvasUI loaded');
        } catch (e) {
            console.log('[Illustrious] PromptCanvasUI not available:', e?.message || e);
        }

        // Load KSampler inline preview
        try {
            await import('./ksampler_inline_preview.js');
            console.log('[Illustrious] KSampler Inline Preview loaded');
        } catch (e) {
            console.log('[Illustrious] KSampler Inline Preview not available:', e?.message || e);
        }

        // Setup advanced UI enhancements
        setupWorkflowDetection();
        setupContextMenus();
        setupQueueButton();
    setupLoadMoreListener();

        // Enhance existing nodes (after workflow load)
        setTimeout(applySearchToExistingNodes, 300);
        setTimeout(applySearchToExistingNodes, 1500);
    },
    
    // Bottom panel help documentation
    bottomPanelTabs: [
        {
            id: "illustrious-help",
            title: "Illustrious Help",
            type: "custom",
            icon: "🎨",
            render: (el) => {
                createHelpPanel(el);
            }
        }
    ],
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const nodesWithSearch = [
            "IllustriousPrompt", "IllustriousCharacters", "IllustriousArtists",
            "IllustriousE621Characters", "IllustriousE621Artists"
        ];
        
        const nodesWithKSamplerEnhancements = [
            "IllustriousKSamplerPro", "IllustriousKSamplerPresets",
            "IllustriousMultiPassSampler", "IllustriousTriplePassSampler"
        ];
        
        const nodesWithSchedulerEnhancements = [
            "IllustriousScheduler"
        ];
        
        const nodesWithCLIPEnhancements = [
            "IllustriousCLIPTextEncoder", "IllustriousNegativeCLIPEncoder"
        ];
        
        const nodesWithLatentEnhancements = [
            "IllustriousEmptyLatentImage", "IllustriousLatentUpscale"
        ];
        
        const nodesWithColorCorrectorEnhancements = [
            "IllustriousColorCorrector"
        ];
        
        const shouldEnhance = nodesWithSearch.includes(nodeData.name) || 
                             nodesWithKSamplerEnhancements.includes(nodeData.name) ||
                             nodesWithSchedulerEnhancements.includes(nodeData.name) ||
                             nodesWithCLIPEnhancements.includes(nodeData.name) ||
                             nodesWithLatentEnhancements.includes(nodeData.name) ||
                             nodesWithColorCorrectorEnhancements.includes(nodeData.name);
        
        if (!shouldEnhance) return;
        
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        
        nodeType.prototype.onNodeCreated = function() {
            const result = onNodeCreated?.apply(this, arguments);
            
            // Handle search functionality for character/artist nodes
            if (nodesWithSearch.includes(nodeData.name)) {
                this.widgets?.forEach(widget => {
                    const searchFields = [
                        "Character", "Artist", "E621 Character", "E621 Artist",
                        "Character (Base)", "Character (Secondary)", "Character (Tertiary)",
                        "Character (Quaternary)", "Character (Quinary)",
                        "Artist (Base)", "Artist (Secondary)", "Artist (Tertiary)", 
                        "Artist (Quaternary)", "Artist (Quinary)", "Artist (Primary)"
                    ];
                    
                    if (searchFields.includes(widget.name) && (widget.type === "text" || widget.type === "STRING")) {
                        setupSearchSuggestions.call(this, widget);
                    }
                    
                    // Setup enhanced weight controls only if enabled
                    if (createWeightControl && __illustriousEnhancementsEnabled() && widget.name.toLowerCase().includes('weight')) {
                        try {
                            const weightControl = createWeightControl(this, widget);
                            if (weightControl) {
                                console.log(`[Illustrious] Enhanced weight control attached to ${widget.name}`);
                            }
                        } catch (e) {
                            console.warn("[Illustrious] Failed to create weight control:", e.message);
                        }
                    }
                });
            }
            
            // Handle KSampler enhancements
            if (nodesWithKSamplerEnhancements.includes(nodeData.name)) {
                setupKSamplerEnhancements.call(this, nodeData.name);
                
                // Add enhanced weight controls for KSampler parameters
                if (createWeightControl && __illustriousEnhancementsEnabled()) {
                    this.widgets?.forEach(widget => {
                        const samplerWeightFields = ['cfg', 'denoise', 'steps'];
                        if (samplerWeightFields.includes(widget.name.toLowerCase()) && 
                            (widget.type === 'number' || widget.type === 'slider')) {
                            try {
                                const weightControl = createWeightControl(this, widget);
                                if (weightControl) {
                                    console.log(`[Illustrious] Enhanced control for ${widget.name} in ${nodeData.name}`);
                                }
                            } catch (e) {
                                console.warn(`[Illustrious] Failed to enhance ${widget.name}:`, e.message);
                            }
                        }
                    });
                }
            }
            
            // Add context menu enhancements for all Illustrious nodes
            if (shouldEnhance) {
                setupNodeContextMenu.call(this, nodeData.name);
            }
            
            return result;
        };
    },

    // Called for nodes when a workflow is loaded
    loadedGraphNode(node /*, app */) {
        try {
            const targetNodes = ["IllustriousPrompt", "IllustriousCharacters", "IllustriousArtists", "IllustriousE621Characters", "IllustriousE621Artists"];
            const ksamplerNodes = ["IllustriousKSamplerPro", "IllustriousKSamplerPresets",
                           "IllustriousMultiPassSampler", "IllustriousTriplePassSampler"];
            const schedulerNodes = ["IllustriousScheduler"];
            const clipNodes = ["IllustriousCLIPTextEncoder", "IllustriousNegativeCLIPEncoder"];
            const latentNodes = ["IllustriousEmptyLatentImage", "IllustriousLatentUpscale"];
            const colorCorrectorNodes = ["IllustriousColorCorrector"];

            if (!node?.comfyClass) return;

            if (targetNodes.includes(node.comfyClass)) {
                console.log(`[Illustrious] ${node.comfyClass} loaded; attaching search enhancement`);
                // Attach search suggestions to relevant widgets
                node.widgets?.forEach(w => {
                    const searchFields = [
                        "Character", "Artist", "E621 Character", "E621 Artist",
                        "Character (Base)", "Character (Secondary)", "Character (Tertiary)",
                        "Character (Quaternary)", "Character (Quinary)",
                        "Artist (Base)", "Artist (Secondary)", "Artist (Tertiary)",
                        "Artist (Quaternary)", "Artist (Quinary)", "Artist (Primary)"
                    ];
                    if (searchFields.includes(w?.name) && (w.type === "text" || w.type === "STRING") && !w._hasSearchEnhancement) {
                        setupSearchSuggestions.call(node, w);
                    }
                });
            } else if (ksamplerNodes.includes(node.comfyClass)) {
                console.log(`[Illustrious] ${node.comfyClass} loaded with KSampler UI enhancements`);
            } else if (schedulerNodes.includes(node.comfyClass)) {
                console.log(`[Illustrious] ${node.comfyClass} Scheduler loaded with enhancements`);
            } else if (clipNodes.includes(node.comfyClass)) {
                console.log(`[Illustrious] ${node.comfyClass} CLIP encoder loaded with enhancements`);
            } else if (latentNodes.includes(node.comfyClass)) {
                console.log(`[Illustrious] ${node.comfyClass} Latent node loaded with enhancements`);
            } else if (colorCorrectorNodes.includes(node.comfyClass)) {
                console.log(`[Illustrious] ${node.comfyClass} Color Corrector loaded with enhancements`);
            }
        } catch (e) {
            console.warn("[Illustrious] loadedGraphNode hook error:", e?.message || e);
        }
    },
    
    async nodeCreated(node) {
    const targetNodes = ["IllustriousPrompt", "IllustriousCharacters", "IllustriousArtists", "IllustriousE621Characters", "IllustriousE621Artists"];
    const ksamplerNodes = ["IllustriousKSamplerPro", "IllustriousKSamplerPresets",
                   "IllustriousMultiPassSampler", "IllustriousTriplePassSampler"];
    const schedulerNodes = ["IllustriousScheduler"];
    const clipNodes = ["IllustriousCLIPTextEncoder", "IllustriousNegativeCLIPEncoder"];
    const latentNodes = ["IllustriousEmptyLatentImage", "IllustriousLatentUpscale"];
        const colorCorrectorNodes = ["IllustriousColorCorrector"];
        
        if (targetNodes.includes(node?.comfyClass)) {
            console.log(`[Illustrious] ${node.comfyClass} node created with search enhancement`);
        } else if (ksamplerNodes.includes(node?.comfyClass)) {
            console.log(`[Illustrious] ${node.comfyClass} KSampler node created with UI enhancements`);
        } else if (schedulerNodes.includes(node?.comfyClass)) {
            console.log(`[Illustrious] ${node.comfyClass} Scheduler node created with enhancements`);
        } else if (clipNodes.includes(node?.comfyClass)) {
            console.log(`[Illustrious] ${node.comfyClass} CLIP encoder node created with enhancements`);
        } else if (latentNodes.includes(node?.comfyClass)) {
            console.log(`[Illustrious] ${node.comfyClass} Latent node created with enhancements`);
        } else if (colorCorrectorNodes.includes(node?.comfyClass)) {
            console.log(`[Illustrious] ${node.comfyClass} Color Corrector node created with enhancements`);
        }
    }
    });
})();

// --- KSAMPLER ENHANCEMENT FUNCTIONS ---
function setupKSamplerEnhancements(nodeType) {
    console.log(`[Illustrious] Setting up KSampler enhancements for ${nodeType}`);
    
    // Setup enhanced tooltips if available
    if (TooltipManager) {
        try {
            this.enhancedTooltips = new TooltipManager();
            setupEnhancedKSamplerTooltips.call(this, this.enhancedTooltips);
            console.log(`[Illustrious] Enhanced tooltips enabled for ${nodeType}`);
        } catch (e) {
            console.warn("[Illustrious] Failed to setup enhanced tooltips:", e.message);
        }
    }
    
    // Add tooltips and UI improvements for KSampler nodes
    this.widgets?.forEach(widget => {
        // Add comprehensive tooltips for all KSampler types
        addKSamplerTooltips.call(this, widget);
        
        // Add visual indicators for optimal settings
        addKSamplerVisualIndicators.call(this, widget, nodeType);
        
        // Add parameter validation
        addKSamplerValidation.call(this, widget, nodeType);
        
        // Add multi-pass specific tooltips
        addMultiPassTooltips.call(this, widget, nodeType);
        
        // Add scheduler specific tooltips
        addSchedulerTooltips.call(this, widget, nodeType);
        
        // Add CLIP encoder specific tooltips
        addCLIPEncoderTooltips.call(this, widget, nodeType);
        
        // Add latent node specific tooltips  
        addLatentTooltips.call(this, widget, nodeType);
        
        // Add color corrector specific tooltips
        addColorCorrectorTooltips.call(this, widget, nodeType);
    });
    
    // Add node-level enhancements
    addKSamplerNodeEnhancements.call(this, nodeType);
}

function addKSamplerTooltips(widget) {
    // Base tooltips that apply to all KSampler variants
    const baseTooltips = {
        "model": "The AI model to use for generation. Connect from a model loader or checkpoint loader.",
        "seed": "Random seed for noise generation. Same seed = same result. 0 = random seed each time.",
    "steps": "Number of denoising steps. More steps = higher quality but slower generation. Optimal: 20-35 for Illustrious.",
    "cfg": "Classifier-Free Guidance scale. Controls how closely the AI follows your prompt.\n• 4.0-6.5: Optimal range for Illustrious\n• Higher: More prompt adherence, risk of oversaturation\n• Lower: More creative freedom, risk of ignoring prompt",
        "sampler_name": "Denoising algorithm used during generation:\n• euler_ancestral: Fast, good quality (recommended for Illustrious)\n• dpm_2s_ancestral: High quality, slightly slower\n• dpm_fast: Very fast, good for testing\n• dpm_adaptive: Adaptive step size, experimental",
        "scheduler": "Controls the noise schedule during denoising:\n• normal: Standard linear schedule\n• karras: Better for high resolution (recommended for 1536px+)\n• exponential: Steep curve, experimental\n• sgm_uniform: Uniform distribution",
        "positive": "Positive conditioning (what you want). Connect from CLIP Text Encode with your prompt.",
        "negative": "Negative conditioning (what you don't want). Connect from CLIP Text Encode with negative prompt.",
        "latent_image": "Input latent image to denoise. Connect from Empty Latent Image or VAE Encode.",
        "denoise": "Denoising strength. 1.0 = full denoising, 0.0 = no denoising. Use <1.0 for img2img workflows.",
    "Model Type": "Select your specific model version for optimized parameters:\n• Auto Detect: Automatically optimize based on model\n• Illustrious versions: Optimize for Illustrious model behavior",
    "Auto CFG": "Automatically adjust CFG to optimal range for Illustrious (4.0-6.5). Prevents oversaturation and foggy results.",
        "Resolution Adaptive": "Automatically scale steps and CFG based on image resolution. Higher resolution = more steps, lower CFG.",
        "Advanced Mode": "Enable advanced warnings and parameter suggestions in the console. Helpful for learning optimal settings."
    };
    
    // Advanced-only tooltips
    const advancedTooltips = {
        "Quality Mode": "Preset configurations for different use cases:\n• Standard: Balanced speed/quality\n• High Quality: Slower but better results\n• Speed: Fast generation for testing\n• Experimental: Cutting-edge settings",
    "CFG Schedule": "Dynamic CFG scheduling during generation:\n• Constant: Fixed CFG throughout\n• Linear Decay: Gradually reduce CFG over time\n• Cosine Decay: Smooth CFG curve reduction\n• Illustrious Adaptive: Optimized curve for Illustrious models",
        "Dynamic Thresholding": "Prevents oversaturation by clamping extreme values based on percentile thresholds. Useful for high CFG values or problematic prompts.",
        "DT Percentile": "Percentile threshold for dynamic thresholding (0.995 = 99.5th percentile).\n• Higher values: Allow more extreme values\n• Lower values: More conservative clamping",
        "DT Clamp Range": "Multiplier for clamping range after threshold calculation.\n• 1.0: Standard clamping\n• <1.0: More aggressive clamping\n• >1.0: More lenient clamping",
        "Danbooru Boost": "Enhances recognition of common Danbooru tags that Illustrious was specifically trained on. Improves quality for anime/manga content.",
        "Resolution Mode": "Resolution-specific optimizations:\n• Auto: Detect optimal settings from image size\n• Manual modes: Override with specific resolution optimizations\n• Higher resolutions need different CFG and step counts",
        "Anime Enhancement": "Applies anime/illustration-specific optimizations during and after sampling. Preserves line art quality and color vibrancy.",
        "Ancestral Eta": "Controls noise scaling for ancestral samplers.\n• 1.0: Standard ancestral behavior\n• <1.0: Less randomness, more deterministic\n• >1.0: More randomness, more variation"
    };
    
    // Simple sampler tooltips
    const simpleTooltips = {
    "Preset": "One-click generation presets:\n• Balanced: Good speed/quality balance\n• High Quality: Best results, slower\n• Fast: Quick generation for testing\n• Portrait Focus: Optimized for character portraits\n• Landscape: Better for backgrounds/scenes\n• Anime Style: Enhanced for anime/manga content"
    };
    
    // Apply appropriate tooltips based on widget name
    const allTooltips = { ...baseTooltips, ...advancedTooltips, ...simpleTooltips };
    
    if (allTooltips[widget.name]) {
        // Try different ways to set tooltip depending on widget type
        if (widget.element) {
            widget.element.title = allTooltips[widget.name];
            if (widget.element.setAttribute) {
                widget.element.setAttribute('title', allTooltips[widget.name]);
            }
        }
        
        // Also try to set on the widget itself
        if (widget.tooltip !== undefined) {
            widget.tooltip = allTooltips[widget.name];
        }
        
        // For dropdowns, try to enhance the options
        if (widget.type === 'combo' && widget.options && widget.options.values) {
            // Add tooltip to the widget container
            setTimeout(() => {
                const widgetElement = document.querySelector(`[data-widget-name="${widget.name}"]`);
                if (widgetElement) {
                    widgetElement.title = allTooltips[widget.name];
                }
            }, 100);
        }
    }
}

function addMultiPassTooltips(widget, nodeType) {
    // Multi-pass sampler specific tooltips
    const multiPassTooltips = {
        "structure_steps": "Steps for the initial structure building pass. Fewer steps focus on overall composition (8-20).",
        "structure_cfg": "CFG for structure pass. Higher values (6.0-8.0) help establish strong composition.",
        "structure_denoise": "Denoising strength for structure pass. Higher values (0.7-1.0) for more dramatic changes.",
        "detail_steps": "Steps for detail refinement pass. More steps (15-35) for finer detail work.",
        "detail_cfg": "CFG for detail pass. Lower values (4.0-6.0) prevent over-processing of fine details.",
        "detail_denoise": "Denoising for detail pass. Lower values (0.4-0.7) preserve structure while adding detail.",
        "comp_steps": "Steps for initial composition layout. Very few steps (4-12) to establish basic structure.",
        "comp_cfg": "CFG for composition pass. High values (6.0-9.0) to strongly establish layout.",
        "comp_denoise": "Denoising for composition. High values (0.8-1.0) for major structural changes.",
        "struct_steps": "Steps for structure building. Medium step count (8-20) to develop forms.",
        "struct_cfg": "CFG for structure development. Medium values (5.0-7.0) for balanced development.",
        "struct_denoise": "Structure pass denoising. Medium values (0.6-0.8) to build on composition.",
        "use_different_samplers": "Use different samplers for each pass. Recommended for optimal results.",
        "adaptive_transition": "Automatically adjust parameters based on previous pass results.",
        "preserve_composition": "Maintain composition integrity across passes.",
    "model_version": "Select Illustrious model version for parameter optimization.",
        "progressive_cfg_decay": "Gradually reduce CFG across passes for natural development.",
        "adaptive_steps": "Adjust step counts based on image complexity analysis."
    };
    
    // Apply multi-pass tooltips if node is a multi-pass sampler
    if ((nodeType === "IllustriousMultiPassSampler" || nodeType === "IllustriousTriplePassSampler") && 
        multiPassTooltips[widget.name]) {
        
        if (widget.element) {
            widget.element.title = multiPassTooltips[widget.name];
            if (widget.element.setAttribute) {
                widget.element.setAttribute('title', multiPassTooltips[widget.name]);
            }
        }
        
        if (widget.tooltip !== undefined) {
            widget.tooltip = multiPassTooltips[widget.name];
        }
    }
}

function addSchedulerTooltips(widget, nodeType) {
    // Scheduler-specific tooltips
    const schedulerTooltips = {
        "steps": "Number of denoising steps. More steps generally improve quality but increase generation time (16-40 recommended).",
    "scheduler_type": "Type of scheduling algorithm: aware (custom optimized), adaptive_content (adapts to complexity), distance (distance-based), cosine_annealing (smooth curve), hybrid_optimized (combines multiple approaches).",
        "model_version": "Illustrious model version for parameter optimization. Auto-detects if possible.",
        "complexity_bias": "Bias towards complex (1.0) or simple (0.0) content scheduling. 0.5 is neutral.",
        "distance_power": "Power curve for distance scheduler. Higher values emphasize early denoising steps (0.5-2.0).",
        "content_analysis": "Describe your prompt content for optimized scheduling (e.g., 'character portrait', 'complex scene', 'anime illustration').",
        "adaptive_mode": "Enable adaptive parameter adjustments based on content analysis and model version."
    };
    
    // Apply scheduler tooltips if node is a scheduler
    if (nodeType === "IllustriousScheduler" && schedulerTooltips[widget.name]) {
        
        if (widget.element) {
            widget.element.title = schedulerTooltips[widget.name];
            if (widget.element.setAttribute) {
                widget.element.setAttribute('title', schedulerTooltips[widget.name]);
            }
        }
        
        if (widget.tooltip !== undefined) {
            widget.tooltip = schedulerTooltips[widget.name];
        }
    }
}

function addCLIPEncoderTooltips(widget, nodeType) {
    // CLIP encoder-specific tooltips
    const clipEncoderTooltips = {
        "text": "Input text/prompt to encode. Supports both natural language and Danbooru tags.",
        "model_version": "Illustrious model version for optimized encoding. Auto-detects from model if possible.",
        "enable_tag_optimization": "Enable automatic tag format optimization for Illustrious compatibility.",
        "enable_padding_fix": "Fix padding token issues that can affect encoding quality.",
        "auto_tag_ordering": "Automatically reorder tags for optimal tag hierarchy (Quality → Character → Composition).",
        "danbooru_weighting": "Apply Danbooru-based tag weighting for better results with anime/illustration content.",
        "clip_skip": "CLIP layer to stop at. clip_skip=2 is a common sweet spot. Higher values = more abstract, lower = more literal.",
        "quality_boost": "Boost weight for quality-related tags (masterpiece, best_quality, etc.). 1.0 = no change.",
        "character_boost": "Boost weight for character-related tags (1girl, character_features, etc.). 1.0 = no change.",
        "composition_boost": "Boost weight for composition tags (portrait, upper_body, etc.). 1.0 = no change.",
        "enable_hybrid_mode": "Enable hybrid natural language + tag processing.",
        "preset": "Negative prompt preset optimized for different use cases.",
        "strength_multiplier": "Overall strength multiplier for negative prompt. Higher values = stronger negative guidance."
    };
    
    // Apply CLIP encoder tooltips if node is a CLIP encoder
    if ((nodeType === "IllustriousCLIPTextEncoder" || nodeType === "IllustriousNegativeCLIPEncoder") && 
        clipEncoderTooltips[widget.name]) {
        
        if (widget.element) {
            widget.element.title = clipEncoderTooltips[widget.name];
            if (widget.element.setAttribute) {
                widget.element.setAttribute('title', clipEncoderTooltips[widget.name]);
            }
        }
        
        if (widget.tooltip !== undefined) {
            widget.tooltip = clipEncoderTooltips[widget.name];
        }
    }
}

function addLatentTooltips(widget, nodeType) {
    // Latent node specific tooltips
    const latentTooltips = {
        'IllustriousEmptyLatentImage': {
            'width': 'Image width in pixels. Use multiples of 64, optimally 128 for high-res.',
            'height': 'Image height in pixels. Use multiples of 64, optimally 128 for high-res.',
            'batch_size': 'Number of images to generate simultaneously.',
            'model_version': 'Illustrious model version. Auto-detects optimal version based on resolution.',
            'optimization_mode': 'Optimization strategy: auto (balanced), quality (best results), speed (faster), compatibility (stable).',
            'aspect_ratio_mode': 'Aspect ratio optimization for anime content. Auto-detects from dimensions.',
            'noise_pattern': 'Initial noise pattern optimized for anime/illustration results.',
            'resolution_tier': 'Resolution tier for optimization. Higher tiers use more advanced noise patterns.',
            'enable_native_resolution': 'Optimize dimensions for Illustrious native training resolutions.',
            'seed': 'Random seed for reproducible noise generation. 0 for random.'
        },
        'IllustriousLatentUpscale': {
            'samples': 'Input latent image to upscale.',
            'upscale_method': 'Interpolation method. Bilinear recommended for anime content.',
            'width': 'Target width. Will be rounded to latent-compatible size (÷8).',
            'height': 'Target height. Will be rounded to latent-compatible size (÷8).',
            'model_version': 'Illustrious version for optimization.',
            'preserve_anime_details': 'Apply anime-specific detail preservation during upscaling.',
            'crop': 'Cropping mode if dimensions don\'t match exactly.',
            'noise_augmentation': 'Add slight noise for detail enhancement. 0.05-0.1 recommended for anime.'
        }
    };
    
    // Apply latent tooltips if node is a latent node
    const nodeTooltips = latentTooltips[nodeType];
    if (nodeTooltips && nodeTooltips[widget.name]) {
        
        if (widget.element) {
            widget.element.title = nodeTooltips[widget.name];
            if (widget.element.setAttribute) {
                widget.element.setAttribute('title', nodeTooltips[widget.name]);
            }
        }
        
        if (widget.tooltip !== undefined) {
            widget.tooltip = nodeTooltips[widget.name];
        }
    }
}

function addColorCorrectorTooltips(widget, nodeType) {
    // Color corrector specific tooltips
    const colorCorrectorTooltips = {
        'IllustriousColorCorrector': {
            'images': 'Input images to color correct.',
            'model_version': 'Illustrious model version for optimized correction. Auto-detects from image characteristics.',
            'correction_strength': 'Overall correction strength. 0.0 = no correction, 2.0 = maximum correction.',
            'auto_detect_issues': 'Automatically detect and fix common issues like oversaturation.',
            'preserve_anime_aesthetic': 'Preserve anime/illustration characteristics during correction.',
            'fix_oversaturation': 'Fix oversaturation issues in outputs.',
            'enhance_details': 'Enhance image details while preserving anime aesthetic.',
            'balance_colors': 'Remove color casts and balance colors for natural appearance.',
            'adjust_contrast': 'Adjust contrast based on model version characteristics.',
            'custom_preset': 'Apply preset optimizations: character_portrait (anime characters), detailed_scene (complex backgrounds), soft_illustration (gentle correction), vibrant_anime (enhanced colors), natural_colors (realistic look).',
            'show_corrections': 'Display detailed correction report in output.'
        }
    };
    
    // Apply color corrector tooltips if node is a color corrector
    const nodeTooltips = colorCorrectorTooltips[nodeType];
    if (nodeTooltips && nodeTooltips[widget.name]) {
        
        if (widget.element) {
            widget.element.title = nodeTooltips[widget.name];
            if (widget.element.setAttribute) {
                widget.element.setAttribute('title', nodeTooltips[widget.name]);
            }
        }
        
        if (widget.tooltip !== undefined) {
            widget.tooltip = nodeTooltips[widget.name];
        }
    }
}

// --- ENHANCED UI TOOLTIP SETUP ---
function setupEnhancedKSamplerTooltips(tooltipManager) {
    // Import tooltip presets
    if (typeof TOOLTIP_PRESETS !== 'undefined') {
        // Register enhanced tooltips for KSampler parameters
        tooltipManager.registerBatch({
            cfg: TOOLTIP_PRESETS.cfg,
            steps: TOOLTIP_PRESETS.steps,
            seed: {
                title: "Seed Control",
                text: "Random seed for generation consistency.\n• Same seed = identical results\n• -1 or 0 = random each time\n• Use fixed seeds for comparisons"
            },
            sampler_name: {
                title: "Sampling Algorithm",
                text: "Controls how noise is removed during generation:\n• euler_ancestral: Fast, good for most cases\n• dpm_2s_ancestral: Higher quality, slower\n• dpm_fast: Very fast, good for testing\n• Choose based on speed vs quality needs"
            },
            scheduler: {
                title: "Noise Schedule",
                text: "Controls how noise is reduced over time:\n• normal: Standard linear reduction\n• karras: Better for high resolution\n• exponential: Steep noise curve\n• sgm_uniform: Uniform distribution"
            },
            denoise: {
                title: "Denoising Strength",
                text: "Amount of noise to remove:\n• 1.0: Full denoising (text-to-image)\n• 0.7-0.9: Strong changes (img2img)\n• 0.3-0.6: Moderate changes\n• 0.1-0.3: Subtle refinements"
            }
        });
        
        console.log("[Illustrious] Enhanced tooltips registered for KSampler parameters");
    }
}

function addKSamplerVisualIndicators(widget, nodeType) {
    // Add visual indicators for optimal parameter ranges
    const optimalRanges = {
        "cfg": { min: 4.0, max: 6.5, ideal: 5.0 },
        "steps": { min: 20, max: 35, ideal: 24 },
    };
    
    if (widget.name === "cfg" || widget.name === "steps") {
        const range = optimalRanges[widget.name];
        if (range && widget.element) {
            // Add a subtle color indicator
            const originalCallback = widget.callback;
            widget.callback = function(value) {
                if (originalCallback) originalCallback.call(this, value);
                
                // Visual feedback for parameter ranges
                if (widget.element && widget.element.style) {
                    if (value >= range.min && value <= range.max) {
                        widget.element.style.boxShadow = "0 0 2px #4CAF50"; // Green for optimal
                    } else if (Math.abs(value - range.ideal) > (range.max - range.min)) {
                        widget.element.style.boxShadow = "0 0 2px #FF9800"; // Orange for suboptimal
                    } else {
                        widget.element.style.boxShadow = ""; // Default
                    }
                }
            };
        }
    }
}

function addKSamplerValidation(widget, nodeType) {
    // Add parameter validation and warnings
    if (widget.name === "sampler_name") {
        const originalCallback = widget.callback;
        widget.callback = function(value) {
            if (originalCallback) originalCallback.call(this, value);
            
            // Warn about suboptimal samplers for Illustrious
            const suboptimalSamplers = ["ddim", "plms", "k_lms"];
            if (suboptimalSamplers.includes(value)) {
                console.warn(`[Illustrious] Warning: ${value} may not be optimal for Illustrious models. Consider euler_ancestral or dpm_2s_ancestral.`);
            }
        };
    }
    
    // CFG validation
    if (widget.name === "cfg") {
        const originalCallback = widget.callback;
        widget.callback = function(value) {
            if (originalCallback) originalCallback.call(this, value);
            
            if (value > 8.0) {
                console.warn(`[Illustrious] CFG ${value} may cause oversaturation. Consider 4.0-6.5 range for optimal results.`);
            } else if (value < 3.0) {
                console.warn(`[Illustrious] CFG ${value} may produce foggy results. Consider 4.0-6.5 range for optimal results.`);
            }
        };
    }
}

function addKSamplerNodeEnhancements(nodeType) {
    // Add node-level enhancements based on type
    if (nodeType === "IllustriousKSamplerPro") {
        // Add a visual indicator that this is the advanced sampler
        if (this.title) {
            this.title += " 🚀";
        }
    } else if (nodeType === "IllustriousKSamplerPresets") {
        // Add presets indicator
        if (this.title) {
            this.title += " 🎯";
        }
    } else if (nodeType && nodeType.includes("KSampler")) {
        // Add standard indicator for other samplers
        if (this.title) {
            this.title += " ⚡";
        }
    }

    // Add model detection info
    if (this.properties) {
        this.properties.illustrious_enhanced = true;
        this.properties.enhancement_version = "4.0";
    }
}

// --- HELPER FUNCTIONS ---
function setupSearchSuggestions(widget) {
    const originalCallback = widget.callback;
    let searchTimeout;
    
    // Store original widget reference
    widget._originalWidget = widget;
    
    // Add search enhancement
    widget.callback = (value) => {
        // Call original callback first
        if (originalCallback) {
            originalCallback.call(widget, value);
        }
        
        // Remember last interacted widget for this data type
        try {
            const dt = getDataType(widget.name);
            __illustrious_lastWidgetByType[dt] = widget;
        } catch {}

        // Add search suggestions with debouncing
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(async () => {
            if (value && value.length >= 2) {
                await showSearchSuggestions(widget, value);
            } else {
                hideSuggestionsDropdown();
            }
        }, 300); // Reduced delay for better responsiveness
    };
    
    // Also handle direct input events if widget has an input element
    if (widget.inputEl) {
        widget.inputEl.addEventListener('input', (e) => {
            widget.value = e.target.value;
            if (widget.callback) {
                widget.callback(e.target.value);
            }
        });
        // Track focus to aid load_more targeting
        widget.inputEl.addEventListener('focus', () => {
            try {
                const dt = getDataType(widget.name);
                __illustrious_lastWidgetByType[dt] = widget;
            } catch {}
        });
    }
    
    // Store reference for potential future use
    widget._hasSearchEnhancement = true;
}

async function showSearchSuggestions(widget, searchValue) {
    try {
        const dataType = getDataType(widget.name);
        const cacheKey = `${dataType}_${searchValue}`;
        
        // Check cache first
        let suggestions = getCachedData(cacheKey);
        
        if (!suggestions) {
            // Fetch from API with fallback
            const url = `/illustrious/search?data_type=${dataType}&query=${encodeURIComponent(searchValue)}&limit=50`;
            const result = await fetchIllustriousJson(url);
            if (result && result.success && Array.isArray(result.data) && result.data.length > 0) {
                suggestions = result.data.filter(item => item !== "-");
                setCachedData(cacheKey, suggestions);
            }
        }
        
        if (suggestions && suggestions.length > 0) {
            // Show suggestions dropdown
            showSuggestionsDropdown(widget, suggestions, searchValue);
        } else {
            // Hide dropdown if no suggestions
            hideSuggestionsDropdown();
            // Notify user once in a while that nothing matched
            try {
                __illustrious_notify_no_results(widget, searchValue);
            } catch {}
        }
        
    } catch (error) {
    console.log("[Illustrious] Search suggestion error:", error?.message || error);
    }
}

// Unified fetch helper supporting both window.api.fetchApi and window.fetch
async function fetchIllustriousJson(url, options) {
    try {
        if (window.api && typeof window.api.fetchApi === "function") {
            const res = await window.api.fetchApi(url, options);
            if (!res || !res.ok) return null;
            return await res.json();
        }
        const res = await fetch(url, options);
        if (!res.ok) return null;
        return await res.json();
    } catch (e) {
        console.warn("[Illustrious] fetch fallback failed:", e?.message || e);
        return null;
    }
}

function getDataType(widgetName) {
    if (widgetName.includes("E621")) {
        return widgetName.includes("Character") ? "e621_characters" : "e621_artists";
    } else {
        return widgetName.includes("Artist") ? "artists" : "characters";
    }
}

function getCachedData(key) {
    const cached = searchCache.get(key);
    if (cached && (Date.now() - cached.timestamp) < CACHE_DURATION) {
        return cached.data;
    }
    return null;
}

function setCachedData(key, data) {
    searchCache.set(key, {
        data: data,
        timestamp: Date.now()
    });
}

// --- DROPDOWN UI FUNCTIONS ---
let currentDropdown = null;
let currentDropdownWidget = null;

function showSuggestionsDropdown(widget, suggestions, searchValue) {
    // Remove any existing dropdown
    hideSuggestionsDropdown();
    
    // Find the node containing this widget
    const node = Array.from((window.app?.graph?._nodes) || []).find(n => n.widgets?.includes(widget));
    if (!node) return;
    
    // Create dropdown container
    const dropdown = document.createElement('div');
    dropdown.className = 'illustrious-dropdown';
    dropdown.style.cssText = `
        position: fixed;
        background: #1a1a1a;
        border: 2px solid #666;
        border-radius: 4px;
        max-height: 400px;
        overflow-y: auto;
        z-index: 2147483647; /* ensure above canvas and panels */
        min-width: 200px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.8);
        font-family: Arial, sans-serif;
    `;
    
    // Store references
    currentDropdown = dropdown;
    currentDropdownWidget = widget;
    
    // Add suggestions to dropdown
    suggestions.slice(0, 30).forEach(suggestion => {
        const item = document.createElement('div');
    item.className = 'illustrious-dropdown-item';
        item.textContent = suggestion;
        item.style.cssText = `
            padding: 8px 12px;
            cursor: pointer;
            color: #ccc;
            font-size: 12px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            border-bottom: 1px solid #333;
        `;
        
        // Highlight matching part
        const matchIndex = suggestion.toLowerCase().indexOf(searchValue.toLowerCase());
        if (matchIndex >= 0) {
            const before = suggestion.substring(0, matchIndex);
            const match = suggestion.substring(matchIndex, matchIndex + searchValue.length);
            const after = suggestion.substring(matchIndex + searchValue.length);
            item.innerHTML = `${before}<strong style="color: #4a90e2">${match}</strong>${after}`;
        }
        
        // Add hover effect
        item.addEventListener('mouseenter', () => {
            item.style.backgroundColor = '#333';
            item.style.color = '#fff';
        });
        item.addEventListener('mouseleave', () => {
            item.style.backgroundColor = 'transparent';
            item.style.color = '#ccc';
        });
        
        // Handle click
        item.addEventListener('click', () => {
            widget.value = suggestion;
            if (widget.callback) {
                widget.callback(suggestion);
            }
            hideSuggestionsDropdown();
        });
        
        dropdown.appendChild(item);
    });
    
    // Position dropdown using canvas transform (account for scale and offset)
    const canvas = window.app?.canvas;
    if (!canvas || !canvas.canvas || !canvas.ds) {
        console.warn("[Illustrious] Canvas not ready; cannot place dropdown");
        return;
    }
    const rect = canvas.canvas.getBoundingClientRect();
    const transform = canvas.ds;
    const scale = transform.scale || 1;
    const offsetX = (transform.offset?.[0] || 0);
    const offsetY = (transform.offset?.[1] || 0);
    
    // Calculate widget position within the node
    const widgetIndex = node.widgets.indexOf(widget);
    let widgetY = LiteGraph.NODE_TITLE_HEIGHT + 10;
    for (let i = 0; i < widgetIndex; i++) {
        widgetY += LiteGraph.NODE_WIDGET_HEIGHT + 4;
    }
    
    // Calculate absolute position: screen = rect + (canvas_pos + offset) * scale
    const x = rect.left + (node.pos[0] + 15 + offsetX) * scale;
    const y = rect.top + (node.pos[1] + widgetY + 25 + offsetY) * scale;
    const width = (node.size[0] - 30) * scale;
    
    let finalLeft = x;
    let finalTop = y;
    let finalWidth = Math.max(width, 250);

    // If we have the underlying input element, prefer its bounding rect for precise placement
    if (widget.inputEl && typeof widget.inputEl.getBoundingClientRect === 'function') {
        const irect = widget.inputEl.getBoundingClientRect();
        if (irect && irect.width && irect.height) {
            finalLeft = irect.left;
            finalTop = irect.bottom + 4; // just below the input
            finalWidth = Math.max(irect.width, 250);
        }
    }

    dropdown.style.left = finalLeft + 'px';
    dropdown.style.top = finalTop + 'px';
    dropdown.style.width = finalWidth + 'px';
    
    // Ensure dropdown stays on screen
    const dropdownHeight = Math.min(suggestions.length * 33, 400); // Estimate height
    const viewBottom = window.innerHeight;
    if (finalTop + dropdownHeight > viewBottom) {
        dropdown.style.top = (finalTop - dropdownHeight - 8) + 'px'; // show above input
    }
    const viewRight = window.innerWidth;
    if (finalLeft + finalWidth > viewRight) {
        dropdown.style.left = (viewRight - finalWidth - 10) + 'px';
    }
    
    document.body.appendChild(dropdown);
    try { console.log(`[Illustrious] Dropdown at (${Math.round(x)}, ${Math.round(y)}) w=${Math.round(Math.max(width, 250))}, scale=${scale.toFixed(2)}`); } catch {}
    
    // Close dropdown when clicking outside
    const closeHandler = (e) => {
        if (!dropdown.contains(e.target)) {
            hideSuggestionsDropdown();
            document.removeEventListener('click', closeHandler);
        }
    };
    setTimeout(() => {
        document.addEventListener('click', closeHandler);
    }, 100);
}

function hideSuggestionsDropdown() {
    if (currentDropdown) {
        currentDropdown.remove();
        currentDropdown = null;
        currentDropdownWidget = null;
    }
}

// Throttled no-results notifier (toast if available)
let __illustrious_last_nores = 0;
function __illustrious_notify_no_results(widget, query) {
    const now = Date.now();
    if (now - __illustrious_last_nores < 2000) return; // throttle
    __illustrious_last_nores = now;
    const type = (widget?.name || '').toLowerCase().includes('artist') ? 'artist' : 'character';
    const msg = `No ${type} matches for "${String(query).slice(0,50)}"`;
    try {
        const app = window.app;
        const toast = app?.extensionManager?.toast;
        if (toast && typeof toast.add === 'function') {
            toast.add({
                severity: 'info',
                summary: 'No matches',
                content: msg,
                life: 2000
            });
            return;
        }
    } catch {}
    try { console.info('[Illustrious] ' + msg); } catch {}
}

// Enhance search widgets on existing nodes (useful after workflow load)
function applySearchToExistingNodes() {
    if (!window.app || !window.app.graph) return;
    try {
        const count = window.app.graph?.nodes?.length || 0;
        console.log(`[Illustrious] Enhancement sweep: scanning ${count} nodes for search attachment`);
    } catch {}
    const targetNodes = [
        "IllustriousPrompt", "IllustriousCharacters", "IllustriousArtists",
        "IllustriousE621Characters", "IllustriousE621Artists"
    ];
    const searchFields = [
        "Character", "Artist", "E621 Character", "E621 Artist",
        "Character (Base)", "Character (Secondary)", "Character (Tertiary)",
        "Character (Quaternary)", "Character (Quinary)",
        "Artist (Base)", "Artist (Secondary)", "Artist (Tertiary)",
        "Artist (Quaternary)", "Artist (Quinary)", "Artist (Primary)"
    ];

    (window.app.graph.nodes || []).forEach(node => {
        if (!node?.comfyClass || !targetNodes.includes(node.comfyClass)) return;
        node.widgets?.forEach(widget => {
            if (searchFields.includes(widget.name) && (widget.type === "text" || widget.type === "STRING") && !widget._hasSearchEnhancement) {
                setupSearchSuggestions.call(node, widget);
                console.log(`[Illustrious] Attached search enhancement to ${node.comfyClass} → ${widget.name}`);
            }
        });
    });
}

// --- HELP PANEL CREATION ---
function createHelpPanel(el) {
    el.innerHTML = `
    <div id="illustrious-help-container" style="
            height: 100%;
            overflow-y: auto;
            padding: 15px;
            background: #1e1e1e;
            color: #ffffff;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
        ">
            <div id="help-header" style="
                border-bottom: 2px solid #4a90e2;
                padding-bottom: 10px;
                margin-bottom: 20px;
            ">
                <h1 style="margin: 0; color: #4a90e2; font-size: 24px;">
                    🎨 Illustrious Help & Documentation
                </h1>
                <p style="margin: 5px 0 0 0; color: #cccccc; font-size: 14px;">
                    Comprehensive guide for Illustrious XL models
                </p>
            </div>
            
            <div id="help-navigation" style="
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
                flex-wrap: wrap;
            ">
                <button class="help-nav-btn active" data-section="overview">Overview</button>
                <button class="help-nav-btn" data-section="ksampler">KSampler Guide</button>
                <button class="help-nav-btn" data-section="prompt-nodes">Prompt Nodes</button>
                <button class="help-nav-btn" data-section="optimal-settings">Optimal Settings</button>
                <button class="help-nav-btn" data-section="troubleshooting">Troubleshooting</button>
                <button class="help-nav-btn" data-section="tips">Tips & Tricks</button>
            </div>
            
            <div id="help-content">
                <!-- Content will be populated by navigation -->
            </div>
        </div>
    `;
    
    // Add CSS for navigation buttons
    const style = document.createElement('style');
    style.textContent = `
        .help-nav-btn {
            padding: 8px 16px;
            background: #333;
            color: #fff;
            border: 1px solid #555;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s ease;
        }
        
        .help-nav-btn:hover {
            background: #444;
            border-color: #666;
        }
        
        .help-nav-btn.active {
            background: #4a90e2;
            border-color: #4a90e2;
        }
        
        .help-section {
            display: none;
        }
        
        .help-section.active {
            display: block;
        }
        
        .help-code {
            background: #2d2d2d;
            padding: 10px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            margin: 10px 0;
            border-left: 3px solid #4a90e2;
        }
        
        .help-warning {
            background: #4a2c00;
            border-left: 3px solid #ff9800;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
        }
        
        .help-tip {
            background: #1a4a1a;
            border-left: 3px solid #4caf50;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
        }
        
        .param-table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
            font-size: 12px;
        }
        
        .param-table th,
        .param-table td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #444;
        }
        
        .param-table th {
            background: #333;
            color: #4a90e2;
        }
    `;
    document.head.appendChild(style);
    
    // Initialize help content
    setupHelpNavigation(el);
    showHelpSection('overview');
}

function setupHelpNavigation(el) {
    const navButtons = el.querySelectorAll('.help-nav-btn');
    const contentDiv = el.querySelector('#help-content');
    
    navButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            // Update active button
            navButtons.forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            
            // Show corresponding section
            const section = e.target.dataset.section;
            showHelpSection(section);
        });
    });
}

function showHelpSection(section) {
    const contentDiv = document.querySelector('#help-content');
    if (!contentDiv) return;
    
    const sections = {
        overview: createOverviewSection(),
        ksampler: createKSamplerSection(),
        'prompt-nodes': createPromptNodesSection(),
        'optimal-settings': createOptimalSettingsSection(),
        troubleshooting: createTroubleshootingSection(),
        tips: createTipsSection()
    };
    
    contentDiv.innerHTML = sections[section] || sections.overview;
}

function createOverviewSection() {
    return `
        <div class="help-section active">
            <h2 style="color: #4a90e2;">📖 Illustrious Overview</h2>
            
            <p>Illustrious is a comprehensive node suite for working with <strong>Illustrious</strong> models in ComfyUI. It provides specialized tools for prompt construction, parameter optimization, and model-specific enhancements.</p>
            
            <h3 style="color: #4a90e2;">🎯 Available Nodes</h3>
            <ul>
                <li><strong>Illustrious Model</strong> - Model loader with optimized settings</li>
                <li><strong>Illustrious Prompt</strong> - Main prompt construction node with chaining</li>
                <li><strong>Characters/Artists</strong> - Multi-character and artist selection nodes</li>
                <li><strong>E621 Characters/Artists</strong> - E621-specific database nodes</li>
                <li><strong>Hairstyles</strong> - Comprehensive hairstyle selection</li>
                <li><strong>Clothing</strong> - Clothing and outfit options</li>
                <li><strong>Poses</strong> - Pose tokens and descriptions</li>
                <li><strong>Pony Tokens</strong> - Pony model compatibility tokens</li>
                <li><strong>KSampler (Illustrious)</strong> - Optimized sampler for Illustrious</li>
                <li><strong>KSampler (Simple)</strong> - Beginner-friendly preset sampler</li>
                <li><strong>KSampler (Advanced)</strong> - Advanced sampler with fine controls</li>
            </ul>
            
            <h3 style="color: #4a90e2;">🔗 Chaining System</h3>
            <p>Illustrious nodes support a powerful chaining system using the <code>[EN122112_CHAIN]</code> token:</p>
            
            <div class="help-tip">
                <strong>💡 Tip:</strong> Use the "Add Chain Insert" option to create connection points for other Illustrious nodes. This allows you to build complex prompts by connecting multiple specialized nodes together.
            </div>
            
            <h3 style="color: #4a90e2;">🎨 Model Compatibility</h3>
            <p>Optimized for Illustrious XL model family.</p>
            
            <div class="help-warning">
                <strong>⚠️ Important:</strong> These nodes are specifically designed for Illustrious XL models. Using them with other model types (like SD1.5, SDXL, etc.) may not produce optimal results.
            </div>
        </div>
    `;
}

function createKSamplerSection() {
    return `
        <div class="help-section">
            <h2 style="color: #4a90e2;">⚡ KSampler Guide</h2>
            
            <h3 style="color: #4a90e2;">🎯 Which KSampler to Use?</h3>
            
            <table class="param-table">
                <tr>
                    <th>KSampler Type</th>
                    <th>Best For</th>
                    <th>Key Features</th>
                </tr>
                <tr>
                    <td><strong>KSampler (Simple)</strong> 🎯</td>
                    <td>Beginners, quick results</td>
                    <td>One-click presets, automatic optimization</td>
                </tr>
                <tr>
                    <td><strong>KSampler (Illustrious)</strong> ⚡</td>
                    <td>Most users, balanced control</td>
                    <td>Auto CFG, resolution scaling, model detection</td>
                </tr>
                <tr>
                    <td><strong>KSampler (Advanced)</strong> 🚀</td>
                    <td>Power users, fine-tuning</td>
                    <td>Dynamic CFG, thresholding, advanced controls</td>
                </tr>
            </table>
            
            <h3 style="color: #4a90e2;">📊 Optimal Parameter Ranges</h3>
            
            <table class="param-table">
                <tr>
                    <th>Parameter</th>
                    <th>Optimal Range</th>
                    <th>Notes</th>
                </tr>
                <tr>
                    <td><strong>CFG Scale</strong></td>
                    <td>4.0 - 6.5</td>
                    <td>Sweet spot for Illustrious. Higher = oversaturation risk</td>
                </tr>
                <tr>
                    <td><strong>Steps</strong></td>
                    <td>20 - 35</td>
                    <td>24-28 typically sufficient. More steps for high-res</td>
                </tr>
                <tr>
                    <td><strong>Sampler</strong></td>
                    <td>euler_ancestral</td>
                    <td>Best balance of speed and quality for these models</td>
                </tr>
                <tr>
                    <td><strong>Scheduler</strong></td>
                    <td>karras (high-res)<br>normal (standard)</td>
                    <td>Karras better for 1536px+ images</td>
                </tr>
            </table>
            
            <h3 style="color: #4a90e2;">🔧 Advanced Features</h3>
            
            <h4>Dynamic CFG Scheduling</h4>
            <p>Advanced KSampler supports dynamic CFG adjustment during generation:</p>
            <ul>
                <li><strong>Illustrious Adaptive</strong> - Higher CFG early for structure, lower late for details</li>
                <li><strong>Linear Decay</strong> - Gradually reduce CFG over time</li>
                <li><strong>Cosine Decay</strong> - Smooth curved reduction</li>
            </ul>
            
            <h4>Dynamic Thresholding</h4>
            <p>Prevents oversaturation by clamping extreme values:</p>
            <div class="help-code">
DT Percentile: 0.995 (99.5th percentile threshold)
DT Clamp Range: 1.0 (standard clamping strength)
            </div>
            
            <div class="help-tip">
                <strong>💡 Pro Tip:</strong> Enable "Danbooru Boost" in Advanced KSampler to enhance recognition of common anime/manga tags that Illustrious was specifically trained on.
            </div>
        </div>
    `;
}

function createPromptNodesSection() {
    return `
        <div class="help-section">
            <h2 style="color: #4a90e2;">📝 Prompt Construction Nodes</h2>
            
            <h3 style="color: #4a90e2;">🎯 Main Prompt Node</h3>
            <p>The <strong>Illustrious Prompt</strong> node is your central hub for prompt construction:</p>
            
            <h4>Key Features:</h4>
            <ul>
                <li><strong>Character/Artist Search</strong> - Type to search 5000+ characters and artists</li>
                <li><strong>Quality Presets</strong> - Multiple negative prompt templates</li>
                <li><strong>Style Controls</strong> - Lighting, framing, composition options</li>
                <li><strong>Chain Support</strong> - Connect with other Illustrious nodes</li>
            </ul>
            
            <h3 style="color: #4a90e2;">👥 Character & Artist Nodes</h3>
            
            <h4>Characters Node</h4>
            <p>Supports up to 5 characters with weighted blending:</p>
            <div class="help-code">
Base Character: 1.2 weight (strongest influence)
Secondary: 1.15 weight
Tertiary: 1.125 weight
Quaternary: 1.1 weight  
Quinary: 1.0 weight (weakest influence)
            </div>
            
            <h4>Artists Node</h4>
            <p>Similar 5-tier system for artist styles with automatic "artist:" prefixing.</p>
            
            <div class="help-tip">
                <strong>💡 Mixing Tip:</strong> Use "Weighted Average" mode to automatically balance influences for smoother blends.
            </div>
            
            <h3 style="color: #4a90e2;">🎨 Style & Detail Nodes</h3>
            
            <h4>Hairstyles Node</h4>
            <p>Comprehensive hair system with smart injection:</p>
            <ul>
                <li><strong>Length & Volume</strong> - Basic hair properties</li>
                <li><strong>Cuts & Styles</strong> - Specific hairstyles</li>
                <li><strong>Colors</strong> - Hair color options</li>
                <li><strong>Smart Placement</strong> - Automatically inserts after hair color mentions</li>
            </ul>
            
            <h4>Clothing Node</h4>
            <p>Organized clothing system:</p>
            <ul>
                <li><strong>Outfits</strong> - Complete outfit sets</li>
                <li><strong>Tops</strong> - Upper body clothing</li>
                <li><strong>Bottoms</strong> - Lower body clothing</li>
            </ul>
            
            <h4>Poses Node</h4>
            <p>Pose tokens with descriptive options:</p>
            <ul>
                <li><strong>Token Mode</strong> - Simple pose keywords</li>
                <li><strong>Descriptive Mode</strong> - Detailed pose descriptions</li>
                <li><strong>Weight Control</strong> - Adjust pose influence</li>
            </ul>
            
            <h3 style="color: #4a90e2;">🔗 Chaining Workflow</h3>
            <p>Recommended node connection order:</p>
            <div class="help-code">
1. Main Prompt Node (with "Add Chain Insert" enabled)
   ↓
2. Characters Node 
   ↓  
3. Artists Node
   ↓
4. Hairstyles Node
   ↓
5. Clothing Node
   ↓
6. Poses Node
   ↓
7. KSampler
            </div>
            
            <div class="help-warning">
                <strong>⚠️ Chain Order Matters:</strong> Nodes process in sequence, so character information should come before style elements for best results.
            </div>
        </div>
    `;
}

function createOptimalSettingsSection() {
    return `
        <div class="help-section">
            <h2 style="color: #4a90e2;">⚙️ Optimal Settings Guide</h2>
            
            <h3 style="color: #4a90e2;">🎯 Resolution-Based Settings</h3>
            
            <table class="param-table">
                <tr>
                    <th>Resolution</th>
                    <th>CFG Scale</th>
                    <th>Steps</th>
                    <th>Scheduler</th>
                    <th>Notes</th>
                </tr>
                <tr>
                    <td><strong>512x512</strong></td>
                    <td>5.5 - 6.5</td>
                    <td>20-24</td>
                    <td>normal</td>
                    <td>Higher CFG for low-res</td>
                </tr>
                <tr>
                    <td><strong>768x768</strong></td>
                    <td>5.0 - 6.0</td>
                    <td>22-26</td>
                    <td>normal</td>
                    <td>Balanced settings</td>
                </tr>
                <tr>
                    <td><strong>1024x1024</strong></td>
                    <td>4.5 - 5.5</td>
                    <td>24-28</td>
                    <td>karras</td>
                    <td>Standard high quality</td>
                </tr>
                <tr>
                    <td><strong>1536x1536</strong></td>
                    <td>4.0 - 5.0</td>
                    <td>26-32</td>
                    <td>karras</td>
                    <td>Illustrious native resolution</td>
                </tr>
                <tr>
                    <td><strong>2048x2048+</strong></td>
                    <td>4.0 - 4.8</td>
                    <td>28-35</td>
                    <td>karras</td>
                    <td>Lower CFG prevents artifacts</td>
                </tr>
            </table>
            
            <h3 style="color: #4a90e2;">🤖 Model-Specific Settings</h3>
            
            <h4>Illustrious v1.0</h4>
            <div class="help-code">
CFG: 4.0-5.5 (balanced guidance)
Sampler: euler_ancestral preferred  
Scheduler: karras for high-res, normal for standard
Negative: "Article (Recommended)" template
            </div>
            
            <h3 style="color: #4a90e2;">💡 Quality Optimization Tips</h3>
            
            <div class="help-tip">
                <strong>🎨 For Anime/Illustration:</strong>
                <ul>
                    <li>Enable "Anime Enhancement" in Advanced KSampler</li>
                    <li>Use "Danbooru Boost" for better tag recognition</li>
                    <li>Consider "Quality Boost" option in main prompt node</li>
                </ul>
            </div>
            
            <div class="help-tip">
                <strong>📸 For Realistic Results:</strong>
                <ul>
                    <li>Lower CFG (4.0-4.5) to avoid over-processing</li>
                    <li>Use "Article (Recommended)" negative prompt</li>
                    <li>Consider Dynamic Thresholding for problem prompts</li>
                </ul>
            </div>
            
            <h3 style="color: #4a90e2;">🚀 Speed vs Quality Balance</h3>
            
            <table class="param-table">
                <tr>
                    <th>Priority</th>
                    <th>Settings</th>
                    <th>Use Case</th>
                </tr>
                <tr>
                    <td><strong>Speed</strong></td>
                    <td>16-20 steps, CFG 5.5, dpm_fast</td>
                    <td>Testing, iteration, previews</td>
                </tr>
                <tr>
                    <td><strong>Balanced</strong></td>
                    <td>24-28 steps, CFG 5.0, euler_ancestral</td>
                    <td>General use, good quality</td>
                </tr>
                <tr>
                    <td><strong>Quality</strong></td>
                    <td>30-35 steps, CFG 4.8, karras scheduler</td>
                    <td>Final outputs, high detail needed</td>
                </tr>
            </table>
        </div>
    `;
}

function createTroubleshootingSection() {
    return `
        <div class="help-section">
            <h2 style="color: #4a90e2;">🔧 Troubleshooting Guide</h2>
            
            <h3 style="color: #4a90e2;">❌ Common Issues</h3>
            
            <h4>Problem: Oversaturated/Overcooked Images</h4>
            <div class="help-warning">
                <strong>Symptoms:</strong> Images look too processed, colors too intense, artificial appearance
            </div>
            <p><strong>Solutions:</strong></p>
            <ul>
                <li>Lower CFG to 4.0-5.0 range</li>
                <li>Enable "Dynamic Thresholding" in Advanced KSampler</li>
                <li>Use "karras" scheduler instead of "normal"</li>
                <li>Try fewer steps (20-24 instead of 30+)</li>
            </ul>
            
            <h4>Problem: Foggy/Undefined Images</h4>
            <div class="help-warning">
                <strong>Symptoms:</strong> Images lack detail, look washed out, soft focus
            </div>
            <p><strong>Solutions:</strong></p>
            <ul>
                <li>Increase CFG to 5.5-6.5 range</li>
                <li>Add more steps (28-35)</li>
                <li>Check if you're using the right model type</li>
                <li>Enable "Quality Boost" in prompt node</li>
            </ul>
            
            <h4>Problem: Character/Artist Not Working</h4>
            <div class="help-warning">
                <strong>Symptoms:</strong> Searched character/artist doesn't appear in results
            </div>
            <p><strong>Solutions:</strong></p>
            <ul>
                <li>Try different spelling variations (use underscores)</li>
                <li>Search for partial names</li>
                <li>Check if character is in E621 database instead</li>
                <li>Refresh ComfyUI if search isn't working</li>
            </ul>
            
            <h4>Problem: Nodes Not Chaining Properly</h4>
            <div class="help-warning">
                <strong>Symptoms:</strong> Chain connections don't work, content not inserting
            </div>
            <p><strong>Solutions:</strong></p>
            <ul>
                <li>Enable "Add Chain Insert" on the first node</li>
                <li>Ensure chain connections are in correct order</li>
                <li>Check for [EN122112_CHAIN] token in prompts</li>
                <li>Verify all nodes are Illustrious variants</li>
            </ul>
            
            <h3 style="color: #4a90e2;">⚠️ Model Compatibility Issues</h3>
            
            <h4>Wrong Model Type Selected</h4>
            <p>If auto-detection isn't working:</p>
            <div class="help-code">
1. Check your model filename/metadata
2. Manually select model type in KSampler
3. Adjust CFG range accordingly:
    - Illustrious: 4.0-5.5 CFG
            </div>
            
            <h4>Using Non-Compatible Models</h4>
            <div class="help-warning">
                <strong>⚠️ Important:</strong> Illustrious is optimized for Illustrious XL models only. Using with SDXL, SD1.5, or other models may produce suboptimal results.
            </div>
            
            <h3 style="color: #4a90e2;">🐛 Debug Information</h3>
            
            <p>Enable "Advanced Mode" in KSampler nodes to see debug information in browser console (F12):</p>
            <ul>
                <li>Parameter optimization warnings</li>
                <li>Sampler compatibility alerts</li>
                <li>CFG range suggestions</li>
                <li>Model detection results</li>
            </ul>
            
            <h3 style="color: #4a90e2;">🔄 Reset to Defaults</h3>
            
            <p>If you're having persistent issues, try these default settings:</p>
            <div class="help-code">
KSampler (Illustrious):
- CFG: 5.0
- Steps: 24  
- Sampler: euler_ancestral
- Scheduler: normal
- Model Type: Auto Detect
- Auto CFG: Enabled
- Resolution Adaptive: Enabled

Main Prompt Node:
- Negative Prompt: "Latest (Underscores)"
- Quality Boost: Disabled initially
- Format Tag: Enabled
            </div>
            
            <div class="help-tip">
                <strong>💡 Still Having Issues?</strong> Check the ComfyUI console output (terminal where you launched ComfyUI) for detailed error messages and warnings.
            </div>
        </div>
    `;
}

function createTipsSection() {
    return `
        <div class="help-section">
            <h2 style="color: #4a90e2;">💡 Tips & Advanced Techniques</h2>
            
            <h3 style="color: #4a90e2;">🎨 Character Mixing Techniques</h3>
            
            <h4>Weighted Character Blending</h4>
            <p>Use the Characters node to blend multiple characters:</p>
            <div class="help-code">
Base Character: hatsune_miku (1.2 weight)
Secondary: saber_(fate) (1.15 weight)  
Result: Fusion with Miku's dominant features + Saber's elements
            </div>
            
            <div class="help-tip">
                <strong>💡 Pro Tip:</strong> Enable "Weighted Average" to automatically balance influences for smoother blends.
            </div>
            
            <h4>Artist Style Layering</h4>
            <p>Combine artists for unique styles:</p>
            <div class="help-code">
Base Artist: wlop (1.2) - Realistic rendering
Secondary: kantoku (1.15) - Soft anime style
Tertiary: artgerm (1.1) - Professional polish
Result: Realistic anime with professional finish
            </div>
            
            <h3 style="color: #4a90e2;">⚡ Advanced KSampler Techniques</h3>
            
            <h4>Dynamic CFG for Better Results</h4>
            <p>Use "Illustrious Adaptive" CFG scheduling:</p>
            <ul>
                <li><strong>Early steps:</strong> Higher CFG for structure</li>
                <li><strong>Middle steps:</strong> Stable CFG for development</li>
                <li><strong>Late steps:</strong> Lower CFG for detail refinement</li>
            </ul>
            
            <h4>Resolution-Aware Workflows</h4>
            <div class="help-code">
Low-res testing (512px): High CFG (6.0), fewer steps (20)
High-res final (1536px): Low CFG (4.5), more steps (30)
            </div>
            
            <h3 style="color: #4a90e2;">🎯 Prompt Construction Strategies</h3>
            
            <h4>Effective Tag Ordering</h4>
            <p>Place important elements early in the chain:</p>
            <div class="help-code">
1. Character/Subject
2. Artist style  
3. Pose/Action
4. Clothing/Appearance
5. Hairstyle details
6. Background/Environment
7. Quality/Style modifiers
            </div>
            
            <h4>Negative Prompt Selection</h4>
            <table class="param-table">
                <tr>
                    <th>Template</th>
                    <th>Best For</th>
                    <th>Characteristics</th>
                </tr>
                <tr>
                    <td><strong>Latest (Underscores)</strong></td>
                    <td>General use</td>
                    <td>Modern format, comprehensive</td>
                </tr>
                <tr>
                    <td><strong>Article (Recommended)</strong></td>
                    <td>Illustrious, high quality</td>
                    <td>Research-based, effective</td>
                </tr>
                <tr>
                    <td><strong>Boost</strong></td>
                    <td>Older models, problematic prompts</td>
                    <td>Aggressive filtering</td>
                </tr>
            </table>
            
            <h3 style="color: #4a90e2;">🔧 Workflow Optimization</h3>
            
            <h4>Testing Workflow</h4>
            <div class="help-tip">
                <strong>Efficient Testing Process:</strong>
                <ol>
                    <li>Start with Simple KSampler + "Fast" preset</li>
                    <li>Test character/style combinations at 512px</li>
                    <li>Once satisfied, switch to Illustrious KSampler</li>
                    <li>Increase to target resolution</li>
                    <li>Fine-tune with Advanced KSampler if needed</li>
                </ol>
            </div>
            
            <h4>Production Workflow</h4>
            <div class="help-code">
1. Characters Node → Set primary character
2. Artists Node → Add 1-2 complementary artists
3. Main Prompt → Add specific details, quality settings
4. Hairstyles → Enhance character appearance
5. Poses → Add dynamic elements
6. Advanced KSampler → Final quality generation
            </div>
            
            <h3 style="color: #4a90e2;">📊 Performance Tips</h3>
            
            <h4>Memory Management</h4>
            <ul>
                <li>Use "Resolution Adaptive" to automatically optimize for your hardware</li>
                <li>Start with lower resolutions for testing</li>
                <li>Enable "Auto CFG" to prevent problematic settings</li>
            </ul>
            
            <h4>Batch Processing</h4>
            <p>For multiple variations:</p>
            <div class="help-code">
1. Set up base workflow with Illustrious nodes
2. Use different seeds in KSampler  
3. Vary character weights slightly
4. Try different CFG values (4.0, 4.5, 5.0, 5.5)
            </div>
            
            <h3 style="color: #4a90e2;">🎪 Creative Techniques</h3>
            
            <h4>Style Fusion</h4>
            <p>Combine anime and realistic elements:</p>
            <div class="help-code">
Artist 1: makoto_shinkai (anime backgrounds)
Artist 2: artgerm (realistic rendering)
Result: Realistic character in anime-style environment
            </div>
            
            <h4>Era Mixing</h4>
            <p>Use the Year field creatively:</p>
            <div class="help-code">
Year: 2010-2015 → Early digital art aesthetic
Year: 2020+ → Modern high-quality style
            </div>
            
            <div class="help-warning">
                <strong>🎨 Experimental Zone:</strong> Don't be afraid to try unconventional combinations! The node system is designed to handle complex prompt structures safely.
            </div>
        </div>
    `;
}

// --- ADVANCED UI ENHANCEMENTS ---
function setupWorkflowDetection() {
    // Guard environments where window.api may be missing
    if (!window.api || typeof window.api.addEventListener !== "function") {
        console.warn("[Illustrious] window.api not available; workflow detection disabled.");
        return;
    }

    // Detect workflow execution start
    window.api.addEventListener("execution_start", () => {
    console.log("[Illustrious] Workflow execution started");
        
        // Check if any Illustrious nodes are in the workflow
    const graph = window.app?.graph;
        const illustriousNodes = [];
        
        graph.nodes.forEach(node => {
            if (node.comfyClass && node.comfyClass.includes("Illustrious")) {
                illustriousNodes.push({
                    type: node.comfyClass,
                    title: node.title,
                    id: node.id
                });
            }
        });
        
        if (illustriousNodes.length > 0) {
            console.log(`[Illustrious] Found ${illustriousNodes.length} Illustrious nodes in workflow:`, illustriousNodes);
            
            // Show optimization suggestions if needed
            checkWorkflowOptimization(illustriousNodes);
        }
    });
    
    // Detect workflow interruption
    if (typeof window.api.interrupt === "function") {
        const original_api_interrupt = window.api.interrupt;
        window.api.interrupt = function () {
        console.log("[Illustrious] Workflow interrupted");
            return original_api_interrupt.apply(this, arguments);
        }
    }
    
    // Detect execution errors
    window.api.addEventListener("execution_error", (error) => {
        console.log("[Illustrious] Execution error detected:", error);
        
        // Check if error is related to Illustrious nodes
        if (error.node_id) {
            const node = window.app?.graph?.getNodeById(error.node_id);
            if (node && node.comfyClass?.includes("Illustrious")) {
                showIllustriousErrorHelp(node, error);
            }
        }
    });
}

function setupContextMenus() {
    // Add Illustrious options to background context menu
    const original_getCanvasMenuOptions = LGraphCanvas.prototype.getCanvasMenuOptions;
    LGraphCanvas.prototype.getCanvasMenuOptions = function () {
        const options = original_getCanvasMenuOptions.apply(this, arguments);
        
        // Add Illustrious section
        options.push(null); // Divider
        options.push({
            content: "🎨 Illustrious",
            has_submenu: true,
            callback: createIllustriousSubmenu,
        });
        
        return options;
    }
}

function setupQueueButton() {
    // Enhanced queue button monitoring
    const queueButton = document.getElementById("queue-button");
    if (queueButton) {
        queueButton.addEventListener("click", () => {
            console.log("[Illustrious] Queue button pressed - analyzing workflow");
            analyzeWorkflowBeforeQueue();
        });
    }
}

// --- LOAD MORE (FULL LIST) MODAL & LISTENER ---
function setupLoadMoreListener() {
    // Listen for backend signal to open the full list modal
    if (!window.api || typeof window.api.addEventListener !== 'function') {
        console.warn('[Illustrious] window.api not available; load_more listener disabled.');
        return;
    }

    window.api.addEventListener('illustrious.load_more', ({ detail }) => {
        const data_type = detail?.data_type || 'characters';
        try { console.log('[Illustrious] load_more event for', data_type); } catch {}
        openFullListModal(data_type);
    });
}

function openFullListModal(data_type) {
    // Build overlay and modal
    const overlay = document.createElement('div');
    overlay.style.cssText = `position:fixed;inset:0;z-index:2147483646;background:rgba(0,0,0,.55)`;

    const modal = document.createElement('div');
    modal.style.cssText = `position:fixed;left:50%;top:50%;transform:translate(-50%,-50%);z-index:2147483647;`+
        `background:#1f1f1f;border:1px solid #555;border-radius:10px;box-shadow:0 12px 48px rgba(0,0,0,.8);`+
        `width:min(820px,90vw);max-height:80vh;display:flex;flex-direction:column;overflow:hidden;font-family:Arial, sans-serif;`;

    const header = document.createElement('div');
    header.style.cssText = `display:flex;align-items:center;justify-content:space-between;padding:12px 14px;`+
        `background:linear-gradient(180deg,#2a2a2a,#212121);border-bottom:1px solid #444;color:#e6e6e6`;
    header.innerHTML = `<div style="font-weight:700;font-size:14px;letter-spacing:.2px">Illustrious: Select ${humanizeDataType(data_type)}</div>`+
                       `<button aria-label="Close" style="background:#3a3a3a;border:1px solid #555;color:#ddd;`+
                       `padding:6px 10px;border-radius:6px;cursor:pointer">Close</button>`;

    const closeBtn = header.querySelector('button');
    const close = () => { try { document.body.removeChild(modal); document.body.removeChild(overlay); } catch {} };
    overlay.addEventListener('click', close);
    closeBtn.addEventListener('click', close);

    // Search bar
    const controls = document.createElement('div');
    controls.style.cssText = `padding:10px 14px;display:flex;gap:10px;align-items:center;background:#232323;border-bottom:1px solid #3a3a3a`;
    const searchInput = document.createElement('input');
    searchInput.type = 'text';
    searchInput.placeholder = `Search ${humanizeDataType(data_type)}…`;
    searchInput.style.cssText = `flex:1;min-width:0;background:#2b2b2b;color:#eee;border:1px solid #4a4a4a;`+
        `padding:8px 10px;border-radius:6px;outline:none`;
    const pageSizeSelect = document.createElement('select');
    pageSizeSelect.style.cssText = `background:#2b2b2b;color:#eee;border:1px solid #4a4a4a;padding:8px;border-radius:6px`;
    ;['100','200','500'].forEach(v=>{ const o=document.createElement('option'); o.value=v; o.textContent=v+'/page'; pageSizeSelect.appendChild(o); });
    controls.appendChild(searchInput);
    controls.appendChild(pageSizeSelect);

    // List area
    const list = document.createElement('div');
    list.style.cssText = `padding:8px 0;overflow:auto;flex:1;background:#1c1c1c`;

    // Footer with pagination
    const footer = document.createElement('div');
    footer.style.cssText = `display:flex;align-items:center;justify-content:space-between;padding:10px 14px;`+
        `background:#222;border-top:1px solid #3a3a3a;color:#bbb;font-size:12px`;
    const left = document.createElement('div');
    const right = document.createElement('div');
    const prevBtn = document.createElement('button');
    const nextBtn = document.createElement('button');
    [prevBtn,nextBtn].forEach(b=>b.style.cssText=`background:#333;border:1px solid #555;color:#ddd;padding:6px 10px;border-radius:6px;cursor:pointer;margin:0 4px`);
    prevBtn.textContent = 'Prev'; nextBtn.textContent = 'Next';
    right.appendChild(prevBtn); right.appendChild(nextBtn);
    footer.appendChild(left); footer.appendChild(right);

    modal.appendChild(header);
    modal.appendChild(controls);
    modal.appendChild(list);
    modal.appendChild(footer);
    document.body.appendChild(overlay);
    document.body.appendChild(modal);

    // State
    let page = 0;
    let page_size = 200;
    let query = '';
    pageSizeSelect.value = String(page_size);

    const renderItems = (items) => {
        list.innerHTML = '';
        const frag = document.createDocumentFragment();
        items.forEach(name => {
            if (!name || name === '-') return;
            const row = document.createElement('div');
            row.textContent = name;
            row.style.cssText = `padding:8px 14px;color:#ddd;border-bottom:1px solid #2a2a2a;cursor:pointer;`;
            row.addEventListener('mouseenter',()=>row.style.background='#2a2a2a');
            row.addEventListener('mouseleave',()=>row.style.background='');
            row.addEventListener('click',()=>{
                applySelectionToWidget(data_type, name);
                close();
            });
            frag.appendChild(row);
        });
        list.appendChild(frag);
    };

    const refresh = async () => {
        const url = `/illustrious/get_full_list?data_type=${encodeURIComponent(data_type)}&page=${page}&page_size=${page_size}` + (query?`&search=${encodeURIComponent(query)}`:'');
        const res = await fetchIllustriousJson(url);
        if (!res || !res.success) { renderItems([]); left.textContent = 'Error loading list'; return; }
        renderItems(res.data || []);
        const total = res.total || 0; const tp = res.total_pages || 1; const returned = res.returned || 0;
        left.textContent = `Page ${res.page+1}/${tp} • ${returned}/${total} shown`;
        prevBtn.disabled = (res.page<=0); nextBtn.disabled = !(res.has_more);
    };

    // Wire controls
    let searchDebounce;
    searchInput.addEventListener('input', () => {
        clearTimeout(searchDebounce);
        searchDebounce = setTimeout(()=>{ query = searchInput.value.trim(); page = 0; refresh(); }, 250);
    });
    pageSizeSelect.addEventListener('change', () => { page_size = parseInt(pageSizeSelect.value,10)||200; page = 0; refresh(); });
    prevBtn.addEventListener('click', () => { if (page>0){ page--; refresh(); }});
    nextBtn.addEventListener('click', () => { page++; refresh(); });

    // Initial load
    refresh();
    setTimeout(()=>searchInput.focus(), 50);
}

function humanizeDataType(dt){
    switch(dt){
        case 'artists': return 'Artist';
        case 'characters': return 'Character';
        case 'e621_artists': return 'E621 Artist';
        case 'e621_characters': return 'E621 Character';
        default: return dt;
    }
}

function applySelectionToWidget(data_type, value){
    // Prefer the last interacted widget for this data type
    let widget = __illustrious_lastWidgetByType[data_type];

    // If none, try to infer a reasonable target in the current graph
    if (!widget && window.app?.graph?.nodes?.length){
        const namesByType = {
            characters: ["Character","Character (Base)","Character (Secondary)","Character (Tertiary)","Character (Quaternary)","Character (Quinary)","E621 Character"],
            e621_characters: ["E621 Character","Character"],
            artists: ["Artist","Artist (Base)","Artist (Secondary)","Artist (Tertiary)","Artist (Quaternary)","Artist (Quinary)","Artist (Primary)","E621 Artist"],
            e621_artists: ["E621 Artist","Artist"]
        };
        const candidates = namesByType[data_type] || [];
        outer: for (const node of window.app.graph.nodes){
            if (!node.widgets) continue;
            for (const w of node.widgets){
                if ((w.type === 'text' || w.type === 'STRING') && candidates.includes(w.name)) { widget = w; break outer; }
            }
        }
    }

    if (widget){
        widget.value = value;
        try { widget.callback && widget.callback(value); } catch {}
        if (widget.inputEl) { widget.inputEl.value = value; }
        toastInfo(`Selected ${value}`);
        return;
    }

    // Fallback: copy to clipboard so user can paste
    try { navigator.clipboard?.writeText?.(value); toastInfo(`Copied ${value} to clipboard`); } catch {}
}

function toastInfo(msg){
    try {
        const toast = window.app?.extensionManager?.toast;
        if (toast && typeof toast.add === 'function') {
            toast.add({ severity: 'info', summary: 'Illustrious', content: msg, life: 1800 });
        } else {
            console.info('[Illustrious]', msg);
        }
    } catch {}
}

function setupNodeContextMenu(nodeType) {
    const original_getExtraMenuOptions = this.constructor.prototype.getExtraMenuOptions;
    this.constructor.prototype.getExtraMenuOptions = function(_, options) {
        original_getExtraMenuOptions?.apply(this, arguments);
        
    // Add Illustrious-specific options
        options.push(null); // Divider
        
        if (nodeType.includes("KSampler")) {
            options.push({
                content: "🔧 Reset to Optimal Settings",
                callback: () => resetKSamplerToOptimal(this)
            });
            
            options.push({
                content: "📊 Show Parameter Guide",
                callback: () => showParameterGuide(this)
            });
        }
        
    if (nodeType === "IllustriousPrompt") {
            options.push({
                content: "🎯 Quick Setup Wizard",
                callback: () => showQuickSetupWizard(this)
            });
        }
        
        if (nodeType.includes("Characters") || nodeType.includes("Artists")) {
            options.push({
                content: "🔍 Popular Suggestions",
                callback: () => showPopularSuggestions(this, nodeType)
            });
        }
        
        options.push({
            content: "❓ Help for this Node",
            callback: () => openHelpForNode(nodeType)
        });
    }
}

function createIllustriousSubmenu(value, options, e, menu, node) {
    const submenuOptions = [
        "Add Prompt Chain Workflow",
        "Add Character Focus Workflow", 
        "Add KSampler Comparison",
        "Open Help Documentation",
        "Reset All Illustrious Nodes",
        "Export Illustrious Settings"
    ];
    
    const submenu = new LiteGraph.ContextMenu(
        submenuOptions,
        { 
            event: e, 
            callback: function (v) { 
                handleIllustriousSubmenuAction(v);
            }, 
            parentMenu: menu, 
            node: node
        }
    )
}

function handleIllustriousSubmenuAction(action) {
    console.log(`[Illustrious] Context menu action: ${action}`);
    
    switch(action) {
        case "Add Prompt Chain Workflow":
            addPromptChainWorkflow();
            break;
        case "Add Character Focus Workflow":
            addCharacterFocusWorkflow();
            break;
        case "Add KSampler Comparison":
            addKSamplerComparison();
            break;
        case "Open Help Documentation":
            openHelpDocumentation();
            break;
        case "Reset All Illustrious Nodes":
            resetAllIllustriousNodes();
            break;
        case "Export Illustrious Settings":
            exportIllustriousSettings();
            break;
    }
}

function checkWorkflowOptimization(nodes) {
    const suggestions = [];
    
    // Check for common optimization opportunities
    const ksamplerNodes = nodes.filter(n => n.type.includes("KSampler"));
    const mainPromptNodes = nodes.filter(n => n.type === "IllustriousPrompt");
    const characterNodes = nodes.filter(n => n.type.includes("Characters"));
    
    if (ksamplerNodes.length === 0) {
    suggestions.push("⚠️ No Illustrious KSampler found. Consider using KSampler (Illustrious) for optimal results.");
    }
    
    if (mainPromptNodes.length > 1) {
        suggestions.push("💡 Multiple main prompt nodes detected. Consider using chaining for better organization.");
    }
    
    if (characterNodes.length > 1) {
        suggestions.push("🎨 Multiple character nodes detected. You can combine up to 5 characters in a single Characters node.");
    }
    
    if (suggestions.length > 0) {
        console.log("[Illustrious] Workflow optimization suggestions:", suggestions);
        
        // Show suggestions in a subtle way
        setTimeout(() => {
            suggestions.forEach(suggestion => {
                console.warn(`[Illustrious] ${suggestion}`);
            });
        }, 1000);
    }
}

function showIllustriousErrorHelp(node, error) {
    console.error(`[Illustrious] Error in ${node.comfyClass}:`, error);
    
    // Show helpful error context
    const errorHelp = getErrorHelp(node.comfyClass, error);
    if (errorHelp) {
        console.log(`[Illustrious] Help: ${errorHelp}`);
    }
}

function getErrorHelp(nodeType, error) {
    const errorPatterns = {
    "model": "Check that you're using an Illustrious model. Other model types may not work correctly.",
        "conditioning": "Ensure CLIP Text Encode nodes are connected to positive/negative inputs.",
        "cfg": "CFG values outside 3.0-8.0 range may cause issues. Try 4.0-6.5 for optimal results.",
        "steps": "Very high step counts (>50) may cause memory issues. Try 20-35 steps.",
        "sampler": "If using custom samplers, try switching to euler_ancestral for compatibility."
    };
    
    const errorMsg = error.message || error.toString();
    
    for (const [pattern, help] of Object.entries(errorPatterns)) {
        if (errorMsg.toLowerCase().includes(pattern)) {
            return help;
        }
    }
    
    return null;
}

function analyzeWorkflowBeforeQueue() {
    const graph = window.app?.graph;
    const illustriousNodes = graph.nodes.filter(node => 
        node.comfyClass && node.comfyClass.includes("Illustrious")
    );
    
    if (illustriousNodes.length > 0) {
        console.log(`[Illustrious] Analyzing ${illustriousNodes.length} nodes before queue...`);
        
        // Quick validation checks
        const issues = [];
        
        illustriousNodes.forEach(node => {
            if (node.comfyClass.includes("KSampler")) {
                const cfgWidget = node.widgets?.find(w => w.name === "cfg");
                if (cfgWidget && (cfgWidget.value > 8.0 || cfgWidget.value < 3.0)) {
                    issues.push(`${node.title || node.comfyClass}: CFG ${cfgWidget.value} may cause issues (recommended: 4.0-6.5)`);
                }
            }
        });
        
        if (issues.length > 0) {
            console.warn("[Illustrious] Potential issues detected:");
            issues.forEach(issue => console.warn(`  • ${issue}`));
        }
    }
}

// Helper functions for context menu actions
function addPromptChainWorkflow() {
    // Add a basic prompt chain workflow to canvas
    console.log("[Illustrious] Adding prompt chain workflow template");
    // Implementation would add nodes with proper connections
}

function addCharacterFocusWorkflow() {
    console.log("[Illustrious] Adding character focus workflow template");
    // Implementation would add character-focused node setup
}

function addKSamplerComparison() {
    console.log("[Illustrious] Adding KSampler comparison setup");
    // Implementation would add multiple KSamplers for comparison
}

function openHelpDocumentation() {
    // Focus on the help panel tab if it exists
    const helpTab = document.querySelector('[data-tab-id="illustrious-help"]');
    if (helpTab) {
        helpTab.click();
    } else {
        console.log("[Illustrious] Help panel not found - opening in browser");
        window.open("https://github.com/itsjustregi/ComfyUI-EasyIllustrious", "_blank");
    }
}

function resetAllIllustriousNodes() {
    console.log("[Illustrious] Resetting all Illustrious nodes to defaults");
    
    const graph = window.app?.graph;
    const illustriousNodes = graph.nodes.filter(node => 
        node.comfyClass && node.comfyClass.includes("Illustrious")
    );
    
    illustriousNodes.forEach(node => {
        if (node.comfyClass.includes("KSampler")) {
            resetKSamplerToOptimal(node);
        }
    });
}

function exportIllustriousSettings() {
    console.log("[Illustrious] Exporting settings");
    
    const graph = window.app?.graph;
    const illustriousNodes = graph.nodes.filter(node => 
        node.comfyClass && node.comfyClass.includes("Illustrious")
    );
    
    const settings = illustriousNodes.map(node => ({
        type: node.comfyClass,
        title: node.title,
        widgets: node.widgets?.map(w => ({ name: w.name, value: w.value })) || []
    }));
    
    const blob = new Blob([JSON.stringify(settings, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'illustrious-settings.json';
    a.click();
    URL.revokeObjectURL(url);
}

// Legacy basic reset kept for backwards-compatibility; renamed to avoid duplicate identifier
function resetKSamplerToOptimalLegacy(node) {
    console.log(`[Illustrious] Resetting ${node.comfyClass} to optimal settings`);
    
    const optimalSettings = {
        "cfg": 5.0,
        "steps": 24,
        "sampler_name": "euler_ancestral",
        "scheduler": "normal",
        "denoise": 1.0
    };
    
    node.widgets?.forEach(widget => {
        if (optimalSettings[widget.name] !== undefined) {
            widget.value = optimalSettings[widget.name];
            if (widget.callback) {
                widget.callback(widget.value);
            }
        }
    });
    
    console.log("[Illustrious] Reset complete");
}

function showParameterGuide(node) {
    console.log(`[Illustrious] Showing parameter guide for ${node.comfyClass}`);
    
    // Focus on help panel and switch to KSampler section
    const helpTab = document.querySelector('[data-tab-id="illustrious-help"]');
    if (helpTab) {
        helpTab.click();
        setTimeout(() => {
            const ksamplerBtn = document.querySelector('[data-section="ksampler"]');
            if (ksamplerBtn) {
                ksamplerBtn.click();
            }
        }, 100);
    }
}

function showQuickSetupWizard(node) {
    console.log("[Illustrious] Opening quick setup wizard for main prompt node");
    
    // This could show a modal with common setups
    alert("Quick Setup Wizard\n\nChoose a preset:\n\n1. Anime Character Portrait\n2. Realistic Character\n3. Landscape/Background\n4. Character Mixing\n\n(Full implementation would show proper UI)");
}

function showPopularSuggestions(node, nodeType) {
    console.log(`[Illustrious] Showing popular suggestions for ${nodeType}`);
    
    const suggestions = {
        "Characters": ["hatsune_miku", "saber", "rem_(re:zero)", "zero_two_(darling_in_the_franxx)", "nezuko_kamado"],
        "Artists": ["wlop", "artgerm", "kantoku", "makoto_shinkai", "greg_rutkowski"]
    };
    
    const type = nodeType.includes("Characters") ? "Characters" : "Artists";
    const popularItems = suggestions[type] || [];
    
    console.log(`[Illustrious] Popular ${type}:`, popularItems);
    
    // This could show a popup with clickable suggestions
    alert(`Popular ${type}:\n\n${popularItems.join('\n')}\n\n(Click to add to node - full implementation would provide clickable interface)`);
}

function openHelpForNode(nodeType) {
    console.log(`[Illustrious] Opening help for ${nodeType}`);
    
    // Map node types to help sections
    const helpSections = {
    "IllustriousKSamplerPro": "ksampler",
    "IllustriousKSamplerPresets": "ksampler",
    "IllustriousPrompt": "prompt-nodes",
    "IllustriousCharacters": "prompt-nodes",
    "IllustriousArtists": "prompt-nodes",
    "IllustriousHairstyles": "prompt-nodes",
    "IllustriousClothing": "prompt-nodes",
    "IllustriousPoses": "prompt-nodes"
    };
    
    const section = helpSections[nodeType] || "overview";
    
    // Focus help panel and navigate to appropriate section
    const helpTab = document.querySelector('[data-tab-id="illustrious-help"]');
    if (helpTab) {
        helpTab.click();
        setTimeout(() => {
            const sectionBtn = document.querySelector(`[data-section="${section}"]`);
            if (sectionBtn) {
                sectionBtn.click();
            }
        }, 100);
    }
}

// ==========================================================================
// --- Illustrious KSampler UI Enhancements ---
// ==========================================================================

// Defer KSampler UI registration similarly
(function registerIllustriousKSampler(){
    if (window.__illustrious_ui_ksampler_registered) return;
    const reg = __getRegisterExtension();
    if (!window.app || !reg) { setTimeout(registerIllustriousKSampler, 50); return; }
    window.__illustrious_ui_ksampler_registered = true;
    reg.fn.call(reg.ctx, {
    name: "Illustrious.KSamplerUI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const illustriousNodes = [
            "IllustriousKSamplerPro",
            "IllustriousKSamplerPresets"
        ];

        // Attach enhancement setup
        if (illustriousNodes.includes(nodeData.name)) {
            setupIllustriousKSamplerEnhancements.call(this, nodeData.name);
        }
    },

    async nodeCreated(node) {
        const illustriousNodes = [
            "IllustriousKSamplerPro",
            "IllustriousKSamplerPresets"
        ];

        if (illustriousNodes.includes(node?.comfyClass)) {
            console.log(`[IllustriousUI] ${node.comfyClass} created with UI enhancements`);
        }
    }
    });
})();

// Safety fallback: if registration is delayed, still attach search enhancements once app.graph is ready
(function __illustrious_fallback_attach(){
    let attempts = 0;
    function tick() {
        attempts++;
        const app = window.app;
        if (app && app.graph && !window.__illustrious_fallback_attached) {
            window.__illustrious_fallback_attached = true;
            try {
                console.log('[Illustrious] Fallback attach: scanning graph for search widgets');
                applySearchToExistingNodes();
                const orig = app.canvas && app.canvas.onNodeAdded;
                if (app.canvas) {
                    app.canvas.onNodeAdded = function(node){
                        if (orig) orig.call(this, node);
                        try {
                            const relevant = [
                                'IllustriousPrompt','IllustriousCharacters','IllustriousArtists','IllustriousE621Characters','IllustriousE621Artists'
                            ];
                            if (node && relevant.includes(node.comfyClass)) {
                                node.widgets?.forEach(w => {
                                    const names = ['Character','Artist','E621 Character','E621 Artist','Character (Base)','Character (Secondary)','Character (Tertiary)','Character (Quaternary)','Character (Quinary)','Artist (Base)','Artist (Secondary)','Artist (Tertiary)','Artist (Quaternary)','Artist (Quinary)','Artist (Primary)'];
                                    if (names.includes(w?.name) && (w.type === 'text' || w.type === 'STRING') && !w._hasSearchEnhancement) {
                                        setupSearchSuggestions.call(node, w);
                                        console.log(`[Illustrious] Fallback attached → ${node.comfyClass} :: ${w.name}`);
                                    }
                                });
                            }
                        } catch(e) { console.warn('[Illustrious] Fallback onNodeAdded error:', e?.message||e); }
                    };
                }
            } catch(e){ console.warn('[Illustrious] Fallback attach error:', e?.message||e); }
        }
        if (!window.__illustrious_ui_main_registered && attempts < 200) setTimeout(tick, 200);
    }
    setTimeout(tick, 200);
})();

// --- Setup function for Illustrious enhancements ---
function setupIllustriousKSamplerEnhancements(nodeType) {
    // Add tooltip handling
    this.onNodeWidgetAdded = function (node, widget) {
        addIllustriousKSamplerTooltips(widget, nodeType);
    };

    // Add context menu button for optimal reset
    const origGetMenuOptions = this.getMenuOptions;
    this.getMenuOptions = function () {
        const options = origGetMenuOptions ? origGetMenuOptions.apply(this, arguments) : [];
        options.push({
            content: "Reset to Optimal (Illustrious)",
            callback: () => resetKSamplerToOptimal(this)
        });
        return options;
    };
}

// --- Illustrious Tooltip maps ---
function addIllustriousKSamplerTooltips(widget, nodeType) {
    const coreTooltips = {
        "cfg": "Classifier-Free Guidance. Higher = more adherence to prompt, but can oversaturate or oversharpen. Illustrious sweet spot: 4.8–5.4",
        "steps": "Number of denoising steps. More = finer detail, slower. Illustrious sweet spot: 24–32",
        "sampler_name": "Algorithm for stepping through the denoising process. Euler Ancestral & DPM++ 2S a are great choices.",
        "scheduler": "Controls sigma scheduling curve. Karras = smoother transitions and better contrast retention.",
        "denoise": "Strength of noise application. 1.0 = full generation. Lower values blend with the input latent."
    };

    const advancedTooltips = {
        "Model Version": "Choose specific Illustrious model tuning or let Auto Detect adjust for resolution.",
        "Resolution Adaptive": "Auto-tune steps & CFG based on resolution to preserve quality at large sizes.",
        "Auto CFG": "Gently adjusts CFG to keep contrast & saturation in the optimal range.",
        "Color Safe Mode": "Bypass automation to use exactly your values; ideal for A/B testing.",
        "Seed Offset": "Offsets the seed for quick variations without changing base seed.",
        "Advanced Logs": "Log sampler settings and detected parameters for debugging.",
        "Resolution Mode": "Override auto detection to force specific Illustrious tuning tier.",
        "Ancestral Eta": "Noise scaling for ancestral samplers. Lower = steadier, higher = wilder."
    };

    if (coreTooltips[widget.name]) {
        widget.tooltip = coreTooltips[widget.name];
    } else if (advancedTooltips[widget.name]) {
        widget.tooltip = advancedTooltips[widget.name];
    }
}

// --- Reset to Optimal for Illustrious ---
function resetKSamplerToOptimal(node) {
    console.log(`[IllustriousUI] Resetting ${node.comfyClass} to optimal Illustrious settings`);
    const optimalSettings = {
        cfg: 5.2,
        steps: 26,
        sampler_name: "euler_ancestral",
        scheduler: "karras",
        denoise: 1.0
    };

    node.widgets?.forEach(w => {
        if (optimalSettings[w.name] !== undefined) {
            w.value = optimalSettings[w.name];
            w.callback?.(w.value);
        }
    });

    console.log("[IllustriousUI] Reset complete");
}
