// ============================================================================
//  Illustrious KSampler Inline Preview (512x512)
// ----------------------------------------------------------------------------
//  Adds an inline, square preview area to Illustrious KSampler nodes similar
//  to the core ComfyUI KSampler. It listens to execution progress/executed
//  events and renders the latest preview image inline on the node.
// ============================================================================
(function registerIllustriousKSamplerInlinePreview(){
    if (window.__illustrious_ksampler_inline_preview_registered) return;
    const app = window.app;
    const api = window.api;
    if (!app || !api) { setTimeout(registerIllustriousKSamplerInlinePreview, 50); return; }
    window.__illustrious_ksampler_inline_preview_registered = true;

    const KSAMPLER_CLASSES = new Set([
        'IllustriousKSamplerPro',
        'IllustriousKSamplerPresets',
        'IllustriousMultiPassSampler',
        'IllustriousTriplePassSampler'
    ]);

    // Keep a weak map of node -> preview image element
    const nodePreviewImg = new WeakMap();
    const nodeHasPreview = (node) => !!nodePreviewImg.get(node);

    function ensurePreviewArea(node) {
        // Reserve space for a 512x512 preview (scaled to node width if smaller)
        // Use some padding around the preview area
        const PAD = 8;
        const desiredW = Math.max(node.size?.[0] || 320, 540); // ensure width large enough (~512 + padding)
        const desiredH = Math.max(node.size?.[1] || 200, 512 + 90); // extra for widgets/title
        node.size = [desiredW, desiredH];

        // Provide helpers for layout
        node.__getPreviewRect = function() {
            const w = Math.min(512, this.size[0] - PAD * 2);
            const h = w; // square
            // place preview below title and initial widgets
            // Try to detect first widget y offset roughly using LiteGraph constants
            const headerH = (window.LiteGraph?.NODE_TITLE_HEIGHT || 20);
            // leave some room for first few widgets; we draw at about 100px below title
            const topOffset = headerH + 90;
            const x = PAD;
            const y = topOffset;
            return { x, y, w, h };
        };
    }

    function drawPreview(node, ctx) {
        if (!node || !ctx || node.flags?.collapsed) return;
        const img = nodePreviewImg.get(node);
        const { x, y, w, h } = node.__getPreviewRect ? node.__getPreviewRect() : { x: 8, y: 90, w: 256, h: 256 };

        // Background
        ctx.save();
        ctx.fillStyle = '#1f1f1f';
        ctx.strokeStyle = '#444';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.rect(x - 2, y - 2, w + 4, h + 4);
        ctx.fill();
        ctx.stroke();

        if (img && img.complete) {
            // draw centered with cover fit
            const iw = img.naturalWidth || img.width;
            const ih = img.naturalHeight || img.height;
            if (iw && ih) {
                const scale = Math.max(w / iw, h / ih);
                const dw = Math.floor(iw * scale);
                const dh = Math.floor(ih * scale);
                const dx = x + Math.floor((w - dw) / 2);
                const dy = y + Math.floor((h - dh) / 2);
                ctx.imageSmoothingEnabled = true;
                ctx.imageSmoothingQuality = 'high';
                ctx.drawImage(img, dx, dy, dw, dh);
            }
        } else {
            // placeholder
            ctx.fillStyle = '#2a2a2a';
            ctx.fillRect(x, y, w, h);
            ctx.fillStyle = '#7c9cff';
            ctx.font = '13px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('Illustrious KSampler Preview', x + w / 2, y + h / 2 - 10);
            ctx.fillStyle = '#aaa';
            ctx.font = '11px Arial';
            ctx.fillText('Waiting for image…', x + w / 2, y + h / 2 + 12);
        }
        ctx.restore();
    }

    function attachInlinePreview(node) {
        if (!node) return;
        if (node.__illustrious_preview_attached) return;
        node.__illustrious_preview_attached = true;

        // Initial layout reserve
        ensurePreviewArea(node);

        // Override onDrawForeground to render preview
        const originalOnDrawForeground = node.onDrawForeground;
        node.onDrawForeground = function(ctx) {
            originalOnDrawForeground?.call(this, ctx);
            try { drawPreview(this, ctx); } catch(e) { console.warn('[Illustrious] Preview draw error', e); }
        };

        // Expand computeSize slightly to keep min height
        const originalComputeSize = node.computeSize;
        node.computeSize = function() {
            const res = originalComputeSize?.apply(this, arguments);
            const rect = this.__getPreviewRect?.();
            if (rect) {
                const minH = rect.y + rect.h + 12;
                if (this.size[1] < minH) this.size[1] = minH;
                if (this.size[0] < 540) this.size[0] = 540;
            }
            return res;
        };

        // Helper to set preview from various payloads
        node.__setPreviewFromDetail = function(detail) {
            if (!detail) return;
            // 1) Direct base64 (preview_image)
            if (detail.preview_image && typeof detail.preview_image === 'string') {
                const img = new Image();
                img.onload = () => { nodePreviewImg.set(this, img); this.setDirtyCanvas(true, true); };
                img.src = detail.preview_image.startsWith('data:') ? detail.preview_image : `data:image/png;base64,${detail.preview_image}`;
                return;
            }
            // 1b) Direct base64 (preview)
            if (detail.preview && typeof detail.preview === 'string') {
                const img = new Image();
                img.onload = () => { nodePreviewImg.set(this, img); this.setDirtyCanvas(true, true); };
                img.src = detail.preview.startsWith('data:') ? detail.preview : `data:image/png;base64,${detail.preview}`;
                return;
            }
            // 2) detail.images[0].url or build from filename
            const imgObj = (detail.images && detail.images[0]) || (Array.isArray(detail.preview) && detail.preview[0]) || null;
            if (imgObj) {
                let url = imgObj.url;
                if (!url && (imgObj.filename || imgObj.name)) {
                    const filename = imgObj.filename || imgObj.name;
                    const subfolder = imgObj.subfolder || '';
                    const type = imgObj.type || 'output';
                    const base = (api && api.apiURL) ? api.apiURL : '';
                    const params = new URLSearchParams({ filename });
                    if (subfolder) params.set('subfolder', subfolder);
                    if (type) params.set('type', type);
                    url = `${base}/view?${params.toString()}`;
                }
                if (url) {
                    const img = new Image();
                    img.crossOrigin = 'anonymous';
                    img.onload = () => { nodePreviewImg.set(this, img); this.setDirtyCanvas(true, true); };
                    img.src = url;
                    return;
                }
            }
        };

    // Also try to pick up final images from onExecuted if provided
        const originalOnExecuted = node.onExecuted;
        node.onExecuted = function(message) {
            const r = originalOnExecuted?.apply(this, arguments);
            try { this.__setPreviewFromDetail(message); } catch {}
            this.setDirtyCanvas(true, true);
            return r;
        };
    }

    // Register as an extension to hook KSampler nodes as they are defined
    const reg = (function() {
        const app = window.app;
        if (!app) return null;
        if (typeof app.registerExtension === 'function') return { fn: app.registerExtension, ctx: app };
        const mgr = app.extensionManager;
        if (mgr && typeof mgr.registerExtension === 'function') return { fn: mgr.registerExtension, ctx: mgr };
        return null;
    })();

    if (!reg) return;

    reg.fn.call(reg.ctx, {
        name: 'itsjustregi.KSamplerInlinePreview',
        async beforeRegisterNodeDef(nodeType, nodeData/*, app*/){
            if (!nodeData || !nodeData.name || !KSAMPLER_CLASSES.has(nodeData.name)) return;
            const origCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const res = origCreated?.apply(this, arguments);
                attachInlinePreview(this);
                return res;
            };
        },
        async loadedGraphNode(node) {
            if (node && node.comfyClass && KSAMPLER_CLASSES.has(node.comfyClass)) {
                attachInlinePreview(node);
            }
        },
        async setup(){
            // Listen to progress-like events and route preview to nodes
            const tryAttach = (detail) => {
                if (!detail) return;
                const nodeId = detail.node || detail.node_id || detail.id;
                if (nodeId == null) return;
                const node = app?.graph?.getNodeById(nodeId);
                if (!node || !KSAMPLER_CLASSES.has(node.comfyClass)) return;
                if (!node.__illustrious_preview_attached) attachInlinePreview(node);
                try { node.__setPreviewFromDetail(detail); } catch {}
                node.setDirtyCanvas(true, true);
            };

            // Core ComfyUI events
            api?.addEventListener && api.addEventListener('progress', ({ detail }) => tryAttach(detail));
            api?.addEventListener && api.addEventListener('executed', ({ detail }) => {
                // First, try direct attach (if executed node is a KSampler variant in some forks)
                tryAttach(detail);
                // Fallback: If the executed node produced images, map to nearest upstream Illustrious KSampler
                try {
                    const executedNodeId = detail.node || detail.node_id || detail.id;
                    const executedNode = app?.graph?.getNodeById(executedNodeId);
                    const hasImages = (detail.images && detail.images.length) || (typeof detail.preview === 'string') || (Array.isArray(detail.preview) && detail.preview.length);
                    if (!executedNode || !hasImages) return;

                    // BFS upstream to find nearest KSampler
                    const visited = new Set();
                    const queue = [executedNode];
                    let found = null;
                    const MAX_STEPS = 50;
                    let steps = 0;
                    while (queue.length && steps++ < MAX_STEPS) {
                        const n = queue.shift();
                        if (!n || visited.has(n.id)) continue;
                        visited.add(n.id);
                        if (n.comfyClass && KSAMPLER_CLASSES.has(n.comfyClass)) { found = n; break; }
                        // enqueue origins of inputs
                        if (n.inputs) {
                            for (const inp of n.inputs) {
                                if (!inp?.link) continue;
                                const link = app?.graph?.links?.[inp.link];
                                if (!link) continue;
                                const origin = app?.graph?.getNodeById(link.origin_id);
                                if (origin && !visited.has(origin.id)) queue.push(origin);
                            }
                        }
                    }
                    if (found) {
                        if (!found.__illustrious_preview_attached) attachInlinePreview(found);
                        try { found.__setPreviewFromDetail(detail); } catch {}
                        found.setDirtyCanvas(true, true);
                    }
                } catch {}
            });
            api?.addEventListener && api.addEventListener('execution_cached', ({ detail }) => tryAttach(detail));
        }
    });

    console.log('🎨 Illustrious KSampler Inline Preview registered');
})();
