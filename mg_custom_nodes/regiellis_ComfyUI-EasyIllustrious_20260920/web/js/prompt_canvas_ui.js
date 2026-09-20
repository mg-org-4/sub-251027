// Lightweight canvas UI for IllustriousPrompt inspired by ResolutionMaster
// Non-invasive: draws a compact control bar and mirrors existing widgets
// No overlay, no global patches, keeps ComfyUI stable

// Helper: resolve the proper registerExtension across frontend versions
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

// Defer registration until window.app is ready
(function registerPromptCanvasUI(){
    if (window.__illustrious_prompt_canvas_ui_registered) return;
    const reg = __getRegisterExtension();
    if (!window.app || !reg) {
        setTimeout(registerPromptCanvasUI, 50);
        return;
    }
    window.__illustrious_prompt_canvas_ui_registered = true;

    class PromptCanvasUI {
        constructor(node) {
            this.node = node;
            this.controls = {}; // name -> {x,y,w,h}
            this.hover = null;

            // cache relevant widgets by name
            this.w = {};
            this._resolveWidgets();

            // chain handlers without breaking existing ones
            this._patchHandlers();
        }

        _resolveWidgets() {
            const names = [
                'SFW', 'Format Tag', 'Break Format', 'Quality Boost (Beta)', 'Cinematic (Beta)',
                'Quality Style', 'Quality Variation', 'Add Chain Insert'
            ];
            const widgets = this.node.widgets || [];
            names.forEach(n => {
                this.w[n] = widgets.find(w => w && w.name === n);
            });
        }

        _patchHandlers() {
            const n = this.node; const ui = this;
            const origDraw = n.onDrawForeground;
            n.onDrawForeground = function(ctx) {
                // call original first to keep existing visuals, then draw our bar on top
                if (origDraw) origDraw.call(this, ctx);
                if (this.flags?.collapsed) return;
                try { ui.draw(ctx); } catch(e) { /* no-op */ }
            };

            const origMD = n.onMouseDown;
            n.onMouseDown = function(e, pos, canvas) {
                try { if (ui.mouseDown(e, pos, canvas)) return true; } catch {}
                return origMD ? origMD.call(this, e, pos, canvas) : false;
            };

            const origMM = n.onMouseMove;
            n.onMouseMove = function(e, pos, canvas) {
                try { if (ui.mouseMove(e, pos, canvas)) return true; } catch {}
                return origMM ? origMM.call(this, e, pos, canvas) : false;
            };

            const origMU = n.onMouseUp;
            n.onMouseUp = function(e, pos, canvas) {
                try { if (ui.mouseUp(e, pos, canvas)) return true; } catch {}
                return origMU ? origMU.call(this, e, pos, canvas) : false;
            };
        }

        // Drawing primitives
        _pill(ctx, x, y, w, h, active, hover) {
            ctx.save();
            const r = Math.min(12, h/2);
            const grad = ctx.createLinearGradient(x, y, x, y+h);
            if (active) {
                grad.addColorStop(0, hover ? '#4f7ad8' : '#3f6ac8');
                grad.addColorStop(1, hover ? '#3c66c4' : '#355cab');
            } else {
                grad.addColorStop(0, hover ? '#585858' : '#505050');
                grad.addColorStop(1, hover ? '#4a4a4a' : '#444');
            }
            ctx.fillStyle = grad;
            ctx.strokeStyle = hover ? 'rgba(255,255,255,0.25)' : 'rgba(255,255,255,0.15)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.roundRect(x, y, w, h, r);
            ctx.fill();
            ctx.stroke();
            ctx.restore();
        }

        _text(ctx, text, x, y, color = '#ddd', size = 11, align = 'center') {
            ctx.fillStyle = color;
            ctx.font = `bold ${size}px Arial`;
            ctx.textAlign = align;
            ctx.textBaseline = 'middle';
            ctx.fillText(text, x, y);
        }

        _measure(ctx, text, padding = 14) {
            ctx.font = 'bold 11px Arial';
            const w = ctx.measureText(text).width + padding;
            return Math.max(34, Math.min(160, w));
        }

        draw(ctx) {
            // top bar layout
            const x0 = 10;
            const y0 = (window.LiteGraph?.NODE_TITLE_HEIGHT || 24) + 6;
            const h = 22;
            const gap = 6;
            let x = x0;

            this.controls = {};

            // Toggles (mirror boolean widgets)
            const toggles = [
                { key: 'SFW', label: 'SFW' },
                { key: 'Format Tag', label: 'Format' },
                { key: 'Break Format', label: 'Break' },
                { key: 'Quality Boost (Beta)', label: 'QB' },
                { key: 'Cinematic (Beta)', label: 'Cinema' },
                { key: 'Add Chain Insert', label: 'Chain+' },
            ];

            ctx.save();
            for (const t of toggles) {
                const w = this._measure(ctx, t.label);
                const active = !!this.w[t.key]?.value;
                const hover = this.hover === t.key;
                this._pill(ctx, x, y0, w, h, active, hover);
                this._text(ctx, t.label, x + w/2, y0 + h/2, active ? '#fff' : (hover ? '#eee' : '#ddd'));
                this.controls[t.key] = { x, y: y0, w, h, type: 'toggle' };
                x += w + gap;
            }

            // Dropdowns for quality selections
            const dd = [
                { key: 'Quality Style', label: 'Style' },
                { key: 'Quality Variation', label: 'Variation' },
            ];

            for (const d of dd) {
                const widget = this.w[d.key];
                const current = widget?.value && widget.value !== '-' ? String(widget.value) : d.label;
                // truncate long text
                const text = current.length > 16 ? current.slice(0, 15) + '…' : current;
                const w = this._measure(ctx, text + ' ▾');
                const hover = this.hover === d.key;
                this._pill(ctx, x, y0, w, h, false, hover);
                this._text(ctx, text + ' ▾', x + w/2, y0 + h/2, hover ? '#fff' : '#ddd');
                this.controls[d.key] = { x, y: y0, w, h, type: 'dropdown' };
                x += w + gap;
            }
            ctx.restore();

            // Ensure bar area is visible
            const needed = y0 + h + 6;
            if (Array.isArray(this.node.size) && this.node.size[1] < needed) {
                this.node.size[1] = needed;
            }
        }

        _rel(e) {
            return { x: e.canvasX - this.node.pos[0], y: e.canvasY - this.node.pos[1] };
        }

        _hit(x, y, c) {
            return c && x >= c.x && x <= c.x + c.w && y >= c.y && y <= c.y + c.h;
        }

        mouseDown(e, _pos, _canvas) {
            const { x, y } = this._rel(e);
            // Early exit if above title or outside node body
            if (y < (window.LiteGraph?.NODE_TITLE_HEIGHT || 24)) return false;
            for (const [name, c] of Object.entries(this.controls)) {
                if (this._hit(x, y, c)) {
                    if (c.type === 'toggle') {
                        const w = this.w[name];
                        if (w && typeof w.value === 'boolean') {
                            w.value = !w.value;
                            if (typeof w.callback === 'function') w.callback(w.value);
                            window.app?.graph?.setDirtyCanvas(true);
                            return true;
                        }
                    } else if (c.type === 'dropdown') {
                        this._openDropdown(name, e);
                        return true;
                    }
                }
            }
            return false;
        }

        mouseMove(e, _pos, _canvas) {
            const { x, y } = this._rel(e);
            let h = null;
            for (const [name, c] of Object.entries(this.controls)) {
                if (this._hit(x, y, c)) { h = name; break; }
            }
            if (h !== this.hover) {
                this.hover = h;
                window.app?.graph?.setDirtyCanvas(true);
            }
            return !!h; // if hovering our bar, consume move so cursor feels responsive
        }

        mouseUp(_e, _pos, _canvas) { return false; }

        _openDropdown(name, e) {
            const w = this.w[name];
            if (!w) return;
            // Try to get value list from widget; fall back to cached from options
            let items = [];
            if (Array.isArray(w.options)) items = w.options;
            else if (w.options && Array.isArray(w.options.values)) items = w.options.values;
            // Last resort: if current value looks like '-', keep just '-'
            if (!items || items.length === 0) items = ['-'];
            new LiteGraph.ContextMenu(items, {
                event: e.originalEvent || e,
                callback: (val) => {
                    if (w) {
                        w.value = val;
                        if (typeof w.callback === 'function') w.callback(val);
                        window.app?.graph?.setDirtyCanvas(true);
                    }
                }
            });
        }
    }

    reg.fn.call(reg.ctx, {
        name: 'itsjustregi.PromptCanvasUI',
        beforeRegisterNodeDef(nodeType, nodeData /*, app */) {
            if (nodeData?.name !== 'IllustriousPrompt') return;
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const res = onNodeCreated?.apply(this, arguments);
                try {
                    this.__promptCanvasUI = new PromptCanvasUI(this);
                } catch (e) {
                    console.warn('[Illustrious] PromptCanvasUI failed to init:', e?.message || e);
                }
                return res;
            };
        }
    });
})();
